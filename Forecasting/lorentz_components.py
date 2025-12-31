# lorentz_components_v4.py

import torch
import torch.nn as nn
import geoopt

# HyperCore doesn't have pre-built high-level layers like HLorentzMLP
# It provides manifolds and utilities, so we build Lorentz-native ops manually

from spec import safe_expmap, safe_expmap0


class LorentzLinear(nn.Module):
    """
    Lorentz-native linear layer using exponential/logarithmic maps.
    Maps Lorentz ? Tangent ? Transform ? Lorentz
    """
    def __init__(self, in_features, out_features, manifold):
        super().__init__()
        self.manifold = manifold
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x_hyp):
        # Map to tangent space at origin
        x_tan = self.manifold.logmap0(x_hyp)
        
        # Apply Euclidean transformation
        y_tan = self.linear(x_tan)
        
        # Map back to manifold
        y_hyp = safe_expmap0(self.manifold, y_tan)
        return self.manifold.projx(y_hyp)


class LorentzMLP(nn.Module):
    """Lorentz MLP using native operations"""
    def __init__(self, in_features, out_features, hidden_dim, manifold, 
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.manifold = manifold
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(LorentzLinear(in_features, hidden_dim, manifold))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            self.layers.append(LorentzLinear(hidden_dim, hidden_dim, manifold))
        
        # Output layer
        self.layers.append(LorentzLinear(hidden_dim, out_features, manifold))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_hyp):
        for i, layer in enumerate(self.layers[:-1]):
            x_hyp = layer(x_hyp)
            # Apply dropout in tangent space
            x_tan = self.manifold.logmap0(x_hyp)
            x_tan = self.dropout(x_tan)
            x_hyp = safe_expmap0(self.manifold, x_tan)
            x_hyp = self.manifold.projx(x_hyp)
        
        # Final layer (no dropout/activation)
        x_hyp = self.layers[-1](x_hyp)
        return x_hyp


class LorentzSegmentEncoder(nn.Module):
    """Lorentz encoder using native Lorentz operations"""
    
    def __init__(self, lookback, input_dim, encode_dim, curvature=1.0,
                 segment_length=24, dropout=0.1, use_segment_norm=True):
        super().__init__()
        
        self.lookback = lookback
        self.input_dim = input_dim
        self.encode_dim = encode_dim
        self.segment_length = segment_length
        self.num_segments = lookback // segment_length
        self.use_segment_norm = use_segment_norm
        
        self.pad_len = 0
        if lookback % segment_length != 0:
            self.num_segments += 1
            self.pad_len = self.num_segments * segment_length - lookback
        
        self.manifold = geoopt.Lorentz(k=curvature)
        
        # Lorentz-native encoders
        input_size = segment_length * input_dim
        
        self.trend_encoder = LorentzMLP(
            input_size, encode_dim, hidden_dim=128, 
            manifold=self.manifold, n_layers=2, dropout=dropout
        )
        self.coarse_encoder = LorentzMLP(
            input_size, encode_dim, hidden_dim=128,
            manifold=self.manifold, n_layers=2, dropout=dropout
        )
        self.fine_encoder = LorentzMLP(
            input_size, encode_dim, hidden_dim=128,
            manifold=self.manifold, n_layers=2, dropout=dropout
        )
        self.residual_encoder = LorentzMLP(
            input_size, encode_dim, hidden_dim=128,
            manifold=self.manifold, n_layers=2, dropout=dropout
        )
        
        self.effective_scale = nn.Parameter(torch.tensor(0.5))
        self.fusion_weights = nn.Parameter(torch.ones(4) * 0.25)
    
    def _encode_component(self, x, encoder):
        """Encode component to Lorentz manifold"""
        B, T, C = x.shape
        
        if self.pad_len > 0:
            pad = x[:, -self.pad_len:, :]
            x = torch.cat([x, pad], dim=1)
        
        x = x.view(B, self.num_segments, self.segment_length, C)
        x_flat = x.reshape(B, self.num_segments, -1)
        
        # Segment normalization
        if self.use_segment_norm:
            mean = x_flat.mean(dim=-1, keepdim=True)
            std = x_flat.std(dim=-1, keepdim=True) + 1e-6
            x_flat = (x_flat - mean) / std
        
        # Encode each segment
        z_list = []
        for i in range(self.num_segments):
            x_euclidean = x_flat[:, i, :]  # [B, input_size]
            
            # Scale and map to Lorentz origin
            scale = torch.tanh(self.effective_scale)
            x_hyp_init = safe_expmap0(self.manifold, scale * x_euclidean)
            x_hyp_init = self.manifold.projx(x_hyp_init)
            
            # Pass through Lorentz encoder
            z_encoded = encoder(x_hyp_init)  # [B, encode_dim+1]
            z_list.append(z_encoded)
        
        z = torch.stack(z_list, dim=1)  # [B, num_segments, encode_dim+1]
        return z
    
    def lorentz_frechet_mean(self, points):
        """Compute Fr chet mean via Karcher flow"""
        B, N, D = points.shape
        
        # Initialize at tangent space mean
        tangents = torch.stack([
            self.manifold.logmap0(points[:, i, :]) for i in range(N)
        ], dim=1)
        mean_tangent = tangents.mean(dim=1)
        current = safe_expmap0(self.manifold, mean_tangent)
        current = self.manifold.projx(current)
        
        # Karcher flow iterations
        for _ in range(10):
            grad = torch.zeros_like(current)
            for i in range(N):
                v = self.manifold.logmap(current, points[: , i, :])
                grad = grad + v
            grad = grad / N
            
            if torch.norm(grad, dim=-1).max() < 1e-5:
                break
            
            current = safe_expmap(self.manifold, current, 0.5 * grad)
            current = self.manifold.projx(current)
        
        return current
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        # Encode each component
        z_trend_h = self._encode_component(trend, self.trend_encoder)
        z_coarse_h = self._encode_component(seasonal_coarse, self.coarse_encoder)
        z_fine_h = self._encode_component(seasonal_fine, self.fine_encoder)
        z_residual_h = self._encode_component(residual, self.residual_encoder)
        
        # Pool each to single point using Fr chet mean
        z_trend_pooled = self.lorentz_frechet_mean(z_trend_h)
        z_coarse_pooled = self.lorentz_frechet_mean(z_coarse_h)
        z_fine_pooled = self.lorentz_frechet_mean(z_fine_h)
        z_residual_pooled = self.lorentz_frechet_mean(z_residual_h)
        
        # Fuse using weighted Fr chet mean
        weights = torch.softmax(self.fusion_weights, dim=0)
        stacked = torch.stack([
            z_trend_pooled, z_coarse_pooled, z_fine_pooled, z_residual_pooled
        ], dim=1)
        
        combined = self.lorentz_frechet_mean(stacked)
        
        # Expand to trajectory
        combined_traj = combined.unsqueeze(1).repeat(1, z_trend_h.shape[1], 1)
        
        return {
            'trend': z_trend_h,
            'seasonal_coarse': z_coarse_h,
            'seasonal_fine': z_fine_h,
            'residual': z_residual_h,
            'combined': combined_traj
        }


class LorentzDynamics(nn.Module):
    """Lorentz dynamics using native operations"""
    
    def __init__(self, encode_dim, manifold, hidden_dim=64, dropout=0.1, n_layers=2):
        super().__init__()
        self.manifold = manifold
        
        # Velocity network (operates in tangent space, maps back to manifold)
        self.velocity_net = LorentzMLP(
            encode_dim, encode_dim, hidden_dim=hidden_dim,
            manifold=manifold, n_layers=n_layers, dropout=dropout
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.7))
        self.step_size = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, z_h, x_previous=None, average_velocity=None):
        # Compute velocity in tangent space
        if average_velocity is not None:
            v_tangent = average_velocity
        elif x_previous is None:
            v_tangent = self.manifold.logmap0(z_h)
        else:
            v_tangent = self.manifold.logmap(x_previous, z_h)
        
        # Map to manifold, transform, map back
        v_hyp = safe_expmap0(self.manifold, v_tangent)
        v_hyp = self.manifold.projx(v_hyp)
        
        v_transformed_hyp = self.velocity_net(v_hyp)
        v_transformed = self.manifold.logmap0(v_transformed_hyp)
        
        step = torch.sigmoid(self.step_size)
        v_final = step * v_transformed
        
        # Extrapolate
        z_update = safe_expmap(self.manifold, z_h, v_final)
        z_update = self.manifold.projx(z_update)
        
        # Geodesic interpolation
        alpha = torch.sigmoid(self.alpha)
        tangent_direction = self.manifold.logmap(z_h, z_update)
        scaled_tangent = (1 - alpha) * tangent_direction
        z_next = safe_expmap(self.manifold, z_h, scaled_tangent)
        z_next = self.manifold.projx(z_next)
        
        return z_next, z_h


class LorentzReconstructor(nn.Module):
    """Lorentz to Euclidean reconstructor"""
    
    def __init__(self, encode_dim, output_dim, segment_length, manifold,
                 hidden_dim=256, dropout=0.1):
        super().__init__()
        self.manifold = manifold
        self.segment_length = segment_length
        self.output_dim = output_dim
        
        # Decode via tangent space projection
        self.decoder = nn.Sequential(
            nn.Linear(encode_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, segment_length * output_dim)
        )
    
    def forward(self, z_h):
        B = z_h.shape[0]
        
        # Project to tangent space at origin
        z_tangent = self.manifold.logmap0(z_h)
        
        # Decode in Euclidean space
        segment_flat = self.decoder(z_tangent)
        segment = segment_flat.reshape(B, self.segment_length, self.output_dim)
        
        return segment


class LorentzForecaster(nn.Module):
    """Channel-DEPENDENT Lorentz forecaster"""
    
    def __init__(self, lookback, pred_len, n_features, encode_dim, hidden_dim,
                 curvature=1.0, segment_length=24, dropout=0.1,
                 use_truncated_bptt=True, truncate_every=4):
        super().__init__()
        
        self.pred_len = pred_len
        self.segment_length = segment_length
        self.num_pred_steps = (pred_len + segment_length - 1) // segment_length
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        
        self.encoder = LorentzSegmentEncoder(
            lookback, n_features, encode_dim, curvature, segment_length, dropout
        )
        self.manifold = self.encoder.manifold
        self.dynamics = LorentzDynamics(encode_dim, self.manifold, hidden_dim, dropout)
        self.reconstructor = LorentzReconstructor(
            encode_dim, n_features, segment_length, self.manifold, hidden_dim, dropout
        )
    
    def forward(self, trend, coarse, fine, residual):
        encoded = self.encoder(trend, coarse, fine, residual)
        
        z_combined = encoded['combined']
        z_trend = encoded['trend']
        z_coarse = encoded['seasonal_coarse']
        z_fine = encoded['seasonal_fine']
        z_residual = encoded['residual']
        
        predictions = []
        trend_predictions = []
        coarse_predictions = []
        fine_predictions = []
        residual_predictions = []
        
        for step_idx in range(self.num_pred_steps):
            if self.use_truncated_bptt and (step_idx + 1) % self.truncate_every == 0:
                if step_idx < self.num_pred_steps - 1:
                    z_combined = z_combined.detach()
                    z_trend = z_trend.detach()
                    z_coarse = z_coarse.detach()
                    z_fine = z_fine.detach()
                    z_residual = z_residual.detach()
            
            # Combined
            z_last = z_combined[:, -1, :]
            z_prev = z_combined[:, -2, : ] if z_combined.shape[1] > 1 else None
            z_next, _ = self.dynamics(z_last, z_prev)
            pred_seg = self.reconstructor(z_next)
            predictions.append(pred_seg)
            z_combined = torch.cat([z_combined[: , 1:, :], z_next.unsqueeze(1)], dim=1)
            
            # Trend
            z_last_trend = z_trend[:, -1, :]
            z_prev_trend = z_trend[: , -2, :] if z_trend.shape[1] > 1 else None
            z_next_trend, _ = self.dynamics(z_last_trend, z_prev_trend)
            trend_pred_seg = self.reconstructor(z_next_trend)
            trend_predictions.append(trend_pred_seg)
            z_trend = torch.cat([z_trend[:, 1:, :], z_next_trend.unsqueeze(1)], dim=1)
            
            # Coarse
            z_last_coarse = z_coarse[:, -1, :]
            z_prev_coarse = z_coarse[:, -2, : ] if z_coarse.shape[1] > 1 else None
            z_next_coarse, _ = self.dynamics(z_last_coarse, z_prev_coarse)
            coarse_pred_seg = self.reconstructor(z_next_coarse)
            coarse_predictions.append(coarse_pred_seg)
            z_coarse = torch.cat([z_coarse[:, 1:, :], z_next_coarse.unsqueeze(1)], dim=1)
            
            # Fine
            z_last_fine = z_fine[:, -1, :]
            z_prev_fine = z_fine[: , -2, :] if z_fine.shape[1] > 1 else None
            z_next_fine, _ = self.dynamics(z_last_fine, z_prev_fine)
            fine_pred_seg = self.reconstructor(z_next_fine)
            fine_predictions.append(fine_pred_seg)
            z_fine = torch.cat([z_fine[:, 1:, :], z_next_fine.unsqueeze(1)], dim=1)
            
            # Residual
            z_last_resid = z_residual[:, -1, :]
            z_prev_resid = z_residual[:, -2, : ] if z_residual.shape[1] > 1 else None
            z_next_resid, _ = self.dynamics(z_last_resid, z_prev_resid)
            resid_pred_seg = self.reconstructor(z_next_resid)
            residual_predictions.append(resid_pred_seg)
            z_residual = torch.cat([z_residual[:, 1:, : ], z_next_resid.unsqueeze(1)], dim=1)
        
        predictions = torch.cat(predictions, dim=1)[:, : self.pred_len, :]
        trend_predictions = torch.cat(trend_predictions, dim=1)[:, :self.pred_len, :]
        coarse_predictions = torch.cat(coarse_predictions, dim=1)[:, :self.pred_len, :]
        fine_predictions = torch.cat(fine_predictions, dim=1)[:, :self.pred_len, :]
        residual_predictions = torch.cat(residual_predictions, dim=1)[:, :self.pred_len, :]
        
        return {
            'predictions': predictions,
            'trend_predictions': trend_predictions,
            'coarse_predictions': coarse_predictions,
            'fine_predictions': fine_predictions,
            'residual_predictions': residual_predictions,
            'hyperbolic_states': {'combined_h': z_combined}
        }


class LorentzMovingWindowForecaster(nn.Module):
    """Channel-INDEPENDENT Lorentz forecaster"""
    
    def __init__(self, lookback, pred_len, n_features, encode_dim, hidden_dim,
                 curvature=1.0, segment_length=24, window_size=5, dropout=0.1,
                 use_truncated_bptt=True, truncate_every=4):
        super().__init__()
        
        self.pred_len = pred_len
        self.segment_length = segment_length
        self.n_features = n_features
        self.window_size = window_size
        self.num_pred_steps = (pred_len + segment_length - 1) // segment_length
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        
        # Channel-independent:  input_dim=1
        self.encoder = LorentzSegmentEncoder(
            lookback, 1, encode_dim, curvature, segment_length, dropout
        )
        self.manifold = self.encoder.manifold
        self.dynamics = LorentzDynamics(encode_dim, self.manifold, hidden_dim, dropout)
        self.reconstructor = LorentzReconstructor(
            encode_dim, 1, segment_length, self.manifold, hidden_dim, dropout
        )
    
    def compute_average_velocity(self, z_current, decay=0.9):
        """Compute exponentially weighted velocity in tangent space"""
        if self.window_size is None:
            return None
        
        B, N, D = z_current.shape
        if N < 2:
            return None
        
        # Compute velocities in tangent space
        velocities = []
        for i in range(N - 1):
            v = self.manifold.logmap(z_current[:, i, : ], z_current[:, i+1, :])
            velocities.append(v)
        
        velocities = torch.stack(velocities, dim=1)
        
        # Exponential weighting
        indices = torch.arange(N-1, dtype=torch.float32, device=z_current.device)
        weights = decay ** (N - 2 - indices)
        weights = weights / (weights.sum() + 1e-8)
        
        weights = weights.view(1, -1, 1)
        avg_velocity = (velocities * weights).sum(dim=1)
        
        return avg_velocity
    
    def forward(self, trend, coarse, fine, residual):
        B, T, C = trend.shape
        
        all_predictions = []
        all_trend_predictions = []
        all_coarse_predictions = []
        all_fine_predictions = []
        all_residual_predictions = []
        
        # Process each channel independently
        for c in range(C):
            trend_c = trend[:, :, c: c+1]
            coarse_c = coarse[:, : , c:c+1]
            fine_c = fine[:, :, c:c+1]
            residual_c = residual[:, :, c:c+1]
            
            encoded = self.encoder(trend_c, coarse_c, fine_c, residual_c)
            
            z_combined = encoded['combined']
            z_trend = encoded['trend']
            z_coarse = encoded['seasonal_coarse']
            z_fine = encoded['seasonal_fine']
            z_residual = encoded['residual']
            
            # Limit window
            if z_combined.shape[1] > self.window_size:
                z_combined = z_combined[: , -self.window_size:, :]
                z_trend = z_trend[: , -self.window_size:, :]
                z_coarse = z_coarse[:, -self.window_size:, :]
                z_fine = z_fine[:, -self.window_size:, :]
                z_residual = z_residual[:, -self.window_size:, :]
            
            predictions = []
            trend_predictions = []
            coarse_predictions = []
            fine_predictions = []
            residual_predictions = []
            
            for step_idx in range(self.num_pred_steps):
                if self.use_truncated_bptt and (step_idx + 1) % self.truncate_every == 0:
                    if step_idx < self.num_pred_steps - 1:
                        z_combined = z_combined.detach()
                        z_trend = z_trend.detach()
                        z_coarse = z_coarse.detach()
                        z_fine = z_fine.detach()
                        z_residual = z_residual.detach()
                
                # Combined with velocity
                avg_vel = self.compute_average_velocity(z_combined)
                z_last = z_combined[:, -1, :]
                z_prev = z_combined[:, -2, :] if z_combined.shape[1] > 1 else None
                z_next, _ = self.dynamics(z_last, z_prev, avg_vel)
                pred_seg = self.reconstructor(z_next)
                predictions.append(pred_seg)
                z_combined = torch.cat([z_combined[:, 1:, :], z_next.unsqueeze(1)], dim=1)
                
                # Trend with velocity
                avg_vel_trend = self.compute_average_velocity(z_trend)
                z_last_trend = z_trend[:, -1, :]
                z_prev_trend = z_trend[:, -2, :] if z_trend.shape[1] > 1 else None
                z_next_trend, _ = self.dynamics(z_last_trend, z_prev_trend, avg_vel_trend)
                trend_pred_seg = self.reconstructor(z_next_trend)
                trend_predictions.append(trend_pred_seg)
                z_trend = torch.cat([z_trend[: , 1:, :], z_next_trend.unsqueeze(1)], dim=1)
                
                # Coarse with velocity
                avg_vel_coarse = self.compute_average_velocity(z_coarse)
                z_last_coarse = z_coarse[:, -1, :]
                z_prev_coarse = z_coarse[:, -2, :] if z_coarse.shape[1] > 1 else None
                z_next_coarse, _ = self.dynamics(z_last_coarse, z_prev_coarse, avg_vel_coarse)
                coarse_pred_seg = self.reconstructor(z_next_coarse)
                coarse_predictions.append(coarse_pred_seg)
                z_coarse = torch.cat([z_coarse[:, 1:, :], z_next_coarse.unsqueeze(1)], dim=1)
                
                # Fine with velocity
                avg_vel_fine = self.compute_average_velocity(z_fine)
                z_last_fine = z_fine[:, -1, :]
                z_prev_fine = z_fine[:, -2, :] if z_fine.shape[1] > 1 else None
                z_next_fine, _ = self.dynamics(z_last_fine, z_prev_fine, avg_vel_fine)
                fine_pred_seg = self.reconstructor(z_next_fine)
                fine_predictions.append(fine_pred_seg)
                z_fine = torch.cat([z_fine[: , 1:, :], z_next_fine.unsqueeze(1)], dim=1)
                
                # Residual with velocity
                avg_vel_resid = self.compute_average_velocity(z_residual)
                z_last_resid = z_residual[:, -1, :]
                z_prev_resid = z_residual[:, -2, :] if z_residual.shape[1] > 1 else None
                z_next_resid, _ = self.dynamics(z_last_resid, z_prev_resid, avg_vel_resid)
                resid_pred_seg = self.reconstructor(z_next_resid)
                residual_predictions.append(resid_pred_seg)
                z_residual = torch.cat([z_residual[: , 1:, :], z_next_resid.unsqueeze(1)], dim=1)
            
            pred_c = torch.cat(predictions, dim=1)[:, :self.pred_len, :]
            trend_c = torch.cat(trend_predictions, dim=1)[:, :self.pred_len, :]
            coarse_c = torch.cat(coarse_predictions, dim=1)[:, :self.pred_len, :]
            fine_c = torch.cat(fine_predictions, dim=1)[:, :self.pred_len, :]
            resid_c = torch.cat(residual_predictions, dim=1)[:, :self.pred_len, :]
            
            all_predictions.append(pred_c)
            all_trend_predictions.append(trend_c)
            all_coarse_predictions.append(coarse_c)
            all_fine_predictions.append(fine_c)
            all_residual_predictions.append(resid_c)
        
        predictions = torch.cat(all_predictions, dim=2)
        trend_predictions = torch.cat(all_trend_predictions, dim=2)
        coarse_predictions = torch.cat(all_coarse_predictions, dim=2)
        fine_predictions = torch.cat(all_fine_predictions, dim=2)
        residual_predictions = torch.cat(all_residual_predictions, dim=2)
        
        return {
            'predictions':  predictions,
            'trend_predictions': trend_predictions,
            'coarse_predictions': coarse_predictions,
            'fine_predictions': fine_predictions,
            'residual_predictions': residual_predictions,
            'hyperbolic_states': {}
        }