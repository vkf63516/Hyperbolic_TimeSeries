import torch
import torch.nn as nn
import geoopt


class PoincareSegmentEncoder(nn.Module):
    """Poincare encoder with proper Mobius operations"""
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
        
        self.manifold = geoopt.PoincareBall(c=curvature)
        
        # Euclidean encoders (map to tangent space first)
        self.trend_encoder = self._build_encoder(dropout)
        self.coarse_encoder = self._build_encoder(dropout)
        self.fine_encoder = self._build_encoder(dropout)
        self.residual_encoder = self._build_encoder(dropout)
        
        self.effective_scale = nn.Parameter(torch.tensor(0.5))
        self.fusion_weights = nn.Parameter(torch.ones(4) * 0.25)
    
    def _build_encoder(self, dropout):
        return nn.Sequential(
            nn.Linear(self.segment_length * self.input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.encode_dim)
        )
    
    def _encode_component(self, x, encoder):
        B, T, C = x.shape
        
        if self.pad_len > 0:
            pad = x[:, -self.pad_len:, :]
            x = torch.cat([x, pad], dim=1)
        
        x = x.view(B, self.num_segments, self.segment_length, C)
        x_flat = x.reshape(B, self.num_segments, -1)
        
        if self.use_segment_norm:
            mean = x_flat.mean(dim=-1, keepdim=True)
            std = x_flat.std(dim=-1, keepdim=True) + 1e-6
            x_flat = (x_flat - mean) / std
        
        z_list = []
        for i in range(self.num_segments):
            z_euclidean = encoder(x_flat[:, i, :])
            # Map to Poincare ball
            scale = torch.tanh(self.effective_scale)
            z_hyp = self.manifold.expmap0(scale * z_euclidean)
            z_hyp = self.manifold.projx(z_hyp)
            z_list.append(z_hyp)
        
        z = torch.stack(z_list, dim=1)
        return z
    
    def mobius_mean(self, points):
        """Compute Frechet mean in Poincare ball"""
        B, N, D = points.shape
        
        # Initialize at tangent space mean
        tangents = torch.stack([
            self.manifold.logmap0(points[:, i, :]) for i in range(N)
        ], dim=1)
        mean_tangent = tangents.mean(dim=1)
        current = self.manifold.expmap0(mean_tangent)
        current = self.manifold.projx(current)
        
        # Karcher flow
        for _ in range(10):
            grad = torch.zeros_like(current)
            for i in range(N):
                v = self.manifold.logmap(current, points[: , i, :])
                grad = grad + v
            grad = grad / N
            
            if torch.norm(grad, dim=-1).max() < 1e-5:
                break
            
            current = self.manifold.expmap(current, 0.5 * grad)
            current = self.manifold.projx(current)
        
        return current
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        z_trend_h = self._encode_component(trend, self.trend_encoder)
        z_coarse_h = self._encode_component(seasonal_coarse, self.coarse_encoder)
        z_fine_h = self._encode_component(seasonal_fine, self.fine_encoder)
        z_residual_h = self._encode_component(residual, self.residual_encoder)
        
        # Pool each component to single point
        z_trend_pooled = self.mobius_mean(z_trend_h)
        z_coarse_pooled = self.mobius_mean(z_coarse_h)
        z_fine_pooled = self.mobius_mean(z_fine_h)
        z_residual_pooled = self.mobius_mean(z_residual_h)
        
        # Fuse using weighted Mobius addition
        weights = torch.softmax(self.fusion_weights, dim=0)
        
        combined = self.manifold.mobius_scalar_mul(weights[0], z_trend_pooled)
        scaled = self.manifold.mobius_scalar_mul(weights[1], z_coarse_pooled)
        combined = self.manifold.mobius_add(combined, scaled)
        scaled = self.manifold.mobius_scalar_mul(weights[2], z_fine_pooled)
        combined = self.manifold.mobius_add(combined, scaled)
        scaled = self.manifold.mobius_scalar_mul(weights[3], z_residual_pooled)
        combined = self.manifold.mobius_add(combined, scaled)
        combined = self.manifold.projx(combined)
        
        # Expand back to trajectory for moving window
        combined_traj = combined.unsqueeze(1).repeat(1, z_trend_h.shape[1], 1)
        
        return {
            'trend': z_trend_h,
            'seasonal_coarse': z_coarse_h,
            'seasonal_fine': z_fine_h,
            'residual': z_residual_h,
            'combined': combined_traj
        }


class PoincareDynamics(nn.Module):
    """Poincare dynamics with Mobius residual update"""
    def __init__(self, encode_dim, manifold, hidden_dim=64, dropout=0.1, n_layers=2):
        super().__init__()
        self.manifold = manifold
        
        # Euclidean velocity network (operates in tangent space)
        layers = []
        layers.append(nn.Linear(encode_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, encode_dim))
        
        self.velocity_net = nn.Sequential(*layers)
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
        
        # Transform velocity
        v_transformed = self.velocity_net(v_tangent)
        step = torch.sigmoid(self.step_size)
        v_final = step * v_transformed
        
        # Extrapolate on manifold
        z_update = self.manifold.expmap(z_h, v_final)
        z_update = self.manifold.projx(z_update)
        
        # Mobius residual update
        alpha = torch.sigmoid(self.alpha)
        alpha_z = self.manifold.mobius_scalar_mul(alpha, z_h)
        beta_z = self.manifold.mobius_scalar_mul(1 - alpha, z_update)
        z_next = self.manifold.mobius_add(alpha_z, beta_z)
        z_next = self.manifold.projx(z_next)
        
        return z_next, z_h


class PoincareReconstructor(nn.Module):
    """Poincare reconstructor"""
    def __init__(self, encode_dim, output_dim, segment_length, manifold,
                 hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.manifold = manifold
        self.segment_length = segment_length
        self.output_dim = output_dim
        
        layers = []
        layers.append(nn.Linear(encode_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, segment_length * output_dim))
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, z_h):
        B = z_h.shape[0]
        # Map to tangent space
        v = self.manifold.logmap0(z_h)
        # Reconstruct
        segment_flat = self.fc(v)
        segment = segment_flat.reshape(B, self.segment_length, self.output_dim)
        return segment


class PoincareForecaster(nn.Module):
    """Channel-DEPENDENT Poincare forecaster (original mode)"""
    def __init__(self, lookback, pred_len, n_features, encode_dim, hidden_dim,
                 curvature=1.0, segment_length=24, dropout=0.1,
                 use_truncated_bptt=True, truncate_every=4):
        super().__init__()
        
        self.pred_len = pred_len
        self.segment_length = segment_length
        self.num_pred_steps = (pred_len + segment_length - 1) // segment_length
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        
        self.encoder = PoincareSegmentEncoder(
            lookback, n_features, encode_dim, curvature, segment_length, dropout
        )
        self.manifold = self.encoder.manifold
        self.dynamics = PoincareDynamics(encode_dim, self.manifold, hidden_dim, dropout)
        self.reconstructor = PoincareReconstructor(
            encode_dim, n_features, segment_length, self.manifold, hidden_dim, dropout=dropout
        )
        self.step_size = nn.Parameter(torch.tensor(0.1))
    
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
        
        step_size = torch.sigmoid(self.step_size)
        
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
            z_combined = torch.cat([z_combined[:, 1:, :], z_next.unsqueeze(1)], dim=1)
            
            # Trend
            z_last_trend = z_trend[:, -1, :]
            z_prev_trend = z_trend[:, -2, :] if z_trend.shape[1] > 1 else None
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
            z_residual = torch.cat([z_residual[:, 1:, :], z_next_resid.unsqueeze(1)], dim=1)
        
        predictions = torch.cat(predictions, dim=1)[:, :self.pred_len, :]
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
            'hyperbolic_states':  {'combined_h': z_combined}
        }


class PoincareMovingWindowForecaster(nn.Module):
    """Channel-INDEPENDENT Poincare forecaster (moving window mode)"""
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
        self.encoder = PoincareSegmentEncoder(
            lookback, 1, encode_dim, curvature, segment_length, dropout
        )
        self.manifold = self.encoder.manifold
        self.dynamics = PoincareDynamics(encode_dim, self.manifold, hidden_dim, dropout)
        self.reconstructor = PoincareReconstructor(
            encode_dim, 1, segment_length, self.manifold, hidden_dim, dropout=dropout
        )
        self.step_size = nn.Parameter(torch.tensor(0.1))
    
    def compute_average_velocity(self, z_current, decay=0.9):
        """Compute velocity in tangent space"""
        if self.window_size is None:
            return None
        
        B, N, D = z_current.shape
        if N < 2:
            return None
        
        velocities = []
        for i in range(N - 1):
            v = self.manifold.logmap(z_current[:, i, : ], z_current[:, i+1, :])
            velocities.append(v)
        
        velocities = torch.stack(velocities, dim=1)
        
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
        
        for c in range(C):
            trend_c = trend[:, : , c: c+1]
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
                
                # Combined
                avg_vel = self.compute_average_velocity(z_combined)
                z_last = z_combined[:, -1, :]
                z_prev = z_combined[:, -2, :] if z_combined.shape[1] > 1 else None
                z_next, _ = self.dynamics(z_last, z_prev, avg_vel)
                pred_seg = self.reconstructor(z_next)
                predictions.append(pred_seg)
                z_combined = torch.cat([z_combined[:, 1:, :], z_next.unsqueeze(1)], dim=1)
                
                # Trend
                avg_vel_trend = self.compute_average_velocity(z_trend)
                z_last_trend = z_trend[:, -1, :]
                z_prev_trend = z_trend[:, -2, :] if z_trend.shape[1] > 1 else None
                z_next_trend, _ = self.dynamics(z_last_trend, z_prev_trend, avg_vel_trend)
                trend_pred_seg = self.reconstructor(z_next_trend)
                trend_predictions.append(trend_pred_seg)
                z_trend = torch.cat([z_trend[: , 1:, :], z_next_trend.unsqueeze(1)], dim=1)
                
                # Coarse
                avg_vel_coarse = self.compute_average_velocity(z_coarse)
                z_last_coarse = z_coarse[:, -1, :]
                z_prev_coarse = z_coarse[: , -2, :] if z_coarse.shape[1] > 1 else None
                z_next_coarse, _ = self.dynamics(z_last_coarse, z_prev_coarse, avg_vel_coarse)
                coarse_pred_seg = self.reconstructor(z_next_coarse)
                coarse_predictions.append(coarse_pred_seg)
                z_coarse = torch.cat([z_coarse[:, 1:, :], z_next_coarse.unsqueeze(1)], dim=1)
                
                # Fine
                avg_vel_fine = self.compute_average_velocity(z_fine)
                z_last_fine = z_fine[:, -1, :]
                z_prev_fine = z_fine[: , -2, :] if z_fine.shape[1] > 1 else None
                z_next_fine, _ = self.dynamics(z_last_fine, z_prev_fine, avg_vel_fine)
                fine_pred_seg = self.reconstructor(z_next_fine)
                fine_predictions.append(fine_pred_seg)
                z_fine = torch.cat([z_fine[:, 1:, :], z_next_fine.unsqueeze(1)], dim=1)
                
                # Residual
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