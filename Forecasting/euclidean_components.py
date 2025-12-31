import torch
import torch.nn as nn


class EuclideanSegmentEncoder(nn.Module):
    """Standard MLP encoder for Euclidean space."""
    def __init__(self, lookback, input_dim, encode_dim, segment_length=24, dropout=0.1, use_segment_norm=True):
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
        
        self.trend_encoder = self._build_encoder(dropout)
        self.coarse_encoder = self._build_encoder(dropout)
        self.fine_encoder = self._build_encoder(dropout)
        self.residual_encoder = self._build_encoder(dropout)
        
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
            pad = x[: , -self.pad_len:, :]
            x = torch.cat([x, pad], dim=1)
        
        x = x.view(B, self.num_segments, self.segment_length, C)
        x_flat = x.reshape(B, self.num_segments, -1)
        
        # ? ADD:  Segment normalization
        if self.use_segment_norm:
            mean = x_flat.mean(dim=-1, keepdim=True)
            std = x_flat.std(dim=-1, keepdim=True) + 1e-6
            x_flat = (x_flat - mean) / std
        
        z_list = [encoder(x_flat[:, i, :]) for i in range(self.num_segments)]
        z = torch.stack(z_list, dim=1)  # ? RETURN TRAJECTORY
        
        return z
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        z_trend = self._encode_component(trend, self.trend_encoder)
        z_coarse = self._encode_component(seasonal_coarse, self.coarse_encoder)
        z_fine = self._encode_component(seasonal_fine, self.fine_encoder)
        z_residual = self._encode_component(residual, self.residual_encoder)
        
        weights = torch.softmax(self.fusion_weights, dim=0)
        z_combined = (weights[0] * z_trend + weights[1] * z_coarse +
                     weights[2] * z_fine + weights[3] * z_residual)
        
        return {
            'trend': z_trend,
            'seasonal_coarse': z_coarse,
            'seasonal_fine': z_fine,
            'residual':  z_residual,
            'combined': z_combined
        }


class EuclideanDynamics(nn.Module):
    """Residual dynamics in Euclidean space."""
    def __init__(self, encode_dim, hidden_dim=64, dropout=0.1, n_layers=2):
        super().__init__()
        
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
        
        self.residual_net = nn.Sequential(*layers)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, z, x_previous=None, average_velocity=None):
        if average_velocity is not None: 
            backward_trajectory = average_velocity
        else:  
            backward_trajectory = z
        
        residual = self.residual_net(backward_trajectory)
        return backward_trajectory + self.residual_weight * residual, z


class EuclideanReconstructor(nn.Module):
    """Euclidean reconstructor - outputs [B, segment_length, output_dim]"""
    def __init__(self, encode_dim, output_dim, segment_length, 
                 hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
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
    
    def forward(self, z):
        B = z.shape[0]
        segment_flat = self.fc(z)
        segment = segment_flat.reshape(B, self.segment_length, self.output_dim)
        return segment


class EuclideanForecaster(nn.Module):
    """Channel-DEPENDENT Euclidean forecaster (original mode)."""
    def __init__(self, lookback, pred_len, n_features, encode_dim, hidden_dim,
                 segment_length=24, dropout=0.1, use_truncated_bptt=True, truncate_every=4):
        super().__init__()
        
        self.pred_len = pred_len
        self.segment_length = segment_length
        self.num_pred_steps = (pred_len + segment_length - 1) // segment_length
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        
        self.encoder = EuclideanSegmentEncoder(
            lookback, n_features, encode_dim, segment_length, dropout
        )
        self.dynamics = EuclideanDynamics(encode_dim, hidden_dim, dropout)
        self.reconstructor = EuclideanReconstructor(
            encode_dim, n_features, segment_length, hidden_dim, dropout=dropout
        )
        self.step_size = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, trend, coarse, fine, residual):
        encoded = self.encoder(trend, coarse, fine, residual)
        
        # ? PROCESS ALL COMPONENTS SEPARATELY
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
            # ? Truncated BPTT
            if self.use_truncated_bptt and (step_idx + 1) % self.truncate_every == 0:
                if step_idx < self.num_pred_steps - 1:
                    z_combined = z_combined.detach()
                    z_trend = z_trend.detach()
                    z_coarse = z_coarse.detach()
                    z_fine = z_fine.detach()
                    z_residual = z_residual.detach()
            
            # Combined
            z_last = z_combined[:, -1, :]
            z_next_raw, _ = self.dynamics(z_last)
            z_next = z_last + step_size * (z_next_raw - z_last)
            pred_seg = self.reconstructor(z_next)
            predictions.append(pred_seg)
            z_combined = torch.cat([z_combined[: , 1:, :], z_next.unsqueeze(1)], dim=1)
            
            # Trend
            z_last_trend = z_trend[:, -1, :]
            z_next_trend_raw, _ = self.dynamics(z_last_trend)
            z_next_trend = z_last_trend + step_size * (z_next_trend_raw - z_last_trend)
            trend_pred_seg = self.reconstructor(z_next_trend)
            trend_predictions.append(trend_pred_seg)
            z_trend = torch.cat([z_trend[:, 1:, :], z_next_trend.unsqueeze(1)], dim=1)
            
            # Coarse
            z_last_coarse = z_coarse[:, -1, :]
            z_next_coarse_raw, _ = self.dynamics(z_last_coarse)
            z_next_coarse = z_last_coarse + step_size * (z_next_coarse_raw - z_last_coarse)
            coarse_pred_seg = self.reconstructor(z_next_coarse)
            coarse_predictions.append(coarse_pred_seg)
            z_coarse = torch.cat([z_coarse[:, 1:, :], z_next_coarse.unsqueeze(1)], dim=1)
            
            # Fine
            z_last_fine = z_fine[:, -1, :]
            z_next_fine_raw, _ = self.dynamics(z_last_fine)
            z_next_fine = z_last_fine + step_size * (z_next_fine_raw - z_last_fine)
            fine_pred_seg = self.reconstructor(z_next_fine)
            fine_predictions.append(fine_pred_seg)
            z_fine = torch.cat([z_fine[:, 1:, :], z_next_fine.unsqueeze(1)], dim=1)
            
            # Residual
            z_last_resid = z_residual[:, -1, :]
            z_next_resid_raw, _ = self.dynamics(z_last_resid)
            z_next_resid = z_last_resid + step_size * (z_next_resid_raw - z_last_resid)
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
            'hyperbolic_states': {}
        }


class EuclideanMovingWindowForecaster(nn.Module):
    """Channel-INDEPENDENT Euclidean forecaster (moving window mode)."""
    def __init__(self, lookback, pred_len, n_features, encode_dim, hidden_dim,
                 segment_length=24, window_size=5, dropout=0.1, 
                 use_truncated_bptt=True, truncate_every=4):
        super().__init__()
        
        self.pred_len = pred_len
        self.segment_length = segment_length
        self.n_features = n_features
        self.window_size = window_size
        self.num_pred_steps = (pred_len + segment_length - 1) // segment_length
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        
        # ? CHANNEL-INDEPENDENT:  input_dim=1
        self.encoder = EuclideanSegmentEncoder(
            lookback, 1, encode_dim, segment_length, dropout
        )
        self.dynamics = EuclideanDynamics(encode_dim, hidden_dim, dropout)
        self.reconstructor = EuclideanReconstructor(
            encode_dim, 1, segment_length, hidden_dim, dropout=dropout
        )
        self.step_size = nn.Parameter(torch.tensor(0.1))
    
    def compute_average_velocity(self, z_current, decay=0.9):
        """Compute exponentially weighted velocity"""
        if self.window_size is None:
            return None
        
        B, N, D = z_current.shape
        if N < 2:
            return None
        
        velocities = []
        for i in range(N - 1):
            v = z_current[:, i+1, : ] - z_current[:, i, :]
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
        
        # ? CHANNEL-INDEPENDENT: Process each feature separately
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
                z_trend = z_trend[:, -self.window_size:, :]
                z_coarse = z_coarse[:, -self.window_size:, :]
                z_fine = z_fine[:, -self.window_size:, :]
                z_residual = z_residual[: , -self.window_size:, :]
            
            predictions = []
            trend_predictions = []
            coarse_predictions = []
            fine_predictions = []
            residual_predictions = []
            
            step_size = torch.sigmoid(self.step_size)
            
            for step_idx in range(self.num_pred_steps):
                # ? Truncated BPTT
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
                z_next_raw, _ = self.dynamics(z_last, average_velocity=avg_vel)
                z_next = z_last + step_size * (z_next_raw - z_last)
                pred_seg = self.reconstructor(z_next)
                predictions.append(pred_seg)
                z_combined = torch.cat([z_combined[:, 1:, :], z_next.unsqueeze(1)], dim=1)
                
                # Trend
                avg_vel_trend = self.compute_average_velocity(z_trend)
                z_last_trend = z_trend[:, -1, :]
                z_next_trend_raw, _ = self.dynamics(z_last_trend, average_velocity=avg_vel_trend)
                z_next_trend = z_last_trend + step_size * (z_next_trend_raw - z_last_trend)
                trend_pred_seg = self.reconstructor(z_next_trend)
                trend_predictions.append(trend_pred_seg)
                z_trend = torch.cat([z_trend[: , 1:, :], z_next_trend.unsqueeze(1)], dim=1)
                
                # Coarse
                avg_vel_coarse = self.compute_average_velocity(z_coarse)
                z_last_coarse = z_coarse[:, -1, :]
                z_next_coarse_raw, _ = self.dynamics(z_last_coarse, average_velocity=avg_vel_coarse)
                z_next_coarse = z_last_coarse + step_size * (z_next_coarse_raw - z_last_coarse)
                coarse_pred_seg = self.reconstructor(z_next_coarse)
                coarse_predictions.append(coarse_pred_seg)
                z_coarse = torch.cat([z_coarse[:, 1:, :], z_next_coarse.unsqueeze(1)], dim=1)
                
                # Fine
                avg_vel_fine = self.compute_average_velocity(z_fine)
                z_last_fine = z_fine[:, -1, :]
                z_next_fine_raw, _ = self.dynamics(z_last_fine, average_velocity=avg_vel_fine)
                z_next_fine = z_last_fine + step_size * (z_next_fine_raw - z_last_fine)
                fine_pred_seg = self.reconstructor(z_next_fine)
                fine_predictions.append(fine_pred_seg)
                z_fine = torch.cat([z_fine[:, 1:, :], z_next_fine.unsqueeze(1)], dim=1)
                
                # Residual
                avg_vel_resid = self.compute_average_velocity(z_residual)
                z_last_resid = z_residual[:, -1, :]
                z_next_resid_raw, _ = self.dynamics(z_last_resid, average_velocity=avg_vel_resid)
                z_next_resid = z_last_resid + step_size * (z_next_resid_raw - z_last_resid)
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