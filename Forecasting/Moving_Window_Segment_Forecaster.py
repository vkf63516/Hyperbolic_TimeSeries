import torch
import torch.nn as nn
from embed.Linear.moving_segment_linear_embed_poincare import SegmentedParallelPoincareMovingWindow
from embed.Linear.segment_linear_embed_lorentz import SegmentedParallelLorentz
from DynamicsMvar.Poincare_Residual_Dynamics import HyperbolicPoincareDynamics
from DynamicsMvar.Lorentz_Residual_Dynamics import HyperbolicLorentzDynamics
from Lifting.moving_hyperbolic_segment_reconstructor import HyperbolicSegmentReconstructionHead
from spec import RevIN


class MovingWindowHyperbolicForecaster(nn.Module):
    """
    Segment-aware hyperbolic forecaster with moving window trajectory modeling.
    
    Key difference: Maintains a sliding window of segment embeddings and computes
    velocity from the entire trajectory at each prediction step.
    """
    def __init__(self, lookback, pred_len, n_features, embed_dim, hidden_dim, 
                 curvature, manifold_type, segment_length=24, 
                 use_revin=False, dynamic_dropout=0.3, embed_dropout=0.5,
                 num_layers=2, use_segment_norm=True, 
                 share_feature_weights=False, window_size=30):
        """
        Args:
            window_size: int - number of segments to keep in moving window
                        (default: 4, matches typical lookback of 720 timesteps / 24 segment_length)
        """
        super().__init__()

        if pred_len % segment_length != 0:
            raise ValueError(f"pred_len ({pred_len}) must be divisible by segment_length ({segment_length})")

        self.lookback = lookback
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.n_features = n_features
        self.segment_length = segment_length
        self.num_pred_segments = pred_len // segment_length
        self.use_revin = use_revin
        self.manifold_type = manifold_type
        self.window_size = self.lookback // self.segment_length
        
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)
        
        # Encoder that outputs [B, num_segments, embed_dim] (NOT aggregated)
        if manifold_type == "Poincare":
            self.embed_hyperbolic = SegmentedParallelPoincareMovingWindow(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                curvature=curvature,
                segment_length=segment_length,
                use_segment_norm=use_segment_norm,
                embed_dropout=embed_dropout,
                share_feature_weights=share_feature_weights,
            )
        # elif manifold_type == "Lorentzian":
        #     self.embed_hyperbolic = SegmentedParallelLorentzMovingWindow(
        #         lookback=lookback,
        #         input_dim=n_features,
        #         embed_dim=embed_dim,
        #         curvature=curvature,
        #         segment_length=segment_length,
        #         use_segment_norm=use_segment_norm,
        #         embed_dropout=embed_dropout,
        #         share_feature_weights=share_feature_weights
        #     )
        
        self.manifold = self.embed_hyperbolic.manifold
        
        # Dynamics that accepts avg_velocity
        if manifold_type == "Poincare":
            self.dynamics = HyperbolicPoincareDynamics(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                manifold=self.manifold,
                dropout=dynamic_dropout,
                n_layers=num_layers
            )
        # elif manifold_type == "Lorentzian":
        #     self.dynamics = HyperbolicLorentzDynamics(
        #         embed_dim=embed_dim,
        #         manifold=self.manifold,
        #         dropout=dynamic_dropout,
        #         n_layers=num_layers
        #     )
        
        self.reconstructor = HyperbolicSegmentReconstructionHead(
            embed_dim=embed_dim,
            output_dim=n_features,
            segment_length=segment_length,
            manifold=self.manifold
        )
    
    def compute_trajectory_velocity(self, z_window):
        """
        Compute average velocity from a window of segments.
        
        Args:
            z_window: [B, window_size, embed_dim] - sequence of segment embeddings
        
        Returns:
            avg_velocity: [B, embed_dim] - average velocity across trajectory
        """
        B, N, D = z_window.shape
        
        # Compute velocities between consecutive segments
        velocities = []
        for i in range(N - 1):
            v = self.manifold.logmap(z_window[:, i, :], z_window[:, i+1, :])
            velocities.append(v)
        
        # Average velocities
        avg_velocity = torch.stack(velocities, dim=1).mean(dim=1)  # [B, embed_dim]
        
        return avg_velocity
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Forecasting with moving window trajectory modeling.
        
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len, n_features]
        
        Returns:
            dict with predictions for each component
        """
        # Store RevIN stats
        x_combined = trend + seasonal_coarse + seasonal_fine + residual
        if self.use_revin:
            self.revin(x_combined, mode='norm')

        # Encode to hyperbolic segments: [B, num_segments, embed_dim]
        embed_h = self.embed_hyperbolic(trend, seasonal_coarse, seasonal_fine, residual)
        
        # Initialize moving windows with historical segments
        z_current = embed_h["combined_h"]  # [B, num_segments, embed_dim]
        z_current_trend = embed_h["trend_h"]
        z_current_coarse = embed_h["seasonal_coarse_h"]
        z_current_fine = embed_h["seasonal_fine_h"]
        z_current_resid = embed_h["residual_h"]
        
        # If we have more segments than window_size, take the last window_size
        if z_current.shape[1] > self.window_size:
            z_current = z_current[:, -self.window_size:, :]
            z_current_trend = z_current_trend[:, -self.window_size:, :]
            z_current_coarse = z_current_coarse[:, -self.window_size:, :]
            z_current_fine = z_current_fine[:, -self.window_size:, :]
            z_current_resid = z_current_resid[:, -self.window_size:, :]
        
        # Storage for predicted latents
        latent_z = []
        latent_trend = []
        latent_coarse = []
        latent_fine = []
        latent_resid = []

        # Autoregressive forecasting with moving window
        for seg_step in range(self.num_pred_segments):
            # === Combined signal ===
            # Compute average velocity from current window
            avg_velocity = self.compute_trajectory_velocity(z_current)
            
            # Get last segment for prediction
            z_last = z_current[:, -1, :]  # [B, embed_dim]
            z_prev = z_current[:, -2, :] if z_current.shape[1] > 1 else None
            
            # Predict next segment
            z_next, _ = self.dynamics(z_last, z_prev, avg_velocity)
            latent_z.append(z_next)
            
            # Update window: drop oldest, add newest
            z_current = torch.cat([z_current[:, 1:, :], z_next.unsqueeze(1)], dim=1)
            
            # === Trend ===
            avg_velocity_trend = self.compute_trajectory_velocity(z_current_trend)
            z_last_trend = z_current_trend[:, -1, :]
            z_prev_trend = z_current_trend[:, -2, :] if z_current_trend.shape[1] > 1 else None
            z_next_trend, _ = self.dynamics(z_last_trend, z_prev_trend, avg_velocity_trend)
            latent_trend.append(z_next_trend)
            z_current_trend = torch.cat([z_current_trend[:, 1:, :], z_next_trend.unsqueeze(1)], dim=1)
            
            # === Seasonal Coarse ===
            avg_velocity_coarse = self.compute_trajectory_velocity(z_current_coarse)
            z_last_coarse = z_current_coarse[:, -1, :]
            z_prev_coarse = z_current_coarse[:, -2, :] if z_current_coarse.shape[1] > 1 else None
            z_next_coarse, _ = self.dynamics(z_last_coarse, z_prev_coarse, avg_velocity_coarse)
            latent_coarse.append(z_next_coarse)
            z_current_coarse = torch.cat([z_current_coarse[:, 1:, :], z_next_coarse.unsqueeze(1)], dim=1)
            
            # === Seasonal Fine ===
            avg_velocity_fine = self.compute_trajectory_velocity(z_current_fine)
            z_last_fine = z_current_fine[:, -1, :]
            z_prev_fine = z_current_fine[:, -2, :] if z_current_fine.shape[1] > 1 else None
            z_next_fine, _ = self.dynamics(z_last_fine, z_prev_fine, avg_velocity_fine)
            latent_fine.append(z_next_fine)
            z_current_fine = torch.cat([z_current_fine[:, 1:, :], z_next_fine.unsqueeze(1)], dim=1)
            
            # === Residual ===
            avg_velocity_resid = self.compute_trajectory_velocity(z_current_resid)
            z_last_resid = z_current_resid[:, -1, :]
            z_prev_resid = z_current_resid[:, -2, :] if z_current_resid.shape[1] > 1 else None
            z_next_resid, _ = self.dynamics(z_last_resid, z_prev_resid, avg_velocity_resid)
            latent_resid.append(z_next_resid)
            z_current_resid = torch.cat([z_current_resid[:, 1:, :], z_next_resid.unsqueeze(1)], dim=1)

        # Stack predicted segments
        latent_z_segments = torch.stack(latent_z, dim=1)  # [B, num_pred_segments, embed_dim]
        latent_trend_segments = torch.stack(latent_trend, dim=1)
        latent_coarse_segments = torch.stack(latent_coarse, dim=1)
        latent_fine_segments = torch.stack(latent_fine, dim=1)
        latent_resid_segments = torch.stack(latent_resid, dim=1)

        # Batch decode all segments
        predictions_norm = self.reconstructor(latent_z_segments)  # [B, num_pred_segments, segment_length, n_features]
        trend_predictions = self.reconstructor(latent_trend_segments)
        coarse_predictions = self.reconstructor(latent_coarse_segments)
        fine_predictions = self.reconstructor(latent_fine_segments)
        residual_predictions = self.reconstructor(latent_resid_segments)

        # Flatten to [B, pred_len, n_features]
        B = predictions_norm.shape[0]
        predictions_norm = predictions_norm.reshape(B, self.num_pred_segments * self.segment_length, self.n_features)[:, :self.pred_len, :]
        trend_predictions = trend_predictions.reshape(B, self.num_pred_segments * self.segment_length, self.n_features)[:, :self.pred_len, :]
        coarse_predictions = coarse_predictions.reshape(B, self.num_pred_segments * self.segment_length, self.n_features)[:, :self.pred_len, :]
        fine_predictions = fine_predictions.reshape(B, self.num_pred_segments * self.segment_length, self.n_features)[:, :self.pred_len, :]
        residual_predictions = residual_predictions.reshape(B, self.num_pred_segments * self.segment_length, self.n_features)[:, :self.pred_len, :]

        # Apply RevIN denormalization
        if self.use_revin:
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm
            
        return {
            'predictions': predictions,
            'trend_predictions': trend_predictions,
            'coarse_predictions': coarse_predictions,
            'fine_predictions': fine_predictions,
            'residual_predictions': residual_predictions,
            'hyperbolic_states': {
                "combined_h": latent_z_segments,
                "trend_h": latent_trend_segments,
                "coarse_h": latent_coarse_segments,
                "fine_h": latent_fine_segments,
                "resid_h": latent_resid_segments,
            }
        }


