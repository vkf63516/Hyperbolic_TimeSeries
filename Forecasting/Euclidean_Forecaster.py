import torch
import torch.nn as nn
from embed.mamba_embed_euclidean import ParallelEuclideanEmbed

class PointForecastEuclidean(nn.Module):
    """
    Point-level Euclidean forecasting
    NO manifold operations - pure Euclidean arithmetic
    """
    def __init__(self, lookback, n_features, pred_len, embed_dim=32, hidden_dim=64,
                 use_hierarchy=False, hierarchy_scales=[0.5, 1.0, 1.0, 1.5]):
        super().__init__()
        self.lookback = lookback
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        
        # Encoder: 4 parallel Mamba blocks
        self.embed = ParallelEuclideanEmbed(
            lookback=lookback,
            input_dim=n_features,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            use_hierarchy=use_hierarchy,
            hierarchy_scales=hierarchy_scales
        )
        
        # Reconstructor: simple MLP (no logmap/expmap needed!)
        self.reconstructor = EuclideanReconstructor(
            embed_dim=embed_dim,
            output_dim=n_features,
            hidden_dim=hidden_dim
        )
    
    def forward(self, trend, weekly, daily, residual, teacher_forcing=False, target=None):
        """
        Args:
            trend, weekly, daily, residual: [B, lookback, input_dim]
            horizon: number of steps to forecast
        
        Returns:
            forecast: [B, horizon, input_dim]
            euclidean_trajectory: [B, horizon, embed_dim]
        """
        # Initial encoding
        embed_output = self.embed(trend, weekly, daily, residual)
        z_current = embed_output['combined_e']  # [B, embed_dim]
        
        predictions = []
        euclidean_trajectory = []
        
        # Keep rolling history
        trend_hist = trend.clone()
        weekly_hist = weekly.clone()
        daily_hist = daily.clone()
        residual_hist = residual.clone()
        
        for step in range(self.pred_len):
            # Reconstruct: just MLP, no manifold operations!
            x_pred = self.reconstructor(z_current)  # [B, input_dim]
            predictions.append(x_pred)
            euclidean_trajectory.append(z_current)

            if teacher_forcing and target is not None:
                x_next = target[:, step:, :]
            
            # Update history
            trend_hist = torch.cat([trend_hist[:, 1:, :], x_next.unsqueeze(1)], dim=1)
            weekly_hist = torch.cat([weekly_hist[:, 1:, :], x_next.unsqueeze(1)], dim=1)
            daily_hist = torch.cat([daily_hist[:, 1:, :], x_next.unsqueeze(1)], dim=1)
            residual_hist = torch.cat([residual_hist[:, 1:, :], x_next.unsqueeze(1)], dim=1)
            
            # Re-encode
            embed_output = self.embed(trend_hist, weekly_hist, daily_hist, residual_hist)
            z_current = embed_output['combined_e']
        
        return {
            'predictions': torch.stack(predictions, dim=1),          # [B, horizon, input_dim]
            'euclidean_trajectory': torch.stack(euclidean_trajectory, dim=1),  # [B, horizon, embed_dim]
        }