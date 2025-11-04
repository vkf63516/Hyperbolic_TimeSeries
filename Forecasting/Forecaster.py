import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add project root
from hyperbolic_mvar.mamba_mvar_lorentz import HyperbolicMambaLorentz
from Lifting.reconstructor import HyperbolicReconstructionHead
import pandas as pd
import torch
import torch.nn as nn
import geoopt
# forecast_points_lorentz.py
import torch
import torch.nn as nn
import geoopt
from embed.mamba_embed_lorentz import ParallelLorentz
from embed.mamba_embed_poincare import ParallelPoincare
from Lifting.hyperbolic_reconstructor import HyperbolicReconstructionHead


class HyperbolicPointForecaster(nn.Module):
    """
    Forecasts single points in Lorentz hyperbolic space
    Reuses ParallelLorentz or ParallelPoincare for both embedding and rolling prediction
    """
    def __init__(self, lookback, pred_len, n_features, embed_dim, hidden_dim, 
                 curvature, manifold_type, use_hierarchical=False, 
                 hierarchy_scales=[0.5,1.0,1.0,1.5]):
        super().__init__()
        self.lookback = lookback
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.n_features = n_features
        
        # ParallelLorentzBlock: 4 Mamba encoders + manifold fusion
        if manifold_type == "Poincare":
            self.embed_hyperbolic = ParallelPoincare(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                curvature=curvature, 
                use_hierarchy=use_hierarchical,
                hierarchy_scales=hierarchy_scales
                
            )
        else:
            self.embed_hyperbolic = ParallelLorentz(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                curvature=curvature,
                use_hierarchy=use_hierarchical,
                hierarchy_scales=hierarchy_scales
            )

        
        # Reconstruction head (MLP in tangent space)
        self.reconstructor = HyperbolicReconstructionHead(
            embed_dim=embed_dim,
            output_dim=n_features,
            manifold=self.embed_hyperbolic.manifold
        )
    
    def forward(self, trend, seasonal_weekly, seasonal_daily, residual, teacher_forcing=False, target=None):
        """
        Rolling prediction of single points
        
        Args:
            trend, seasonal_weekly, seasonal_daily, residual: [B, segment_length, input_dim]
            horizon: number of steps to forecast
        
        Returns:
            forecast: [B, horizon, input_dim]
            embed_trajectory: [B, horizon, embed_dim] points on manifold
        """
        # Initial encoding
        embed_output = self.embed_hyperbolic(trend, seasonal_weekly, seasonal_daily, residual)
        z_current = embed_output['combined_h']  # [B, embed_dim]
        
        predictions = []
        embed_trajectory = []
        
        # Keep rolling history
        trend_hist = trend.clone()
        weekly_hist = seasonal_weekly.clone()
        daily_hist = seasonal_daily.clone()
        residual_hist = residual.clone()
        
        for step in range(self.pred_len):
            # Reconstruct single point
            x_pred = self.reconstructor(z_current)  # [B, input_dim]
            predictions.append(x_pred)
            embed_trajectory.append(z_current)
            if teacher_forcing and target is not None:
                x_next = target[:, step, :]
            else:
                x_next = x_pred
            
            # Update history: shift and append prediction
            trend_hist = torch.cat([
                trend_hist[:, 1:, :],
                x_next.unsqueeze(1)
            ], dim=1)
            
            weekly_hist = torch.cat([
                weekly_hist[:, 1:, :],
                x_next.unsqueeze(1)
            ], dim=1)
            
            daily_hist = torch.cat([
                daily_hist[:, 1:, :],
                x_next.unsqueeze(1)
            ], dim=1)
            
            residual_hist = torch.cat([
                residual_hist[:, 1:, :],
                x_next.unsqueeze(1)
            ], dim=1)
            
            # Encode and get next point
            embed_output = self.embed_hyperbolic(trend_hist, weekly_hist, daily_hist, residual_hist)
            z_current = embed_output['combined_h']
        
        return {
            'predictions': torch.stack(predictions, dim=1),          # [B, horizon, input_dim]
            'embed_trajectory': torch.stack(embed_trajectory, dim=1),  # [B, horizon, embed_dim]
        }
