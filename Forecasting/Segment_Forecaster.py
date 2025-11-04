import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # add project root
import pandas as pd
import torch
import torch.nn as nn
import geoopt

class HyperbolicSegmentForecaster(nn.Module):
    def __init__(self, embed_dim, hidden_dim, seg_len, manifold):
        super().__init__()
        self.manifold = manifold
        self.hidden_dim = hidden_dim
        self.seg_len = seg_len
        
        # Dynamics on hyperbolic manifold
        self.mvar = HyperbolicMambaLorentz(embed_dim, hidden_dim, manifold)
        
        # Reconstruct segment from hyperbolic point
        self.recon = HyperbolicReconstructionHead(embed_dim, seg_len, manifold)

    def combine_branches(self, trend_z, seasonal_weekly_z, seasonal_daily_z, residual_z):
        """Combine individual component embeddings."""
        u_trend = self.manifold.logmap0(trend_z)
        u_weekly = self.manifold.logmap0(seasonal_weekly_z)
        u_daily = self.manifold.logmap0(seasonal_daily_z)
        u_resid = self.manifold.logmap0(residual_z)
        
        combined_tangent = u_trend + u_weekly + u_daily + u_resid
        combined_h = segment_safe_expmap0(self.manifold, combined_tangent)
        combined_h = self.manifold.projx(combined_h)
        
        return combined_h

    def forward(self, pred_len, trend_z=None, seasonal_weekly_z=None, 
                seasonal_daily_z=None, residual_z=None, z0=None,
                teacher_forcing=False, z_true_seq=None, K=6):
        """
        Args:
            pred_len: Number of timesteps to forecast
            trend_z, weekly_z, daily_z, resid_z: [B, num_seg, D+1]
            z0: [B, num_seg, D+1] or [B, D+1] (pre-combined from encoder)
            teacher_forcing: bool - if True, always use ground truth when available
            z_true_seq: [B, num_pred_segments, D+1] - ground truth hyperbolic states
            K: Gradient truncation interval (in segments)
            
        Returns:
            x_hat: [B, pred_len]
            z_pred: [B, num_pred_segments, D+1]
        """
        num_pred_segments = (pred_len + self.seg_len - 1) // self.seg_len
        
        # Get initial state
        z_cur = None
        if z0 is None:
            if trend_z is None:
                raise ValueError("Must provide either z0 or individual components")
            combined = self.combine_branches(trend_z, seasonal_weekly_z, seasonal_daily_z, residual_z)
            z_cur = combined[:, -1, :]
        else:
            if z0.dim() == 3:
                z_cur = z0[:, -1, :]
            else:
                z_cur = z0

        preds_x = []
        preds_z = []
        
        for seg in range(num_pred_segments):
            z_next, _ = self.mvar(z_cur)
            x_hat_seg = self.recon(z_next)
            
            preds_x.append(x_hat_seg.unsqueeze(1))
            preds_z.append(z_next.unsqueeze(1))
            
            # Teacher forcing: if enabled and ground truth available, use it
            if teacher_forcing and z_true_seq is not None and seg < num_pred_segments - 1:
                z_cur = z_true_seq[:, seg, :]
            else:
                z_cur = z_next
            
            # Gradient truncation
            if K > 0 and (seg + 1) % K == 0:
                z_cur = z_cur.detach()
        
        x_hat = torch.cat(preds_x, dim=1)
        z_pred = torch.cat(preds_z, dim=1)
        
        return x_hat, z_pred
