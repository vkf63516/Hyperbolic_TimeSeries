import torch
import torch.nn as nn
import geoopt
from mamba_ssm import Mamba  
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0

# ---------------------------------------------------
# 1. Mamba encoder block
# ---------------------------------------------------
class MambaEmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=3, lookback=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([ 
            Mamba(
                d_model=hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2
            ) for _ in range(n_layer)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.lookback = lookback

    def forward(self, x):
        """
        x: [B, T, input_dim]
        returns: [B, output_dim]  (Euclidean latent, tangent vectors)
        """
        if self.lookback is not None and x.size(1) > self.lookback:
            x = x[:, -self.lookback:, :]
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        # x = x.mean(dim=1)              # mean pooling
        return torch.tanh(self.output_proj(x))


# ---------------------------------------------------
# 2. Parallel Hyperbolic Encoder
# ---------------------------------------------------
class ParallelHyperbolicEncoder(nn.Module):
    def __init__(self, lookback=None, embed_dim=32, hidden_dim=64, curvature=1.0):
        super().__init__()
        
        # Three parallel Mamba encoder branches
        self.trend_embed = MambaEmbed(1, hidden_dim, embed_dim, lookback=lookback)
        self.weekly_embed = MambaEmbed(1, hidden_dim, embed_dim, lookback=lookback)  # hourly, daily, weekly
        self.daily_embed = MambaEmbed(1, hidden_dim, embed_dim, lookback=lookback)
        self.resid_embed = MambaEmbed(1, hidden_dim, embed_dim, lookback=lookback)
        self.manifold = geoopt.PoincareBall(c=curvature)

    def forward(self, trend, weekly, daily, resid):
        """
        trend:     [batch, seq_len, 1]
        seasonal:  [batch, seq_len, 3] (hourly, daily, weekly)
        resid:     [batch, seq_len, 1]
        """
        # --- Parallel encoders ---
        z_trend_t = self.trend_embed(trend)
        z_weekly_t = self.weekly_embed(weekly)
        z_daily_t = self.daily_embed(daily)
        z_resid_t = self.resid_embed(resid)
        
        # --- Project to hyperbolic space ---
        z_trend_h = safe_expmap0(self.manifold, z_trend_t)
        z_weekly_h = safe_expmap0(self.manifold, z_weekly_t)
        z_daily_h = safe_expmap0(self.manifold, z_daily_t)
        z_resid_h = safe_expmap0(self.manifold, z_resid_t)

        z_trend_h = self.manifold.projx(z_trend_h)
        z_weekly_h = self.manifold.projx(z_weekly_h)
        z_daily_h = self.manifold.projx(z_daily_h)
        z_resid_h = self.manifold.projx(z_resid_h)
        
        # --- Combine components in hyperbolic space ---
        z_combined = self.manifold.mobius_add(self.manifold.mobius_add(z_trend_h, self.mobius_add(z_weekly_h, z_daily_h)), z_resid_h)
        z_combined = self.manifold.projx(z_combined) # projects the point on the mainfold
        
        return {
            "trend_h": z_trend_h,
            "weekly_h": z_weekly_h,
            "daily_h": z_daily_h,
            "resid_h": z_resid_h,
            "combined_h": z_combined
        }

