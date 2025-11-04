import torch
import torch.nn as nn
from mamba_ssm import Mamba  # pip install mamba-ssm

# --------------------------
# Mamba encoder block
# --------------------------
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
        x = x.mean(dim=1)              # mean pooling
        return self.output_proj(x)

class ParallelEuclideanEmbed(nn.Module):
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64,
                use_hierarchy=False, hierarchy_scales=[0.5,1.0,1.5,2.0], n_layer=3):
        super().__init__()
        # 5 parallel Mamba encoder blocks
        self.use_hierarchy = use_hierarchy
        if self.use_hierarchy:
            self.hierarchy_scales = hierarchy_scales
            self.trend_scale = nn.Parameter(torch.tensor(hierarchy_scales[0]))
            self.seasonal_weekly_scale = nn.Parameter(torch.tensor(hierarchy_scales[1]))
            self.seasonal_daily_scale = nn.Parameter(torch.tensor(hierarchy_scales[2]))
            self.residual_scale = nn.Parameter(torch.tensor(hierarchy_scales[-1]))
        self.trend_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback, n_layer=n_layer)
        self.daily_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback, n_layer=n_layer)
        self.weekly_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback, n_layer=n_layer)
        self.residual_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback, n_layer=n_layer)

    def forward(self, trend, daily, weekly, residual):
        # Embed each branch to Euclidean latent vector
        e_trend = self.trend_embed(trend)
        e_daily = self.daily_embed(daily)
        e_weekly = self.weekly_embed(weekly)
        e_residual = self.residual_embed(residual)
        if self.use_hierarchy:
            e_trend *= self.trend_scale
            e_weekly *= self.seasonal_weekly_scale
            e_daily *= self.seasonal_daily_scale
            e_residual *= self.residual_scale
        # Simple Euclidean fusion: sum or concatenation
        # Option 1: sum
        combined_e = e_trend + e_daily + e_weekly + e_residual

        # Option 2: concatenation (preserves branches separately)
        # combined = torch.cat([z_trend, z_daily, z_weekly, z_monthly, z_resid], dim=-1)

        return {
            "trend_e": e_trend,
            "seasonal_daily_e": e_daily,
            "seasonal_weekly_e": e_weekly,
            "residual_e": e_residual,
            "combined_e": combined_e
        }