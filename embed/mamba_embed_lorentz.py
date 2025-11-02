import torch
import torch.nn as nn
import geoopt
from mamba_ssm import Mamba  # pip install mamba-ssm
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0

# --------------------------
# Mamba block
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
        self.lookback = lookback #  number of timesteps to look back

    def forward(self, x):
        """
        x: [B, T, input_dim]
        returns: [B, output_dim]  (Euclidean latent, tangent vectors)
        """
        if self.lookback is not None and x.size(1) > self.lookback:
            print(self.lookback)
            x = x[:, -self.lookback:, :]
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = torch.tanh(x)
        x = x.mean(dim=1)              # mean pooling would be good for point level forecasting
        return torch.tanh(self.output_proj(x))


# --------------------------
# Parallel encoders + Lorentz manifold fusion
# --------------------------
class ParallelLorentzBlock(nn.Module):
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                curvature=-1.0, use_hierarchy=False, hierarchy_scales=[0.5,1.0,1.0,1.5]):
        """
        embed_dim: dimensionality of tangent-space vectors (intrinsic manifold dimension)
        For Lorentz model geoopt expects expmap/logmap shapes [B, embed_dim].
        Internally manifold points live in R^{embed_dim + 1}, but geoopt APIs hide that.
        """
        super().__init__()
        self.use_hierarchy = use_hierarchy
        if use_hierarchy:
            self.hierarchy_scales = hierarchy_scales
            self.trend_scale = nn.Parameter(torch.tensor(hierarchy_scales[0]))
            self.seasonal_weekly_scale = nn.Parameter(torch.tensor(hierarchy_scales[1]))
            self.seasonal_daily_scale = nn.Parameter(torch.tensor(hierarchy_scales[2]))
            self.residual_scale = nn.Parameter(torch.tensor(hierarchy_scales[-1]))

        # Branch encoders
        self.trend_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)
        self.seasonal_weekly_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)  
        self.seasonal_daily_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)
        self.residual_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)
        
        # Lorentz manifold (k controls scale; curvature = -1/k)
        # Passing k = curvature (1.0 gives standard curvature -1)
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)

    def forward(self, trend, seasonal_weekly, seasonal_daily, residual):
        """
        Inputs:
          trend:    [B, seq_len, 1]
          seasonal_weekly: [B, seq_len, 1]  
          seasonal_daily:  [B, seq_len, 1]
          residual:    [B, seq_len, 1]

        Returns dict with:
          - trend_h, seasonal_weekly_h, seasonal_daily_h, residual_h : points on Lorentz manifold
          - combined_h : fused point on Lorentz manifold (via tangent-sum)
          - combined_tangent : the summed tangent vector 
        """
        # 1) encode to Euclidean latent (interpreted as tangent vectors at origin)
        z_trend_t = self.trend_embed(trend)       # [B, D]
        z_seasonal_weekly_t = self.seasonal_weekly_embed(seasonal_weekly) # [B, D]
        z_seasonal_daily_t = self.seasonal_daily_embed(seasonal_daily) # [B, D]
        z_residual_t = self.residual_embed(residual)       # [B, D]
        if self.use_hierarchy:
            z_trend_t = z_trend_t * self.trend_scale.abs()  # abs() ensures positive
            z_seasonal_weekly_t = z_seasonal_weekly_t * self.seasonal_weekly_scale.abs()
            z_seasonal_daily_t = z_seasonal_daily_t * self.seasonal_daily_scale.abs()
            z_residual_t = z_residual_t * self.residual_scale.abs()
            
        z_trend_h = safe_expmap0(self.manifold, z_trend_t)    # manifold point
        z_seasonal_weekly_h = safe_expmap0(self.manifold, z_seasonal_weekly_t)
        z_seasonal_daily_h = safe_expmap0(self.manifold, z_seasonal_daily_t)
        z_residual_h = safe_expmap0(self.manifold, z_residual_t)

        # # (Optional) project to manifold numerically safely
        z_trend_h = self.manifold.projx(z_trend_h)
        z_seasonal_weekly_h = self.manifold.projx(z_seasonal_weekly_h)
        z_seasonal_daily_h = self.manifold.projx(z_seasonal_daily_h)
        z_residual_h = self.manifold.projx(z_residual_h)

        # 3) Fuse: logmap0(manifold) -> tangent vectors -> sum -> expmap back
        u_trend = self.manifold.logmap0(z_trend_h)    # [B, D]
        u_seasonal_weekly = self.manifold.logmap0(z_seasonal_weekly_h)
        u_seasonal_daily = self.manifold.logmap0(z_seasonal_daily_h)
        u_residual = self.manifold.logmap0(z_residual_h)

        combined_tangent = u_trend + u_seasonal_weekly + u_seasonal_daily + u_residual  # tangent-space fusion (Euclidean sum)
        combined_h = safe_expmap(self.manifold, combined_tangent)
        # combined_h = self.manifold.expmap0(combined_tangent)
        combined_h = self.manifold.projx(combined_h)

        return {
            "trend_tangent": z_trend_t,
            "seasonal_weekly_tangent": z_seasonal_weekly_t,
            "seasonal_daily_tangent": z_seasonal_daily_t,
            "residual_tangent": z_residual_t,
            "trend_h": z_trend_h,
            "seasonal_weekly_h": z_seasonal_weekly_h,
            "seasonal_daily_h": z_seasonal_daily_h,
            "residual_h": z_residual_h,
            "combined_tangent": combined_tangent,
            "combined_h": combined_h
        }


