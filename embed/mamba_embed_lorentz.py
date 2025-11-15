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
        x = x.mean(dim=1)              # mean pooling would be good for point level forecasting
        return self.output_proj(x)


# --------------------------
# Parallel encoders + Lorentz manifold fusion
# --------------------------
class ParallelLorentz(nn.Module):
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                curvature=1.0, use_hierarchy=False, hierarchy_scales=[0.5,1.0,1.0,1.5]):
        """
        embed_dim: dimensionality of tangent-space vectors (intrinsic manifold dimension)
        For Lorentz model geoopt expects expmap/logmap shapes [B, embed_dim].
        Internally manifold points live in R^{embed_dim + 1}, but geoopt APIs hide that.
        """
        super().__init__()
        
        self.use_hierarchy = use_hierarchy
        if self.use_hierarchy:
            self.log_scales = nn.ParameterList([
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[0]))),
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[1]))),
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[2]))),
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[3])))
            ])


        # Branch encoders
        self.trend_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)
        self.seasonal_coarse_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)  
        self.seasonal_fine_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)
        self.residual_embed = MambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)
        # Lorentz manifold (k controls scale; curvature = -1/k)
        # Passing k = curvature (1.0 gives standard curvature -1)
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.effective_scale = nn.Parameter(torch.tensor(0.1))

    def apply_hierarchy_scaling(self, manifold_point, scale):
        """Scale manifold point radially from origin"""
        tangent = self.manifold.logmap0(manifold_point)
        scaled_tangent = tangent * scale
        scaled_point = safe_expmap0(self.manifold, scaled_tangent)
        return self.manifold.projx(scaled_point)

    def hierarchical_combine(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """
        Build up from general (trend) to specific (residual)
        Each component modifies the previous state
        """
        # Start from trend (root of hierarchy)
        z_current = z_trend_h

        # Incorporate coarse (moves from trend toward coarse)
        v_to_coarse = self.manifold.logmap(z_current, z_coarse_h)
        z_current = safe_expmap(self.manifold, z_current, 0.25 * v_to_coarse)
        z_current = self.manifold.projx(z_current)
    
        # Incorporate fine
        v_to_fine = self.manifold.logmap(z_current, z_fine_h)
        z_current = safe_expmap(self.manifold, z_current, 0.25 * v_to_fine)
        z_current = self.manifold.projx(z_current)
    
        # Incorporate residual
        v_to_residual = self.manifold.logmap(z_current, z_residual_h)
        z_current = safe_expmap(self.manifold, z_current, 0.25 * v_to_residual)
        z_current = self.manifold.projx(z_current)
    
        return z_current
        
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Inputs:
          trend
          seasonal_coarse
          seasonal_fine
          residual

        Returns dict with:
          - trend_h, seasonal_coarse_h, seasonal_fine_h, residual_h : points on Lorentz manifold
          - combined_h : fused point on Lorentz manifold (via tangent-sum)
          - combined_tangent : the summed tangent vector 
        """
        # 1) encode to Euclidean latent (interpreted as tangent vectors at origin)
        z_trend_t = self.trend_embed(trend)       # [B, D]
        z_seasonal_coarse_t = self.seasonal_coarse_embed(seasonal_coarse) # [B, D]
        z_seasonal_fine_t = self.seasonal_fine_embed(seasonal_fine) # [B, D]
        z_residual_t = self.residual_embed(residual)       # [B, D]
        
        # Scaling to prevent numerical instability
        effective_scale = torch.tanh(self.effective_scale)
        scaled_trend_embed = z_trend_t * effective_scale
        scaled_coarse_embed = z_seasonal_coarse_t * effective_scale
        scaled_fine_embed = z_seasonal_fine_t * effective_scale
        scaled_residual_embed = z_residual_t * effective_scale

        z_trend_h = safe_expmap0(self.manifold, scaled_trend_embed)    # manifold point
        z_seasonal_coarse_h = safe_expmap0(self.manifold, scaled_coarse_embed)
        z_seasonal_fine_h = safe_expmap0(self.manifold, scaled_fine_embed)
        z_residual_h = safe_expmap0(self.manifold, scaled_residual_embed)

        # # (Optional) project to manifold numerically safely
        z_trend_h = self.manifold.projx(z_trend_h)
        z_seasonal_coarse_h = self.manifold.projx(z_seasonal_coarse_h)
        z_seasonal_fine_h = self.manifold.projx(z_seasonal_fine_h)
        z_residual_h = self.manifold.projx(z_residual_h)

        if self.use_hierarchy:
            trend_scale = torch.exp(self.log_scales[0])
            coarse_scale = torch.exp(self.log_scales[1])
            fine_scale = torch.exp(self.log_scales[2])
            residual_scale = torch.exp(self.log_scales[3])
    
            z_trend_h = self.apply_hierarchy_scaling(z_trend_h, trend_scale)
            z_seasonal_coarse_h = self.apply_hierarchy_scaling(z_seasonal_coarse_h, coarse_scale)
            z_seasonal_fine_h = self.apply_hierarchy_scaling(z_seasonal_fine_h, fine_scale)
            z_residual_h = self.apply_hierarchy_scaling(z_residual_h, residual_scale)

        # 3) Fuse: logmap0(manifold) -> tangent vectors -> sum -> expmap back
        u_trend = self.manifold.logmap0(z_trend_h)    # [B, D]
        u_seasonal_coarse = self.manifold.logmap0(z_seasonal_coarse_h)
        u_seasonal_fine = self.manifold.logmap0(z_seasonal_fine_h)
        u_residual = self.manifold.logmap0(z_residual_h)

        combined_tangent = u_trend + u_seasonal_coarse + u_seasonal_fine + u_residual  # tangent-space fusion (Euclidean sum)
        scaled_tangent = combined_tangent * effective_scale
        combined_h = safe_expmap0(self.manifold, scaled_tangent)
        # combined_h = self.manifold.expmap0(combined_tangent)
        combined_h = self.manifold.projx(combined_h)

        return {
            "trend_tangent": z_trend_t,
            "seasonal_coarse_tangent": z_seasonal_coarse_t,
            "seasonal_fine_tangent": z_seasonal_fine_t,
            "residual_tangent": z_residual_t,
            "trend_h": z_trend_h,
            "seasonal_coarse_h": z_seasonal_coarse_h,
            "seasonal_fine_h": z_seasonal_fine_h,
            "residual_h": z_residual_h,
            "combined_tangent": combined_tangent,
            "combined_h": combined_h
        }


