import torch
import torch.nn as nn
import geoopt
from mamba_ssm import Mamba  
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap0

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
        x = x.mean(dim=1)              # mean pooling
        return torch.tanh(self.output_proj(x))


# ---------------------------------------------------
# 2. Parallel Hyperbolic Encoder
# ---------------------------------------------------
class ParallelPoincare(nn.Module):
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                curvature=1.0, n_layer=3, use_hierarchy=False, hierarchy_scales=[0.5,1.0,1.0,1.5]):
        super().__init__()
        
        # Three parallel Mamba encoder branches
        self.use_hierarchy = use_hierarchy
        if use_hierarchy:
            self.hierarchy_scales = hierarchy_scales
            trend_scale = torch.exp(self.log_scales[0])
            coarse_scale = torch.exp(self.log_scales[1])
            fine_scale = torch.exp(self.log_scales[2])
            residual_scale = torch.exp(self.log_scales[3])

        self.trend_embed = MambaEmbed(input_dim, hidden_dim, embed_dim, lookback=lookback, n_layer=n_layer)
        self.coarse_embed = MambaEmbed(input_dim, hidden_dim, embed_dim, lookback=lookback, n_layer=n_layer)  # hourly, fine, coarse
        self.fine_embed = MambaEmbed(input_dim, hidden_dim, embed_dim, lookback=lookback, n_layer=n_layer)
        self.residual_embed = MambaEmbed(input_dim, hidden_dim, embed_dim, lookback=lookback, n_layer=n_layer)
        self.manifold = geoopt.PoincareBall(c=curvature)
        

    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        trend:     [batch, seq_len, 1]
        seasonal:  [batch, seq_len, 3] (hourly, fine, coarse)
        resid:     [batch, seq_len, 1]
        """
        # --- Parallel encoders ---
        z_trend_t = self.trend_embed(trend)
        z_coarse_t = self.coarse_embed(coarse)
        z_fine_t = self.fine_embed(fine)
        z_residual_t = self.residual_embed(residual)
        
        # --- Project to hyperbolic space ---
        z_trend_h = safe_expmap0(self.manifold, z_trend_t)
        z_coarse_h = safe_expmap0(self.manifold, z_coarse_t)
        z_fine_h = safe_expmap0(self.manifold, z_fine_t)
        z_residual_h = safe_expmap0(self.manifold, z_residual_t)

        z_trend_h = self.manifold.projx(z_trend_h)
        z_coarse_h = self.manifold.projx(z_coarse_h)
        z_fine_h = self.manifold.projx(z_fine_h)
        z_residual_h = self.manifold.projx(z_residual_h)
        
        # --- Combine components in hyperbolic space ---
        z_combined = self.manifold.mobius_add(self.manifold.mobius_add(z_trend_h,
                                             self.manifold.mobius_add(z_coarse_h, z_fine_h)),
                                             z_residual_h)
        z_combined = self.manifold.projx(z_combined) # projects the point on the mainfold
        
        return {
            "trend_h": z_trend_h,
            "coarse_h": z_coarse_h,
            "fine_h": z_fine_h,
            "residual_h": z_residual_h,
            "combined_h": z_combined
        }

