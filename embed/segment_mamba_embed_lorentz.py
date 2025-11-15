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
class SegmentMambaEmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=3, lookback_segment=None):
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
        self.lookback_segment = lookback_segment #  number of segments to look back

    def forward(self, x):
        """
        x: [B, num_seg, seg_len, C]
        returns: [B, output_dim]  (Euclidean latent, tangent vectors)
        """
        B,num_seg, seg_len, C = x.shape
        if self.lookback_segment is not None and x.size(1) > self.lookback_segment:
            x = x[:, -self.lookback_segment:, :]
        print(x.shape)
        x = x.reshape(B * num_seg, seg_len, C)
        print(x.shape)
        for layer in self.layers:
            x = layer(x)
        # Pool to single vector per segment
        x = x.mean(dim=1)
        print(x.shape)
        return torch.tanh(self.output_proj(x))


# --------------------------
# Parallel encoders + Lorentz manifold fusion
# --------------------------
class SegmentParallelLorentzBlock(nn.Module):
    def __init__(self, lookback_steps, seg_len, input_dim, embed_dim=32, hidden_dim=64,
                curvature=-1.0, use_hierarchy=True, hierarchy_scales=[0.5,1.0,1.0,1.5]):
        """
        embed_dim: dimensionality of tangent-space vectors (intrinsic manifold dimension)
        For Lorentz model geoopt expects expmap/logmap shapes [B, embed_dim].
        Internally manifold points live in R^{embed_dim + 1}, but geoopt APIs hide that.
        """
        super().__init__()
        self.use_hierarchy = use_hierarchy
        
        if use_hierarchy:
            self.hierarchy_scales = hierarchy_scales
            trend_scale = torch.exp(self.log_scales[0])
            coarse_scale = torch.exp(self.log_scales[1])
            fine_scale = torch.exp(self.log_scales[2])
            residual_scale = torch.exp(self.log_scales[3])

        # Branch encoders
        lookback_segments = lookback_steps // seg_len if lookback_steps else None
        self.trend_embed = SegmentMambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback_segment=lookback_segments)
        self.seasonal_coarse_embed = SegmentMambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback_segment=lookback_segments)  
        self.seasonal_fine_embed = SegmentMambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback_segment=lookback_segments)
        self.residual_embed = SegmentMambaEmbed(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=embed_dim, lookback_segment=lookback_segments)

        # Lorentz manifold (k controls scale; curvature = -1/k)
        # Passing k = curvature (1.0 gives standard curvature -1)
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)

    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Inputs:
          trend:    [B, seq_len, 1]
          seasonal_coarse: [B, seq_len, 1]  
          seasonal_fine:  [B, seq_len, 1]
          residual:    [B, seq_len, 1]

        Returns dict with:
          - trend_h, seasonal_coarse_h, seasonal_fine_h, residual_h : points on Lorentz manifold
          - combined_h : fused point on Lorentz manifold (via tangent-sum)
          - combined_tangent : the summed tangent vector 
        """
        # 1) encode to Euclidean latent (interpreted as tangent vectors at origin)
        B, num_seg, seg_len, C = trend.shape

    # ========================================
    # 2. Flatten Segments for Batch Processing
    # ========================================
        total_segs = B * num_seg

        trend_flat = trend.reshape(total_segs, seg_len, C)
        seasonal_coarse_flat = seasonal_coarse.reshape(total_segs, seg_len, C)
        seasonal_fine_flat = seasonal_fine.reshape(total_segs, seg_len, C)
        residual_flat = residual.reshape(total_segs, seg_len, C)

        z_trend_t = self.trend_embed(trend)       # [B, D]
        z_seasonal_coarse_t = self.seasonal_coarse_embed(seasonal_coarse) # [B, D]
        z_seasonal_fine_t = self.seasonal_fine_embed(seasonal_fine) # [B, D]
        z_residual_t = self.residual_embed(residual)       # [B, D]

        z_trend_h = safe_expmap0(self.manifold, z_trend_t)    # manifold point
        z_seasonal_coarse_h = safe_expmap0(self.manifold, z_seasonal_coarse_t)
        z_seasonal_fine_h = safe_expmap0(self.manifold, z_seasonal_fine_t)
        z_residual_h = safe_expmap0(self.manifold, z_residual_t)

        # # (Optional) project to manifold numerically safely
        z_trend_h = self.manifold.projx(z_trend_h)
        z_seasonal_coarse_h = self.manifold.projx(z_seasonal_coarse_h)
        z_seasonal_fine_h = self.manifold.projx(z_seasonal_fine_h)
        z_residual_h = self.manifold.projx(z_residual_h)
        if self.use_hierarchy:
            z_trend_ = z_trend_h * self.trend_scale 
            z_seasonal_coarse_h = z_seasonal_coarse_h * self.seasonal_coarse_scale
            z_seasonal_fine_h = z_seasonal_fine_h * self.seasonal_fine_scale
            z_residual_h = z_residual_h * self.residual_scale

        # 3) Fuse: logmap0(manifold) -> tangent vectors -> sum -> expmap back
        u_trend = self.manifold.logmap0(z_trend_h)    # [B, D]
        u_seasonal_coarse = self.manifold.logmap0(z_seasonal_coarse_h)
        u_seasonal_fine = self.manifold.logmap0(z_seasonal_fine_h)
        u_residual = self.manifold.logmap0(z_residual_h)

        combined_tangent = u_trend + u_seasonal_coarse + u_seasonal_fine + u_residual  # tangent-space fusion (Euclidean sum)
        combined_h = safe_expmap0(self.manifold,combined_tangent)
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

