import torch
import torch.nn as nn
import geoopt
from mamba_ssm import Mamba  # pip install mamba-ssm
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils import safe_expmap

# --------------------------
# Mamba encoder block
# --------------------------
class MambaEncoder(nn.Module):
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
            x = x[:, -self.lookback:, :]
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        # x = x.mean(dim=1)              # mean pooling would be good for point level forecasting
        return torch.tanh(self.output_proj(x))


# --------------------------
# Parallel encoders + Lorentz manifold fusion
# --------------------------
class ParallelLorentzEncoder(nn.Module):
    def __init__(self, lookback=None, embed_dim=32, hidden_dim=64, curvature=1.0):
        """
        embed_dim: dimensionality of tangent-space vectors (intrinsic manifold dimension)
        For Lorentz model geoopt expects expmap/logmap shapes [B, embed_dim].
        Internally manifold points live in R^{embed_dim + 1}, but geoopt APIs hide that.
        """
        super().__init__()

        # Branch encoders
        self.trend_encoder = MambaEncoder(input_dim=1, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)
        self.seasonal_encoder = MambaEncoder(input_dim=1, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)  # hourly, daily, weekly
        self.resid_encoder = MambaEncoder(input_dim=1, hidden_dim=hidden_dim, output_dim=embed_dim, lookback=lookback)

        # Lorentz manifold (k controls scale; curvature = -1/k)
        # Passing k = curvature (1.0 gives standard curvature -1)
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)

    def forward(self, trend, seasonal, resid):
        """
        Inputs:
          trend:    [B, seq_len, 1]
          seasonal: [B, seq_len, 2]  (daily, weekly)
          resid:    [B, seq_len, 1]

        Returns dict with:
          - trend_h, season_h, resid_h : points on Lorentz manifold
          - combined_h : fused point on Lorentz manifold (via tangent-sum)
          - combined_tangent : the summed tangent vector (optional)
        """
        # 1) encode to Euclidean latent (interpreted as tangent vectors at origin)
        z_trend_t = self.trend_encoder(trend)       # [B, D]
        z_season_t = self.seasonal_encoder(seasonal) # [B, D]
        z_resid_t = self.resid_encoder(resid)       # [B, D]

        # 2) map to manifold points (safe expmap0: tangent -> manifold)
        # z_trend_h = self.manifold.expmap0(z_trend_t)
        # z_season_h = self.manifold.expmap0(z_season_t)
        # z_resid_h = self.manifold.expmap0(z_resid_t)
        z_trend_h = safe_expmap(self.manifold, z_trend_t)    # manifold point
        z_season_h = safe_expmap(self.manifold, z_season_t)
        z_resid_h = safe_expmap(self.manifold, z_resid_t)

        # # (Optional) project to manifold numerically safely
        z_trend_h = self.manifold.projx(z_trend_h)
        z_season_h = self.manifold.projx(z_season_h)
        z_resid_h = self.manifold.projx(z_resid_h)

        # 3) Fuse: logmap0(manifold) -> tangent vectors -> sum -> expmap0 back
        u_trend = self.manifold.logmap0(z_trend_h)    # [B, D]
        u_season = self.manifold.logmap0(z_season_h)
        u_resid = self.manifold.logmap0(z_resid_h)

        combined_tangent = u_trend + u_season + u_resid  # tangent-space fusion (Euclidean sum)
        combined_h = safe_expmap(self.manifold, combined_tangent)
        # combined_h = self.manifold.expmap0(combined_tangent)
        combined_h = self.manifold.projx(combined_h)

        return {
            "trend_tangent": z_trend_t,
            "season_tangent": z_season_t,
            "resid_tangent": z_resid_t,
            "trend_h": z_trend_h,
            "season_h": z_season_h,
            "resid_h": z_resid_h,
            "combined_tangent": combined_tangent,
            "combined_h": combined_h
        }


# # --------------------------
# # Example usage
# # --------------------------
# if __name__ == "__main__":
#     batch_size = 4
#     seq_len = 128
#     embed_dim = 16   # intrinsic manifold dimension (tangent dim)
#     hidden_dim = 64

#     model = ParallelLorentzEncoder(seq_len=seq_len, embed_dim=embed_dim, hidden_dim=hidden_dim, curvature=1.0)

#     # Dummy inputs (normalized decomposition windows)
#     trend = torch.randn(batch_size, seq_len, 1)
#     seasonal = torch.randn(batch_size, seq_len, 3)  # hourly, daily, weekly (3 channels)
#     resid = torch.randn(batch_size, seq_len, 1)

#     out = model(trend, seasonal, resid)

#     print("trend_h shape:", out["trend_h"].shape)          # manifold point (geoopt handles embedding dim)
#     print("season_h shape:", out["season_h"].shape)
#     print("resid_h shape:", out["resid_h"].shape)
#     print("combined_h shape:", out["combined_h"].shape)
#     # geodesic distance example
#     d = model.manifold.dist(out["combined_h"][0], out["combined_h"][1])
#     print("geodesic distance between combined samples:", d.item())
