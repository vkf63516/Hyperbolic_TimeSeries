import torch
import torch.nn as nn
import geoopt
from mamba_ssm import Mamba  

# ---------------------------------------------------
# 1. Mamba encoder block
# ---------------------------------------------------
class MambaEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
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

    def forward(self, x):
        """
        x: [B, T, input_dim]
        returns: [B, output_dim]  (Euclidean latent, tangent vectors)
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)              # mean pooling
        return self.output_proj(x)


# ---------------------------------------------------
# 2. Parallel Hyperbolic Encoder
# ---------------------------------------------------
class ParallelHyperbolicEncoder(nn.Module):
    def __init__(self, seq_len, embed_dim=32, hidden_dim=64, curvature=1.0):
        super().__init__()
        
        # Three parallel Mamba encoder branches
        self.trend_encoder = MambaEncoder(1, hidden_dim, embed_dim)
        self.seasonal_encoder = MambaEncoder(3, hidden_dim, embed_dim)  # daily, weekly, monthly
        self.resid_encoder = MambaEncoder(1, hidden_dim, embed_dim)
        
        self.manifold = geoopt.PoincareBall(c=curvature)

    def forward(self, trend, seasonal, resid):
        """
        trend:     [batch, seq_len, 1]
        seasonal:  [batch, seq_len, 3] (daily, weekly, monthly)
        resid:     [batch, seq_len, 1]
        """
        # --- Parallel encoders ---
        z_trend = self.trend_encoder(trend)
        z_season = self.seasonal_encoder(seasonal)
        z_resid = self.resid_encoder(resid)
        
        # --- Project to hyperbolic space ---
        z_trend_h = self.manifold.expmap0(z_trend)
        z_season_h = self.manifold.expmap0(z_season)
        z_resid_h = self.manifold.expmap0(z_resid)
        
        # --- Combine components in hyperbolic space ---
        z_combined = self.manifold.mobius_add(self.manifold.mobius_add(z_trend_h, z_season_h), z_resid_h)
        z_combined = self.manifold.projx(z_combined) # projects the point on the mainfold
        
        return {
            "trend_h": z_trend_h,
            "season_h": z_season_h,
            "resid_h": z_resid_h,
            "combined_h": z_combined
        }


# # ---------------------------------------------------
# # 3. Example usage
# # ---------------------------------------------------
# if __name__ == "__main__":
#     batch_size = 8
#     seq_len = 128

#     model = ParallelHyperbolicEncoder(seq_len=seq_len, embed_dim=16, hidden_dim=64, curvature=1.0)
    
#     # Dummy decomposition input
#     trend = torch.randn(batch_size, seq_len, 1)
#     seasonal = torch.randn(batch_size, seq_len, 3)  # daily, weekly, monthly
#     resid = torch.randn(batch_size, seq_len, 1)
    
#     out = model(trend, seasonal, resid)
    
#     print("Trend hyperbolic embedding:", out["trend_h"].shape)
#     print("Seasonal hyperbolic embedding:", out["season_h"].shape)
#     print("Residual hyperbolic embedding:", out["resid_h"].shape)
#     print("Combined hyperbolic embedding:", out["combined_h"].shape)
    
#     dist = model.manifold.dist(out["combined_h"][0], out["combined_h"][1])
#     print("Geodesic distance between two samples:", dist.item())
