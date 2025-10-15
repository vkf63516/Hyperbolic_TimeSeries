import torch 
import torch
import torch.nn as nn
import geoopt
from mamba_ssm import Mamba  # pip install mamba-ssm

# --------------------------
# Mamba encoder block
# --------------------------
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

class ParallelEuclideanEncoder(nn.Module):
    def __init__(self, seq_len, embed_dim=32, hidden_dim=64):
        super().__init__()
        # 5 parallel Mamba encoder blocks
        self.trend_encoder = MambaEncoder(input_dim=1, hidden_dim=hidden_dim, output_dim=embed_dim)
        self.daily_encoder = MambaEncoder(input_dim=1, hidden_dim=hidden_dim, output_dim=embed_dim)
        self.weekly_encoder = MambaEncoder(input_dim=1, hidden_dim=hidden_dim, output_dim=embed_dim)
        self.monthly_encoder = MambaEncoder(input_dim=1, hidden_dim=hidden_dim, output_dim=embed_dim)
        self.resid_encoder = MambaEncoder(input_dim=1, hidden_dim=hidden_dim, output_dim=embed_dim)

    def forward(self, trend, daily, weekly, monthly, resid):
        # Encode each branch to Euclidean latent vector
        z_trend = self.trend_encoder(trend)
        z_daily = self.daily_encoder(daily)
        z_weekly = self.weekly_encoder(weekly)
        z_monthly = self.monthly_encoder(monthly)
        z_resid = self.resid_encoder(resid)

        # Simple Euclidean fusion: sum or concatenation
        # Option 1: sum
        combined = z_trend + z_daily + z_weekly + z_monthly + z_resid

        # Option 2: concatenation (preserves branches separately)
        # combined = torch.cat([z_trend, z_daily, z_weekly, z_monthly, z_resid], dim=-1)

        return {
            "trend_h": z_trend,
            "daily_h": z_daily,
            "weekly_h": z_weekly,
            "monthly_h": z_monthly,
            "resid_h": z_resid,
            "combined": combined
        }