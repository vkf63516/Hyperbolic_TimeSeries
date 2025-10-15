import torch
import torch.nn as nn
import geoopt

class HyperbolicReconstructionHead(nn.Module):
    def __init__(self, embed_dim, output_dim, manifold):
        super().__init__()
        self.manifold = manifold
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, z_t):
        # Map from manifold → tangent space at origin
        v = self.manifold.logmap0(z_t)  # [B, embed_dim]
        return self.fc(v)
