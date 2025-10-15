import torch
import torch.nn as nn
import geoopt

class HyperbolicMambaDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, manifold):
        super().__init__()
        self.manifold = manifold 
        self.state_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.gate = nn.Linear(embed_dim, embed_dim)

    def forward(self, z_t):
        """
        z_t : [B, D+1] current manifold point(Lorentz coordinates)
        returns: z_next [B, D+1], v_proj [B, D+1]
        """
        # map to tangent space at origin
        tangent_zt = self.manifold.logmap0(z_t) # [B, D]

        # predict tangent update (Euclidean)
        v_pred = self.start_net(tangent_zt)
        gate = torch.sigmoid(self.gate(tangent_zt))
        v_pred = gate * v_pred # selective gating (Mamba-like)

        # move tangent to the correct tangent space and project
        v_proj = v_pred + torch.sum(-v_pred[:, :1] * z_t[:, :1] + v_pred[:, 1:] * z_t[:, 1:], dim=-1, keepdim=True) * z_t 

        # Exponential map to get the next manifold point
        norm = torch.sqrt(torch.clamp(
            -v_proj[:, :1]**2 + (v_proj[:, 1:]**2).sum(-1, keepdim=True), min=1e-6
        ))
        cosh = torch.cosh(norm)
        sinh = torch.sinh(norm)
        z_next = cosh * z_t + sinh * (v_proj / (norm + 1e-6))
        z_next = self.manifold.projx(z_next)
        return z_next, v_proj

