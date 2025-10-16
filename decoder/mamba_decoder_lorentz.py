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
        z_t : [B, D+1] current manifold point (Lorentz coordinates)
        returns:
            z_next [B, D+1] : next manifold point
            v_proj [B, D+1] : projected tangent update
        """
        # 1) Map to tangent space at origin
        tangent_zt = self.manifold.logmap0(z_t)  # [B, D]

        # 2) Predict tangent update
        v_pred = self.state_net(tangent_zt)
        gate = torch.sigmoid(self.gate(tangent_zt))
        v_pred = gate * v_pred

        # 3) Project tangent vector to tangent space at z_t
        lorentz_dot = (-z_t[:, :1] * v_pred[:, :1] + (z_t[:, 1:] * v_pred[:, 1:]).sum(dim=-1, keepdim=True))
        v_proj = v_pred + lorentz_dot * z_t

        # 4) Exponential map to next point on manifold
        z_next = self.manifold.expmap(z_t, v_proj)
        z_next = self.manifold.projx(z_next)
        return z_next, v_proj

