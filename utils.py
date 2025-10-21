import torch
import torch.nn as nn
import geoopt
# --------------------------
# Clamping with safe Exponential map
# --------------------------
def safe_expmap0(manifold, u, max_norm=10.0, eps=1e-8):
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(norm, max=max_norm) / norm
    u = u * scale
    x = manifold.expmap0(u)
    return manifold.projx(x)

