import torch
import torch.nn.functional as Func
from geoopt.manifolds import PoincareBall

def hyperbolic_velocity_consistency_loss(z_trajectory, manifold, beta=1.0):
    """
    Optimized version with batched parallel transport.
    """
    # Handle dimensions
    if z_trajectory.dim() == 4:
        B, F, N, D = z_trajectory.shape
        z_trajectory = z_trajectory. reshape(B * F, N, D)
        B = B * F
    elif z_trajectory.dim() == 3:
        B, N, D = z_trajectory.shape
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {z_trajectory. shape}")
    
    if N < 3:
        return torch.tensor(0.0, device=z_trajectory.device, dtype=z_trajectory.dtype)
    
    # Compute velocities
    z_start = z_trajectory[:, :-1, :].reshape(-1, D)  # [B*(N-1), D]
    z_end = z_trajectory[:, 1:, :].reshape(-1, D)     # [B*(N-1), D]
    
    velocities_flat = manifold.logmap(z_start, z_end)  # [B*(N-1), D]
    velocities = velocities_flat.view(B, N-1, D)       # [B, N-1, D]
    
    # Batched parallel transport
    # Transport all v[t] from z[t] to z[t+1] at once
    v_curr_all = velocities[: , :-1, : ].reshape(-1, D)      # [B*(N-2), D]
    v_next_all = velocities[:, 1:, :].reshape(-1, D)       # [B*(N-2), D]
    
    z_from_all = z_trajectory[:, :-2, : ].reshape(-1, D)    # [B*(N-2), D]
    z_to_all = z_trajectory[:, 1:-1, :].reshape(-1, D)     # [B*(N-2), D]
    
    # Parallel transport in batch (geoopt supports this!)
    v_curr_transported = manifold.transp(z_from_all, z_to_all, v_curr_all)  # [B*(N-2), D]
    
    # Compute accelerations
    acceleration = v_next_all - v_curr_transported  # [B*(N-2), D]
    acceleration_norm_sq = torch.sum(acceleration ** 2, dim=-1)  # [B*(N-2)]
    
    # Reshape and average
    acceleration_norm_sq = acceleration_norm_sq.view(B, N-2)  # [B, N-2]
    loss = beta * torch.mean(acceleration_norm_sq)
    
    return loss

