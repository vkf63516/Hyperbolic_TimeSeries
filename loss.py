import torch
import torch.nn.functional as Func
from geoopt.manifolds import Lorentz, PoincareBall

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
    
    # ✅ Batched parallel transport
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

def euclidean_velocity_consistency_loss(z_trajectory, beta=1.0):
    """
    Euclidean equivalent of hyperbolic_velocity_consistency_loss.
    Penalizes acceleration in trajectory — encourages smooth linear dynamics.
    
    Args:
        z_trajectory: [B, F, N, D] or [B, N, D]
        beta: weighting factor
    
    Returns:
        loss: scalar consistency loss
    """
    # Handle dimensions
    if z_trajectory.dim() == 4:
        B, F, N, D = z_trajectory.shape
        z_trajectory = z_trajectory.reshape(B * F, N, D)
        B = B * F
    elif z_trajectory.dim() == 3:
        B, N, D = z_trajectory.shape
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {z_trajectory.shape}")
    
    if N < 3:
        return torch.tensor(0.0, device=z_trajectory.device, dtype=z_trajectory.dtype)
    
    # Euclidean velocities — simple finite difference
    # logmap(x, y) = y - x in Euclidean space
    velocities = z_trajectory[:, 1:, :] - z_trajectory[:, :-1, :]  # [B, N-1, D]
    
    # Euclidean parallel transport is identity — v transported = v unchanged
    # So acceleration = v[t+1] - v[t] directly
    v_curr = velocities[:, :-1, :]  # [B, N-2, D]
    v_next = velocities[:, 1:, :]   # [B, N-2, D]
    
    # Acceleration — no parallel transport needed
    acceleration = v_next - v_curr  # [B, N-2, D]
    
    acceleration_norm_sq = torch.sum(acceleration ** 2, dim=-1)  # [B, N-2]
    loss = beta * torch.mean(acceleration_norm_sq)
    
    return loss

def radial_diversity_loss(z_trajectory, manifold, target_variance=0.1, beta=1.0):
    """
    Encourage diversity in radial distances from the origin in hyperbolic space.
    
    Penalizes when embeddings cluster at similar radii from the origin.
    
    Args:
        z_trajectory: Hyperbolic embeddings
                     [B, F, N, D] - batch, features, segments, embedding_dim
                     [B, N, D] - batch, segments, embedding_dim
                     [B, D] - batch, embedding_dim
        manifold: geoopt manifold (Poincare or Lorentz)
        target_variance: Target variance for radii (σ²_target)
        beta: weighting factor for the loss
    
    Returns:
        loss: scalar radial diversity loss
    """
    # Handle different input dimensions
    original_shape = z_trajectory.shape
    
    if z_trajectory.dim() == 4: 
        B, F, N, D = z_trajectory.shape
        # Reshape to [B*F*N, D] for radius computation
        z_flat = z_trajectory.reshape(B * F * N, D)
        B = B * F
    elif z_trajectory. dim() == 3:
        B, N, D = z_trajectory.shape
        z_flat = z_trajectory.reshape(B * N, D)
    elif z_trajectory.dim() == 2:
        B, D = z_trajectory.shape
        z_flat = z_trajectory
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got shape {z_trajectory.shape}")
    
    # Compute radii for all points
    radii = compute_hyperbolic_radius(z_flat, manifold)  # [B*F*N] or [B*N] or [B]
    
    # Compute variance of radii
    mean_radius = radii.mean()
    radius_variance = ((radii - mean_radius) ** 2).mean()
    
    # Penalize if variance is below target (encourages spreading)
    # L_div = max(0, σ²_target - actual_variance)
    loss = Func.relu(target_variance - radius_variance)
    
    return beta * loss


def compute_hyperbolic_radius(points, manifold):
    """
    Compute radius (distance from origin) for hyperbolic points.
    
    Args:
        points: [N, D] tensor of hyperbolic points
        manifold: geoopt manifold
    
    Returns:
        radii: [N] tensor of radii
    """
    curvature = manifold.c
    if isinstance(manifold, Lorentz):
        # Lorentzian model: r = (1/√c) * arccosh(-√c * x_0)
        # Origin in Lorentz:  o = (1/√c, 0, ..., 0)
        # Inner product: <o, x>_L = -x_0/√c
        # Distance: d(o, x) = (1/√c) * arccosh(-√c * x_0)
        
        sqrt_c = torch.sqrt(torch.tensor(curvature, device=points.device))
        x_0 = points[:, 0]  # Time component
        
        # For numerical stability:  arccosh(x) requires x >= 1
        # In Lorentz model:  -√c * x_0 should be >= 1
        inner_arg = -sqrt_c * x_0
        
        # Clamp to ensure numerical stability
        inner_arg = torch.clamp(inner_arg, min=1.0 + 1e-7)
        
        radii = (1.0 / sqrt_c) * torch.acosh(inner_arg)
        
    elif isinstance(manifold, PoincareBall):
        # Poincaré ball:  r = (1/√c) * artanh(√c * ||x||)
        # where ||x|| is Euclidean norm
        
        sqrt_c = torch.sqrt(torch.tensor(curvature, device=points.device))
        
        # Euclidean norm
        norms = torch.norm(points, dim=-1)  # [N]
        
        # For numerical stability: artanh(x) requires |x| < 1
        # In Poincaré:  √c * ||x|| should be < 1
        inner_arg = sqrt_c * norms
        
        # Clamp to stay within Poincaré ball
        
        radii = (1.0 / sqrt_c) * torch.atanh(inner_arg)
        
    else:
        raise ValueError(f"Unsupported manifold type: {type(manifold)}")
    
    return radii


def compute_radial_diversity_metrics(z_trajectory, manifold):
    """
    Compute metrics for analyzing radial diversity (for logging).
    
    Returns:
        dict with mean_radius, radius_variance, radius_std, radius_range
    """
    # Flatten to [N, D]
    if z_trajectory.dim() == 4:
        B, F, N, D = z_trajectory.shape
        z_flat = z_trajectory.reshape(B * F * N, D)
    elif z_trajectory.dim() == 3:
        B, N, D = z_trajectory.shape
        z_flat = z_trajectory. reshape(B * N, D)
    else:
        z_flat = z_trajectory
    
    radii = compute_hyperbolic_radius(z_flat, manifold)
    
    return {
        'mean_radius': radii.mean().item(),
        'radius_variance': radii.var().item(),
        'radius_std': radii. std().item(),
        'radius_min': radii.min().item(),
        'radius_max': radii.max().item(),
        'radius_range': (radii.max() - radii.min()).item()
    }

def curvature_regularization_loss(z_trajectory, manifold, r_threshold=0.5, 
                                   formulation='mean', r_min_per_point=0.3, beta=1.0):
    """
    Prevent embeddings from collapsing to the origin in hyperbolic space. 
    
    Ensures the model actually utilizes hyperbolic geometry by keeping
    embeddings at a minimum distance from the origin.
    
    Args:
        z_trajectory: Hyperbolic embeddings
                     [B, F, N, D] - batch, features, segments, embedding_dim
                     [B, N, D] - batch, segments, embedding_dim
                     [B, D] - batch, embedding_dim
        manifold: geoopt manifold (Poincare or Lorentz)
        r_threshold:  Minimum desired mean radius (default:  0.5)
        formulation: 'mean' (weaker) or 'per_point' (stronger)
        r_min_per_point:  Minimum radius per point (for per_point formulation)
        beta: weighting factor
    
    Returns:
        loss: scalar curvature regularization loss
    """
    # Handle different input dimensions
    if z_trajectory.dim() == 4:  
        B, F, N, D = z_trajectory.shape
        z_flat = z_trajectory.reshape(B * F * N, D)
    elif z_trajectory.dim() == 3:
        B, N, D = z_trajectory.shape
        z_flat = z_trajectory.reshape(B * N, D)
    elif z_trajectory.dim() == 2:
        B, D = z_trajectory.shape
        z_flat = z_trajectory
    else:
        raise ValueError(f"Expected 2D, 3D or 4D tensor, got shape {z_trajectory.shape}")
    
    # Compute radii for all points
    radii = compute_hyperbolic_radius(z_flat, manifold)  # [N_total]
    
    if formulation == 'mean': 
        # Mean radius formulation (from image):
        # L_curv = max(0, r_threshold - (1/B) * Σ r_b)
        mean_radius = radii.mean()
        loss = Func.relu(r_threshold - mean_radius)
        
    elif formulation == 'per_point':
        # Per-point formulation (stronger):
        # L_curv = (1/B) * Σ max(0, r_min - r_b)
        per_point_violations = Func.relu(r_min_per_point - radii)
        loss = per_point_violations.mean()
        
    else:
        raise ValueError(f"Unknown formulation:  {formulation}. Use 'mean' or 'per_point'")
    
    return beta * loss


def compute_curvature_metrics(z_trajectory, manifold, r_threshold=0.5):
    """
    Compute metrics for analyzing curvature utilization (for logging).
    
    Returns:
        dict with radius statistics and utilization assessment
    """
    # Flatten to [N, D]
    if z_trajectory.dim() == 4:
        B, F, N, D = z_trajectory.shape
        z_flat = z_trajectory.reshape(B * F * N, D)
    elif z_trajectory.dim() == 3:
        B, N, D = z_trajectory.shape
        z_flat = z_trajectory. reshape(B * N, D)
    else:
        z_flat = z_trajectory
    
    radii = compute_hyperbolic_radius(z_flat, manifold)
    
    mean_radius = radii.mean().item()
    
    # Assess if hyperbolic geometry is being utilized
    if mean_radius < 0.1:
        status = "CRITICAL:  Near-origin collapse"
    elif mean_radius < r_threshold:
        status = "WARNING: Underutilizing hyperbolic space"
    else:
        status = "OK: Good hyperbolic utilization"
    
    return {
        'mean_radius': mean_radius,
        'radius_std': radii.std().item(),
        'radius_min': radii.min().item(),
        'radius_max': radii. max().item(),
        'pct_below_threshold': (radii < r_threshold).float().mean().item() * 100,
        'status': status
    }