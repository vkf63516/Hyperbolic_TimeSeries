import geoopt
import torch
import torch.nn as nn
from spec import safe_expmap
# ============================================
# Poincaré Ball Operations
# ============================================

def poincare_residual_update(x_current, x_update, manifold, alpha=0.7):
    """
    Poincaré ball residual update using Möbius operations:
    
    x_next = (α ⊗ x_current) ⊕_c ((1-α) ⊗ x_update)
    
    Args:
        x_current: [B, n] current state in Poincaré ball
        x_update: [B, n] predicted update in Poincaré ball
        manifold: geoopt.PoincareBall instance
        alpha: weight for current state (0 < alpha < 1)
    
    Returns:
        x_next: [B, n] updated state in Poincaré ball
        x_current: [B, n] current state (for next iteration)
    """
    # Ensure inputs are on manifold
    x_current = manifold.projx(x_current)
    x_update = manifold.projx(x_update)
    
    # Möbius scalar multiplications
    alpha_x = manifold.mobius_scalar_mul(alpha, x_current)
    beta_x = manifold.mobius_scalar_mul(1 - alpha, x_update)
    
    # Möbius addition
    x_next = manifold.mobius_add(alpha_x, beta_x)
    
    # Project back to manifold
    x_next = manifold.projx(x_next)
    
    return x_next, x_current

class HyperbolicPoincareDynamics(nn.Module):
    """
    Hyperbolic dynamics network using Poincaré residual updates.
    
    Only 2 learnable parameters:
    - alpha: Poincaré residual blending weight
    - step_size: velocity scaling factor
    """
    
    def __init__(self, encode_dim, manifold):
        """
        Args:
            encode_dim: dimension of encodedings (from your segment encodeder)
            manifold: geoopt.PoincareBall instance
        """
        super().__init__()
        self.manifold = manifold
        self.encode_dim = encode_dim
        
        # Only 2 learnable parameters and one linear layer!
        self.alpha = nn.Parameter(torch.tensor(0.7))
        self.step_size = nn.Parameter(torch.tensor(1.0))
        self.velocity_net = nn.Linear(encode_dim, encode_dim)
    
    def forward(self, x_current, x_previous=None, average_velocity=None):
        """
        Compute next state using LINEAR velocity + Poincaré residual update.
        
        Same interface as your original HyperbolicPoincareDynamics!
        
        Args:
            x_current: [B, encode_dim] current encodeding on manifold
            x_previous: [B, encode_dim] previous encodeding (for velocity)
            avg_velocity: [B, encode_dim] pre-computed velocity (optional)
        
        Returns:
            x_next: [B, encode_dim] next encodeding
            x_current: [B, encode_dim] current (to use as previous in next step)
        """
        # ========================================
        # Step 1: Compute velocity (finite difference)
        # ========================================
        if average_velocity is not None:
            backward_trajectory = average_velocity
        elif x_previous is None:
            # First iteration: velocity from origin
            backward_trajectory = self.manifold.logmap0(x_current)
        else:
            # Geodesic velocity from previous to current
            backward_trajectory = self.manifold.logmap(x_previous, x_current)
        
        # ========================================
        # Step 2: Scale velocity
        # ========================================
        step = torch.sigmoid(self.step_size)
        velocity = self.velocity_net(backward_trajectory)
        scaled_velocity = step * velocity
        
        # ========================================
        # Step 3: Extrapolate on geodesic
        # ========================================
        x_update = self.manifold.expmap(x_current, scaled_velocity)
        x_update = self.manifold.projx(x_update)
        
        # ========================================
        # Step 4: Poincaré residual update
        # ========================================
        alpha = torch.sigmoid(self.alpha)
        x_next, x_current = poincare_residual_update(
            x_current, x_update, self.manifold, alpha=alpha
        )
        
        return x_next, x_current
