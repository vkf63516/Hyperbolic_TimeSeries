import geoopt
import torch
import torch.nn as nn
from spec import safe_expmap
# ============================================
# Poincaré Ball Operations
# ============================================

def poincare_norm(x, c=1.0, eps=1e-8):
    """
    Compute Euclidean norm for Poincaré ball.
    
    Args:
        x: [B, n] points in Poincaré ball
        c: curvature (positive)
        eps: numerical stability
    
    Returns:
        norm: [B] Euclidean norms
    """
    return torch.sqrt((x ** 2).sum(dim=-1) + eps)


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
    
    Supports optional avg_velocity parameter for trajectory-aware prediction.
    """
    
    def __init__(self, embed_dim, hidden_dim, manifold, n_layers=3, dropout=0.3):
        """
        Args:
            embed_dim: dimension of tangent space (manifold has embed_dim)
            hidden_dim: hidden layer size
            manifold: geoopt.Lorentz or geoopt.PoincareBall instance
            n_layers: number of layers in velocity network
            dropout: dropout probability
        """
        super().__init__()
        self.manifold = manifold
        self.embed_dim = embed_dim
        
        # Learnable residual weight (initialized to 0.7)
        self.alpha = nn.Parameter(torch.tensor(0.7))
        self.velocity_scale = nn.Parameter(torch.tensor(1.0))

        
        # Velocity network: predicts update in tangent space
        layers = []
        
        # Input layer
        layers.append(nn.Linear(embed_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, embed_dim))
        
        self.velocity_net = nn.Sequential(*layers)
    
    def forward(self, x_current, x_previous=None, avg_velocity=None):
        """
        Compute next state using Poincaré residual update.
        
        Args:
            x_current: [B, embed_dim] current state on manifold
            x_previous: [B, embed_dim] previous state on manifold (for velocity computation)
            avg_velocity: [B, embed_dim] average velocity from trajectory (optional, for moving window)
        
        Returns:
            x_next: [B, embed_dim] next state on manifold
            x_current: [B, embed_dim] current state (to use as previous in next iteration)
        """
        # Compute backward trajectory (velocity input to network)
        if avg_velocity is not None:
            # Use provided average velocity from moving window
            backward_trajectory = avg_velocity
        elif x_previous is None:
            # First iteration: velocity from origin to current
            backward_trajectory = self.manifold.logmap0(x_current)
        else:
            # Compute velocity from previous to current
            backward_trajectory = self.manifold.logmap(x_previous, x_current)
        
        # Predict velocity in tangent space
        velocity = self.velocity_net(backward_trajectory)  # [B, embed_dim]
        
        # Scale velocity
        scale = torch.sigmoid(self.velocity_scale)
        # velocity = velocity * scale 
        
        # Map velocity to manifold via exponential map
        x_update = self.manifold.expmap(x_current, velocity)  # [B, embed_dim]
        x_update = self.manifold.projx(x_update)
        
        # Apply Poincaré residual update (blending)
        alpha = torch.sigmoid(self.alpha)  # Constrain to (0, 1)
        x_next, x_current = poincare_residual_update(
            x_current, x_update, self.manifold, alpha=alpha
        )
        
        return x_next, x_current
