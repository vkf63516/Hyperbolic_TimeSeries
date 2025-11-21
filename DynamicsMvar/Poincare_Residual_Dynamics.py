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


def mobius_scalar_mult(alpha, x, c=1.0, eps=1e-8):
    """
    Möbius scalar multiplication: α ⊗ x
    
    Formula: α ⊗ x = tanh(α * arctanh(sqrt(c)||x||)) * x / (sqrt(c)||x||)
    
    Args:
        alpha: scalar or [B] tensor
        x: [B, n] points in Poincaré ball
        c: curvature
        eps: numerical stability
    
    Returns:
        result: [B, n] scaled points
    """
    x_norm = poincare_norm(x, c, eps=eps)  # [B]
    sqrt_c = torch.sqrt(torch.tensor(c, device=x.device))
    
    # arctanh(sqrt(c)||x||)
    arctanh_arg = sqrt_c * x_norm
    arctanh_arg = torch.clamp(arctanh_arg, -1 + eps, 1 - eps)  # Numerical stability
    arctanh_val = torch.atanh(arctanh_arg)
    
    # tanh(α * arctanh(...))
    tanh_arg = alpha * arctanh_val
    tanh_val = torch.tanh(tanh_arg)
    
    # Result: tanh(...) * x / (sqrt(c)||x||)
    coeff = tanh_val / (sqrt_c * x_norm + eps)
    result = coeff.unsqueeze(-1) * x
    
    return result


def mobius_add(x, y, c=1.0, eps=1e-8):
    """
    Möbius addition (Einstein midpoint): x ⊕_c y
    
    Formula:
    x ⊕_c y = [(1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y] / [1 + 2c<x,y> + c²||x||²||y||²]
    
    Args:
        x: [B, n] points in Poincaré ball
        y: [B, n] points in Poincaré ball
        c: curvature
        eps: numerical stability
    
    Returns:
        result: [B, n] sum in Poincaré ball
    """
    x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    y_norm_sq = (y ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    xy_dot = (x * y).sum(dim=-1, keepdim=True)      # [B, 1]
    
    # Numerator terms
    term1_coeff = 1 + 2 * c * xy_dot + c * y_norm_sq  # [B, 1]
    term2_coeff = 1 - c * x_norm_sq                   # [B, 1]
    
    numerator = term1_coeff * x + term2_coeff * y
    
    # Denominator
    denominator = 1 + 2 * c * xy_dot + c**2 * x_norm_sq * y_norm_sq + eps
    
    result = numerator / denominator
    
    return result


def poincare_residual_update(x_current, x_update, manifold, alpha=0.7, eps=1e-8):
    """
    Poincaré ball residual update using Möbius operations:
    
    x_next = (α ⊗ x_current) ⊕_c ((1-α) ⊗ x_update)
    
    Args:
        x_current: [B, n] current state in Poincaré ball
        x_update: [B, n] predicted update in Poincaré ball
        manifold: geoopt.PoincareBall instance
        alpha: weight for current state (0 < alpha < 1)
        eps: numerical stability
    
    Returns:
        x_next: [B, n] updated state in Poincaré ball
    """
    # Ensure inputs are on manifold
    x_current = manifold.projx(x_current)
    x_update = manifold.projx(x_update)
    
    # Get curvature
    c = 1.0
    
    # Möbius scalar multiplications
    alpha_x = mobius_scalar_mult(alpha, x_current, c=c, eps=eps)
    beta_x = mobius_scalar_mult(1 - alpha, x_update, c=c, eps=eps)
    
    # Möbius addition
    x_next = mobius_add(alpha_x, beta_x, c=c, eps=eps)
    
    # Project back to manifold
    x_next = manifold.projx(x_next)
    
    return x_next, x_current
    
class HyperbolicPoincareDynamics(nn.Module):
    """
    Hyperbolic dynamics network using Lorentzian residual updates.
    
    Properly respects hyperbolic geometry by:
    1. Predicting velocity in tangent space
    2. Mapping velocity to manifold
    3. Applying Lorentzian residual update
    """
    
    def __init__(self, embed_dim, hidden_dim, manifold, n_layers=3, dropout=0.3):
        """
        Args:
            embed_dim: dimension of tangent space (manifold has embed_dim+1)
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
        self.velocity_scale = nn.Parameter(torch.tensor(0.1))

        
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
    
    def forward(self, x_current, x_previous=None):
        """
        Compute next state using Lorentzian residual update.
        
        Args:
            x_current: [B, embed_dim+1] current state on manifold
            x_previous: [B, embed_dim+1] previous state on manifold
        
        Returns:
            x_next: [B, embed_dim+1] next state on manifold
        """
        # Map to tangent space at origin]
        if x_previous is None:
            backward_trajectory = self.manifold.logmap0(x_current)  # [B, embed_dim]
        else:
             # Map to tangent space from current step
            dist = self.manifold.dist(x_current, x_previous)
        
            backward_trajectory = self.manifold.logmap(x_previous, x_current)
        # Predict velocity in tangent space
        velocity = self.velocity_net(backward_trajectory)  # [B, embed_dim]
        velocity = torch.clamp(velocity, min=-5.0, max=5.0)

        scale = torch.sigmoid(self.velocity_scale)  # 0.5 initially
        velocity = velocity * scale * 0.3
        # print(velocity)
        # Map velocity to manifold
        x_update = safe_expmap(self.manifold, x_current, velocity)  # [B, embed_dim+1]
        
        x_update = self.manifold.projx(x_update)
        # Apply Poincare residual update
        alpha = torch.sigmoid(self.alpha)  # Constrain to (0, 1)
        x_next, x_current = poincare_residual_update(
            x_current, x_update, self.manifold, alpha=alpha
        )
        
        return x_next, x_current


class HyperbolicPoincareMultiStepDynamics(nn.Module):
    """
    Apply multiple Poinc residual updates for iterative refinement.
    
    Useful for:
    - Building up complex predictions gradually
    - Refining embeddings before decoding
    - Capturing multi-scale temporal patterns
    """
    
    def __init__(self, embed_dim, hidden_dim, manifold, num_steps=4, 
                 n_layers=3, dropout=0.3, shared_weights=False):
        """
        Args:
            num_steps: number of residual update steps
            shared_weights: if True, use same network for all steps
        """
        super().__init__()
        self.manifold = manifold
        self.num_steps = num_steps
        self.shared_weights = shared_weights
        
        if shared_weights:
            # Single dynamics network shared across all steps
            self.dynamics = HyperbolicPoincareDynamics(
                embed_dim, hidden_dim, manifold, n_layers, dropout
            )
        else:
            # Separate dynamics network for each step
            self.dynamics_list = nn.ModuleList([
                HyperbolicPoincareDynamics(embed_dim, hidden_dim, manifold, n_layers, dropout)
                for _ in range(num_steps)
            ])
    
    def forward(self, x_init):
        """
        Apply multiple residual updates.
        
        Args:
            x_init: [B, embed_dim+1] initial state
        
        Returns:
            x_final: [B, embed_dim+1] final refined state
            trajectory: list of intermediate states
        """
        x_current = x_init
        trajectory = [x_init]
        
        for step in range(self.num_steps):
            # Get dynamics network for this step
            if self.shared_weights:
                dynamics = self.dynamics
            else:
                dynamics = self.dynamics_list[step]
            
            # Apply residual update
            x_current = dynamics(x_current)
            trajectory.append(x_current)
        
        return x_current, trajectory