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
        eps: numerical stability
    
    Returns:
        x_next: [B, n] updated state in Poincaré ball
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
        # velocity = torch.clamp(velocity, min=-5.0, max=5.0)

        scale = torch.sigmoid(self.velocity_scale)  # 0.5 initially
        velocity = velocity * scale * 0.2
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