import geoopt 
import torch 
import torch.nn as nn
from spec import safe_expmap
def lorentzian_norm(x, manifold, eps=1e-8):
    """
    Compute Lorentzian norm: ||x||_L = sqrt(x_0^2 - sum(x_i^2))
    
    Args:
        x: [B, n+1] points in Lorentz model (time-like component first)
        manifold: geoopt.Lorentz instance
        eps: numerical stability constant
    
    Returns:
        norm: [B] Lorentzian norms
    """
    # Lorentzian inner product: <x, x>_L = -x_0^2 + x_1^2 + ... + x_n^2
    lorentz_inner = -x[:, 0]**2 + (x[:, 1:]**2).sum(dim=-1)
    
    # Lorentzian norm: ||x||_L = sqrt(-<x,x>_L) = sqrt(x_0^2 - sum(x_i^2))
    norm = torch.sqrt(-lorentz_inner + eps)
    
    return norm


def lorentzian_residual_update(x_current, x_update, manifold, alpha=0.7, eps=1e-8, c=1.0):
    """
    Lorentzian residual update (LResNet formula):
    
    x_next = (α*x_current + (1-α)*x_update) / (sqrt(c) * ||α*x_current + (1-α)*x_update||_L)
    
    This is the hyperbolic analog of the Euclidean residual: x_next = x + f(x)
    
    Args:
        x_current: [B, n+1] current state on Lorentz manifold
        x_update: [B, n+1] predicted update on Lorentz manifold  
        manifold: geoopt.Lorentz instance
        alpha: weight for current state (0 < alpha < 1)
        eps: numerical stability constant
    
    Returns:
        x_next: [B, n+1] updated state on Lorentz manifold
    """
    # Ensure inputs are on manifolddef lorentzian_residual_update(x_current, x_update, manifold, alpha=0.7, eps=1e-8, c=1.0):
    """Geodesic interpolation between x_current and x_update"""
    x_current = manifold.projx(x_current)
    x_update = manifold.projx(x_update)
    
    # Compute tangent direction from current to update
    tangent_direction = manifold.logmap(x_current, x_update)
    
    
    # Scale by (1-alpha) to interpolate
    scaled_tangent = (1 - alpha) * tangent_direction
    
    # Map back to manifold
    x_next = safe_expmap(manifold, x_current, scaled_tangent)
    x_next = manifold.projx(x_next)

    return x_next, x_current

    
class HyperbolicLorentzDynamics(nn.Module):
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
        # print(f"backward trajectoru: {backward_trajectory}")
        velocity = self.velocity_net(backward_trajectory)  # [B, embed_dim]
        print(f"velocity net: {velocity}")
        velocity = torch.clamp(velocity, min=-5.0, max=5.0)

        scale = torch.sigmoid(self.velocity_scale)  # 0.5 initially
        velocity = velocity * scale * 0.3
        print(f"Velocity: {velocity}")
        # Map velocity to manifold
        x_update = safe_expmap(self.manifold, x_current, velocity)  # [B, embed_dim+1]
        print(f"X update {x_update}")
        x_update = self.manifold.projx(x_update)
        print(f"X update after projx {x_update}")
        # Apply Lorentzian residual update
        alpha = torch.sigmoid(self.alpha)  # Constrain to (0, 1)
        x_next, x_current = lorentzian_residual_update(
            x_current, x_update, self.manifold, alpha=alpha
        )
        
        return x_next, x_current


class HyperbolicLorentzMultiStepDynamics(nn.Module):
    """
    Apply multiple Lorentzian residual updates for iterative refinement.
    
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
            self.dynamics = HyperbolicLorentzDynamics(
                embed_dim, hidden_dim, manifold, n_layers, dropout
            )
        else:
            # Separate dynamics network for each step
            self.dynamics_list = nn.ModuleList([
                HyperbolicLorentzDynamics(embed_dim, hidden_dim, manifold, n_layers, dropout)
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
