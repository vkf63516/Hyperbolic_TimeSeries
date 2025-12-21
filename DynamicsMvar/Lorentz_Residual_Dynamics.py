import geoopt 
import torch 
import torch.nn as nn
from spec import safe_expmap

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
    
    def __init__(self, encode_dim, segment_length, manifold):
        """
        Args:
            encode_dim: dimension of tangent space (manifold has encode_dim+1)
            hidden_dim: hidden layer size
            manifold: geoopt.Lorentz or geoopt.PoincareBall instance
            n_layers: number of layers in velocity network
            dropout: dropout probability
        """
        super().__init__()    
        self.manifold = manifold
        self.encode_dim = encode_dim
        
        # Only 2 learnable parameters!
        self.alpha = nn.Parameter(torch.tensor(0.7))
        self.step_size = nn.Parameter(torch.tensor(1.0))
        self.temp_lin = nn.Linear(encode_dim, segment_length)
        self.lin_temp = nn.Linear(segment_length, encode_dim)
    
         
    def forward(self, x_current, x_previous=None):
        """
        Compute next state using Lorentzian residual update.
        
        Args:
            x_current: [B, encode_dim+1] current state on manifold
            x_previous: [B, encode_dim+1] previous state on manifold
        
        Returns:
            x_next: [B, encode_dim+1] next state on manifold
        """
        # Map to tangent space at origin]
        if x_previous is None:
            backward_trajectory = self.manifold.logmap0(x_current)  # [B, encode_dim]
        else:
             # Map to tangent space from current step
            dist = self.manifold.dist(x_current, x_previous)
        
            backward_trajectory = self.manifold.logmap(x_previous, x_current)
        # Predict velocity in tangent space
        # print(f"backward trajectoru: {backward_trajectory}")
        basis = self.temp_lin(backward_trajectory)  # [B, encode_dim]
        velocity = self.lin_temp(backward_trajectory)
        print(f"velocity net: {velocity}")
        velocity = torch.clamp(velocity, min=-5.0, max=5.0)

        scale = torch.sigmoid(self.velocity_scale)  # 0.5 initially
        velocity = velocity * scale * 0.3
        print(f"Velocity: {velocity}")
        # Map velocity to manifold
        x_update = safe_expmap(self.manifold, x_current, velocity)  # [B, encode_dim+1]
        print(f"X update {x_update}")
        x_update = self.manifold.projx(x_update)
        print(f"X update after projx {x_update}")
        # Apply Lorentzian residual update
        alpha = torch.sigmoid(self.alpha)  # Constrain to (0, 1)
        x_next, x_current = lorentzian_residual_update(
            x_current, x_update, self.manifold, alpha=alpha
        )
        
        return x_next, x_current


