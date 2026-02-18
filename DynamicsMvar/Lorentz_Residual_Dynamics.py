import geoopt 
import torch 
import torch.nn as nn
from spec import safe_expmap, safe_expmap0
# from HyperCore.hypercore.nn.linear.lorentz_linear

class LorentzLinear(nn.Module):
    """
    Lorentz-native linear layer using exponential/logarithmic maps.
    Maps Lorentz ? Tangent ? Transform ? Lorentz
    """
    def __init__(self, in_features, out_features, manifold):
        super().__init__()
        self.manifold = manifold
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x_hyp):
        # Map to tangent space at origin
        x_tan = self.manifold.logmap0(x_hyp)
        
        # Apply Euclidean transformation
        y_tan = self.linear(x_tan)
        
        # Map back to manifold
        y_hyp = safe_expmap0(self.manifold, y_tan)
        return self.manifold.projx(y_hyp)

def lorentzian_residual_update(x_current, x_update, manifold, alpha=0.7, eps=1e-8):
    """
    Geodesic interpolation in Lorentz space (stable version).
    
    Implements:  x_next = exp_{x_current}((1-alpha) * log_{x_current}(x_update))
    
    This is more stable than direct weighted combination.
    """
    # Project to manifold
    x_current = manifold.projx(x_current)
    x_update = manifold.projx(x_update)
    
    # Compute tangent direction from current to update
    tangent_direction = manifold.logmap(x_current, x_update)
    
    # Clamp to prevent explosion
    tangent_norm = torch.norm(tangent_direction, dim=-1, keepdim=True).clamp(min=eps)
    max_norm = 5.0
    tangent_direction = tangent_direction / tangent_norm * torch.clamp(tangent_norm, max=max_norm)
    
    # Scale by (1-alpha) to interpolate
    scaled_tangent = (1 - alpha) * tangent_direction
    
    # Map back to manifold
    x_next = safe_expmap(manifold, x_current, scaled_tangent)
    x_next = manifold.projx(x_next)
    
    return x_next, x_current

    
class HyperbolicLorentzDynamics(nn.Module):
    def __init__(self, encode_dim, manifold):
        super().__init__()    
        self.manifold = manifold
        self.encode_dim = encode_dim
        
        self.alpha = nn.Parameter(torch.tensor(0.7))
        self.step_size = nn.Parameter(torch.tensor(0.3))  # Conservative for Exchange
        
        self.velocity_net = nn.Sequential(
            nn.Linear(encode_dim, encode_dim),
            nn.Tanh()  # Already bounds to (-1, 1)
        )
        
        # Small weight initialization
        for m in self.velocity_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.005)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x_current, x_previous=None, average_velocity=None):
        eps = 1e-8
        
        # Get velocity in tangent space
        if average_velocity is not None: 
            backward_trajectory = average_velocity
        elif x_previous is None: 
            backward_trajectory = self.manifold.logmap0(x_current)
        else:
            backward_trajectory = self.manifold.logmap(x_previous, x_current)
        
        # ✅ FIX 11:  Clamp INPUT to velocity network (this one IS needed)
        backward_trajectory = torch.clamp(backward_trajectory, min=-3.0, max=3.0)
        
        # Predict velocity (already bounded by Tanh)
        velocity = self.velocity_net(backward_trajectory)  # ∈ (-1, 1)
        
        # Scale velocity (still bounded)
        step = torch.sigmoid(self.step_size)  # ∈ (0, 1)
        velocity = velocity * step             # ∈ (-1, 1)
        
        # ❌ REMOVED FIX 12: Redundant clamp - velocity already ∈ (-1, 1)
        
        # Apply velocity
        x_update = safe_expmap(self.manifold, x_current, velocity)
        x_update = self.manifold.projx(x_update)
        
        # NaN handling
        if torch.isnan(x_update).any():
            print("⚠️ NaN detected in x_update, returning x_current")
            return x_current, x_current
        
        # Residual update
        alpha = torch.sigmoid(self.alpha)
        x_next, x_current = lorentzian_residual_update(
            x_current, x_update, self.manifold, alpha=alpha
        )
        
        return x_next, x_current
