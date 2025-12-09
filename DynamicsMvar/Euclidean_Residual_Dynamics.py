import torch
import torch.nn as nn
from spec import RevIN

class ResidualDynamics(nn.Module):
    """
    Residual block for dynamics prediction.
    Outputs: z_next = z + weighted_residual
    This prevents drift by keeping updates close to identity mapping.
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.3, n_layers=2):
        super().__init__()
        
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
        
        # Output layer (maps back to embed_dim)
        layers.append(nn.Linear(hidden_dim, embed_dim))
        
        self.residual_net = nn.Sequential(*layers)
        
        # Learnable residual weight (initialized near 0 for stability)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, z, average_velocity=None):
        """
        Args:
            z: [B, embed_dim] - current state
        Returns:
            z_next: [B, embed_dim] - next state with residual connection
        """
        if average_velocity is not None:
            backward_trajectory = average_velocity
        else:
            backward_trajectory = z
        residual = self.residual_net(backward_trajectory)
        # Residual connection: z_next = z + α * f(z)
        # where α is learnable and starts small
        return backward_trajectory + self.residual_weight * residual

