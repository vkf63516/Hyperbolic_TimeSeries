import torch
import torch.nn as nn
import geoopt


class EuclideanSegmentReconstructionHead(nn.Module):
    """
    Simple segment reconstructor - follows HyperbolicReconstructionHead design.
    Outputs flattened segment then reshapes.
    """
    def __init__(self, embed_dim, output_dim, segment_length, 
                 hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.segment_length = segment_length
        self.output_dim = output_dim
        
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
        layers.append(nn.Linear(hidden_dim, segment_length * output_dim))
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, z_t):
        """
        z_t: [B, embed_dim] - point on hyperbolic manifold
        returns: [B, segment_length, output_dim]
        """
        B = z_t.shape[0]
        segment_flat = self.fc(z_t)
        segment = segment_flat.reshape(B, self.segment_length, self.output_dim)
        return segment


