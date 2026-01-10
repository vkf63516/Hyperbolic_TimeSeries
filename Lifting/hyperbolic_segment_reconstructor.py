import torch
import torch.nn as nn
import geoopt


class HyperbolicSegmentReconstructionHead(nn.Module):
    """
    Simple segment reconstructor - follows HyperbolicReconstructionHead design.
    Outputs flattened segment then reshapes.
    """
    def __init__(self, encode_dim, output_dim, segment_length, manifold, 
                 hidden_dim=64, n_layers=1, dropout=0.1):
        super().__init__()
        self.manifold = manifold
        self.segment_length = segment_length
        self.output_dim = output_dim
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(encode_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        # Output layer
        layers.append(nn.Linear(hidden_dim, segment_length * output_dim))
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, z_t):
        """
        z_t: [B, encode_dim] - point on hyperbolic manifold
        returns: [B, segment_length, output_dim]
        """
        B = z_t.shape[0]
        v = self.manifold.logmap0(z_t)
        segment_flat = self.fc(v)
        segment = segment_flat.reshape(B, self.segment_length, self.output_dim)
        return segment.squeeze(-1)


