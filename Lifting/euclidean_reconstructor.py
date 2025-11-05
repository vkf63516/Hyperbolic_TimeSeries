import torch
import torch.nn as nn

class EuclideanReconstructor(nn.Module):
    """
    MLP reconstructor for Euclidean embeddings.
    Maps from latent space back to time series observations.
    """
    def __init__(self, embed_dim, output_dim, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(embed_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))  # Better than BatchNorm for sequences
        layers.append(nn.GELU())                 # Smoother than ReLU
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.fully_connected = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Args:
            z: [B, embed_dim] - Euclidean embedding
        Returns:
            x_recon: [B, output_dim] - reconstructed time series point
        """
        return self.fully_connected(z)