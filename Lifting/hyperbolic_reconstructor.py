import torch
import torch.nn as nn
import geoopt

class HyperbolicReconstructionHead(nn.Module):
    def __init__(self, embed_dim, output_dim, manifold, hidden_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.manifold = manifold
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(embed_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))  # Match Euclidean
        layers.append(nn.GELU())                 # Match Euclidean
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.fc = nn.Sequential(*layers)
    
    def forward(self, z_t):
        v = self.manifold.logmap0(z_t)
        return self.fc(v)