import torch
import torch.nn as nn
import geoopt

class EuclideanHorizonSegmentReconstructionHead(nn.Module):
    """
    Euclidean equivalent of HorizonHyperbolicSegmentReconstructionHead.
    Removes logmap0 since Euclidean space needs no manifold mapping.
    """
    def __init__(self, encode_dim, output_dim, num_pred_segments, segment_length,
                 hidden_dim=64, dropout=0.1):
        super().__init__()
        self.num_pred_segments = num_pred_segments
        self.segment_length = segment_length
        self.output_dim = output_dim
        self.encode_dim = encode_dim
        
        # Identical MLP architecture — only logmap0 removed
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
        z_t: [B, num_pred_segments, encode_dim] - Euclidean point
        returns: [B*output_dim, pred_len]
        """
        B, N, D = z_t.shape
        
        # No logmap0 needed — z_t is already in Euclidean space
        segment_flat = self.fc(z_t)
        segment = segment_flat.reshape(B * self.output_dim, N * self.segment_length)
        return segment