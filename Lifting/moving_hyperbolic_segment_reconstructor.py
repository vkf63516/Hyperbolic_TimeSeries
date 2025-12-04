import torch
import torch.nn as nn
import geoopt


class HyperbolicSegmentReconstructionHead(nn.Module):
    def __init__(self, embed_dim, output_dim, segment_length, manifold):
        super().__init__()
        self.manifold = manifold
        self.segment_length = segment_length
        self.output_dim = output_dim
        
        # Single linear layer
        self.decoder = nn.Linear(embed_dim, segment_length * output_dim)
    
    def forward(self, z_t):
        # Handle both [B, embed_dim] and [B, N, embed_dim]
        if len(z_t.shape) == 2:  # [B, embed_dim]
            B = z_t.shape[0]
            v = self.manifold.logmap0(z_t)
            segment_flat = self.decoder(v)
            return segment_flat.reshape(B, self.segment_length, self.output_dim)
        else:  # [B, N, embed_dim]
            B, N, D = z_t.shape
            z_flat = z_t.reshape(B * N, D)
            v = self.manifold.logmap0(z_flat)
            segment_flat = self.decoder(v)
            return segment_flat.reshape(B, N, self.segment_length, self.output_dim)