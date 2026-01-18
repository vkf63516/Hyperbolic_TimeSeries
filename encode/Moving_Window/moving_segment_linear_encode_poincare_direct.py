import torch
import torch.nn as nn
import geoopt
from spec import safe_expmap0
from encode.Moving_Window.moving_segment_linear_encode_poincare import SegmentLinearencodeMovingWindow

class SegmentLinearencodeMovingWindow(nn.Module):
    """
    This done for each feature 
    Produces ONE encodeding per segment.
    Input:  [B, seq_len]
    Output: [B, num_segments, encode_dim]  # One encodeding per segment
    """
    
    def __init__(self, lookback, encode_dim, num_channels, segment_length=24, dropout=0.1, individual=False):
        super().__init__()
        
        self.encode_dim = encode_dim
        self.segment_length = segment_length
        self.lookback = lookback
        self.num_segments = lookback // segment_length
        self.pad_seq_len = 0
        self.num_channels = num_channels
        self.individual = individual
        
        if self.lookback > self.num_segments * self.segment_length:
            self.pad_seq_len = (self.num_segments + 1) * self.segment_length - self.lookback
            self.num_segments += 1
        if self.individual:
            self.temporal_linears = nn.ModuleList()
            for _ in range(self.num_channels):
                self.temporal_linears.append(nn.Linear(segment_length, encode_dim))

        self.temporal_linears = nn.Linear(segment_length, encode_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len] - single feature
        Returns:
            z: [B, num_segments, encode_dim] - trajectory
        """
        B = x.shape[0]
        
        # Pad if necessary
        if self.pad_seq_len > 0:
            pad = torch.zeros(B, self.pad_seq_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        # Reshape into segments
        x_seg = x.view(B, self.num_segments, self.segment_length)  # [B, num_seg, seg_len]
        
        # encode each segment (KEEP segment structure!)
    
        seg_encode = self.temporal_linears(x_seg)  # [B, num_segments, encode_dim]
        seg_encode = self.dropout(seg_encode)
        
        return seg_encode

class DirectPoincareMovingWindow(nn.Module):
    """
    ABLATION: No decomposition - encode raw signal directly. 
    Single encoder instead of 4 component encoders.
    """
    
    def __init__(self, lookback, num_channels, encode_dim, curvature=1.0, 
                 segment_length=24, encode_dropout=0.1):
        super().__init__()
        
        self.encode_dim = encode_dim
        
        # SINGLE encoder for raw input
        self.direct_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, 
            lookback=lookback, 
            num_channels=num_channels,
            segment_length=segment_length, 
            dropout=encode_dropout
        )
        
        # Same manifold as decomposed version
        self.manifold = geoopt.manifolds.PoincareBall(c=curvature)
        
        # Same scaling parameter
        self.effective_scale = nn.Parameter(torch.tensor(1.0))
    
    def map_segments_to_hyperbolic(self, segment_encodes):
        """
        Same as decomposed version - map segments to hyperbolic space. 
        """
        B, N, D = segment_encodes.shape
        encodes_flat = segment_encodes.reshape(B * N, D)
        
        effective_scale = torch.tanh(self.effective_scale)
        scaled_encodes = encodes_flat * effective_scale
        
        hyperbolic_flat = safe_expmap0(self.manifold, scaled_encodes)
        hyperbolic_flat = self.manifold.projx(hyperbolic_flat)
        
        return hyperbolic_flat.view(B, N, D)
    
    def forward(self, x_raw):
        """
        Args:
            x_raw: [B, seq_len] - raw signal (NO decomposition)
        
        Returns:
            dict with hyperbolic encoding [B, num_segments, encode_dim]
        """
        # Encode raw signal directly
        z_segments = self.direct_encode(x_raw)  # [B, num_segments, encode_dim]
        
        # Map to hyperbolic space
        z_h = self.map_segments_to_hyperbolic(z_segments)
        
        # Return in same format as decomposed version for compatibility
        return {
            "combined_h": z_h  # Only this matters for ablation
        }