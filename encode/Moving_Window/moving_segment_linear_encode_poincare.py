import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from spec import safe_expmap0


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


class SegmentedParallelPoincareMovingWindow(nn.Module):
    """
    Encode each feature for moving window that outputs [B, num_segments, encode_dim].
    Each segment is independently mapped to hyperbolic space.
    """
    
    def __init__(self, lookback, num_channels, encode_dim, curvature=1.0, segment_length=24,
                 encode_dropout=0.1):
        super().__init__()
        
        self.encode_dim = encode_dim
        
        # Segment-aware encoders (output per-segment encodedings)
        self.trend_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels, segment_length=segment_length, dropout=encode_dropout
        )
        self.seasonal_coarse_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels, segment_length=segment_length,dropout=encode_dropout
        )
        self.seasonal_fine_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels, segment_length=segment_length, dropout=encode_dropout
        )
        self.residual_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels, segment_length=segment_length, dropout=encode_dropout
        )
        
        # Poincaré ball manifold
        self.manifold = geoopt.manifolds.PoincareBall(c=curvature)
        
        # Scaling parameter
        self.effective_scale = nn.Parameter(torch.tensor(1.0))
        
        # Möbius fusion weights
        self.mobius_weights = nn.Parameter(torch.ones(4) * 0.25)
    
    def map_segments_to_hyperbolic(self, segment_encodes):
        """
        Map each segment encodeding to hyperbolic space independently.
        
        Args:
            segment_encodes: [B, num_segments, encode_dim]
        
        Returns:
            hyperbolic_encodes: [B, num_segments, encode_dim]
        """
        B, N, D = segment_encodes.shape
        
        # Flatten segments for batch processing
        encodes_flat = segment_encodes.reshape(B * N, D)  # [B*N, encode_dim]
        
        # Scale
        effective_scale = torch.tanh(self.effective_scale)
        scaled_encodes = encodes_flat * effective_scale
        
        # Map to hyperbolic space
        hyperbolic_flat = safe_expmap0(self.manifold, scaled_encodes)  # [B*N, encode_dim]
        
        # Project to manifold
        hyperbolic_flat = self.manifold.projx(hyperbolic_flat)
        
        # Reshape back to sequence
        hyperbolic_encodes = hyperbolic_flat.view(B, N, D)  # [B, num_segments, encode_dim]
        
        return hyperbolic_encodes
    
    def mobius_fusion_segments(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """
        Fuse components for each segment independently using Möbius addition.
        
        Args:
            z_trend_h, z_coarse_h, z_fine_h, z_residual_h: [B, num_segments, encode_dim]
        
        Returns:
            combined_h: [B, num_segments, encode_dim]
        """
        B, N, D = z_trend_h.shape
        
        # Flatten for batch Möbius operations
        z_trend_flat = z_trend_h.reshape(B * N, D)
        z_coarse_flat = z_coarse_h.reshape(B * N, D)
        z_fine_flat = z_fine_h.reshape(B * N, D)
        z_residual_flat = z_residual_h.reshape(B * N, D)
        
        # Normalize weights
        weights = torch.softmax(self.mobius_weights, dim=0)
        
        # Sequential Möbius addition with weights
        combined_flat = self.manifold.mobius_scalar_mul(weights[0], z_trend_flat)
        
        scaled_coarse = self.manifold.mobius_scalar_mul(weights[1], z_coarse_flat)
        combined_flat = self.manifold.mobius_add(combined_flat, scaled_coarse)
        
        scaled_fine = self.manifold.mobius_scalar_mul(weights[2], z_fine_flat)
        combined_flat = self.manifold.mobius_add(combined_flat, scaled_fine)
        
        scaled_residual = self.manifold.mobius_scalar_mul(weights[3], z_residual_flat)
        combined_flat = self.manifold.mobius_add(combined_flat, scaled_residual)
        
        # Ensure numerical stability
        combined_flat = self.manifold.projx(combined_flat)
        
        # Reshape back to sequence
        combined_h = combined_flat.view(B, N, D)
        
        return combined_h
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Encode with segment structure preserved.
        
        Args
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len]
        
        Returns:
            dict with hyperbolic encodedings [B, num_segments, encode_dim] for each component
        """
        # Encode to per-segment encodedings: [B, num_segments, encode_dim]
        z_trend_segments = self.trend_encode(trend)
        z_coarse_segments = self.seasonal_coarse_encode(seasonal_coarse)
        z_fine_segments = self.seasonal_fine_encode(seasonal_fine)
        z_residual_segments = self.residual_encode(residual)
        
        # Map each segment to hyperbolic space: [B, num_segments, encode_dim]
        z_trend_h = self.map_segments_to_hyperbolic(z_trend_segments)
        z_coarse_h = self.map_segments_to_hyperbolic(z_coarse_segments)
        z_fine_h = self.map_segments_to_hyperbolic(z_fine_segments)
        z_residual_h = self.map_segments_to_hyperbolic(z_residual_segments)
        
        # Möbius fusion for each segment: [B, num_segments, encode_dim]
        combined_h = self.mobius_fusion_segments(z_trend_h, z_coarse_h, z_fine_h, z_residual_h)
        
        return {
            "trend_h": z_trend_h,
            "seasonal_coarse_h": z_coarse_h,
            "seasonal_fine_h": z_fine_h,
            "residual_h": z_residual_h,
            "combined_h": combined_h
        }