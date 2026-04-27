import torch
import torch.nn as nn
import geoopt

class SegmentLinearencodeMultiHorizon(nn.Module):
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

        self.temporal_linears = nn.Linear(self.segment_length, encode_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, 720] - full historical sequence
        Returns:
            z: [B, 64] - ONE point in hyperbolic space
        """
        B = x.shape[0]
        
        if self.pad_seq_len > 0:
            pad = torch.zeros(B, self.pad_seq_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        # Reshape into segments
        x_seg = x.view(B, self.num_segments, self.segment_length)  # [B, num_seg, seg_len]
        
        # encode each segment (KEEP segment structure!)
    
        seg_encode = self.temporal_linears(x_seg)  # [B, num_segments, encode_dim]
        seg_encode = self.dropout(seg_encode)
        
        return seg_encode


class SegmentedParallelEuclideanMultiHorizon(nn.Module):
    """
    Encode each feature for moving window that outputs [B, num_segments, encode_dim].
    Each segment is independently mapped to hyperbolic space.
    """
    
    def __init__(self, lookback, num_channels, encode_dim, curvature=1.0, segment_length=24,
                 encode_dropout=0.1):
        super().__init__()
        
        self.encode_dim = encode_dim
        
        # Segment-aware encoders (output per-segment encodedings)
        self.trend_encode = SegmentLinearencodeMultiHorizon(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels, segment_length=segment_length, dropout=encode_dropout
        )
        self.seasonal_coarse_encode = SegmentLinearencodeMultiHorizon(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels, segment_length=segment_length,dropout=encode_dropout
        )
        self.seasonal_fine_encode = SegmentLinearencodeMultiHorizon(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels, segment_length=segment_length, dropout=encode_dropout
        )
        self.residual_encode = SegmentLinearencodeMultiHorizon(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels, segment_length=segment_length, dropout=encode_dropout
        )
        
    def forward(self, trend, coarse, fine, residual):
        """
        Encode decomposed time series components to Euclidean space.
        
        Args:
            trend: [B, seq_len] 
            fine: [B, seq_len] 
            coarse: [B, seq_len]
            residual: [B, seq_len]
        
        Returns:
            dict with:
                - trend_e, seasonal_fine_e, seasonal_coarse_e, residual_e: [B, encode_dim]
                - combined_e: [B, num_segments, encode_dim]
        """
        # encode each branch to Euclidean latent vector
        e_trend = self.trend_encode(trend)
        e_coarse = self.seasonal_fine_encode(coarse)
        e_fine = self.seasonal_coarse_encode(fine)
        e_residual = self.residual_encode(residual)
        
        # Simple sum (no hierarchy, all components equally weighted)
        combined_e = e_trend + e_coarse + e_fine + e_residual

        return {
            "trend_e": e_trend,
            "seasonal_fine_e": e_fine,
            "seasonal_coarse_e": e_coarse,
            "residual_e": e_residual,
            "combined_e": combined_e
        }