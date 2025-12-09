import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentLinearEmbedMovingWindow(nn.Module):
    """
    This done for each feature 
    Produces ONE embedding per segment.
    Input:  [B, seq_len]
    Output: [B, num_segments, embed_dim]  # One embedding per segment
    """
    
    def __init__(self, lookback, embed_dim, segment_length=24, dropout=0.1, 
                 use_segment_norm=True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.segment_length = segment_length
        self.lookback = lookback
        self.num_segments = lookback // segment_length
        self.use_segment_norm = use_segment_norm
        self.pad_seq_len = 0
        
        if self.lookback > self.num_segments * self.segment_length:
            self.pad_seq_len = (self.num_segments + 1) * self.segment_length - self.lookback
            self.num_segments += 1
        
        self.temporal_linears = nn.Linear(segment_length, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len] - single feature
        Returns:
            z: [B, num_segments, embed_dim] - trajectory
        """
        B = x.shape[0]
        
        # Pad if necessary
        if self.pad_seq_len > 0:
            pad = torch.zeros(B, self.pad_seq_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        # Reshape into segments
        x_seg = x.view(B, self.num_segments, self.segment_length)  # [B, num_seg, seg_len]
        
        # Segment normalization
        if self.use_segment_norm:
            mean = x_seg.mean(dim=2, keepdim=True)  # [B, num_seg, 1]
            std = x_seg.std(dim=2, keepdim=True) + 1e-6
            x_seg = (x_seg - mean) / std
        
        # Embed each segment (KEEP segment structure!)
        seg_embed = self.temporal_linears(x_seg)  # [B, num_segments, embed_dim]
        seg_embed = self.dropout(seg_embed)
        
        return seg_embed

class SegmentParallelEuclideanMovingWindow(nn.Module):
    def __init__(self, lookback, embed_dim=32, segment_length=24,
                 embed_dropout=0.1, use_segment_norm=True):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Segment-aware encoders (output per-segment embeddings)
        self.trend_embed = SegmentLinearEmbedMovingWindow(
            embed_dim=embed_dim, lookback=lookback, segment_length=segment_length,
            use_segment_norm=use_segment_norm, dropout=embed_dropout
        )
        self.seasonal_coarse_embed = SegmentLinearEmbedMovingWindow(
            embed_dim=embed_dim, lookback=lookback, segment_length=segment_length,
            use_segment_norm=use_segment_norm, dropout=embed_dropout
        )
        self.seasonal_fine_embed = SegmentLinearEmbedMovingWindow(
            embed_dim=embed_dim, lookback=lookback, segment_length=segment_length, 
            use_segment_norm=use_segment_norm, dropout=embed_dropout
        )
        self.residual_embed = SegmentLinearEmbedMovingWindow(
            embed_dim=embed_dim, lookback=lookback, segment_length=segment_length,
            use_segment_norm=use_segment_norm, dropout=embed_dropout
        )
        

    
    def forward(self, trend, fine, coarse, residual):
        """
        Encode decomposed time series components to Euclidean space.
        
        Args:
            trend: [B, seq_len] 
            fine: [B, seq_len] 
            coarse: [B, seq_len]
            residual: [B, seq_len]
        
        Returns:
            dict with:
                - trend_e, seasonal_fine_e, seasonal_coarse_e, residual_e: [B, embed_dim]
                - combined_e: [B, num_segments, embed_dim]
        """
        # Embed each branch to Euclidean latent vector
        e_trend = self.trend_embed(trend)
        e_fine = self.seasonal_fine_embed(fine)
        e_coarse = self.seasonal_coarse_embed(coarse)
        e_residual = self.residual_embed(residual)
        
        # Simple sum (no hierarchy, all components equally weighted)
        combined_e = e_trend + e_fine + e_coarse + e_residual

        return {
            "trend_e": e_trend,
            "seasonal_fine_e": e_fine,
            "seasonal_coarse_e": e_coarse,
            "residual_e": e_residual,
            "combined_e": combined_e
        }
