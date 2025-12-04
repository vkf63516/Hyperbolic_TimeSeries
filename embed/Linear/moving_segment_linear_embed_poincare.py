import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from spec import safe_expmap0


class SegmentLinearEmbedMovingWindow(nn.Module):
    """
    Produces ONE embedding per segment (not aggregated).
    Input:  [B, seq_len, C]
    Output: [B, num_segments, embed_dim]  # One embedding per segment
    """
    
    def __init__(self, input_dim, embed_dim, lookback, segment_length=24, 
                 dropout=0.1, use_segment_norm=True, share_feature_weights=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.segment_length = segment_length
        self.lookback = lookback
        self.num_segments = lookback // segment_length
        self.use_segment_norm = use_segment_norm
        self.pad_seq_len = 0
        self.share_feature_weights = share_feature_weights
        
        if self.lookback > self.num_segments * self.segment_length:
            self.pad_seq_len = (self.num_segments + 1) * self.segment_length - self.lookback
            self.num_segments += 1
        
        if share_feature_weights:
            self.temporal_linear = nn.Linear(self.segment_length, self.embed_dim)
        else:
            self.temporal_linears = nn.ModuleList(
                [nn.Linear(self.segment_length, self.embed_dim) for _ in range(self.input_dim)]
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, lookback, input_dim]
        
        Returns:
            seg_embeds: [B, num_segments, embed_dim]  # One embedding per segment
        """
        B, T, C = x.shape
        total_len = self.num_segments * self.segment_length
        
        # Pad if necessary
        if T < total_len:
            pad_length = total_len - T
            pad = torch.zeros(B, pad_length, C, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        # Reshape into segments: [B, num_segments, segment_length, C]
        x_seg = x.view(B, self.num_segments, self.segment_length, C)
        
        # Optional segment normalization
        if self.use_segment_norm:
            mean = x_seg.mean(dim=2, keepdim=True)
            std = x_seg.std(dim=2, keepdim=True) + 1e-6
            x_seg_norm = (x_seg - mean) / std
        else:
            x_seg_norm = x_seg
        
        if self.share_feature_weights:
            # Process all segments and features with shared weights
            x_flat = x_seg_norm.reshape(B * self.num_segments * C, self.segment_length)
            embeds_flat = self.temporal_linear(x_flat)
            embeds = embeds_flat.view(B, self.num_segments, C, self.embed_dim)
            
            # Pool across features for each segment: [B, num_segments, embed_dim]
            seg_embeds = embeds.mean(dim=2)
        else:
            # Process each feature independently, then aggregate
            seg_embeds = []
            for seg_idx in range(self.num_segments):
                feature_embeds = []
                for feat_idx in range(C):
                    # Extract feature data for this segment: [B, segment_length]
                    feat_data = x_seg_norm[:, seg_idx, :, feat_idx]
                    # Embed: [B, embed_dim]
                    embed = self.temporal_linears[feat_idx](feat_data)
                    feature_embeds.append(embed)
                # Average across features for this segment: [B, embed_dim]
                seg_embed_single = torch.stack(feature_embeds, dim=1).mean(dim=1)
                seg_embeds.append(seg_embed_single)
            # Stack across segments: [B, num_segments, embed_dim]
            seg_embeds = torch.stack(seg_embeds, dim=1)
        
        # Apply dropout
        seg_embeds = self.dropout(seg_embeds)
        
        return seg_embeds


class SegmentedParallelPoincareMovingWindow(nn.Module):
    """
    Encoder for moving window that outputs [B, num_segments, embed_dim].
    Each segment is independently mapped to hyperbolic space.
    """
    
    def __init__(self, lookback, input_dim, embed_dim=32, curvature=1.0, 
                 segment_length=24, embed_dropout=0.1, use_segment_norm=True, 
                 share_feature_weights=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Segment-aware encoders (output per-segment embeddings)
        self.trend_embed = SegmentLinearEmbedMovingWindow(
            input_dim=input_dim, embed_dim=embed_dim, lookback=lookback,
            segment_length=segment_length, use_segment_norm=use_segment_norm,
            dropout=embed_dropout, share_feature_weights=share_feature_weights
        )
        self.seasonal_coarse_embed = SegmentLinearEmbedMovingWindow(
            input_dim=input_dim, embed_dim=embed_dim, lookback=lookback,
            segment_length=segment_length, use_segment_norm=use_segment_norm,
            dropout=embed_dropout, share_feature_weights=share_feature_weights
        )
        self.seasonal_fine_embed = SegmentLinearEmbedMovingWindow(
            input_dim=input_dim, embed_dim=embed_dim, lookback=lookback,
            segment_length=segment_length, use_segment_norm=use_segment_norm,
            dropout=embed_dropout, share_feature_weights=share_feature_weights
        )
        self.residual_embed = SegmentLinearEmbedMovingWindow(
            input_dim=input_dim, embed_dim=embed_dim, lookback=lookback,
            segment_length=segment_length, use_segment_norm=use_segment_norm,
            dropout=embed_dropout, share_feature_weights=share_feature_weights
        )
        
        # Poincaré ball manifold
        self.manifold = geoopt.manifolds.PoincareBall(c=curvature)
        
        # Scaling parameter
        self.effective_scale = nn.Parameter(torch.tensor(0.1))
        
        # Möbius fusion weights
        self.mobius_weights = nn.Parameter(torch.ones(4) * 0.25)
    
    def map_segments_to_hyperbolic(self, segment_embeds):
        """
        Map each segment embedding to hyperbolic space independently.
        
        Args:
            segment_embeds: [B, num_segments, embed_dim]
        
        Returns:
            hyperbolic_embeds: [B, num_segments, embed_dim]
        """
        B, N, D = segment_embeds.shape
        
        # Flatten segments for batch processing
        embeds_flat = segment_embeds.reshape(B * N, D)  # [B*N, embed_dim]
        
        # Scale
        effective_scale = F.softplus(self.effective_scale)
        scaled_embeds = embeds_flat * effective_scale
        
        # Map to hyperbolic space
        hyperbolic_flat = safe_expmap0(self.manifold, scaled_embeds)  # [B*N, embed_dim]
        
        # Project to manifold
        hyperbolic_flat = self.manifold.projx(hyperbolic_flat)
        
        # Reshape back to sequence
        hyperbolic_embeds = hyperbolic_flat.view(B, N, D)  # [B, num_segments, embed_dim]
        
        return hyperbolic_embeds
    
    def mobius_fusion_segments(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """
        Fuse components for each segment independently using Möbius addition.
        
        Args:
            z_trend_h, z_coarse_h, z_fine_h, z_residual_h: [B, num_segments, embed_dim]
        
        Returns:
            combined_h: [B, num_segments, embed_dim]
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
        
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len, input_dim]
        
        Returns:
            dict with hyperbolic embeddings [B, num_segments, embed_dim] for each component
        """
        # Encode to per-segment embeddings: [B, num_segments, embed_dim]
        z_trend_segments = self.trend_embed(trend)
        z_coarse_segments = self.seasonal_coarse_embed(seasonal_coarse)
        z_fine_segments = self.seasonal_fine_embed(seasonal_fine)
        z_residual_segments = self.residual_embed(residual)
        
        # Map each segment to hyperbolic space: [B, num_segments, embed_dim]
        z_trend_h = self.map_segments_to_hyperbolic(z_trend_segments)
        z_coarse_h = self.map_segments_to_hyperbolic(z_coarse_segments)
        z_fine_h = self.map_segments_to_hyperbolic(z_fine_segments)
        z_residual_h = self.map_segments_to_hyperbolic(z_residual_segments)
        
        # Möbius fusion for each segment: [B, num_segments, embed_dim]
        combined_h = self.mobius_fusion_segments(z_trend_h, z_coarse_h, z_fine_h, z_residual_h)
        
        return {
            "trend_h": z_trend_h,
            "seasonal_coarse_h": z_coarse_h,
            "seasonal_fine_h": z_fine_h,
            "residual_h": z_residual_h,
            "combined_h": combined_h
        }