import torch
import torch.nn as nn
import geoopt
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0


class SegmentMLPEmbed(nn.Module):
    """
    MLP encoder for segmented data - follows same logic as MLPEmbed but operates on segments.
    
    Key difference: Instead of pooling across time points, we:
    1. Divide sequence into segments
    2. Encode each segment 
    3. Pool across segments
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=3, dropout=0.5, 
                 lookback=None, segment_length=24, use_attention_pooling=False, 
                 use_segment_norm=True):
        super().__init__()
        self.lookback = lookback
        self.segment_length = segment_length
        self.use_attention_pooling = use_attention_pooling
        self.use_segment_norm = use_segment_norm
        
        # Calculate number of segments (with padding if needed)
        if lookback is not None:
            self.num_segments = lookback // segment_length
            self.pad_len = 0
            if lookback % segment_length != 0:
                self.pad_len = (self.num_segments + 1) * segment_length - lookback
                self.num_segments += 1
        else:
            self.num_segments = None
            self.pad_len = 0
        
        # Instead of projecting single points, we project flattened segments
        # Each segment: [segment_length * input_dim] → [hidden_dim]
        self.input_proj = nn.Linear(input_dim * segment_length, hidden_dim)
        
        # Same layer structure as MLPEmbed
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(n_layer)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Attention over segments (instead of time points)
        if self.use_attention_pooling:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
                nn.Tanh(),
                nn.Linear(max(1, hidden_dim // 2), 1)
            )
    
    def _normalize_segments(self, x):
        """
        Normalize each segment independently (critical for hyperbolic stability).
        x: [B, N_seg, seg_len, C]
        Returns: normalized x
        """
        if self.use_segment_norm:
            # Per-segment z-score normalization
            seg_mean = x.mean(dim=2, keepdim=True)  # [B, N_seg, seg_lem, C]
            seg_std = x.std(dim=2, keepdim=True).clamp_min(1e-5)
            x_norm = (x - seg_mean) / seg_std
            return x_norm
        else:
            # Global normalization across all segments
            B, N_seg, seg_len, C = x.shape
            x_flat = x.reshape(B, -1, C)
            global_mean = x_flat.mean(dim=1, keepdim=True)
            global_std = x_flat.std(dim=1, keepdim=True).clamp_min(1e-5)
            x_norm = (x_flat - global_mean) / global_std
            return x_norm.reshape(B, N_seg, seg_len, C)

    def forward(self, x):
        """
        x: [B, seq_len, input_dim]
        returns: [B, output_dim]  (tangent vectors for Poincaré)
        
        Follows same logic as MLPEmbed but operates on segments:
        1. Segment the sequence
        2. Project each segment to hidden_dim
        3. Process through layers
        4. Pool across segments (attention or mean)
        5. Project to output_dim
        """
        B, seq_len, C = x.shape
        
        # 1. Padding if needed (like orthogonal)
        if self.pad_len > 0:
            pad_start = max(0, seq_len - self.pad_len)
            pad_data = x[:, pad_start:pad_start + self.pad_len, :]
            x = torch.cat([x, pad_data], dim=1)
        
        # 2. Reshape to segments: [B, seq_len, C] → [B, N_seg, seg_len, C]
        x = x.reshape(B, self.num_segments, self.segment_length, C)
        
        # 3. Normalize segments (CRITICAL for hyperbolic!)
        x = self._normalize_segments(x)
        
        # 4. Flatten each segment: [B, N_seg, seg_len, C] → [B, N_seg, seg_len*C]
        x = x.reshape(B, self.num_segments, self.segment_length * C)
        
        # 5. Project segments (analogous to input_proj in MLPEmbed)
        x = self.input_proj(x)  # [B, N_seg, hidden_dim]
        
        # 6. Process through layers (same as MLPEmbed)
        for layer in self.layers:
            x = layer(x)
        
        # 7. Pool across segments (analogous to pooling across time in MLPEmbed)
        if self.use_attention_pooling:
            attn_scores = self.attention(x)  # [B, N_seg, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)  # [B, N_seg, 1]
            x_pooled = (x * attn_weights).sum(dim=1)  # [B, hidden_dim]
        else:
            x_pooled = x.mean(dim=1)  # mean pooling across segments
        
        # 8. Output projection (same as MLPEmbed)
        return self.output_proj(x_pooled)  # [B, output_dim]

class SegmentParallelEuclidean(nn.Module):
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64,
                segment_length=24, embed_dropout=0.1, use_segment_norm=False):
        super().__init__()
        # 5 parallel Mamba encoder blocks
        
        self.trend_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            output_dim=embed_dim, 
            hidden_dim=hidden_dim,
            segment_length=segment_length,
            dropout=embed_dropout,
            use_segment_norm=use_segment_norm
        )
        self.fine_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            output_dim=embed_dim, 
            hidden_dim=hidden_dim,
            segment_length=segment_length,
            dropout=embed_dropout,
            use_segment_norm=use_segment_norm
        )
        self.coarse_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            output_dim=embed_dim, 
            hidden_dim=hidden_dim,
            segment_length=segment_length,
            dropout=embed_dropout,
            use_segment_norm=use_segment_norm
        )
        self.residual_embed = SegmentMLPEmbed(
            lookback=lookback, 
            input_dim=input_dim, 
            output_dim=embed_dim, 
            hidden_dim=hidden_dim,
            segment_length=segment_length,
            dropout=embed_dropout,
            use_segment_norm=use_segment_norm
        )

    
    def forward(self, trend, fine, coarse, residual):
        """
        Encode decomposed time series components to Euclidean space.
        
        Args:
            trend: [B, seq_len, input_dim] 
            fine: [B, seq_len, input_dim] 
            coarse: [B, seq_len, input_dim]
            residual: [B, seq_len, input_dim]
        
        Returns:
            dict with:
                - trend_e, seasonal_fine_e, seasonal_coarse_e, residual_e: [B, embed_dim]
                - combined_e: [B, embed_dim]
        """
        # Embed each branch to Euclidean latent vector
        e_trend = self.trend_embed(trend)
        e_fine = self.fine_embed(fine)
        e_coarse = self.coarse_embed(coarse)
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
