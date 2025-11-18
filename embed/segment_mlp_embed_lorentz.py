import torch
import torch.nn as nn
import geoopt
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0


class SegmentMLPEmbed(nn.Module):
    """MLP encoder for segmented data"""
    def __init__(self, lookback, input_dim, segment_length, hidden_dim, output_dim,
                 n_layer=2, use_attention_pooling=False, embed_dropout=0.5):
        super().__init__()
        self.num_segments = lookback // segment_length
        self.use_attention_pooling = use_attention_pooling
        
        # Encode each segment
        self.segment_encoder = nn.Sequential(
            nn.Linear(input_dim * segment_length, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Process segment encodings
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(embed_dropout)
            ) for _ in range(n_layer)
        ])
        
        # Attention pooling over segments
        if self.use_attention_pooling:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: [B, N_segments, segment_length, input_dim] 
        returns: [B, output_dim]
        """
        
        # Segmented: [B, N_segments, segment_length, n_features]
        B, N_seg, seg_len, C = x.shape
        
        x = x.reshape(B, N_seg, seg_len * C)
        
        # Encode segments
        x = self.segment_encoder(x)  
        # Process
        for layer in self.layers:
            x = layer(x)
        
        # Attention pooling
        if self.use_attention_pooling:
            attn_scores = self.attention(x)
            attn_weights = torch.softmax(attn_scores, dim=1)
            x_pooled = (x * attn_weights).sum(dim=1)
        else:
            x_pooled = x.mean(dim=1)
        
        return self.output_proj(x_pooled)


class SegmentedParallelLorentz(nn.Module):
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, curvature=1.0,
                 segment_length=24):

        super().__init__()
        
        # Segment-aware encoders (different segment lengths for different components)
        self.trend_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            segment_length=segment_length,
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=2
        )
        self.seasonal_coarse_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            segment_length=segment_length,
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=2
        )
        self.seasonal_fine_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            segment_length=segment_length,
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=2
        )
        self.residual_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            segment_length=segment_length,
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=2
        )
        
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.effective_scale = nn.Parameter(torch.tensor(0.1))

    

    def tangent_fusion(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """Non-hierarchical tangent space fusion"""
        u_trend = self.manifold.logmap0(z_trend_h)
        u_seasonal_coarse = self.manifold.logmap0(z_coarse_h)
        u_seasonal_fine = self.manifold.logmap0(z_fine_h)
        u_residual = self.manifold.logmap0(z_residual_h)

        combined_tangent = u_trend + u_seasonal_coarse + u_seasonal_fine + u_residual
        effective_scale = torch.tanh(self.effective_scale)
        scaled_tangent = combined_tangent * effective_scale
        
        combined_h = safe_expmap0(self.manifold, scaled_tangent)
        combined_h = self.manifold.projx(combined_h)
        
        return combined_h, combined_tangent

    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Inputs can be:
        - Segmented: [B, N_segments, segment_length, input_dim]
        - Non-segmented: [B, seq_len, input_dim]
        
        The SegmentMLPEmbed handles both cases.
        """
        # Encode (handles both segmented and non-segmented)
        z_trend_t = self.trend_embed(trend)
        z_seasonal_coarse_t = self.seasonal_coarse_embed(seasonal_coarse)
        z_seasonal_fine_t = self.seasonal_fine_embed(seasonal_fine)
        z_residual_t = self.residual_embed(residual)
        
        # Map to hyperbolic space
        effective_scale = torch.tanh(self.effective_scale)
        
        z_trend_h = safe_expmap0(self.manifold, z_trend_t * effective_scale)
        z_seasonal_coarse_h = safe_expmap0(self.manifold, z_seasonal_coarse_t * effective_scale)
        z_seasonal_fine_h = safe_expmap0(self.manifold, z_seasonal_fine_t * effective_scale)
        z_residual_h = safe_expmap0(self.manifold, z_residual_t * effective_scale)

        z_trend_h = self.manifold.projx(z_trend_h)
        z_seasonal_coarse_h = self.manifold.projx(z_seasonal_coarse_h)
        z_seasonal_fine_h = self.manifold.projx(z_seasonal_fine_h)
        z_residual_h = self.manifold.projx(z_residual_h)

    
        combined_h, combined_tangent = self.tangent_fusion(
            z_trend_h, z_seasonal_coarse_h, z_seasonal_fine_h, z_residual_h
        )

        return {
            "trend_tangent": z_trend_t,
            "seasonal_coarse_tangent": z_seasonal_coarse_t,
            "seasonal_fine_tangent": z_seasonal_fine_t,
            "residual_tangent": z_residual_t,
            "trend_h": z_trend_h,
            "seasonal_coarse_h": z_seasonal_coarse_h,
            "seasonal_fine_h": z_seasonal_fine_h,
            "residual_h": z_residual_h,
            "combined_tangent": combined_tangent,
            "combined_h": combined_h
        }