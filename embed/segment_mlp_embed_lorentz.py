import torch
import torch.nn as nn
import geoopt
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0


class SegmentMLPEmbed(nn.Module):
    """MLP encoder for segmented data"""
    def __init__(self, input_dim, segment_length, hidden_dim, output_dim, n_layer=2):
        super().__init__()
        
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
                nn.Dropout(0.1)
            ) for _ in range(n_layer)
        ])
        
        # Attention pooling over segments
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: [B, N_segments, segment_length, input_dim] OR [B, seq_len, input_dim]
        returns: [B, output_dim]
        """
        if x.ndim == 3:
            # Non-segmented: [B, seq_len, input_dim]
            # Just use mean pooling
            B, T, C = x.shape
            x = x.reshape(B, T * C)
            x = self.segment_encoder(x)
            for layer in self.layers:
                x = x + layer(x)
            return self.output_proj(x)
        
        # Segmented: [B, N_segments, segment_length, input_dim]
        B, N_seg, seg_len, C = x.shape
        
        # Flatten each segment
        x = x.reshape(B, N_seg, seg_len * C)
        
        # Encode segments
        x = self.segment_encoder(x)  # [B, N_segments, hidden_dim]
        
        # Process
        for layer in self.layers:
            x = x + layer(x)
        
        # Attention pooling
        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores, dim=1)
        x_pooled = (x * attn_weights).sum(dim=1)
        
        return self.output_proj(x_pooled)


class SegmentedParallelLorentz(nn.Module):
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                 curvature=1.0, use_hierarchy=False, hierarchy_scales=[0.5,1.0,1.5,2.0],
                 segment_lengths={'trend': 30, 'weekly': 7, 'daily': 24, 'residual': 1}):
        """
        segment_lengths: dict specifying segment length for each component
            e.g., {'trend': 30, 'weekly': 7, 'daily': 24, 'residual': 1}
        """
        super().__init__()
        
        self.use_hierarchy = use_hierarchy
        if self.use_hierarchy:
            self.log_scales = nn.ParameterList([
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[0]))),
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[1]))),
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[2]))),
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[3])))
            ])

        # Segment-aware encoders (different segment lengths for different components)
        self.trend_embed = SegmentMLPEmbed(
            input_dim=input_dim, 
            segment_length=segment_lengths['trend'],
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=2
        )
        self.seasonal_weekly_embed = SegmentMLPEmbed(
            input_dim=input_dim, 
            segment_length=segment_lengths['weekly'],
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=2
        )
        self.seasonal_daily_embed = SegmentMLPEmbed(
            input_dim=input_dim, 
            segment_length=segment_lengths['daily'],
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=2
        )
        self.residual_embed = SegmentMLPEmbed(
            input_dim=input_dim, 
            segment_length=segment_lengths['residual'],
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=2
        )
        
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.effective_scale = nn.Parameter(torch.tensor(0.1))

    def apply_hierarchy_scaling(self, manifold_point, scale):
        """Scale manifold point radially from origin"""
        tangent = self.manifold.logmap0(manifold_point)
        scaled_tangent = tangent * scale
        scaled_point = safe_expmap0(self.manifold, scaled_tangent)
        return self.manifold.projx(scaled_point)

    def hierarchical_combine(self, z_trend_h, z_weekly_h, z_daily_h, z_residual_h):
        """Sequential hierarchical composition"""
        z_current = z_trend_h

        v_to_weekly = self.manifold.logmap(z_current, z_weekly_h)
        z_current = safe_expmap(self.manifold, 0.25 * v_to_weekly, z_current)
        z_current = self.manifold.projx(z_current)

        v_to_daily = self.manifold.logmap(z_current, z_daily_h)
        z_current = safe_expmap(self.manifold, 0.25 * v_to_daily, z_current)
        z_current = self.manifold.projx(z_current)

        v_to_residual = self.manifold.logmap(z_current, z_residual_h)
        z_current = safe_expmap(self.manifold, 0.25 * v_to_residual, z_current)
        z_current = self.manifold.projx(z_current)

        return z_current

    def tangent_fusion(self, z_trend_h, z_weekly_h, z_daily_h, z_residual_h):
        """Non-hierarchical tangent space fusion"""
        u_trend = self.manifold.logmap0(z_trend_h)
        u_seasonal_weekly = self.manifold.logmap0(z_weekly_h)
        u_seasonal_daily = self.manifold.logmap0(z_daily_h)
        u_residual = self.manifold.logmap0(z_residual_h)

        combined_tangent = u_trend + u_seasonal_weekly + u_seasonal_daily + u_residual
        effective_scale = torch.tanh(self.effective_scale)
        scaled_tangent = combined_tangent * effective_scale
        
        combined_h = safe_expmap0(self.manifold, scaled_tangent)
        combined_h = self.manifold.projx(combined_h)
        
        return combined_h, combined_tangent

    def forward(self, trend, seasonal_weekly, seasonal_daily, residual):
        """
        Inputs can be:
        - Segmented: [B, N_segments, segment_length, input_dim]
        - Non-segmented: [B, seq_len, input_dim]
        
        The SegmentMLPEmbed handles both cases.
        """
        # Encode (handles both segmented and non-segmented)
        z_trend_t = self.trend_embed(trend)
        z_seasonal_weekly_t = self.seasonal_weekly_embed(seasonal_weekly)
        z_seasonal_daily_t = self.seasonal_daily_embed(seasonal_daily)
        z_residual_t = self.residual_embed(residual)
        
        # Map to hyperbolic space
        effective_scale = torch.tanh(self.effective_scale)
        
        z_trend_h = safe_expmap0(self.manifold, z_trend_t * effective_scale)
        z_seasonal_weekly_h = safe_expmap0(self.manifold, z_seasonal_weekly_t * effective_scale)
        z_seasonal_daily_h = safe_expmap0(self.manifold, z_seasonal_daily_t * effective_scale)
        z_residual_h = safe_expmap0(self.manifold, z_residual_t * effective_scale)

        z_trend_h = self.manifold.projx(z_trend_h)
        z_seasonal_weekly_h = self.manifold.projx(z_seasonal_weekly_h)
        z_seasonal_daily_h = self.manifold.projx(z_seasonal_daily_h)
        z_residual_h = self.manifold.projx(z_residual_h)

        # Apply hierarchy
        if self.use_hierarchy:
            trend_scale = torch.exp(self.log_scales[0])
            weekly_scale = torch.exp(self.log_scales[1])
            daily_scale = torch.exp(self.log_scales[2])
            residual_scale = torch.exp(self.log_scales[3])
    
            z_trend_h = self.apply_hierarchy_scaling(z_trend_h, trend_scale)
            z_seasonal_weekly_h = self.apply_hierarchy_scaling(z_seasonal_weekly_h, weekly_scale)
            z_seasonal_daily_h = self.apply_hierarchy_scaling(z_seasonal_daily_h, daily_scale)
            z_residual_h = self.apply_hierarchy_scaling(z_residual_h, residual_scale)
            
            combined_h = self.hierarchical_combine(
                z_trend_h, z_seasonal_weekly_h, z_seasonal_daily_h, z_residual_h
            )
            combined_tangent = self.manifold.logmap0(combined_h)
        else:
            combined_h, combined_tangent = self.tangent_fusion(
                z_trend_h, z_seasonal_weekly_h, z_seasonal_daily_h, z_residual_h
            )

        return {
            "trend_tangent": z_trend_t,
            "seasonal_weekly_tangent": z_seasonal_weekly_t,
            "seasonal_daily_tangent": z_seasonal_daily_t,
            "residual_tangent": z_residual_t,
            "trend_h": z_trend_h,
            "seasonal_weekly_h": z_seasonal_weekly_h,
            "seasonal_daily_h": z_seasonal_daily_h,
            "residual_h": z_residual_h,
            "combined_tangent": combined_tangent,
            "combined_h": combined_h
        }