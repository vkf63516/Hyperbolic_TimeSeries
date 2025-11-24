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


# --------------------------
# Segmented Parallel Lorentz Encoder
# --------------------------
class SegmentedParallelLorentz(nn.Module):
    """
    Segmented version of ParallelPoincare - follows exact same logic but uses segments.
    
    Instead of encoding entire sequences point-by-point, we:
    1. Divide sequences into segments
    2. Encode each segment as a unit
    3. Pool across segments
    4. Map to Lorentz curvature
    
    Everything else (Möbius fusion, scaling, projection) is IDENTICAL to ParallelPoincare.
    """
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                 curvature=1.0, segment_length=24, n_layer=2, embed_dropout=0.1,
                 use_attention_pooling=False, use_segment_norm=True):
        """
        Args:
            lookback: int - lookback window size
            input_dim: int - number of input features
            embed_dim: int - dimension of hyperbolic embeddings
            hidden_dim: int - hidden dimension for MLP
            curvature: float - curvature of Poincaré ball (c parameter)
            segment_length: int - length of each segment (e.g., 24 for daily segments in hourly data)
            n_layer: int - number of MLP layers
            use_attention_pooling: bool - use attention pooling over segments
            use_segment_norm: bool - normalize each segment independently
        """
        super().__init__()
        
        # Segment-aware MLP encoders (replacing point-level MLPEmbed)
        self.trend_embed = SegmentMLPEmbed(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            n_layer=n_layer,
            lookback=lookback,
            segment_length=segment_length,
            use_attention_pooling=use_attention_pooling,
            use_segment_norm=use_segment_norm,
            dropout=embed_dropout
        )
        self.seasonal_coarse_embed = SegmentMLPEmbed(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            n_layer=n_layer,
            lookback=lookback,
            segment_length=segment_length,
            use_attention_pooling=use_attention_pooling,
            use_segment_norm=use_segment_norm,
            dropout=embed_dropout
        )
        self.seasonal_fine_embed = SegmentMLPEmbed(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            n_layer=n_layer,
            lookback=lookback,
            segment_length=segment_length,
            use_attention_pooling=use_attention_pooling,
            use_segment_norm=use_segment_norm,
            dropout=embed_dropout
        )
        self.residual_embed = SegmentMLPEmbed(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            n_layer=n_layer,
            lookback=lookback,
            segment_length=segment_length,
            use_attention_pooling=use_attention_pooling,
            use_segment_norm=use_segment_norm,
            dropout=embed_dropout
        )
        
        # Poincaré ball manifold (SAME as ParallelPoincare)
        self.manifold = geoopt.manifolds.PoincareBall(c=curvature)
        
        # Scaling parameter (SAME as ParallelPoincare)
        self.effective_scale = nn.Parameter(torch.tensor(0.1))
        
        # Learnable weights for Möbius combination (SAME as ParallelPoincare)
        self.mobius_weights = nn.Parameter(torch.ones(4) * 0.25)

    def weighted_lorentz_mean(self, points, weights):
        """
        Compute weighted Einstein midpoint (Fréchet mean) in Lorentz manifold.
        This is the proper way to combine multiple points in hyperbolic space.
        
        Uses iterative algorithm to find the point that minimizes weighted distances.
        
        Args:
            points: list of [B, embed_dim+1] - points on Lorentz manifold
            weights: [num_points] - normalized weights
        
        Returns:
            mean_point: [B, embed_dim+1] - weighted mean on manifold
        """
        # Initialize at weighted tangent space mean
        tangents = [self.manifold.logmap0(p) for p in points]
        weighted_tangent = sum(w * t for w, t in zip(weights, tangents))
        current_mean = safe_expmap0(self.manifold, weighted_tangent)
        current_mean = self.manifold.projx(current_mean)
        
        # Iterative refinement (Karcher flow)
        # Usually converges in 5-10 iterations
        for _ in range(10):
            # Compute tangent vectors from current mean to each point
            tangent_vecs = [self.manifold.logmap(current_mean, p) for p in points]
            
            # Weighted sum in tangent space at current mean
            weighted_vec = sum(w * v for w, v in zip(weights, tangent_vecs))
            
            # Check convergence
            if torch.norm(weighted_vec, dim=-1).max() < 1e-5:
                break
            
            # Move along weighted direction
            current_mean = safe_expmap(self.manifold, current_mean, 0.5 * weighted_vec)
            current_mean = self.manifold.projx(current_mean)
        
        return current_mean

    def lorentz_fusion(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """
        Non-hierarchical fusion using weighted Einstein midpoint in Lorentz space.
        This is the geometrically correct way to combine points in hyperbolic space.
        
        Args:
            z_trend_h, z_coarse_h, z_fine_h, z_residual_h: [B, embed_dim+1]
        
        Returns:
            combined_h: [B, embed_dim+1] - combined point on manifold
            combined_tangent: [B, embed_dim] - tangent vector representation at origin
        """
        # Normalize weights to sum to 1
        weights = torch.softmax(self.lorentz_weights, dim=0)
        
        # Collect all points
        points = [z_trend_h, z_coarse_h, z_fine_h, z_residual_h]
        
        # Compute weighted Einstein midpoint
        combined_h = self.weighted_lorentz_mean(points, weights)
                
        return combined_h

    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Encode decomposed time series components to Lorentz hyperbolic space.
        
        Args:
            trend: [B, seq_len, input_dim] 
            seasonal_coarse: [B, seq_len, input_dim] 
            seasonal_fine: [B, seq_len, input_dim] 
            residual: [B, seq_len, input_dim] 
        
        Returns:
            dict with:
                - trend_tangent, seasonal_coarse_tangent, seasonal_fine_tangent, residual_tangent: [B, embed_dim]
                - trend_h, seasonal_coarse_h, seasonal_fine_h, residual_h: [B, embed_dim+1]
                - combined_tangent: [B, embed_dim]
                - combined_h: [B, embed_dim+1]
        """
        # 1) Encode to Euclidean latent (tangent vectors at origin)
        z_trend_t = self.trend_embed(trend)  # [B, embed_dim]
        z_seasonal_coarse_t = self.seasonal_coarse_embed(seasonal_coarse)
        z_seasonal_fine_t = self.seasonal_fine_embed(seasonal_fine)
        z_residual_t = self.residual_embed(residual)
        
        # 2) Scaling to prevent numerical instability
        effective_scale = torch.tanh(self.effective_scale)  # [-1, 1]
        scaled_trend_embed = z_trend_t * effective_scale
        scaled_coarse_embed = z_seasonal_coarse_t * effective_scale
        scaled_fine_embed = z_seasonal_fine_t * effective_scale
        scaled_residual_embed = z_residual_t * effective_scale

        # 3) Map to hyperbolic space (Lorentz model)
        # expmap0: tangent space at origin → manifold
        # Result: [B, embed_dim+1] (extra dimension for Lorentz constraint)
        z_trend_h = safe_expmap0(self.manifold, scaled_trend_embed)
        z_seasonal_coarse_h = safe_expmap0(self.manifold, scaled_coarse_embed)
        z_seasonal_fine_h = safe_expmap0(self.manifold, scaled_fine_embed)
        z_residual_h = safe_expmap0(self.manifold, scaled_residual_embed)

        # 4) Project to manifold (ensure numerical stability)
        # Ensures points satisfy Lorentz constraint: -x_0^2 + x_1^2 + ... = -1/k
        z_trend_h = self.manifold.projx(z_trend_h)
        z_seasonal_coarse_h = self.manifold.projx(z_seasonal_coarse_h)
        z_seasonal_fine_h = self.manifold.projx(z_seasonal_fine_h)
        z_residual_h = self.manifold.projx(z_residual_h)

    
        # Non-hierarchical: Use weighted Einstein midpoint (Fréchet mean)
        combined_h = self.lorentz_fusion(
            z_trend_h, z_seasonal_coarse_h, z_seasonal_fine_h, z_residual_h
        )

        return {
            "trend_h": z_trend_h,
            "seasonal_coarse_h": z_seasonal_coarse_h,
            "seasonal_fine_h": z_seasonal_fine_h,
            "residual_h": z_residual_h,
            "combined_h": combined_h
        }