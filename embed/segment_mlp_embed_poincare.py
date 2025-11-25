import torch
import torch.nn as nn
import geoopt
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0


# --------------------------
# Segment-Level MLP Embedder
# --------------------------
class SegmentMLPEmbed(nn.Module):
    """
    MLP encoder for segmented data with per-feature projections (TimeBase-style).
    
    Key improvements:
    1. Per-feature projections (respects feature heterogeneity)
    2. TimeBase initialization (stable training start)
    3. Feature fusion layer (learns inter-feature relationships)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=3, dropout=0.5, 
                 lookback=None, segment_length=24, use_attention_pooling=False, 
                 use_segment_norm=True, use_per_feature=True):  # ← NEW
        super().__init__()
        self.lookback = lookback
        self.segment_length = segment_length
        self.input_dim = input_dim
        self.use_attention_pooling = use_attention_pooling
        self.use_segment_norm = use_segment_norm
        self.use_per_feature = use_per_feature  # NEW
        self.hidden_dim=hidden_dim
        
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
        
        # ========================================
        # NEW: Per-Feature Projections
        # ========================================
        if use_per_feature:
            # Each feature gets its own projection: [segment_length] → [hidden_dim]
            self.feature_projections = nn.ModuleList([
                nn.Linear(segment_length, hidden_dim)
                for _ in range(input_dim)
            ])
            
            # TimeBase-style initialization: start with mean
            for proj in self.feature_projections:
                with torch.no_grad():
                    proj.weight.data = (1.0 / segment_length) * torch.ones_like(proj.weight.data)
                    proj.bias.data.zero_()
            
            # Feature fusion: [input_dim * hidden_dim] → [hidden_dim]
            self.feature_fusion = nn.Sequential(
                nn.Linear(input_dim * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            # Original: Shared projection across all features
            self.input_proj = nn.Linear(input_dim * segment_length, hidden_dim)
        
        # ========================================
        # Existing: Processing Layers
        # ========================================
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(n_layer)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Attention over segments
        if self.use_attention_pooling:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
                nn.Tanh(),
                nn.Linear(max(1, hidden_dim // 2), 1)
            )
    
    def _normalize_segments(self, x):
        """
        Normalize each segment independently.
        x: [B, N_seg, seg_len, C]
        Returns: normalized x
        """
        if self.use_segment_norm:
            seg_mean = x.mean(dim=2, keepdim=True)
            seg_std = x.std(dim=2, keepdim=True).clamp_min(1e-5)
            x_norm = (x - seg_mean) / seg_std
            return x_norm
        else:
            B, N_seg, seg_len, C = x.shape
            x_flat = x.reshape(B, -1, C)
            global_mean = x_flat.mean(dim=1, keepdim=True)
            global_std = x_flat.std(dim=1, keepdim=True).clamp_min(1e-5)
            x_norm = (x_flat - global_mean) / global_std
            return x_norm.reshape(B, N_seg, seg_len, C)
    
    def _project_per_feature(self, x):
        """
        Project each feature separately using per-feature projections.
        
        Args:
            x: [B, N_seg, seg_len, input_dim]
        
        Returns:
            x_proj: [B, N_seg, hidden_dim]
        """
        B, N_seg, seg_len, C = x.shape
        
        # Process each feature with its own projection
        feature_embeds = []
        for feat_idx in range(C):
            # Extract this feature: [B, N_seg, seg_len]
            x_feat = x[:, :, :, feat_idx]
            
            # Flatten batch and segments: [B*N_seg, seg_len]
            x_feat_flat = x_feat.reshape(B * N_seg, seg_len)
            
            # Project with feature-specific layer
            feat_embed = self.feature_projections[feat_idx](x_feat_flat)  # [B*N_seg, hidden_dim]
            
            # Reshape back: [B, N_seg, hidden_dim]
            feat_embed = feat_embed.reshape(B, N_seg, self.hidden_dim)
            
            feature_embeds.append(feat_embed)
        
        # Stack all features: [B, N_seg, input_dim, hidden_dim]
        x_all_features = torch.stack(feature_embeds, dim=2)
        
        # Flatten features: [B, N_seg, input_dim * hidden_dim]
        x_concat = x_all_features.reshape(B, N_seg, C * self.hidden_dim)
        
        # Fuse features: [B, N_seg, input_dim * hidden_dim] → [B, N_seg, hidden_dim]
        x_fused = self.feature_fusion(x_concat)
        
        return x_fused
    
    def forward(self, x):
        """
        x: [B, seq_len, input_dim]
        returns: [B, output_dim]
        """
        B, seq_len, C = x.shape
        
        # 1. Padding if needed
        if self.pad_len > 0:
            pad_start = max(0, seq_len - self.pad_len)
            pad_data = x[:, pad_start:pad_start + self.pad_len, :]
            x = torch.cat([x, pad_data], dim=1)
        
        # 2. Reshape to segments: [B, seq_len, C] → [B, N_seg, seg_len, C]
        x = x.reshape(B, self.num_segments, self.segment_length, C)
        
        # 3. Normalize segments
        x = self._normalize_segments(x)
        
        # 4. Project segments
        if self.use_per_feature:
            # NEW: Per-feature projection
            x = self._project_per_feature(x)  # [B, N_seg, hidden_dim]
        else:
            # Original: Shared projection
            x = x.reshape(B, self.num_segments, self.segment_length * C)
            x = self.input_proj(x)  # [B, N_seg, hidden_dim]
        
        # 5. Process through layers
        for layer in self.layers:
            x = layer(x)
        
        # 6. Pool across segments
        if self.use_attention_pooling:
            attn_scores = self.attention(x)
            attn_weights = torch.softmax(attn_scores, dim=1)
            x_pooled = (x * attn_weights).sum(dim=1)
        else:
            x_pooled = x.mean(dim=1)
        
        # 7. Output projection
        return self.output_proj(x_pooled)
# --------------------------
# Segmented Parallel Poincaré Encoder
# --------------------------
class SegmentedParallelPoincare(nn.Module):
    """
    Segmented version of ParallelPoincare - follows exact same logic but uses segments.
    
    Instead of encoding entire sequences point-by-point, we:
    1. Divide sequences into segments
    2. Encode each segment as a unit
    3. Pool across segments
    4. Map to Poincaré ball
    
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

    def mobius_fusion(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """
        IDENTICAL to ParallelPoincare.mobius_fusion
        
        Non-hierarchical fusion using weighted Möbius addition.
        Properly combines points in Poincaré ball using gyrovector space operations.
        
        Uses the formula: result = w₁⊗x₁ ⊕ w₂⊗x₂ ⊕ w₃⊗x₃ ⊕ w₄⊗x₄
        where ⊗ is Möbius scalar multiplication and ⊕ is Möbius addition
        """
        # Normalize weights to sum to 1
        weights = torch.softmax(self.mobius_weights, dim=0)
        
        # Sequential Möbius addition with weights
        # Start with scaled trend: w₁ ⊗ x₁
        combined_h = self.manifold.mobius_scalar_mul(weights[0], z_trend_h)
        
        # Add scaled coarse: (w₁⊗x₁) ⊕ (w₂⊗x₂)
        scaled_coarse = self.manifold.mobius_scalar_mul(weights[1], z_coarse_h)
        combined_h = self.manifold.mobius_add(combined_h, scaled_coarse)
        
        # Add scaled fine: (...) ⊕ (w₃⊗x₃)
        scaled_fine = self.manifold.mobius_scalar_mul(weights[2], z_fine_h)
        combined_h = self.manifold.mobius_add(combined_h, scaled_fine)
        
        # Add scaled residual: (...) ⊕ (w₄⊗x₄)
        scaled_residual = self.manifold.mobius_scalar_mul(weights[3], z_residual_h)
        combined_h = self.manifold.mobius_add(combined_h, scaled_residual)
        
        # Ensure numerical stability
        combined_h = self.manifold.projx(combined_h)
        
        # Get tangent representation for downstream tasks
        
        return combined_h

    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        IDENTICAL logic to ParallelPoincare.forward, just uses segment encoders.
        
        Encode decomposed time series components to Poincaré ball.
        
        Args:
            trend: [B, seq_len, input_dim]
            seasonal_coarse: [B, seq_len, input_dim]
            seasonal_fine: [B, seq_len, input_dim]
            residual: [B, seq_len, input_dim]
        
        Returns:
            dict with:
                - trend_h, seasonal_coarse_h, seasonal_fine_h, residual_h: [B, embed_dim]
                - combined_h: [B, embed_dim]
        """
        # 1) Encode to Euclidean latent (tangent vectors at origin)
        # ONLY DIFFERENCE: Uses SegmentMLPEmbed instead of MLPEmbed
        z_trend_t = self.trend_embed(trend)  # [B, embed_dim]
        z_seasonal_coarse_t = self.seasonal_coarse_embed(seasonal_coarse)
        z_seasonal_fine_t = self.seasonal_fine_embed(seasonal_fine)
        z_residual_t = self.residual_embed(residual)
        
        # 2) Scaling (SAME as ParallelPoincare)
        effective_scale = torch.tanh(self.effective_scale)  # [-1, 1]
        scaled_trend_embed = z_trend_t * effective_scale
        scaled_coarse_embed = z_seasonal_coarse_t * effective_scale
        scaled_fine_embed = z_seasonal_fine_t * effective_scale
        scaled_residual_embed = z_residual_t * effective_scale

        # 3) Map to hyperbolic space (SAME as ParallelPoincare)
        z_trend_h = safe_expmap0(self.manifold, scaled_trend_embed)
        z_seasonal_coarse_h = safe_expmap0(self.manifold, scaled_coarse_embed)
        z_seasonal_fine_h = safe_expmap0(self.manifold, scaled_fine_embed)
        z_residual_h = safe_expmap0(self.manifold, scaled_residual_embed)

        # 4) Project to manifold (SAME as ParallelPoincare)
        z_trend_h = self.manifold.projx(z_trend_h)
        z_seasonal_coarse_h = self.manifold.projx(z_seasonal_coarse_h)
        z_seasonal_fine_h = self.manifold.projx(z_seasonal_fine_h)
        z_residual_h = self.manifold.projx(z_residual_h)
        
        # 5) Möbius fusion (SAME as ParallelPoincare)
        combined_h = self.mobius_fusion(
            z_trend_h, z_seasonal_coarse_h, z_seasonal_fine_h, z_residual_h
        )

        return {
            "trend_h": z_trend_h,
            "seasonal_coarse_h": z_seasonal_coarse_h,
            "seasonal_fine_h": z_seasonal_fine_h,
            "residual_h": z_residual_h,
            "combined_h": combined_h
        }