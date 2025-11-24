import torch
import torch.nn as nn
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))


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


class SegmentParallelEuclidean(nn.Module):
    """
    Parallel encoder with per-feature projections for each decomposition component.
    """
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64,
                segment_length=24, embed_dropout=0.1, use_segment_norm=False,
                use_per_feature=True, n_layer=2):  # ← NEW
        super().__init__()
        
        # Store hidden_dim for SegmentMLPEmbed
        self.hidden_dim = hidden_dim
        
        # 4 parallel encoders (one per component)
        self.trend_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            output_dim=embed_dim, 
            hidden_dim=hidden_dim,
            segment_length=segment_length,
            dropout=embed_dropout,
            use_segment_norm=use_segment_norm,
            n_layer=n_layer,
            use_per_feature=use_per_feature  # NEW
        )
        
        self.fine_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            output_dim=embed_dim, 
            hidden_dim=hidden_dim,
            segment_length=segment_length,
            dropout=embed_dropout,
            n_layer=n_layer,
            use_segment_norm=use_segment_norm,
            use_per_feature=use_per_feature  # NEW
        )
        
        self.coarse_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            output_dim=embed_dim, 
            hidden_dim=hidden_dim,
            segment_length=segment_length,
            dropout=embed_dropout,
            n_layer=n_layer,
            use_segment_norm=use_segment_norm,
            use_per_feature=use_per_feature  # NEW
        )
        
        self.residual_embed = SegmentMLPEmbed(
            lookback=lookback, 
            input_dim=input_dim, 
            output_dim=embed_dim, 
            hidden_dim=hidden_dim,
            segment_length=segment_length,
            dropout=embed_dropout,
            n_layer=n_layer,
            use_segment_norm=use_segment_norm,
            use_per_feature=use_per_feature  # NEW
        )
    
    def forward(self, trend, fine, coarse, residual):
        """
        Encode decomposed time series components.
        
        Args:
            trend: [B, seq_len, input_dim] 
            fine: [B, seq_len, input_dim] 
            coarse: [B, seq_len, input_dim]
            residual: [B, seq_len, input_dim]
        
        Returns:
            dict with component and combined embeddings
        """
        # Embed each component
        e_trend = self.trend_embed(trend)
        e_fine = self.fine_embed(fine)
        e_coarse = self.coarse_embed(coarse)
        e_residual = self.residual_embed(residual)
        
        # Combine (equal weighting)
        combined_e = e_trend + e_fine + e_coarse + e_residual

        return {
            "trend_e": e_trend,
            "seasonal_fine_e": e_fine,
            "seasonal_coarse_e": e_coarse,
            "residual_e": e_residual,
            "combined_e": combined_e
        }