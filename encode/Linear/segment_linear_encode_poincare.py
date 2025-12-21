import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0

class SegmentLinearencode(nn.Module):
    def __init__(self, input_dim, output_dim, segment_length, dropout=0.1,
                 lookback=None, use_segment_norm=True, share_feature_weights=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.segment_length = segment_length
        self.use_segment_norm = use_segment_norm
        self.share_feature_weights = share_feature_weights
        
        # Calculate segments
        if lookback is not None:
            self.num_segments = lookback // segment_length
            self.pad_len = 0
            if lookback % segment_length != 0:
                self.pad_len = (self.num_segments + 1) * segment_length - lookback
                self.num_segments += 1
        
        self.total_len = self.num_segments * segment_length
        
        if share_feature_weights:
            self.shared_linear = nn.Linear(self.total_len, output_dim)
            self.shared_linear.weight = nn.Parameter(
                (1 / self.total_len) * torch.ones([output_dim, self.total_len])
            )
            print(f"SegmentLinearencode (SHARED): {input_dim} features → {self.total_len * output_dim} params")
        else:
            self.feature_linears = nn.ModuleList([
                nn.Linear(self.total_len, output_dim) for _ in range(input_dim)
            ])
            print(f"SegmentLinearencode (PER-FEATURE): {input_dim} features → {input_dim * self.total_len * output_dim} params")
        
        self.dropout = nn.Dropout(dropout)
    
    def _normalize_segments(self, x):
        """Period normalization (from TimeBase)"""
        if self.use_segment_norm:
            # Per-period normalization (TimeBase style)
            period_mean = x.mean(dim=2, keepdim=True)
            x = x - period_mean
            return x, period_mean
        else:
            B, N_seg, seg_len, C = x.shape
            x_flat = x.reshape(B, -1, C)
            mean = x_flat.mean(dim=1, keepdim=True)
            x = x_flat - mean
            return x.reshape(B, N_seg, seg_len, C), mean

    def forward(self, x):
        """
        Forward with optional orthogonal loss.
        
        Args:
            x: [B, seq_len, C]
        
        Returns:
            output: [B, output_dim]
        """
        B, seq_len, C = x.shape
        
        # Padding (TimeBase style)
        if self.pad_len > 0:
            pad_start = max(0, seq_len - self.pad_len)
            pad_data = x[:, pad_start:pad_start + self.pad_len, :]
            x = torch.cat([x, pad_data], dim=1)
        
        # Segment
        x = x.reshape(B, self.num_segments, self.segment_length, C)
        
        # Normalize (TimeBase style)
        x, norm_stats = self._normalize_segments(x)
        
        # Flatten time: [B, total_len, C]
        x = x.reshape(B, self.total_len, C)
        
        # Transpose to [B, C, total_len] (TimeBase/DLinear style)
        x = x.permute(0, 2, 1)
        
        if self.share_feature_weights:
            x_flat = x.reshape(-1, self.total_len)
            output_flat = self.shared_linear(x_flat)
            output = output_flat.reshape(B, C, self.output_dim)
        # Instead of looping
        else:
            # Stack all linear weights: [C, output_dim, total_len]
            weights = torch.stack([self.feature_linears[i].weight for i in range(self.input_dim)], dim=0)
            biases = torch.stack([self.feature_linears[i].bias for i in range(self.input_dim)], dim=0)
            
            # Batch matrix multiply: [B, C, total_len] @ [C, total_len, output_dim] -> [B, C, output_dim]
            output = torch.einsum('bci,cio->bco', x, weights.transpose(1, 2)) + biases.unsqueeze(0)
        
        output = output.mean(dim=1)
        return output

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
    def __init__(self, lookback, input_dim, encode_dim=32,
                 curvature=1.0, segment_length=24, encode_dropout=0.1,
                 use_segment_norm=True, share_feature_weights=False):
        """
        Args:
            lookback: int - lookback window size
            input_dim: int - number of input features
            encode_dim: int - dimension of hyperbolic encodedings
            curvature: float - curvature of Poincaré ball (c parameter)
            segment_length: int - length of each segment (e.g., 24 for daily segments in hourly data)
            n_layer: int - number of linear layers
            use_attention_pooling: bool - use attention pooling over segments
            use_segment_norm: bool - normalize each segment independently
        """
        super().__init__()
        
        # Segment-aware Linear encoders 
        self.trend_encode = SegmentLinearencode(
            input_dim=input_dim,
            output_dim=encode_dim,
            lookback=lookback,
            segment_length=segment_length,
            use_segment_norm=use_segment_norm,
            dropout=encode_dropout,
            share_feature_weights=share_feature_weights
        )
        self.seasonal_coarse_encode = SegmentLinearencode(
            input_dim=input_dim,
            output_dim=encode_dim,
            lookback=lookback,
            segment_length=segment_length,
            use_segment_norm=use_segment_norm,
            dropout=encode_dropout,
            share_feature_weights=share_feature_weights
        )
        self.seasonal_fine_encode = SegmentLinearencode(
            input_dim=input_dim,
            output_dim=encode_dim,
            lookback=lookback,
            segment_length=segment_length,
            use_segment_norm=use_segment_norm,
            dropout=encode_dropout,
            share_feature_weights=share_feature_weights
        )
        self.residual_encode = SegmentLinearencode(
            input_dim=input_dim,
            output_dim=encode_dim,
            lookback=lookback,
            segment_length=segment_length,
            use_segment_norm=use_segment_norm,
            dropout=encode_dropout,
            share_feature_weights=share_feature_weights
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
                - trend_h, seasonal_coarse_h, seasonal_fine_h, residual_h: [B, encode_dim]
                - combined_h: [B, encode_dim]
        """
        # 1) Encode to Euclidean latent (tangent vectors at origin)
        # ONLY DIFFERENCE: Uses SegmentMLPencode instead of MLPencode
        z_trend_t = self.trend_encode(trend)  # [B, encode_dim]
        z_seasonal_coarse_t = self.seasonal_coarse_encode(seasonal_coarse)
        z_seasonal_fine_t = self.seasonal_fine_encode(seasonal_fine)
        z_residual_t = self.residual_encode(residual)
        
        # 2) Scaling (SAME as ParallelPoincare)
        effective_scale = torch.tanh(self.effective_scale)
        scaled_trend_encode = z_trend_t * effective_scale
        scaled_coarse_encode = z_seasonal_coarse_t * effective_scale
        scaled_fine_encode = z_seasonal_fine_t * effective_scale
        scaled_residual_encode = z_residual_t * effective_scale

        # 3) Map to hyperbolic space (SAME as ParallelPoincare)
        z_trend_h = safe_expmap0(self.manifold, scaled_trend_encode)
        z_seasonal_coarse_h = safe_expmap0(self.manifold, scaled_coarse_encode)
        z_seasonal_fine_h = safe_expmap0(self.manifold, scaled_fine_encode)
        z_residual_h = safe_expmap0(self.manifold, scaled_residual_encode)

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