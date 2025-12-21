import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
# from pathlib import Path
import sys 
# sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0

class SegmentLinearencode(nn.Module):
    def __init__(self, input_dim, output_dim, segment_length, dropout=0.1,
                 lookback=None, use_segment_norm=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.segment_length = segment_length
        self.use_segment_norm = use_segment_norm
        
        # Calculate segments
        if lookback is not None:
            self.num_segments = lookback // segment_length
            self.pad_len = 0
            if lookback % segment_length != 0:
                self.pad_len = (self.num_segments + 1) * segment_length - lookback
                self.num_segments += 1
        
        self.total_len = self.num_segments * segment_length
        
        print(f"TimeBaseInspiredencode: {input_dim} features, {self.num_segments} segments")
        
        # Per-feature projections 
        self.feature_linears = nn.ModuleList([
            nn.Linear(self.total_len, output_dim)
            for _ in range(input_dim)
        ])
        
        # Initialize (helps training)
        for linear in self.feature_linears:
            linear.weight = nn.Parameter(
                (1 / self.total_len) * torch.ones([output_dim, self.total_len])
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def _normalize_segments(self, x):
        """Period normalization (from TimeBase)"""
        if self.use_segment_norm:
            # Per-period normalization 
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
        Forward
        
        Args:
            x: [B, seq_len, C]
        
        Returns:
            output: [B, output_dim]
        """
        B, seq_len, C = x.shape
        
        # Padding 
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
        
        # Transpose to [B, C, total_len] 
        x = x.permute(0, 2, 1)
        
        # Per-feature processing 
        output = torch.zeros([B, C, self.output_dim], 
                            dtype=x.dtype, device=x.device)
        
        for i in range(self.input_dim):
            feature_i = x[:, i, :]  # [B, total_len]
            output[:, i, :] = self.feature_linears[i](feature_i)  # [B, output_dim]
        
        self.dropout(output)

        
        # Average across features
        output = output.mean(dim=1)  # [B, output_dim]
        
        return output

# --------------------------
# Segmented Parallel Lorentz Encoder
# --------------------------
class SegmentedParallelLorentz(nn.Module):
    """
    Segmented version of ParallelLorentz - follows exact same logic but uses segments.
    
    Instead of encoding entire sequences point-by-point, we:
    1. Divide sequences into segments
    2. Encode each segment as a unit
    3. Pool across segments
    4. Map to Lorentz curvature
    
    Everything else (Lorentz fusion, scaling, projection) is IDENTICAL to ParallelLorentz.
    """
    def __init__(self, lookback, input_dim, encode_dim=32,
                 curvature=1.0, segment_length=24, encode_dropout=0.1,
                 use_segment_norm=True):
        """
        Args:
            lookback: int - lookback window size
            input_dim: int - number of input features
            encode_dim: int - dimension of hyperbolic encodedings
            curvature: float - curvature of Lorentz  (k parameter)
            segment_length: int - length of each segment (e.g., 24 for daily segments in hourly data)
            n_layer: int - number of linear layers
            use_attention_pooling: bool - use attention pooling over segments
            use_segment_norm: bool - normalize each segment independently
        """
        super().__init__()
        
        self.trend_encode = SegmentLinearencode(
            input_dim=input_dim,
            output_dim=encode_dim,
            lookback=lookback,
            segment_length=segment_length,
            use_segment_norm=use_segment_norm,
            dropout=encode_dropout
        )
        self.seasonal_coarse_encode = SegmentLinearencode(
            input_dim=input_dim,
            output_dim=encode_dim,
            lookback=lookback,
            segment_length=segment_length,
            use_segment_norm=use_segment_norm,
            dropout=encode_dropout
        )
        self.seasonal_fine_encode = SegmentLinearencode(
            input_dim=input_dim,
            output_dim=encode_dim,
            lookback=lookback,
            segment_length=segment_length,
            use_segment_norm=use_segment_norm,
            dropout=encode_dropout
        )
        self.residual_encode = SegmentLinearencode(
            input_dim=input_dim,
            output_dim=encode_dim,
            lookback=lookback,
            segment_length=segment_length,
            use_segment_norm=use_segment_norm,
            dropout=encode_dropout
        )

        
        # Lorentz ball manifold (SAME as ParallelLorentz)
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        
        # Scaling parameter (SAME as ParallelLorentz)
        self.effective_scale = nn.Parameter(torch.tensor(0.1))
        
        # Learnable weights for Lorentz combination (SAME as ParallelLorentz)
        self.lorentz_weights = nn.Parameter(torch.ones(4) * 0.25)

    def weighted_lorentz_mean(self, points, weights):
        """
        Compute weighted Einstein midpoint (Fréchet mean) in Lorentz manifold.
        This is the proper way to combine multiple points in hyperbolic space.
        
        Uses iterative algorithm to find the point that minimizes weighted distances.
        
        Args:
            points: list of [B, encode_dim+1] - points on Lorentz manifold
            weights: [num_points] - normalized weights
        
        Returns:
            mean_point: [B, encode_dim+1] - weighted mean on manifold
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
            z_trend_h, z_coarse_h, z_fine_h, z_residual_h: [B, encode_dim+1]
        
        Returns:
            combined_h: [B, encode_dim+1] - combined point on manifold
            combined_tangent: [B, encode_dim] - tangent vector representation at origin
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
                - trend_tangent, seasonal_coarse_tangent, seasonal_fine_tangent, residual_tangent: [B, encode_dim]
                - trend_h, seasonal_coarse_h, seasonal_fine_h, residual_h: [B, encode_dim+1]
                - combined_tangent: [B, encode_dim]
                - combined_h: [B, encode_dim+1]
        """
        # 1) Encode to Euclidean latent (tangent vectors at origin)
        z_trend_t = self.trend_encode(trend)  # [B, encode_dim]
        z_seasonal_coarse_t = self.seasonal_coarse_encode(seasonal_coarse)
        z_seasonal_fine_t = self.seasonal_fine_encode(seasonal_fine)
        z_residual_t = self.residual_encode(residual)
        
        # 2) Scaling to prevent numerical instability
        effective_scale = torch.tanh(self.effective_scale)  # [-1, 1]
        scaled_trend_encode = z_trend_t * effective_scale
        scaled_coarse_encode = z_seasonal_coarse_t * effective_scale
        scaled_fine_encode = z_seasonal_fine_t * effective_scale
        scaled_residual_encode = z_residual_t * effective_scale

        # 3) Map to hyperbolic space (Lorentz model)
        # expmap0: tangent space at origin → manifold
        # Result: [B, encode_dim+1] (extra dimension for Lorentz constraint)
        z_trend_h = safe_expmap0(self.manifold, scaled_trend_encode)
        z_seasonal_coarse_h = safe_expmap0(self.manifold, scaled_coarse_encode)
        z_seasonal_fine_h = safe_expmap0(self.manifold, scaled_fine_encode)
        z_residual_h = safe_expmap0(self.manifold, scaled_residual_encode)

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