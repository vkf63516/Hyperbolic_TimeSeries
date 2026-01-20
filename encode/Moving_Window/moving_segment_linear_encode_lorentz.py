

import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from spec import safe_expmap0, safe_expmap


class SegmentLinearencodeMovingWindow(nn.Module):
    """
    This done for each feature 
    Produces ONE encodeding per segment.
    Input:  [B, seq_len]
    Output: [B, num_segments, encode_dim]  # One encodeding per segment
    """
    
    def __init__(self, lookback, encode_dim, segment_length=24, dropout=0.1):
        super().__init__()
        
        self.encode_dim = encode_dim
        self.segment_length = segment_length
        self.lookback = lookback
        self.num_segments = lookback // segment_length
        self.pad_seq_len = 0
        
        if self.lookback > self.num_segments * self.segment_length:
            self.pad_seq_len = (self.num_segments + 1) * self.segment_length - self.lookback
            self.num_segments += 1
        self.temporal_linears = nn.Linear(segment_length, encode_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len] - single feature
        Returns:
            z: [B, num_segments, encode_dim] - trajectory
        """
        B = x.shape[0]
        
        # Pad if necessary
        if self.pad_seq_len > 0:
            pad = torch.zeros(B, self.pad_seq_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        # Reshape into segments
        x_seg = x.view(B, self.num_segments, self.segment_length)  # [B, num_seg, seg_len]
        
        # encode each segment (KEEP segment structure!)
        seg_encode = self.temporal_linears(x_seg)  # [B, num_segments, encode_dim]
        seg_encode = self.dropout(seg_encode)
        
        return seg_encode


class SegmentedParallelLorentzMovingWindow(nn.Module):
    """
    Moving window Lorentz encoder with ADAPTIVE scaling for high-volatility data.
    """
    
    def __init__(self, lookback, encode_dim=32, curvature=1.0, segment_length=24,
                 encode_dropout=0.1):
        super().__init__()
        
        self.encode_dim = encode_dim
        
        # Segment encoders
        self.trend_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, segment_length=segment_length, dropout=encode_dropout
        )
        self.seasonal_coarse_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, segment_length=segment_length, dropout=encode_dropout
        )
        self.seasonal_fine_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, segment_length=segment_length, dropout=encode_dropout
        )
        self.residual_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, segment_length=segment_length, dropout=encode_dropout
        )
        
        # Lorentz manifold
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        
        # ? FIX 1: Much more conservative initial scaling for high-volatility data
        self.effective_scale = nn.Parameter(torch.tensor(0.001))  # Was 0.1 ? Now 0.01
        
        # Fusion weights
        self.lorentz_weights = nn.Parameter(torch.ones(4) * 0.25)
    
    def map_segments_to_hyperbolic(self, segment_encodes):
        """
        FIXED: More aggressive normalization for high-volatility data.
        """
        B, N, D = segment_encodes.shape
        
        # ? FIX 2: Tanh ensures scale stays in (-1, 1)
        effective_scale = torch.tanh(self.effective_scale)
        
        # Process each segment individually
        hyperbolic_segments = []
        
        for seg_idx in range(N):
            seg = segment_encodes[:, seg_idx, :]  # [B, encode_dim]
            
            # Scale
            scaled_seg = seg * effective_scale
            
            # ? FIX 3: Much stricter clamping (3.0 ? 1.5)
            norm = torch.norm(scaled_seg, dim=-1, keepdim=True).clamp(min=1e-8)
            scaled_seg = scaled_seg / norm * torch.clamp(norm, max=1.5)  # Was 3.0
            
            # Map to Lorentz
            hyp_seg = safe_expmap0(self.manifold, scaled_seg)
            hyp_seg = self.manifold.projx(hyp_seg)
            
            # ? FIX 4: Early NaN detection
            if torch.isnan(hyp_seg).any():
                print(f"?? NaN in map_segments at seg {seg_idx}, using origin")
                # Use origin in Lorentz space:  [sqrt(1/k), 0, 0, ..., 0]
                origin = torch.zeros(B, D + 1, device=seg.device, dtype=seg.dtype)
                origin[:, 0] = torch.sqrt(1.0 / self.manifold.k)
                hyp_seg = origin
            
            hyperbolic_segments.append(hyp_seg)
        
        hyperbolic_encodes = torch.stack(hyperbolic_segments, dim=1)
        return hyperbolic_encodes
 
 
#    def map_segments_to_hyperbolic(self, segment_encodes):
#        """
#        VECTORIZED: Process all segments at once (like Poincare).
#        SAFE: Clamps BEFORE expmap to prevent NaN. 
#        """
#        B, N, D = segment_encodes.shape
#        
#        # Bounded scaling
#        effective_scale = torch.tanh(self.effective_scale)
#        scaled_encodes = segment_encodes * effective_scale  # [B, N, D]
#        
#        # CRITICAL: Clamp norms BEFORE Lorentz projection
#        norms = torch.norm(scaled_encodes, dim=-1, keepdim=True).clamp(min=1e-8)
#        max_norm = 1.0  # Conservative for Lorentz (not 1.5!)
#        scaled_encodes = scaled_encodes / norms * torch.clamp(norms, max=max_norm)
#        
#        # Flatten for batch expmap
#        scaled_flat = scaled_encodes.reshape(B * N, D)  # [B*N, D]
#        
#        # Single batched call (10-30x faster than loop)
#        hyp_flat = safe_expmap0(self. manifold, scaled_flat)  # [B*N, D+1]
#        hyp_flat = self.manifold.projx(hyp_flat)
#        
#        # Reshape back
#        hyperbolic_encodes = hyp_flat.view(B, N, D + 1)  # [B, N, D+1]
#        
#        # NaN check AFTER full batch (avoids per-segment overhead)
#        if torch.isnan(hyperbolic_encodes).any():
#            print(" NaN detected in batch encoding - using conservative fallback")
#            # Fallback: use origin for ALL segments (maintains gradient flow)
#            origin = torch.zeros(B, N, D + 1, device=segment_encodes.device)
#            origin[:, : , 0] = torch.sqrt(1.0 / self. manifold.k)
#            return origin
#        
#        return hyperbolic_encodes
#    
    def weighted_lorentz_mean_segments(self, points, weights):
        """
        ? FIXED: Conservative fusion with multiple fallback strategies.
        """
        B, N, D_plus_1 = points[0].shape
        combined_segments = []
        
        for seg_idx in range(N):
            seg_points = [p[: , seg_idx, :] for p in points]
            
            # Compute weighted tangent
            tangents = [self.manifold.logmap0(p) for p in seg_points]
            weighted_tangent = sum(w * t for w, t in zip(weights, tangents))
            
            # ? FIX 5: Check for NaN/Inf in tangents BEFORE clamping
            if torch.isnan(weighted_tangent).any() or torch.isinf(weighted_tangent).any():
                print(f"?? Invalid tangent at segment {seg_idx}, using first point")
                combined_segments.append(seg_points[0])
                continue
            
            # ? FIX 6: More conservative clamping (5.0 ? 2.0)
            tangent_norm = torch.norm(weighted_tangent, dim=-1, keepdim=True)
            max_norm = 2.0  # Was 5.0
            if (tangent_norm > max_norm).any():
                weighted_tangent = weighted_tangent / tangent_norm * torch.clamp(tangent_norm, max=max_norm)
            
            # Try to map to manifold
            current_mean = safe_expmap0(self.manifold, weighted_tangent)
            current_mean = self.manifold.projx(current_mean)
            
            # ? FIX 7: Check for NaN BEFORE iteration
            if torch.isnan(current_mean).any():
                print(f"?? NaN in initial mean at segment {seg_idx}, using first point")
                combined_segments.append(seg_points[0])
                continue
            
            # ? FIX 8: Reduced iterations (10 ? 5) with smaller steps
            for iteration in range(5):
                tangent_vecs = [self.manifold.logmap(current_mean, p) for p in seg_points]
                weighted_vec = sum(w * v for w, v in zip(weights, tangent_vecs))
                
                # Early convergence
                if torch.norm(weighted_vec, dim=-1).max() < 1e-5:
                    break
                
                # Conservative update
                vec_norm = torch.norm(weighted_vec, dim=-1, keepdim=True)
                if (vec_norm > max_norm).any():
                    weighted_vec = weighted_vec / vec_norm * torch.clamp(vec_norm, max=max_norm)
                
                # ? FIX 9: Smaller step size (0.3 ? 0.1)
                next_mean = safe_expmap(self.manifold, current_mean, 0.1 * weighted_vec)
                next_mean = self.manifold.projx(next_mean)
                
                if torch.isnan(next_mean).any():
                    break  # Keep current_mean
                
                current_mean = next_mean
            
            combined_segments.append(current_mean)
        
        return torch.stack(combined_segments, dim=1)
    
    def lorentz_fusion(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """Fusion across all segment positions."""
        weights = torch.softmax(self.lorentz_weights, dim=0)
        points = [z_trend_h, z_coarse_h, z_fine_h, z_residual_h]
        combined_h = self.weighted_lorentz_mean_segments(points, weights)
        return combined_h
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Forward with fixed dimension handling.
        
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len, 1] 
            
        Returns:
            dict with components of shape [B, num_segments, encode_dim+1]
        """
        # Encode to segments:  [B, num_segments, encode_dim]
        z_trend_segments = self.trend_encode(trend)
        z_coarse_segments = self.seasonal_coarse_encode(seasonal_coarse)
        z_fine_segments = self.seasonal_fine_encode(seasonal_fine)
        z_residual_segments = self.residual_encode(residual)
        
        # Map to hyperbolic (FIXED: now returns [B, num_segments, encode_dim+1])
        z_trend_h = self.map_segments_to_hyperbolic(z_trend_segments)
        z_coarse_h = self.map_segments_to_hyperbolic(z_coarse_segments)
        z_fine_h = self.map_segments_to_hyperbolic(z_fine_segments)
        z_residual_h = self.map_segments_to_hyperbolic(z_residual_segments)
        
        # Fusion
        combined_h = self.lorentz_fusion(z_trend_h, z_coarse_h, z_fine_h, z_residual_h)
        
        return {
            "trend_h": z_trend_h,
            "seasonal_coarse_h": z_coarse_h,
            "seasonal_fine_h": z_fine_h,
            "residual_h": z_residual_h,
            "combined_h": combined_h
        }