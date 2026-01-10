
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
    Encode each feature for moving window that outputs [B, num_segments, encode_dim].
    Each segment is independently mapped to hyperbolic space.
    """
    
    def __init__(self, lookback, encode_dim=32, curvature=1.0, segment_length=24,
                 encode_dropout=0.1):
        super().__init__()
        
        self.encode_dim = encode_dim
        
        # Segment-aware encoders (output per-segment encodedings)
        self.trend_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, segment_length=segment_length, dropout=encode_dropout
        )
        self.seasonal_coarse_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, segment_length=segment_length,dropout=encode_dropout
        )
        self.seasonal_fine_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, segment_length=segment_length, dropout=encode_dropout
        )
        self.residual_encode = SegmentLinearencodeMovingWindow(
            encode_dim=encode_dim, lookback=lookback, segment_length=segment_length, dropout=encode_dropout
        )
        
        # Lorentz manifold
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        
        # Scaling parameter
        self.effective_scale = nn.Parameter(torch.tensor(1.0))
        
        # Möbius fusion weights
        self.lorentz_weights = nn.Parameter(torch.ones(4) * 0.25)
    
    def map_segments_to_hyperbolic(self, segment_encodes):
        """
        Map each segment encodeding to hyperbolic space independently.
        
        Args:
            segment_encodes: [B, num_segments, encode_dim]
        
        Returns:
            hyperbolic_encodes: [B, num_segments, encode_dim]
        """
        B, N, D = segment_encodes.shape
        
        # Flatten segments for batch processing
        encodes_flat = segment_encodes.reshape(B * N, D)  # [B*N, encode_dim]
        
        # Scale
        effective_scale = torch.tanh(self.effective_scale)
        scaled_encodes = encodes_flat * effective_scale
        
        # Map to hyperbolic space
        hyperbolic_flat = safe_expmap0(self.manifold, scaled_encodes)  # [B*N, encode_dim]
        
        # Project to manifold
        hyperbolic_flat = self.manifold.projx(hyperbolic_flat)
        
        # Reshape back to sequence
        hyperbolic_encodes = hyperbolic_flat.view(B, N, D)  # [B, num_segments, encode_dim]
        
        return hyperbolic_encodes
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
        z_trend_segments = self.trend_encode(trend)  # [B, encode_dim]
        z_coarse_segments = self.seasonal_coarse_encode(seasonal_coarse)
        z_fine_segments = self.seasonal_fine_encode(seasonal_fine)
        z_residual_segments = self.residual_encode(residual)

        z_trend_h = self.map_segments_to_hyperbolic(z_trend_segments)
        z_coarse_h = self.map_segments_to_hyperbolic(z_coarse_segments)
        z_fine_h = self.map_segments_to_hyperbolic(z_fine_segments)
        z_residual_h = self.map_segments_to_hyperbolic(z_residual_segments)
            
    
        # Non-hierarchical: Use weighted Einstein midpoint (Fréchet mean)
        combined_h = self.lorentz_fusion(
            z_trend_h, z_coarse_h, z_fine_h, z_residual_h
        )

        return {
            "trend_h": z_trend_h,
            "seasonal_coarse_h": z_coarse_h,
            "seasonal_fine_h": z_fine_h,
            "residual_h": z_residual_h,
            "combined_h": combined_h
        }