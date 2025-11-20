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
                 segment_length=24, num_layers=2, embed_dropout=0.1, use_attention_pooling=False):

        super().__init__()
        
        # Segment-aware encoders (different segment lengths for different components)
        self.trend_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            segment_length=segment_length,
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=num_layers,
            embed_dropout=embed_dropout,
            use_attention_pooling=use_attention_pooling
        )
        self.seasonal_coarse_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            segment_length=segment_length,
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=num_layers,
            embed_dropout=embed_dropout,
            use_attention_pooling=use_attention_pooling
        )
        self.seasonal_fine_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            segment_length=segment_length,
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=num_layers,
            embed_dropout=embed_dropout,
            use_attention_pooling=use_attention_pooling
        )
        self.residual_embed = SegmentMLPEmbed(
            lookback=lookback,
            input_dim=input_dim, 
            segment_length=segment_length,
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=num_layers,
            embed_dropout=embed_dropout,
            use_attention_pooling=use_attention_pooling
        )
        
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        self.effective_scale = nn.Parameter(torch.tensor(0.1))
        self.lorentz_weights = nn.Parameter(torch.ones(4) * 0.25)

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