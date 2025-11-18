import torch
import torch.nn as nn
import geoopt
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0

class MLPEmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=3, dropout=0.5, lookback=None, use_attention_pooling=False):
        super().__init__()
        self.lookback = lookback
        self.use_attention_pooling = use_attention_pooling
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(n_layer)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        if self.use_attention_pooling:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, x):
        """
        x: [B, seqlen, input_dim]
        returns: [B, output_dim]  (Euclidean latent, tangent vectors)
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        # Compresses a series to a vector.
        # Pool across time (attention or mean)
        if self.use_attention_pooling:
            attn_scores = self.attention(x)  # [B, seqlen, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)  # [B, seqlen, 1]
            x_pooled = (x * attn_weights).sum(dim=1)  # [B, hidden_dim]
        else:
            x_pooled = x.mean(dim=1)   # mean pooling
        return self.output_proj(x_pooled)

# --------------------------
# Parallel Lorentz Encoder with Proper Hyperbolic Operations
# --------------------------
class ParallelLorentz(nn.Module):
    """
    Hyperbolic encoder using Lorentz model (hyperboloid) with proper hyperbolic operations.
    
    Encodes decomposed time series components (trend, coarse, fine, residual)
    into hyperbolic space with hierarchical structure.
    
    Uses MLP encoders (memory efficient, no Mamba overhead).
    All operations use proper Lorentz geometry (Einstein midpoint, weighted combinations).
    """
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                 curvature=1.0,n_layer=2, embed_dropout=0.1, use_attention_pooling=False):
        """
        Args:
            lookback: int - lookback window size
            input_dim: int - number of input features (for MVAR)
            embed_dim: int - dimension of hyperbolic embeddings (actual space is embed_dim+1 for Lorentz)
            hidden_dim: int - hidden dimension for MLP
            curvature: float - curvature of Lorentz manifold (k parameter)
            n_layer: int - number of MLP layers
            embed_dropout: float - dropout rate in embedders
            use_attention_pooling: bool - use attention pooling (True) or mean pooling (False)
        """
        super().__init__()

        # MLP encoders for each component
        self.trend_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=n_layer,
            dropout=embed_dropout,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        self.seasonal_coarse_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=n_layer,
            dropout=embed_dropout,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        self.seasonal_fine_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=n_layer,
            dropout=embed_dropout,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        self.residual_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=n_layer,
            dropout=embed_dropout,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        
        # Lorentz manifold (hyperboloid model)
        # Points live in R^(embed_dim+1) with constraint: -x_0^2 + x_1^2 + ... + x_n^2 = -1/k
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        
        # Scaling parameter for mapping to hyperbolic space
        self.effective_scale = nn.Parameter(torch.tensor(0.1))
        
        # Learnable weights for weighted Einstein midpoint 
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
            current_mean = safe_expmap0(self.manifold, 0.5 * weighted_vec, current_mean)
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
        
        # Get tangent representation for downstream tasks
        combined_tangent = self.manifold.logmap0(combined_h)
        
        return combined_h, combined_tangent

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
        combined_h, combined_tangent = self.lorentz_fusion(
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