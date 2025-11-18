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
        returns: [B, output_dim]  (Poincare latent, tangent vectors)
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
# Parallel Poincaré Encoder with Pure Möbius Operations
# --------------------------
class ParallelPoincare(nn.Module):
    """
    Hyperbolic encoder using Poincaré ball model with proper Möbius gyrovector operations.
    
    Encodes decomposed time series components (trend, coarse, fine, residual)
    into hyperbolic space with hierarchical structure.
    
    Uses MLP encoders (memory efficient, no Mamba overhead).
    
    Poincaré ball: Points live in open ball {x ∈ R^n : ||x|| < 1/√c}
    All operations use proper Möbius gyrovector space arithmetic.
    """
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                 curvature=1.0, n_layer=2, use_attention_pooling=False):
        """
        Args:
            lookback: int - lookback window size
            input_dim: int - number of input features (for MVAR)
            embed_dim: int - dimension of hyperbolic embeddings (Poincaré ball in R^embed_dim)
            hidden_dim: int - hidden dimension for MLP
            curvature: float - curvature of Poincaré ball (c parameter)
            n_layer: int - number of MLP layers
            use_attention_pooling: bool - use attention pooling (True) or mean pooling (False)
        """
        super().__init__()


        # MLP encoders for each component
        self.trend_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=n_layer,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        self.seasonal_coarse_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=n_layer,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        self.seasonal_fine_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=n_layer,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        self.residual_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=n_layer,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        
        # Poincaré ball manifold
        # Points live in open ball {x ∈ R^embed_dim : ||x|| < 1/√c}
        self.manifold = geoopt.manifolds.PoincareBall(c=curvature)
        
        # Scaling parameter for mapping to hyperbolic space
        self.effective_scale = nn.Parameter(torch.tensor(0.1))
        
        # Learnable weights for Möbius combination 
        self.mobius_weights = nn.Parameter(torch.ones(4) * 0.25)

    def mobius_fusion(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """
        Non-hierarchical fusion using weighted Möbius addition.
        Properly combines points in Poincaré ball using gyrovector space operations.
        
        Uses the formula: result = w₁⊗x₁ ⊕ w₂⊗x₂ ⊕ w₃⊗x₃ ⊕ w₄⊗x₄
        where ⊗ is Möbius scalar multiplication and ⊕ is Möbius addition
        
        Args:
            z_trend_h, z_coarse_h, z_fine_h, z_residual_h: [B, embed_dim]
        
        Returns:
            combined_h: [B, embed_dim] - combined point on manifold
            combined_tangent: [B, embed_dim] - tangent vector representation at origin
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
        combined_tangent = self.manifold.logmap0(combined_h)
        
        return combined_h, combined_tangent

    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Encode decomposed time series components to Poincaré ball.
        
        Args:
            trend: [B, seq_len, input_dim] or [B, N_seg, seg_len, input_dim]
            seasonal_coarse: [B, seq_len, input_dim] or [B, N_seg, seg_len, input_dim]
            seasonal_fine: [B, seq_len, input_dim] or [B, N_seg, seg_len, input_dim]
            residual: [B, seq_len, input_dim] or [B, N_seg, seg_len, input_dim]
        
        Returns:
            dict with:
                - trend_tangent, seasonal_coarse_tangent, seasonal_fine_tangent, residual_tangent: [B, embed_dim]
                - trend_h, seasonal_coarse_h, seasonal_fine_h, residual_h: [B, embed_dim]
                - combined_tangent: [B, embed_dim]
                - combined_h: [B, embed_dim]
        """
        # 1) Encode to Euclidean latent (tangent vectors at origin)
        z_trend_t = self.trend_embed(trend)  # [B, embed_dim]
        z_seasonal_coarse_t = self.seasonal_coarse_embed(seasonal_coarse)
        z_seasonal_fine_t = self.seasonal_fine_embed(seasonal_fine)
        z_residual_t = self.residual_embed(residual)
        
        # 2) Scaling to prevent numerical instability
        # In Poincaré ball, we need to be careful not to push points too close to boundary
        effective_scale = torch.tanh(self.effective_scale)  # [-1, 1]
        scaled_trend_embed = z_trend_t * effective_scale
        scaled_coarse_embed = z_seasonal_coarse_t * effective_scale
        scaled_fine_embed = z_seasonal_fine_t * effective_scale
        scaled_residual_embed = z_residual_t * effective_scale

        # 3) Map to hyperbolic space (Poincaré ball model)
        # expmap0: tangent space at origin → manifold
        # Result: [B, embed_dim] (points in the ball)
        z_trend_h = safe_expmap0(self.manifold, scaled_trend_embed)
        z_seasonal_coarse_h = safe_expmap0(self.manifold, scaled_coarse_embed)
        z_seasonal_fine_h = safe_expmap0(self.manifold, scaled_fine_embed)
        z_residual_h = safe_expmap0(self.manifold, scaled_residual_embed)

        # 4) Project to manifold (ensure points stay within ball)
        # Ensures points satisfy ||x|| < 1/√c
        z_trend_h = self.manifold.projx(z_trend_h)
        z_seasonal_coarse_h = self.manifold.projx(z_seasonal_coarse_h)
        z_seasonal_fine_h = self.manifold.projx(z_seasonal_fine_h)
        z_residual_h = self.manifold.projx(z_residual_h)
           # Möbius fusion
        combined_h, combined_tangent = self.mobius_fusion(
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
