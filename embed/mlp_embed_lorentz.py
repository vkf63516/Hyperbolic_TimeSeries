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
                 curvature=1.0, use_hierarchy=False, hierarchy_scales=[0.5, 1.0, 1.5, 2.0],
                 n_layer=2, embed_dropout=0.1, use_attention_pooling=False):
        """
        Args:
            lookback: int - lookback window size
            input_dim: int - number of input features (for MVAR)
            embed_dim: int - dimension of hyperbolic embeddings (actual space is embed_dim+1 for Lorentz)
            hidden_dim: int - hidden dimension for MLP
            curvature: float - curvature of Lorentz manifold (k parameter)
            use_hierarchy: bool - whether to use hierarchical scaling
            hierarchy_scales: list - radial scales for [trend, coarse, fine, residual]
                                     Smaller = closer to origin = more general
            n_layer: int - number of MLP layers
            embed_dropout: float - dropout rate in embedders
            use_attention_pooling: bool - use attention pooling (True) or mean pooling (False)
        """
        super().__init__()
        
        self.use_hierarchy = use_hierarchy
        if self.use_hierarchy:
            # Store log of scales (ensures always positive via exp)
            self.log_scales = nn.ParameterList([
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[0]))),  # trend (closest to root)
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[1]))),  # coarse
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[2]))),  # fine
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[3])))   # residual (furthest from root)
            ])

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
        
        # Learnable weights for weighted Einstein midpoint (non-hierarchical)
        self.lorentz_weights = nn.Parameter(torch.ones(4) * 0.25)

    def apply_hierarchy_scaling(self, manifold_point, scale):
        """
        Scale manifold point radially from origin using tangent space scaling.
        Larger scale = further from origin = more specific in hierarchy.
        
        For Lorentz model, we scale in tangent space at origin.
        
        Args:
            manifold_point: [B, embed_dim+1] - point on Lorentz manifold
            scale: float - scaling factor
        
        Returns:
            scaled_point: [B, embed_dim+1] - scaled point on manifold
        """
        # Map to tangent space at origin
        tangent = self.manifold.logmap0(manifold_point)  # [B, embed_dim]
        
        # Scale in tangent space
        scaled_tangent = tangent * scale
        
        # Map back to manifold
        scaled_point = safe_expmap0(self.manifold, scaled_tangent)
        
        # Ensure point is on manifold
        return self.manifold.projx(scaled_point)

    def hierarchical_combine(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """
        Sequential hierarchical composition in Lorentz hyperbolic space.
        Builds from general (trend) to specific (residual) along geodesics.
        
        Each step moves 25% along the geodesic from current state toward the component,
        creating a hierarchical composition: trend → coarse → fine → residual
        
        Args:
            z_trend_h, z_coarse_h, z_fine_h, z_residual_h: [B, embed_dim+1]
        
        Returns:
            z_combined: [B, embed_dim+1] - hierarchically combined point
        """
        # Start from trend (root of hierarchy - most general pattern)
        z_current = z_trend_h

        # Incorporate coarse (second level) - geodesic interpolation
        v_to_coarse = self.manifold.logmap(z_current, z_coarse_h)  # Tangent vector from current to coarse
        z_current = safe_expmap(self.manifold, 0.25 * v_to_coarse, z_current)  # Move 25% along geodesic
        z_current = self.manifold.projx(z_current)

        # Incorporate fine (third level)
        v_to_fine = self.manifold.logmap(z_current, z_fine_h)
        z_current = safe_expmap(self.manifold, 0.25 * v_to_fine, z_current)
        z_current = self.manifold.projx(z_current)

        # Incorporate residual (deepest level - most specific)
        v_to_residual = self.manifold.logmap(z_current, z_residual_h)
        z_current = safe_expmap(self.manifold, 0.25 * v_to_residual, z_current)
        z_current = self.manifold.projx(z_current)

        return z_current

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
            current_mean = safe_expmap(self.manifold, 0.5 * weighted_vec, current_mean)
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

        # 5) Apply hierarchy or hyperbolic fusion
        if self.use_hierarchy:
            # STEP 1: Radial hierarchy scaling
            # Position each component at different distances from origin
            trend_scale = torch.exp(self.log_scales[0])      # e.g., 0.5 (close to origin - general)
            coarse_scale = torch.exp(self.log_scales[1])     # e.g., 1.0
            fine_scale = torch.exp(self.log_scales[2])      # e.g., 1.5
            residual_scale = torch.exp(self.log_scales[3])   # e.g., 2.0 (far from origin - specific)
    
            z_trend_h = self.apply_hierarchy_scaling(z_trend_h, trend_scale)
            z_seasonal_coarse_h = self.apply_hierarchy_scaling(z_seasonal_coarse_h, coarse_scale)
            z_seasonal_fine_h = self.apply_hierarchy_scaling(z_seasonal_fine_h, fine_scale)
            z_residual_h = self.apply_hierarchy_scaling(z_residual_h, residual_scale)
            
            # STEP 2: Sequential geodesic aggregation
            # Combine the radially-positioned components sequentially along geodesics
            combined_h = self.hierarchical_combine(
                z_trend_h, z_seasonal_coarse_h, z_seasonal_fine_h, z_residual_h
            )
            combined_tangent = self.manifold.logmap0(combined_h)
        else:
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



class HybridLorentz(nn.Module):
    """
    Hybrid encoder: One component in Lorentz hyperbolic, others in Euclidean.
    Perfect for ablation studies.
    """
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                 curvature=1.0, hyperbolic_component='seasonal_coarse',
                 n_layer=2, use_attention_pooling=False):
        """
        Args:
            hyperbolic_component: str - which component to encode in hyperbolic space
                Options: 'trend', 'seasonal_coarse', 'seasonal_fine', 'residual'
        """
        super().__init__()
        
        self.hyperbolic_component = hyperbolic_component
        self.components = ['trend', 'seasonal_coarse', 'seasonal_fine', 'residual']
        
        # Lorentz manifold for hyperbolic component
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        
        # Create encoders - one hyperbolic, rest Euclidean
        self.embedding = nn.ModuleDict()
        
        for comp in self.components:
            self.embedding[comp] = MLPEmbed(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=embed_dim,
                n_layer=n_layer,
                lookback=lookback,
                use_attention_pooling=use_attention_pooling
            )
        
        # Scaling parameters
        self.hyp_scale = nn.Parameter(torch.tensor(0.1))  # for hyperbolic component
        self.euc_scale = nn.Parameter(torch.tensor(1.0))  # for Euclidean components
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Encode components with hybrid approach.
        
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len, input_dim]
        
        Returns:
            dict with:
                - tangent_vectors: dict of [B, embed_dim] for each component
                - hyperbolic_point: [B, embed_dim+1] (only for hyperbolic component)
                - combined_tangent: [B, embed_dim] - all components in tangent space
                - component_types: dict showing which space each uses
        """
        components_data = {
            'trend': trend,
            'seasonal_coarse': seasonal_coarse,
            'seasonal_fine': seasonal_fine,
            'residual': residual
        }
        
        tangent_vectors = {}
        hyperbolic_point = None
        
        for comp_name, comp_data in components_data.items():
            # Encode to latent space
            z_latent = self.embedding[comp_name](comp_data)  # [B, embed_dim]
            
            if comp_name == self.hyperbolic_component:
                # Map to hyperbolic space
                scale = torch.tanh(self.hyp_scale)
                z_scaled = z_latent * scale
                z_h = safe_expmap0(self.manifold, z_scaled)
                z_h = self.manifold.projx(z_h)
                hyperbolic_point = z_h
                
                # Get tangent for combination
                tangent_vectors[comp_name] = self.manifold.logmap0(z_h)
            else:
                # Keep in Euclidean space (just scale)
                scale = torch.sigmoid(self.euc_scale)
                tangent_vectors[comp_name] = z_latent * scale
        
        # Combine all in tangent space
        combined_tangent = sum(tangent_vectors.values())
        
        return {
            "tangent_vectors": tangent_vectors,
            "hyperbolic_point": hyperbolic_point,
            "combined_tangent": combined_tangent,
            "hyperbolic_component": self.hyperbolic_component,
            "component_types": {
                comp: ('hyperbolic' if comp == self.hyperbolic_component else 'euclidean')
                for comp in self.components
            }
        }