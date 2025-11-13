import torch
import torch.nn as nn
import geoopt
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0

class MLPEmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=3, lookback=None, use_attention_pooling=False):
        super().__init__()
        self.lookback = lookback
        self.use_attention_pooling = use_attention_pooling
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.5)
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
        #Compresses a series to a vector.
        # STEP 5: Pool across time (attention or mean)
        if self.use_attention_pooling:
            attn_scores = self.attention(x)  # [32, 336, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)  # [32, 336, 1]
            x_pooled = (x * attn_weights).sum(dim=1)  # [32, 64]
        else:
            x_pooled = x.mean(dim=1)   # mean pooling would be good for point level forecasting
        return self.output_proj(x_pooled)

# --------------------------
# Parallel Lorentz Encoder
# --------------------------
class ParallelLorentz(nn.Module):
    """
    Hyperbolic encoder using Lorentz model (hyperboloid).
    
    Encodes decomposed time series components (trend, weekly, daily, residual)
    into hyperbolic space with hierarchical structure.
    
    Uses MLP encoders (memory efficient, no Mamba overhead).
    """
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                 curvature=1.0, use_hierarchy=False, hierarchy_scales=[0.5, 1.0, 1.5, 2.0],
                 n_layer=2, use_attention_pooling=False):
        """
        Args:
            lookback: int - lookback window size
            input_dim: int - number of input features (for MVAR)
            embed_dim: int - dimension of hyperbolic embeddings (actual space is embed_dim+1 for Lorentz)
            hidden_dim: int - hidden dimension for MLP
            curvature: float - curvature of Lorentz manifold (k parameter)
            use_hierarchy: bool - whether to use hierarchical scaling
            hierarchy_scales: list - radial scales for [trend, weekly, daily, residual]
                                     Smaller = closer to origin = more general
            n_layer: int - number of MLP layers
            use_attention_pooling: bool - use attention pooling (True) or mean pooling (False)
        """
        super().__init__()
        
        self.use_hierarchy = use_hierarchy
        if self.use_hierarchy:
            # Store log of scales (ensures always positive via exp)
            self.log_scales = nn.ParameterList([
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[0]))),  # trend (closest to root)
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[1]))),  # weekly
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[2]))),  # daily
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[3])))   # residual (furthest from root)
            ])

        # MLP encoders for each component
        self.trend_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=n_layer,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        self.seasonal_weekly_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            n_layer=n_layer,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        self.seasonal_daily_embed = MLPEmbed(
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
        
        # Lorentz manifold (hyperboloid model)
        # Points live in R^(embed_dim+1) with constraint: -x_0^2 + x_1^2 + ... + x_n^2 = -1/k
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        
        # Scaling parameter for mapping to hyperbolic space
        self.effective_scale = nn.Parameter(torch.tensor(0.1))

    def apply_hierarchy_scaling(self, manifold_point, scale):
        """
        Scale manifold point radially from origin.
        Larger scale = further from origin = more specific in hierarchy.
        
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

    def hierarchical_combine(self, z_trend_h, z_weekly_h, z_daily_h, z_residual_h):
        """
        Sequential hierarchical composition in hyperbolic space.
        Builds from general (trend) to specific (residual) along geodesics.
        
        Each step moves 25% along the geodesic from current state toward the component,
        creating a hierarchical composition: trend → weekly → daily → residual
        
        Args:
            z_trend_h, z_weekly_h, z_daily_h, z_residual_h: [B, embed_dim+1]
        
        Returns:
            z_combined: [B, embed_dim+1] - hierarchically combined point
        """
        # Start from trend (root of hierarchy - most general pattern)
        z_current = z_trend_h

        # Incorporate weekly (second level)
        v_to_weekly = self.manifold.logmap(z_current, z_weekly_h)  # Tangent vector from current to weekly
        z_current = safe_expmap(self.manifold, 0.25 * v_to_weekly, z_current)  # Move 25% along geodesic
        z_current = self.manifold.projx(z_current)

        # Incorporate daily (third level)
        v_to_daily = self.manifold.logmap(z_current, z_daily_h)
        z_current = safe_expmap(self.manifold, 0.25 * v_to_daily, z_current)
        z_current = self.manifold.projx(z_current)

        # Incorporate residual (deepest level - most specific)
        v_to_residual = self.manifold.logmap(z_current, z_residual_h)
        z_current = safe_expmap(self.manifold, 0.25 * v_to_residual, z_current)
        z_current = self.manifold.projx(z_current)

        return z_current

    def tangent_fusion(self, z_trend_h, z_weekly_h, z_daily_h, z_residual_h):
        """
        Non-hierarchical tangent space fusion.
        Maps all points to origin's tangent space, sums them, maps back.
        
        Args:
            z_trend_h, z_weekly_h, z_daily_h, z_residual_h: [B, embed_dim+1]
        
        Returns:
            combined_h: [B, embed_dim+1] - combined point on manifold
            combined_tangent: [B, embed_dim] - tangent vector representation
        """
        # Map to tangent space at origin
        u_trend = self.manifold.logmap0(z_trend_h)
        u_seasonal_weekly = self.manifold.logmap0(z_weekly_h)
        u_seasonal_daily = self.manifold.logmap0(z_daily_h)
        u_residual = self.manifold.logmap0(z_residual_h)

        # Sum in tangent space (Euclidean addition)
        combined_tangent = u_trend + u_seasonal_weekly + u_seasonal_daily + u_residual
        
        # Apply scaling
        effective_scale = torch.tanh(self.effective_scale)  # Keep in (-1, 1)
        scaled_tangent = combined_tangent * effective_scale
        
        # Map back to manifold
        combined_h = safe_expmap0(self.manifold, scaled_tangent)
        combined_h = self.manifold.projx(combined_h)
        
        return combined_h, combined_tangent

    def forward(self, trend, seasonal_weekly, seasonal_daily, residual):
        """
        Encode decomposed time series components to hyperbolic space.
        
        Args:
            trend: [B, seq_len, input_dim] or [B, N_seg, seg_len, input_dim]
            seasonal_weekly: [B, seq_len, input_dim] or [B, N_seg, seg_len, input_dim]
            seasonal_daily: [B, seq_len, input_dim] or [B, N_seg, seg_len, input_dim]
            residual: [B, seq_len, input_dim] or [B, N_seg, seg_len, input_dim]
        
        Returns:
            dict with:
                - trend_tangent, seasonal_weekly_tangent, seasonal_daily_tangent, residual_tangent: [B, embed_dim]
                - trend_h, seasonal_weekly_h, seasonal_daily_h, residual_h: [B, embed_dim+1]
                - combined_tangent: [B, embed_dim]
                - combined_h: [B, embed_dim+1]
        """
        # 1) Encode to Euclidean latent (tangent vectors at origin)
        z_trend_t = self.trend_embed(trend)  # [B, embed_dim]
        z_seasonal_weekly_t = self.seasonal_weekly_embed(seasonal_weekly)
        z_seasonal_daily_t = self.seasonal_daily_embed(seasonal_daily)
        z_residual_t = self.residual_embed(residual)
        
        # 2) Scaling to prevent numerical instability
        effective_scale = torch.tanh(self.effective_scale)  # [-1, 1]
        scaled_trend_embed = z_trend_t * effective_scale
        scaled_weekly_embed = z_seasonal_weekly_t * effective_scale
        scaled_daily_embed = z_seasonal_daily_t * effective_scale
        scaled_residual_embed = z_residual_t * effective_scale

        # 3) Map to hyperbolic space (Lorentz model)
        # expmap0: tangent space at origin → manifold
        # Result: [B, embed_dim+1] (extra dimension for Lorentz constraint)
        z_trend_h = safe_expmap0(self.manifold, scaled_trend_embed)
        z_seasonal_weekly_h = safe_expmap0(self.manifold, scaled_weekly_embed)
        z_seasonal_daily_h = safe_expmap0(self.manifold, scaled_daily_embed)
        z_residual_h = safe_expmap0(self.manifold, scaled_residual_embed)

        # 4) Project to manifold (ensure numerical stability)
        # Ensures points satisfy Lorentz constraint: -x_0^2 + x_1^2 + ... = -1/k
        z_trend_h = self.manifold.projx(z_trend_h)
        z_seasonal_weekly_h = self.manifold.projx(z_seasonal_weekly_h)
        z_seasonal_daily_h = self.manifold.projx(z_seasonal_daily_h)
        z_residual_h = self.manifold.projx(z_residual_h)

        # 5) Apply hierarchy: radial scaling + sequential aggregation
        if self.use_hierarchy:
            # STEP 1: Radial hierarchy scaling
            # Position each component at different distances from origin
            trend_scale = torch.exp(self.log_scales[0])      # e.g., 0.5 (close to origin - general)
            weekly_scale = torch.exp(self.log_scales[1])     # e.g., 1.0
            daily_scale = torch.exp(self.log_scales[2])      # e.g., 1.5
            residual_scale = torch.exp(self.log_scales[3])   # e.g., 2.0 (far from origin - specific)
    
            z_trend_h = self.apply_hierarchy_scaling(z_trend_h, trend_scale)
            z_seasonal_weekly_h = self.apply_hierarchy_scaling(z_seasonal_weekly_h, weekly_scale)
            z_seasonal_daily_h = self.apply_hierarchy_scaling(z_seasonal_daily_h, daily_scale)
            z_residual_h = self.apply_hierarchy_scaling(z_residual_h, residual_scale)
            
            # STEP 2: Sequential geodesic aggregation
            # Combine the radially-positioned components sequentially along geodesics
            combined_h = self.hierarchical_combine(
                z_trend_h, z_seasonal_weekly_h, z_seasonal_daily_h, z_residual_h
            )
            combined_tangent = self.manifold.logmap0(combined_h)
        else:
            # Non-hierarchical tangent space fusion
            combined_h, combined_tangent = self.tangent_fusion(
                z_trend_h, z_seasonal_weekly_h, z_seasonal_daily_h, z_residual_h
            )

        return {
            "trend_tangent": z_trend_t,
            "seasonal_weekly_tangent": z_seasonal_weekly_t,
            "seasonal_daily_tangent": z_seasonal_daily_t,
            "residual_tangent": z_residual_t,
            "trend_h": z_trend_h,
            "seasonal_weekly_h": z_seasonal_weekly_h,
            "seasonal_daily_h": z_seasonal_daily_h,
            "residual_h": z_residual_h,
            "combined_tangent": combined_tangent,
            "combined_h": combined_h
        }

# Add this class to the end of your existing mlp_embed_lorentz.py file

class SingleLorentz(nn.Module):
    """
    Encodes a single component to hyperbolic space.
    Used for testing individual components in isolation.
    """
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                 curvature=1.0, n_layer=2, use_attention_pooling=False):
        """
        Args:
            lookback: int - lookback window size
            input_dim: int - number of input features
            embed_dim: int - dimension of hyperbolic embeddings
            hidden_dim: int - hidden dimension for MLP
            curvature: float - curvature of Lorentz manifold
            n_layer: int - number of MLP layers
            use_attention_pooling: bool - use attention pooling
        """
        super().__init__()
        
        # Single MLP encoder
        self.embed = MLPEmbed(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            n_layer=n_layer,
            lookback=lookback,
            use_attention_pooling=use_attention_pooling
        )
        
        # Lorentz manifold
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        
        # Scaling parameter
        self.effective_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        """
        Encode single component to hyperbolic space.
        
        Args:
            x: [B, seq_len, input_dim] - single time series component
        
        Returns:
            dict with:
                - tangent: [B, embed_dim] - tangent space representation
                - hyperbolic: [B, embed_dim+1] - point on Lorentz manifold
        """
        # 1) Encode to Euclidean latent (tangent vector)
        z_tangent = self.encoder(x)  # [B, embed_dim]
        
        # 2) Scale
        effective_scale = torch.tanh(self.effective_scale)
        scaled_tangent = z_tangent * effective_scale
        
        # 3) Map to hyperbolic space
        z_h = safe_expmap0(self.manifold, scaled_tangent)
        z_h = self.manifold.projx(z_h)
        
        return {
            "tangent": z_tangent,
            "lorentz": z_h
        }


class HybridLorentz(nn.Module):
    """
    Hybrid encoder: One component in hyperbolic, others in Euclidean.
    Perfect for ablation studies.
    """
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64, 
                 curvature=1.0, hyperbolic_component='seasonal_weekly',
                 n_layer=2, use_attention_pooling=False):
        """
        Args:
            hyperbolic_component: str - which component to encode in hyperbolic space
                Options: 'trend', 'seasonal_weekly', 'seasonal_daily', 'residual'
        """
        super().__init__()
        
        self.hyperbolic_component = hyperbolic_component
        self.components = ['trend', 'seasonal_weekly', 'seasonal_daily', 'residual']
        
        # Lorentz manifold for hyperbolic component
        self.manifold = geoopt.manifolds.Lorentz(k=curvature)
        
        # Create encoders - one hyperbolic, rest Euclidean
        self.encoders = nn.ModuleDict()
        
        for comp in self.components:
            self.encoders[comp] = MLPEmbed(
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
    
    def forward(self, trend, seasonal_weekly, seasonal_daily, residual):
        """
        Encode components with hybrid approach.
        
        Args:
            trend, seasonal_weekly, seasonal_daily, residual: [B, seq_len, input_dim]
        
        Returns:
            dict with:
                - tangent_vectors: dict of [B, embed_dim] for each component
                - hyperbolic_point: [B, embed_dim+1] (only for hyperbolic component)
                - combined_tangent: [B, embed_dim] - all components in tangent space
                - component_types: dict showing which space each uses
        """
        components_data = {
            'trend': trend,
            'seasonal_weekly': seasonal_weekly,
            'seasonal_daily': seasonal_daily,
            'residual': residual
        }
        
        tangent_vectors = {}
        hyperbolic_point = None
        
        for comp_name, comp_data in components_data.items():
            # Encode to latent space
            z_latent = self.encoders[comp_name](comp_data)  # [B, embed_dim]
            
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