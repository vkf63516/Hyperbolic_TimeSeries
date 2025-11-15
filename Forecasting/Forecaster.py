import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn
import geoopt
from embed.mlp_embed_lorentz import ParallelLorentz, HybridLorentz
from embed.mlp_embed_poincare import ParallelPoincare, HybridPoincare
from Lifting.hyperbolic_reconstructor import HyperbolicReconstructionHead
from spec import RevIN

class ResidualDynamics(nn.Module):
    """
    Residual block for dynamics prediction.
    Outputs: z_next = z + weighted_residual
    This prevents drift by keeping updates close to identity mapping.
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.3, n_layers=2):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(embed_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer (maps back to embed_dim)
        layers.append(nn.Linear(hidden_dim, embed_dim))
        
        self.residual_net = nn.Sequential(*layers)
        
        # Learnable residual weight (initialized near 0 for stability)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, z):
        """
        Args:
            z: [B, embed_dim] - current state
        Returns:
            z_next: [B, embed_dim] - next state with residual connection
        """
        residual = self.residual_net(z)
        # Residual connection: z_next = z + α * f(z)
        # where α is learnable and starts small
        return z + self.residual_weight * residual


class HyperbolicPointForecaster(nn.Module):
    """
    Hyperbolic space forecaster with learned dynamics.
    Works with any encoder backend (MLP, Mamba, Transformer).
    """
    def __init__(self, lookback, pred_len, n_features, embed_dim, hidden_dim, 
                 curvature, manifold_type, use_hierarchy=False, 
                 hierarchy_scales=[0.5,1.0,1.5,2.0], use_attention_pooling=False, 
                 use_revin=False, use_truncated_bptt=False, truncate_every=16,
                 dynamic_dropout=0.3, embed_dropout=0.5, recon_dropout=0.2, 
                 num_layers=2):
        super().__init__()
        self.lookback = lookback
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.n_features = n_features
        self.use_revin = use_revin
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)

        
        # Encoder (works with MLP, Mamba, or Transformer backend)
        if manifold_type == "Poincare":

            self.embed_hyperbolic = ParallelPoincare(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                curvature=curvature, 
                use_hierarchy=use_hierarchy,
                hierarchy_scales=hierarchy_scales,
                use_attention_pooling=use_attention_pooling
            )
        else:
            self.embed_hyperbolic = ParallelLorentz(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                curvature=curvature,
                use_hierarchy=use_hierarchy,
                hierarchy_scales=hierarchy_scales,
                use_attention_pooling=use_attention_pooling
            )
        
        self.manifold = self.embed_hyperbolic.manifold
        
        # Dynamics network (predicts velocity in tangent space)
        self.dynamics = ResidualDynamics(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=0.3,
            n_layers=3
        )
        
        # Step size (learnable)
        self.step_size = nn.Parameter(torch.tensor(0.1))
        
        # Reconstructor
        self.reconstructor = HyperbolicReconstructionHead(
            embed_dim=embed_dim,
            output_dim=n_features,
            manifold=self.manifold,
            hidden_dim=hidden_dim,
            dropout=0.2
        )
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, lookback, n_features]
            teacher_forcing: Not used in this version (could add latent teacher forcing)
            target: [B, pred_len, n_features] - for computing loss only
        
        Returns:
            'predictions': [B, pred_len, n_features]
            'embed_trajectory': [B, pred_len, embed_dim+1]
        """
        # Encode decomposed components once
        embed_output = self.embed_hyperbolic(trend, seasonal_coarse, seasonal_fine, residual)
        z_current = embed_output['combined_h']  # [B, embed_dim+1] for Lorentz
        
        predictions = []
        embed_trajectory = []
        
        # Autoregressive generation in hyperbolic space
        for step in range(self.pred_len):
            # Decode current state
            x_pred = self.reconstructor(z_current)
            predictions.append(x_pred)
            embed_trajectory.append(z_current)
            
            # Evolve dynamics in tangent space
            tangent_current = self.manifold.logmap0(z_current)  # [B, embed_dim]
            
            # Predict velocity
            velocity = self.dynamics(tangent_current)  # [B, embed_dim]
            
            # Update position in tangent space
            step_size = torch.sigmoid(self.step_size)  # Constrain to (0, 1)
            tangent_next = tangent_current + step_size * velocity
            
            # Map back to manifold
            z_current = self.manifold.expmap0(tangent_next)
            z_current = self.manifold.projx(z_current)
        
        return {
            'predictions': torch.stack(predictions, dim=1),  # [B, pred_len, n_features]
            'embed_trajectory': torch.stack(embed_trajectory, dim=1),  # [B, pred_len, embed_dim]
        }
# Add this new class at the end

class HybridComponentForecaster(nn.Module):
    """
    Forecaster using HybridLorentz encoder.
    One component in hyperbolic, others in Euclidean.
    """
    def __init__(self, lookback, pred_len, n_features, embed_dim, hidden_dim, 
                 curvature, manifold_type, use_hierarchy=False, hierarchy_scales=[0.5,1.0,1.5,2.0],
                 use_attention_pooling=False, use_revin=False, 
                 use_truncated_bptt=False, truncate_every=16,
                 dynamic_dropout=0.3, embed_dropout=0.5, recon_dropout=0.2, 
                 hyperbolic_component='seasonal_coarse'):
        super().__init__()
        self.lookback = lookback
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.n_features = n_features
        self.use_revin = use_revin
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)
        
        # Hybrid encoder
        if manifold_type == 'Lorentz':
            from embed.mlp_embed_lorentz import HybridLorentz
            self.embed_hybrid = HybridLorentz(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                curvature=curvature,
                hyperbolic_component=hyperbolic_component,
                use_attention_pooling=True
            )
        else:  # Poincare
            self.embed_hybrid = HybridPoincare(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                curvature=curvature,
                hyperbolic_component=hyperbolic_component,
                use_attention_pooling=True
            )
        
        # Dynamics in tangent space
        self.dynamics = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        self.step_size = nn.Parameter(torch.tensor(0.1))
        
        # Reconstructor (works in tangent space)
        self.reconstructor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features)
        )
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, lookback, n_features]
        
        Returns:
            dict with predictions and component info
        """
        # Encode with hybrid approach
        embed_output = self.embed_hybrid(trend, seasonal_coarse, seasonal_fine, residual)
        z_current = embed_output['combined_tangent']  # [B, embed_dim]
        
        predictions = []
        
        # Autoregressive forecasting in tangent space
        for step in range(self.pred_len):
            # Reconstruct
            x_pred = self.reconstructor(z_current)
            predictions.append(x_pred)
            
            # Evolve dynamics
            velocity = self.dynamics(z_current)
            step_size = torch.sigmoid(self.step_size)
            z_current = z_current + step_size * velocity
        
        return {
            'predictions': torch.stack(predictions, dim=1),
            'component_info': embed_output['component_types'],
            'hyperbolic_component': embed_output['hyperbolic_component']
        }