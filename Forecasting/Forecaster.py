import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn as nn
import geoopt
from embed.mlp_embed_lorentz import ParallelLorentz
from embed.mlp_embed_poincare import ParallelPoincare
from Lifting.hyperbolic_reconstructor import HyperbolicReconstructionHead


class HyperbolicPointForecaster(nn.Module):
    """
    Hyperbolic space forecaster with learned dynamics.
    Works with any encoder backend (MLP, Mamba, Transformer).
    """
    def __init__(self, lookback, pred_len, n_features, embed_dim, hidden_dim, 
                 curvature, manifold_type, use_hierarchy=False, 
                 hierarchy_scales=[0.5,1.0,1.5,2.0]):
        super().__init__()
        self.lookback = lookback
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.n_features = n_features
        
        # Encoder (works with MLP, Mamba, or Transformer backend)
        if manifold_type == "Poincare":
            self.embed_hyperbolic = ParallelPoincare(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                curvature=curvature, 
                use_hierarchy=use_hierarchy,
                hierarchy_scales=hierarchy_scales
            )
        else:
            self.embed_hyperbolic = ParallelLorentz(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                curvature=curvature,
                use_hierarchy=use_hierarchy,
                hierarchy_scales=hierarchy_scales
            )
        
        self.manifold = self.embed_hyperbolic.manifold
        
        # Dynamics network (predicts velocity in tangent space)
        self.dynamics = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Step size (learnable)
        self.step_size = nn.Parameter(torch.tensor(0.1))
        
        # Reconstructor
        self.reconstructor = HyperbolicReconstructionHead(
            embed_dim=embed_dim,
            output_dim=n_features,
            manifold=self.manifold
        )
    
    def forward(self, trend, seasonal_weekly, seasonal_daily, residual, 
                teacher_forcing=False, target=None):
        """
        Args:
            trend, seasonal_weekly, seasonal_daily, residual: [B, lookback, n_features]
            teacher_forcing: Not used in this version (could add latent teacher forcing)
            target: [B, pred_len, n_features] - for computing loss only
        
        Returns:
            'predictions': [B, pred_len, n_features]
            'embed_trajectory': [B, pred_len, embed_dim+1]
        """
        # Encode decomposed components once
        embed_output = self.embed_hyperbolic(trend, seasonal_weekly, seasonal_daily, residual)
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