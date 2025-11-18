import sys
import torch
import torch.nn as nn
import geoopt
from embed.mlp_embed_euclidean import ParallelEuclideanEmbed
from DynamicsMvar.Lorentz_Residual_Dynamics import HyperbolicLorentzDynamics
from DynamicsMvar.Poincare_Residual_Dynamics import HyperbolicPoincareDynamics
from embed.mlp_embed_lorentz import ParallelLorentz
from embed.mlp_embed_poincare import ParallelPoincare
from Lifting.hyperbolic_reconstructor import HyperbolicReconstructionHead
from spec import RevIN, safe_expmap0, safe_expmap

class ComponentForecaster(nn.Module):
    def __init__(self, lookback, pred_len, n_features, embed_dim, hidden_dim, 
                curvature, manifold_type, use_attention_pooling=False, use_revin=False,
                use_truncated_bptt=False, truncate_every=16,dynamic_dropout=0.3,
                embed_dropout=0.5, recon_dropout=0.2, num_layers=2):
    
        super().__init__()

        self.lookback = lookback
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.n_features = n_features
        self.use_revin = use_revin
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        self.hidden_dim = hidden_dim
        self.manifold_type = manifold_type

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
                use_attention_pooling=use_attention_pooling
            )
        else:
            self.embed_hyperbolic = ParallelLorentz(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                curvature=curvature,
                use_attention_pooling=use_attention_pooling
            )
        
        self.manifold = self.embed_hyperbolic.manifold
        
        # Dynamics network (predicts velocity in tangent space)
        if manifold_type == "Lorentzian":
            self.dynamics = HyperbolicLorentzDynamics(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                manifold=self.manifold,
                dropout=dynamic_dropout,
                n_layers=num_layers
            )
        else:
            self.dynamics = HyperbolicPoincareDynamics(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                manifold=self.manifold,
                dropout=dynamic_dropout,
                n_layers=num_layers
            )
        
        # Reconstructor
        self.reconstructor = HyperbolicReconstructionHead(
            embed_dim=embed_dim,
            output_dim=n_features,
            manifold=self.manifold,
            hidden_dim=hidden_dim,
            dropout=recon_dropout
        )
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):

        embed_h = self.embed_hyperbolic(trend, seasonal_coarse, seasonal_fine, residual)
        z_current_trend = embed_h["trend_h"]
        z_current_coarse = embed_h["seasonal_coarse_h"]
        z_current_fine = embed_h["seasonal_fine_h"]
        z_current_residual = embed_h["residual_h"]

        trend_predictions = []
        trend_embed_trajectory = []
        coarse_predictions = []
        coarse_embed_trajectory = []
        fine_predictions = []
        fine_embed_trajectory = []
        residual_predictions = []
        residual_embed_trajectory = []

        for step in range(self.pred_len):
            # Lift point back to original dimension
            trend_pred = self.reconstructor(z_current_trend)
            trend_predictions.append(trend_pred)
            trend_embed_trajectory.append(z_current_trend)

            coarse_pred = self.reconstructor(z_current_coarse)
            coarse_predictions.append(coarse_pred)
            coarse_embed_trajectory.append(z_current_coarse)

            fine_pred = self.reconstructor(z_current_fine)
            fine_predictions.append(fine_pred)
            fine_embed_trajectory.append(z_current_fine)

            residual_pred = self.reconstructor(z_current_residual)
            residual_predictions.append(residual_pred)
            residual_embed_trajectory.append(z_current_residual)

            
            # Evolve dynamics in tangent space
            tangent_current_trend = self.manifold.logmap0(z_current_trend)  # [B, embed_dim]
            tangent_current_coarse = self.manifold.logmap0(z_current_coarse)
            tangent_current_fine = self.manifold.logmap0(z_current_fine)
            tangent_current_residual = self.manifold.logmap0(z_current_residual)
            
            # Predict velocity
            z_current_trend = self.dynamics(tangent_current_trend)  # [B, embed_dim]
            z_current_coarse = self.dynamics(tangent_current_coarse)
            z_current_fine = self.dynamics(tangent_current_fine)
            z_current_residual = self.dynamics(tangent_current_residual)
            
            # Map back to manifold
            z_current_trend = self.manifold.projx(z_current_trend)
            z_current_coarse = self.manifold.projx(z_current_coarse)
            z_current_fine = self.manifold.projx(z_current_fine)
            z_current_residual = self.manifold.projx(z_current_residual)
        
        return {
            'trend_predictions': torch.stack(trend_predictions, dim=1),  # [B, pred_len, n_features]
            'trend_embed_trajectory': torch.stack(trend_embed_trajectory, dim=1),  # [B, pred_len, embed_dim]
            'coarse_predictions': torch.stack(coarse_predictions, dim=1),
            'coarse_embed_trajectory': torch.stack(coarse_embed_trajectory, dim=1),
            'fine_predictions': torch.stack(fine_predictions, dim=1),
            'fine_embed_trajectory': torch.stack(fine_embed_trajectory, dim=1),
            'residual_predictions': torch.stack(residual_predictions, dim=1),
            'residual_embed_trajectory': torch.stack(residual_embed_trajectory, dim=1)
        }
