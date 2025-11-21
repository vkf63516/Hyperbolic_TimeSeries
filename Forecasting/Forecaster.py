import sys
import torch
import torch.nn as nn
import geoopt
from embed.mlp_embed_euclidean import ParallelEuclideanEmbed
from DynamicsMvar.Lorentz_Residual_Dynamics import HyperbolicLorentzDynamics, HyperbolicLorentzMultiStepDynamics
from DynamicsMvar.Poincare_Residual_Dynamics import HyperbolicPoincareDynamics
from embed.mlp_embed_lorentz import ParallelLorentz
from embed.mlp_embed_poincare import ParallelPoincare
from embed.segment_mlp_embed_lorentz import SegmentedParallelLorentz
from Lifting.hyperbolic_reconstructor import HyperbolicReconstructionHead
from spec import RevIN, safe_expmap

class HyperbolicForecaster(nn.Module):
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
        z_current_resid = embed_h["residual_h"]
        z_current = embed_h["combined_h"]
        z_previous = None
        z_previous_trend = None
        z_previous_coarse = None
        z_previous_fine = None
        z_previous_resid = None
        # print(f"Z Current {z_current}")
        trend_predictions = []
        coarse_predictions = []
        fine_predictions = []
        residual_predictions = []
        predictions = []

        for step in range(self.pred_len):
            # Lift point back to original dimension
            # print(step)
            x_pred = self.reconstructor(z_current)
            predictions.append(x_pred)

            trend_pred = self.reconstructor(z_current_trend)
            trend_predictions.append(trend_pred)

            coarse_pred = self.reconstructor(z_current_coarse)
            coarse_predictions.append(coarse_pred)

            fine_pred = self.reconstructor(z_current_fine)
            fine_predictions.append(fine_pred)

            residual_pred = self.reconstructor(z_current_resid)
            residual_predictions.append(residual_pred)
          
            # Predict velocity and to make sure each step is not isolated
            z_current, z_previous = self.dynamics(z_current, z_previous)
            z_current_trend, z_previous_trend = self.dynamics(z_current_trend, z_previous_trend)  # [B, embed_dim]
            z_current_coarse, z_previous_coarse = self.dynamics(z_current_coarse, z_previous_coarse)
            z_current_fine, z_previous_fine = self.dynamics(z_current_fine, z_previous_fine)
            z_current_resid, z_previous_resid = self.dynamics(z_current_resid, z_previous_resid)
            
        
            if (step + 1) % 8 == 0 and step < self.pred_len - 1:
                z_current = z_current.detach()
                z_current_trend = z_current_trend.detach()
                z_current_coarse = z_current_coarse.detach()
                z_current_fine = z_current_fine.detach()
                z_current_resid = z_current_resid.detach()
            
            if z_previous is not None:
                z_previous = z_previous.detach()
                z_previous_trend = z_previous_trend.detach()
                z_previous_coarse = z_previous_coarse.detach()
                z_previous_fine = z_previous_fine.detach()
                z_previous_resid = z_previous_resid.detach()
    
        
        
        return {
            'predictions': torch.stack(predictions, dim=1),
            'trend_predictions': torch.stack(trend_predictions, dim=1),  # [B, pred_len, n_features]
            'coarse_predictions': torch.stack(coarse_predictions, dim=1),
            'fine_predictions': torch.stack(fine_predictions, dim=1),
            'residual_predictions': torch.stack(residual_predictions, dim=1),
        }
