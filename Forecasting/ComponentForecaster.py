import sys
import torch
import torch.nn as nn
import geoopt
from embed.mlp_embed_euclidean import ParallelEuclideanEmbed
from DynamicsMvar.Lorentz_Residual_Dynamics import HyperbolicLorentzDynamics
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
        z_current = embed_h["trend_h"]

        predictions = []
        embed_trajectory = []

        for step in range(self.pred_len):
            # Lift point back to original dimension
            trend_pred = self.reconstructor(z_current)
            predictions.append(trend_pred)
            embed_trajectory.append(z_current)
            # Evolve dynamics in tangent space
            tangent_current = self.manifold.logmap0(z_current)  # [B, embed_dim]
            
            # Predict velocity
            z_current = self.dynamics(tangent_current)  # [B, embed_dim]
            
            # Map back to manifold
            z_current = self.manifold.projx(z_current)
        
        return {
            'component_predictions': torch.stack(predictions, dim=1),  # [B, pred_len, n_features]
            'embed_trajectory': torch.stack(embed_trajectory, dim=1),  # [B, pred_len, embed_dim]
        }
