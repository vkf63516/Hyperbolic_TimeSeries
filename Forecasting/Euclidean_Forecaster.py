import torch
import torch.nn as nn
from embed.mlp_embed_euclidean import ParallelEuclideanEmbed
from DynamicsMvar.Euclidean_Residual_Dynamics import ResidualDynamics
from Lifting.euclidean_reconstructor import EuclideanReconstructor
from Lifting.euclidean_segment_reconstructor import EuclideanSegmentReconstructionHead
from spec import RevIN

class PointForecastEuclidean(nn.Module):
    """
    Point-level Euclidean forecasting
    NO manifold operations - pure Euclidean arithmetic
    """
    def __init__(self, lookback, n_features, pred_len, embed_dim=32, hidden_dim=64,
                 use_attention_pooling=False, use_revin=False, 
                 use_truncated_bptt=False, truncate_every=16,
                 dynamic_dropout=0.3, embed_dropout=0.5, recon_dropout=0.2, num_layers=2):
        super().__init__()
        self.lookback = lookback
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.use_revin = use_revin
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        self.hidden_dim = hidden_dim
        self.dynamic_dropout = dynamic_dropout
        self.num_layers = num_layers
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)        # Encoder: 4 parallel Mamba blocks
        self.embed = ParallelEuclideanEmbed(
            lookback=lookback,
            input_dim=n_features,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_layer=num_layers,
            embed_dropout=embed_dropout,
            use_attention_pooling=use_attention_pooling
        )
        self.dynamics_combined = self._create_dynamics()
        self.dynamics_trend = self._create_dynamics()
        self.dynamics_coarse = self._create_dynamics()
        self.dynamics_fine = self._create_dynamics()
        self.dynamics_resid = self._create_dynamics()
        self.dynamics = ResidualDynamics(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dynamic_dropout,
            n_layers=self.num_layers
        )
        self.step_size = nn.Parameter(torch.tensor(0.1))
        # Reconstructor: simple MLP (no logmap/expmap needed!)
        self.reconstructor = EuclideanReconstructor(
            embed_dim=embed_dim,
            output_dim=n_features,
            hidden_dim=hidden_dim,
            dropout=recon_dropout
        )
    def _create_dynamics(self):
       
        return ResidualDynamics(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dynamic_dropout,
            n_layers=self.num_layers
        )

    def forward(self, trend, coarse, fine, residual):
        """
        Forecast each component separately, combine at the end.
        """
        # Step 1: Normalize each component SEPARATELY
        x_combined = trend + coarse + fine + residual  # [B, lookback, n_features]
        
        # Step 2: Store RevIN stats from combined signal (NO transformation applied!)
        if self.use_revin:
            # This ONLY stores mean and std, doesn't transform the data
            # We pass mode='norm' but it's just to compute and store stats
            self.revin(x_combined, mode='norm')
        
        # Step 3: Encode RAW components (they're already normalized by StandardScaler!)
        # We DON'T normalize them again!
        embed_output = self.embed(trend, coarse, fine, residual)
        z_current = embed_output['combined_e']  # [B, embed_dim]
        z_current_trend = embed_output["trend_e"]
        z_current_coarse = embed_output["seasonal_coarse_e"]
        z_current_fine = embed_output["seasonal_fine_e"]
        z_current_resid = embed_output["residual_e"]
        
        # Step 4: Autoregressive forecasting (in normalized space)
        predictions_norm = []
        trend_predictions = []
        coarse_predictions = []
        fine_predictions = []
        residual_predictions = []
        step_size = torch.sigmoid(self.step_size)

        for step in range(self.pred_len):
            # Reconstruct in normalized space
            x_pred_norm = self.reconstructor(z_current)  # [B, n_features]
            predictions_norm.append(x_pred_norm)
            
            x_pred_trend = self.reconstructor(z_current_trend)
            trend_predictions.append(x_pred_trend)

            x_pred_coarse = self.reconstructor(z_current_coarse)
            coarse_predictions.append(x_pred_coarse)

            x_pred_fine = self.reconstructor(z_current_fine)
            fine_predictions.append(x_pred_fine)

            x_prend_resid = self.reconstructor(z_current_resid)
            residual_predictions.append(x_prend_resid)
            # Update latent state
            # z_next = self.dynamics(z_current)

            z_next = self.dynamics(z_current)
            z_current = z_current + step_size * (z_next - z_current)
            

            z_next_trend = self.dynamics(z_current_trend)
            z_current_trend = z_current_trend + step_size * (z_next_trend - z_current_trend)

            z_next_coarse = self.dynamics(z_current_coarse)
            z_current_coarse = z_current_coarse + step_size * (z_next_coarse - z_current_coarse)

            z_next_fine  = self.dynamics(z_current_fine)
            z_current_fine = z_current_fine + step_size * (z_next_fine - z_current_fine)

            z_next_resid = self.dynamics(z_current_resid)
            z_current_resid = z_current_resid + step_size * (z_next_resid - z_current_resid)

            
            # Optional: truncated BPTT
            if self.use_truncated_bptt and (step + 1) % self.truncate_every == 0 and step < self.pred_len - 1:
                z_current = z_current.detach()
        
        predictions_norm = torch.stack(predictions_norm, dim=1)  # [B, pred_len, n_features]
        trend_predictions = torch.stack(trend_predictions, dim=1)
        coarse_predictions = torch.stack(coarse_predictions, dim=1)
        fine_predictions = torch.stack(fine_predictions, dim=1)
        residual_predictions = torch.stack(residual_predictions, dim=1)
        
        # Step 5: Denormalize predictions using stored stats
        if self.use_revin:
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm
        
        return {
            'predictions': predictions,  # [B, pred_len, n_features]
            'trend_predictions': trend_predictions,
            'coarse_predictions': coarse_predictions,
            'fine_predictions': fine_predictions,
            'residual_predictions': residual_predictions,
        }

