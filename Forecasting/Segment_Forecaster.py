import sys
import torch
import torch.nn as nn
import geoopt
from embed.Linear.segment_linear_embed_poincare import SegmentedParallelPoincare
from embed.Linear.segment_linear_embed_lorentz import SegmentedParallelLorentz
from DynamicsMvar.Poincare_Residual_Dynamics import HyperbolicPoincareDynamics
from DynamicsMvar.Lorentz_Residual_Dynamics import HyperbolicLorentzDynamics
from Lifting.hyperbolic_segment_reconstructor import HyperbolicSegmentReconstructionHead  # NEW
from spec import RevIN, safe_expmap


class SegmentedHyperbolicForecaster(nn.Module):
    """
    Segment-aware hyperbolic forecaster.
    
    Key improvements over point-level:
    1. Encodes sequences as segments (e.g., daily patterns in hourly data)
    2. Forecasts entire segments at once (not individual points)
    3. Maintains segment structure throughout encoding → dynamics → reconstruction
    """
    def __init__(self, lookback, pred_len, n_features, embed_dim, hidden_dim, 
                 curvature, manifold_type, segment_length=24, 
                 use_attention_pooling=False, use_revin=False,
                 use_truncated_bptt=False, truncate_every=4,  # Truncate every N segments
                 dynamic_dropout=0.3, embed_dropout=0.5, recon_dropout=0.2, 
                 num_layers=2, use_segment_norm=True, share_feature_weights=False):
        """
        Args:
            lookback: int - lookback window (should be divisible by segment_length)
            pred_len: int - prediction horizon (should be divisible by segment_length)
            n_features: int - number of input features
            embed_dim: int - hyperbolic embedding dimension
            hidden_dim: int - hidden dimension for MLPs
            curvature: float - manifold curvature
            manifold_type: str - "Poincare" or "Lorentzian"
            segment_length: int - length of each segment (e.g., 24 for daily in hourly data)
            use_attention_pooling: bool - attention pooling over segments during encoding
            use_revin: bool - use reversible instance normalization
            use_truncated_bptt: bool - truncated backprop through time
            truncate_every: int - truncate gradient every N SEGMENTS (not steps!)
            dynamic_dropout: float - dropout in dynamics
            embed_dropout: float - dropout in embedder
            recon_dropout: float - dropout in reconstructor
            num_layers: int - number of layers
            use_segment_norm: bool - normalize each segment independently
        """
        super().__init__()

        # Validate that lookback and pred_len are divisible by segment_length
        if lookback % segment_length != 0:
            print(f"Warning: lookback ({lookback}) not divisible by segment_length ({segment_length}). "
                  f"Will pad to {(lookback // segment_length + 1) * segment_length}")
        if pred_len % segment_length != 0:
            raise ValueError(f"pred_len ({pred_len}) must be divisible by segment_length ({segment_length})")

        self.lookback = lookback
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.n_features = n_features
        self.segment_length = segment_length
        self.num_pred_segments = pred_len // segment_length  # NEW: forecast in segments
        self.use_revin = use_revin
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        self.hidden_dim = hidden_dim
        self.manifold_type = manifold_type
        self.embed_dropout = embed_dropout
        self.dynamic_dropout = dynamic_dropout
        self.num_layers=num_layers
        self.share_feature_weights = share_feature_weights
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)
        
        # Segmented encoder
        if manifold_type == "Poincare":
            self.embed_hyperbolic = SegmentedParallelPoincare(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                curvature=curvature,
                segment_length=segment_length,
                use_segment_norm=use_segment_norm,
                embed_dropout=self.embed_dropout,
                share_feature_weights=self.share_feature_weights,
            )
        elif manifold_type == "Lorentzian":  # Lorentzian
            self.embed_hyperbolic = SegmentedParallelLorentz(
                lookback=lookback,
                input_dim=n_features,
                embed_dim=embed_dim,
                curvature=curvature,
                segment_length=segment_length,
                use_segment_norm=use_segment_norm,
                embed_dropout=self.embed_dropout,
                share_feature_weights=self.share_feature_weights
            )
        self.manifold = self.embed_hyperbolic.manifold
        self.dynamics = self._create_dynamics()
    
        self.reconstructor = HyperbolicSegmentReconstructionHead(
            embed_dim=embed_dim,
            output_dim=n_features,
            segment_length=segment_length,  # NEW
            manifold=self.manifold,
            hidden_dim=hidden_dim,
            dropout=recon_dropout
        )

    def _create_dynamics(self):
        if self.manifold_type == "Poincare":
            return HyperbolicPoincareDynamics(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                manifold=self.manifold,
                dropout=self.dynamic_dropout,
                n_layers=self.num_layers
            )
        if self.manifold_type == "Lorentzian":
            return HyperbolicLorentzDynamics(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                manifold=self.manifold,
                dropout=self.dynamic_dropout,
                n_layers=self.num_layers
            )
        
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Segment-level forecasting.
        
        Args:
            trend: [B, seq_len, n_features]
            seasonal_coarse: [B, seq_len, n_features]
            seasonal_fine: [B, seq_len, n_features]
            residual: [B, seq_len, n_features]
        
        Returns:
            dict with:
                - predictions: [B, pred_len, n_features]
                - trend_predictions: [B, pred_len, n_features]
                - coarse_predictions: [B, pred_len, n_features]
                - fine_predictions: [B, pred_len, n_features]
                - residual_predictions: [B, pred_len, n_features]
        """
        # Encode (segments handled internally)
        x_combined = trend + seasonal_coarse + seasonal_fine + residual  # [B, lookback, n_features]
        
        # Step 2: Store RevIN stats from combined signal (NO transformation applied!)
        if self.use_revin:
            # This ONLY stores mean and std, doesn't transform the data
            # We pass mode='norm' but it's just to compute and store stats
            self.revin(x_combined, mode='norm')

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
        # Storage for SEGMENT predictions (not point predictions!)
        trend_predictions = []
        coarse_predictions = []
        fine_predictions = []
        residual_predictions = []
        predictions_norm = []

        latent_z = []
        latent_trend   = []
        latent_coarse  = []
        latent_fine    = []
        latent_resid   = []

        # Autoregressive rollout over SEGMENTS
        for seg_step in range(self.num_pred_segments):
            # Predict next state via Dynamics
            z_current, z_previous = self.dynamics(z_current, z_previous)
            z_current_trend, z_previous_trend = self.dynamics(z_current_trend, z_previous_trend)
            z_current_coarse, z_previous_coarse = self.dynamics(z_current_coarse, z_previous_coarse)
            z_current_fine, z_previous_fine = self.dynamics(z_current_fine, z_previous_fine)
            z_current_resid, z_previous_resid = self.dynamics(z_current_resid, z_previous_resid)   
            latent_z.append(z_current)
            latent_trend.append(z_current_trend)
            latent_coarse.append(z_current_coarse)
            latent_fine.append(z_current_fine)
            latent_resid.append(z_current_resid)
            # Truncated BPTT (every N segments, not every N points!)
            if (seg_step + 1) % self.truncate_every == 0 and seg_step < self.num_pred_segments - 1:
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

            # Reconstruct entire SEGMENTS (not single points!)
            x_pred_norm_seg = self.reconstructor(z_current)  # [B, segment_length, n_features]
            predictions_norm.append(x_pred_norm_seg)

            trend_pred_seg = self.reconstructor(z_current_trend)
            trend_predictions.append(trend_pred_seg)

            coarse_pred_seg = self.reconstructor(z_current_coarse)
            coarse_predictions.append(coarse_pred_seg)

            fine_pred_seg = self.reconstructor(z_current_fine)
            fine_predictions.append(fine_pred_seg)

            residual_pred_seg = self.reconstructor(z_current_resid)
            residual_predictions.append(residual_pred_seg)
        
        # Stack segments and reshape to [B, pred_len, n_features]
        # predictions: list of [B, segment_length, n_features] → [B, num_segments, segment_length, n_features]
        predictions_norm = torch.stack(predictions_norm, dim=1)  # [B, num_segments, segment_length, n_features]
        predictions_norm = predictions_norm.reshape(-1, self.pred_len, self.n_features)
        
        trend_predictions = torch.stack(trend_predictions, dim=1)
        trend_predictions = trend_predictions.reshape(-1, self.pred_len, self.n_features)
        
        coarse_predictions = torch.stack(coarse_predictions, dim=1)
        coarse_predictions = coarse_predictions.reshape(-1, self.pred_len, self.n_features)
        
        fine_predictions = torch.stack(fine_predictions, dim=1)
        fine_predictions = fine_predictions.reshape(-1, self.pred_len, self.n_features)
        
        residual_predictions = torch.stack(residual_predictions, dim=1)
        residual_predictions = residual_predictions.reshape(-1, self.pred_len, self.n_features)

        if self.use_revin:
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm
        
        # ---- NEW: Stack hyperbolic latent states ----
        def stack_latents(lst):
            segs = torch.stack(lst, dim=1)  # [B, num_segments, embed_dim+1]
            return segs

        hyperbolic_states = {
            "combined_h": stack_latents(latent_z),
            "trend_h":    stack_latents(latent_trend),
            "coarse_h":   stack_latents(latent_coarse),
            "fine_h":     stack_latents(latent_fine),
            "resid_h":    stack_latents(latent_resid),
        }


        return {
            'predictions': predictions,  # [B, pred_len, n_features]
            'trend_predictions': trend_predictions,
            'coarse_predictions': coarse_predictions,
            'fine_predictions': fine_predictions,
            'residual_predictions': residual_predictions,
            'hyperbolic_states': hyperbolic_states
        }