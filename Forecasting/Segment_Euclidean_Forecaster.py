
import torch
import torch.nn as nn
from encode.Linear.segment_linear_encode_euclidean import SegmentParallelEuclidean
from DynamicsMvar.Euclidean_Residual_Dynamics import ResidualDynamics
from Lifting.euclidean_segment_reconstructor import EuclideanSegmentReconstructionHead
from spec import RevIN

class SegmentForecastEuclidean(nn.Module):
    def __init__(self, lookback, pred_len, n_features, encode_dim,
         hidden_dim, manifold_type, segment_length=24, 
         use_attention_pooling=False, use_revin=False,
         use_truncated_bptt=False, truncate_every=4,  # Truncate every N segments
         dynamic_dropout=0.3, encode_dropout=0.5, recon_dropout=0.2, share_feature_weights=False,
         num_layers=2, use_segment_norm=True):
        """
        Args:
            lookback: int - lookback window (should be divisible by segment_length)
            pred_len: int - prediction horizon (should be divisible by segment_length)
            n_features: int - number of input features
            encode_dim: int - hyperbolic encodeding dimension
            hidden_dim: int - hidden dimension for MLPs
            curvature: float - manifold curvature
            manifold_type: str - "Poincare" or "Lorentzian"
            segment_length: int - length of each segment (e.g., 24 for daily in hourly data)
            use_revin: bool - use reversible instance normalization
            use_truncated_bptt: bool - truncated backprop through time
            truncate_every: int - truncate gradient every N SEGMENTS (not steps!)
            dynamic_dropout: float - dropout in dynamics
            encode_dropout: float - dropout in encodeder
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
        self.encode_dim = encode_dim
        self.pred_len = pred_len
        self.n_features = n_features
        self.segment_length = segment_length
        self.num_pred_segments = pred_len // segment_length  # NEW: forecast in segments
        self.use_revin = use_revin
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        self.hidden_dim = hidden_dim
        self.manifold_type = manifold_type
        self.encode_dropout = encode_dropout
        self.dynamic_dropout = dynamic_dropout
        self.recon_dropout=recon_dropout
        self.num_layers=num_layers
        self.share_feature_weights = share_feature_weights
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)
        
        # Segmented encoder

        self.encode_euclidean = SegmentParallelEuclidean(
            lookback=lookback,
            input_dim=n_features,
            encode_dim=encode_dim,
            segment_length=segment_length,
            encode_dropout=encode_dropout,
            use_segment_norm=use_segment_norm,
            share_feature_weights=share_feature_weights
        )
        self.step_size = nn.Parameter(torch.tensor(0.1))

        self.dynamics = self._create_dynamics()
        
        # Segment reconstructor (NEW: outputs entire segments, not single points)
        self.reconstructor = EuclideanSegmentReconstructionHead(
            encode_dim=encode_dim,
            output_dim=n_features,
            segment_length=segment_length,  # NEW
            hidden_dim=hidden_dim,
            dropout=recon_dropout
        )
 

    def _create_dynamics(self):
       
        return ResidualDynamics(
            encode_dim=self.encode_dim,
            hidden_dim=self.hidden_dim,
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
        
        encode_e = self.encode_euclidean(trend, seasonal_coarse, seasonal_fine, residual)
        z_current = encode_e["combined_e"]
        
        # Storage for SEGMENT predictions (not point predictions!)
        predictions_norm = []
        step_size = torch.sigmoid(self.step_size)


        # Autoregressive rollout over SEGMENTS
        for seg_step in range(self.num_pred_segments):
            # Predict next state via Dynamics

            # Reconstruct entire SEGMENTS (not single points!)
            x_pred_norm_seg = self.reconstructor(z_current)  # [B, segment_length, n_features]
            predictions_norm.append(x_pred_norm_seg)
            # print(x_pred_norm_seg.shape)


            z_next = self.dynamics(z_current)
            z_current = z_current + step_size * (z_next - z_current)

            
            # Truncated BPTT (every N segments, not every N points!)
            if (seg_step + 1) % self.truncate_every == 0 and seg_step < self.num_pred_segments - 1:
                z_current = z_current.detach()
            
        
        # Stack segments and reshape to [B, pred_len, n_features]
        # predictions: list of [B, segment_length, n_features] → [B, num_segments, segment_length, n_features]
        # print(predictions.shape)
        predictions_norm = torch.stack(predictions_norm, dim=1)  # [B, num_segments, segment_length, n_features]
        # print(f"Predictions shape {predictions_norm.shape}")
        predictions_norm = predictions_norm.reshape(-1, self.pred_len, self.n_features)
        # print(f"Reshape Predictions shape {predictions_norm.shape}")
        
        
        if self.use_revin:
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm

        
        return {
            'predictions': predictions,  # [B, pred_len, n_features]

        }
       