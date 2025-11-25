import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))


from Forecasting.Euclidean_Forecaster import PointForecastEuclidean
from Forecasting.Segment_Euclidean_Forecaster import SegmentForecastEuclidean
from Forecasting.Forecaster import HyperbolicForecaster
from Forecasting.Segment_Forecaster import SegmentedHyperbolicForecaster

class Model(nn.Module):
    """
    Hyperbolic Forecasting Model
    
    Architecture:
    1. Decompose time series into trend, seasonal_fine, seasonal_coarse, residual
    2. Encode each component to hyperbolic space via encoders
    3. Combine components in tangent space
    4. Autoregressively forecast in hyperbolic space
    5. Reconstruct to Euclidean space
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.embed_dim = configs.embed_dim
        self.hidden_dim = configs.hidden_dim
        self.curvature = configs.curvature
        self.mstl_period = configs.mstl_period
        self.use_segments = configs.use_segments
        self.manifold_type = configs.manifold_type
        self.use_attention_pooling = configs.use_attention_pooling
        self.use_revin = configs.use_revin
        # Model dimensions
        # Number of input features
        self.enc_in = configs.enc_in
        self.share_feature_weights = configs.share_feature_weights
        
        # Embedding: Maps decomposed components to hyperbolic space
        
        if self.manifold_type == "Euclidean":
            if self.use_segments:
                self.forecaster = SegmentForecastEuclidean(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    manifold_type=self.manifold_type,
                    segment_length=self.mstl_period,
                    use_segment_norm=True,
                    use_revin=self.use_revin,
                    embed_dropout=0.5,
                    dynamic_dropout=0.3,
                    recon_dropout=0.2,
                    share_feature_weights=self.share_feature_weights,
                    num_layers=2
                )
            else:
                self.forecaster = PointForecastEuclidean(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    use_attention_pooling=self.use_attention_pooling,
                    use_revin=self.use_revin,
                    use_truncated_bptt=True,
                    truncate_every=16
                )
        else:
            if self.use_segments:
                self.forecaster = SegmentedHyperbolicForecaster(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    curvature=self.curvature,
                    manifold_type=self.manifold_type,
                    segment_length=self.mstl_period,
                    use_segment_norm=True,
                    use_revin=self.use_revin,
                    embed_dropout=0.5,
                    dynamic_dropout=0.3,
                    recon_dropout=0.2,
                    num_layers=2,
                    share_feature_weights=self.share_feature_weights

                )
            else:
                self.forecaster = HyperbolicForecaster(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    curvature=self.curvature,
                    manifold_type=self.manifold_type,
                    use_attention_pooling=self.use_attention_pooling                
                )

            # Forecaster: Autoregressively predicts in hyperbolic space

   
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Forward pass with explicit decomposed components.
        Use this when you have orthogonalMSTL decomposition.
        
        Args:
            trend: [B, seq_len, C]
            coarse: [B, seq_len, C]
            fine: [B, seq_len, C]
            resid: [B, seq_len, C]
            
        Returns:
            predictions: [B, pred_len, output_dim]
        """

        forecasts = self.forecaster(trend, seasonal_coarse, seasonal_fine, residual)
        # Get individual hyperbolic representations
        x_hat = forecasts["predictions"]
        
        return x_hat
