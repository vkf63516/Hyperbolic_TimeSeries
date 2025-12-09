import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
import os
import sys
from pathlib import Path
from Forecasting.Moving_Window_Segment_Euclidean_Forecaster import MovingWindowEuclideanForecaster
from Forecasting.Moving_Window_Segment_Forecaster import MovingWindowHyperbolicForecaster
from Forecasting.Segment_Euclidean_Forecaster import SegmentForecastEuclidean
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
        self.use_moving_window = configs.use_moving_window
        # Model dimensions
        # Number of input features
        self.enc_in = configs.enc_in
        
        # Embedding: Maps decomposed components to hyperbolic space
        
        if self.manifold_type == "Euclidean":
            if self.use_moving_window:
                self.forecaster = MovingWindowEuclideanForecaster(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    manifold_type=self.manifold_type,
                    segment_length=self.mstl_period,
                    use_segment_norm=True,
                    use_revin=self.use_revin,
                    embed_dropout=0.1,
                    dynamic_dropout=0.3,
                    num_layers=2,
                )
            else:

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
                    embed_dropout=0.3,
                    dynamic_dropout=0.3,
                    recon_dropout=0.2,
                    num_layers=2
                )
        else:
            if self.use_moving_window:
                self.forecaster = MovingWindowHyperbolicForecaster(
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
                    embed_dropout=0.1,
                    dynamic_dropout=0.3,
                    window_size=15,
                    num_layers=2,
                )
            else:

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
                    num_layers=2
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
        if self.manifold_type == "Euclidean":
            return x_hat, []
        x_hyp = forecasts["hyperbolic_states"]["combined_h"]
        
        return x_hat, x_hyp
