import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))


from Forecasting.Euclidean_Forecaster import PointForecastEuclidean
from Forecasting.Forecaster import HyperbolicPointForecaster
from Forecasting.ComponentForecaster import ComponentForecaster
from Forecasting.Segment_Forecaster import HyperbolicSegmentForecaster
from spec import safe_expmap0

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
        
        # Embedding: Maps decomposed components to hyperbolic space
        if self.use_segments:
            self.embedding = SegmentParallelLorentzBlock(
                lookback_steps=self.seq_len,
                input_dim=self.enc_in,
                seg_len=self.mstl_period,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                curvature=self.curvature
            )
            self.forecaster = HyperbolicSegmentForecaster(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                seg_len=self.mstl_period,
                manifold=self.embedding.manifold
            )
   
        else:
            if self.manifold_type == "Euclidean":

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

                self.forecaster = HyperbolicPointForecaster(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    embed_dim=self.embed_dim,
                    hidden_dim=self.hidden_dim,
                    curvature=self.curvature,
                    manifold_type=self.manifold_type,
                    use_attention_pooling=self.use_attention_pooling
                )

                self.component_forecaster = ComponentForecaster(
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
        Use this when you have TimeBaseMSTL decomposition.
        
        Args:
            trend: [B, seq_len, C]
            coarse: [B, seq_len, C]
            fine: [B, seq_len, C]
            resid: [B, seq_len, C]
            
        Returns:
            predictions: [B, pred_len, output_dim]
        """

        forecasts = self.forecaster(trend, seasonal_coarse, seasonal_fine, residual)
        x_hat = forecasts["predictions"]
        comp_forecasts = self.component_forecaster(trend, seasonal_coarse, seasonal_fine, residual)
        # Get individual and combined hyperbolic representations
        comp_xhat = comp_forecasts["component_predictions"]
        return x_hat, comp_xhat
