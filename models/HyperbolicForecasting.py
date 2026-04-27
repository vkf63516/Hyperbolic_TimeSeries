import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
import os
import sys
from pathlib import Path
from Decomposition.Learnable_Decomposition import LearnableMultivariateDecomposition
from loss import hyperbolic_velocity_consistency_loss as hvcl, radial_diversity_loss as rdl, curvature_regularization_loss as crl
from Forecasting.Moving_Window_Segment_Euclidean_Forecaster import MovingWindowEuclideanForecaster
from Forecasting.Moving_Window_Segment_Forecaster import MovingWindowHyperbolicForecaster
from Forecasting.Direct_Moving_Window_Segment_Forecaster import DirectHyperbolicForecaster
from Forecasting.Segment_Euclidean_Forecaster import SegmentForecastEuclidean
from Forecasting.Segment_Forecaster import SegmentedHyperbolicForecaster
from Forecasting.Multi_Horizon_Forecasting import DirectMultiHorizonHyperbolicForecaster
from Forecasting.Euclidean_Multi_Horizon_Forecasting import EuclideanMultiHorizonHyperbolicForecaster
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
        self.encode_dim = configs.encode_dim
        self.hidden_dim = configs.hidden_dim
        self.curvature = configs.curvature
        self.mstl_period = configs.mstl_period
        self.use_segments = configs.use_segments
        self.manifold_type = configs.manifold_type
        self.use_revin = configs.use_revin
        self.use_multi_horizon = configs.use_multi_horizon
        self.use_moving_window = configs.use_moving_window
        self.num_basis = configs.num_basis
        self.window_size = configs.window_size
        self.use_learnable_decomposition = configs.use_learnable_decomposition
        self.use_no_decomposition = configs.use_no_decomposition
        # Model dimensions
        # Number of input features
        self.trend_period = configs.trend_period
        self.enc_in = configs.enc_in
        self.coarse_period = configs.coarse_period
        self.fine_period = configs.fine_period
        self.check = True
        # encodeding: Maps decomposed components to hyperbolic space
        if self.use_learnable_decomposition:
            self.decomposer = LearnableMultivariateDecomposition(
                n_features=self.enc_in,
                kernel_size=self.coarse_period * 2,
                detected_periods=[self.fine_period, self.coarse_period]
            )
        else:
            self.decomposer = None
            

        
        if self.manifold_type == "Euclidean":
            if self.use_moving_window:
                self.forecaster = MovingWindowEuclideanForecaster(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    encode_dim=self.encode_dim,
                    hidden_dim=self.hidden_dim,
                    manifold_type=self.manifold_type,
                    segment_length=self.mstl_period,
                    use_revin=self.use_revin,
                    encode_dropout=0.5,
                    recon_dropout=0.2,
                    num_layers=2,
                    window_size=self.window_size
                )
            elif self.use_multi_horizon:
                self.forecaster = EuclideanMultiHorizonHyperbolicForecaster(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    encode_dim=self.encode_dim,
                    hidden_dim=self.hidden_dim,
                    manifold_type=self.manifold_type,
                    segment_length=self.mstl_period,
                    use_revin=self.use_revin,
                    encode_dropout=0.3,
                    recon_dropout=0.2,
                    
                )
            else:

                self.forecaster = SegmentForecastEuclidean(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    encode_dim=self.encode_dim,
                    hidden_dim=self.hidden_dim,
                    manifold_type=self.manifold_type,
                    segment_length=self.mstl_period,
                    use_segment_norm=True,
                    use_revin=self.use_revin,
                    encode_dropout=0.3,
                    dynamic_dropout=0.3,
                    recon_dropout=0.2,
                    num_layers=2
                )
        else:
            if self.use_moving_window:
                if self.manifold_type == "Poincare":
                    if self.use_no_decomposition:
                        self.forecaster = DirectHyperbolicForecaster(
                            lookback=self.seq_len,
                            pred_len=self.pred_len,
                            n_features=self.enc_in,
                            encode_dim=self.encode_dim,
                            hidden_dim=self.hidden_dim,
                            curvature=self.curvature,
                            manifold_type=self.manifold_type,
                            segment_length=self.mstl_period,
                            use_revin=self.use_revin,
                            window_size=self.window_size,
                            encode_dropout=0.3,
                            recon_dropout=0.2,
                        )
                    else:
                        self.forecaster = MovingWindowHyperbolicForecaster(
                            lookback=self.seq_len,
                            pred_len=self.pred_len,
                            n_features=self.enc_in,
                            encode_dim=self.encode_dim,
                            hidden_dim=self.hidden_dim,
                            curvature=self.curvature,
                            manifold_type=self.manifold_type,
                            segment_length=self.mstl_period,
                            use_revin=self.use_revin,
                            window_size=self.window_size,
                            encode_dropout=0.3,
                            recon_dropout=0.2,
                        )
                elif self.manifold_type == "Lorentzian":
                    self.forecaster = MovingWindowHyperbolicForecaster(
                        lookback=self.seq_len,
                        pred_len=self.pred_len,
                        n_features=self.enc_in,
                        encode_dim=self.encode_dim,
                        hidden_dim=self.hidden_dim,
                        curvature=self.curvature,
                        manifold_type=self.manifold_type,
                        segment_length=self.mstl_period,
                        use_revin=self.use_revin,
                        window_size=self.window_size,
                        encode_dropout=0.3,
                        recon_dropout=0.2,
                    )
            elif self.use_multi_horizon:
                print("********************************")
                self.forecaster = DirectMultiHorizonHyperbolicForecaster(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    encode_dim=self.encode_dim,
                    hidden_dim=self.hidden_dim,
                    curvature=self.curvature,
                    manifold_type=self.manifold_type,
                    segment_length=self.mstl_period,
                    use_revin=self.use_revin,
                    encode_dropout=0.3,
                    recon_dropout=0.2,
                    window_size=self.window_size
                )

            else:

                self.forecaster = SegmentedHyperbolicForecaster(
                    lookback=self.seq_len,
                    pred_len=self.pred_len,
                    n_features=self.enc_in,
                    encode_dim=self.encode_dim,
                    hidden_dim=self.hidden_dim,
                    curvature=self.curvature,
                    manifold_type=self.manifold_type,
                    segment_length=self.mstl_period,
                    use_revin=self.use_revin,
                    encode_dropout=0.3,
                    recon_dropout=0.2,
                    window_size=self.window_size,
                    num_layers=2
                )

   
    
    def forward(self, batch_x=None, trend=None, seasonal_coarse=None, seasonal_fine=None, residual=None):
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
        if self.decomposer is not None and self.manifold_type != "Euclidean":
            components = self.decomposer(batch_x)
            trend = components['trend']
            seasonal_coarse = components['seasonal_coarse']
            seasonal_fine = components['seasonal_fine']
            residual = components['residual']

            forecasts = self.forecaster(trend, seasonal_coarse, seasonal_fine, residual)
            hierarchy_loss = forecasts["hierarchy_loss"]
        else:
            forecasts = self.forecaster(batch_x)
            hierarchy_loss = torch.tensor(0.0, device=x_hat.device)
        # Get individual hyperbolic representations
        x_hat = forecasts["predictions"]
        if self.manifold_type == "Euclidean":
            return x_hat, torch.tensor(0.0, device=x_hat.device), torch.tensor(0.0, device=x_hat.device)
        
        hyperbolic_loss = forecasts["consistency_loss"]
        return x_hat, hyperbolic_loss, hierarchy_loss
