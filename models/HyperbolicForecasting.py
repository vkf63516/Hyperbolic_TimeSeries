# HyperbolicForecasting_v4.py

"""
Unified Hyperbolic Forecasting Model v4
Supports Euclidean, Poincare, and Lorentz manifolds
"""

import torch.nn as nn

# Import the unified forecasters from component files
from Forecasting.euclidean_components import (
    EuclideanForecaster,
    EuclideanMovingWindowForecaster
)
from Forecasting.poincare_components import (
    PoincareForecaster,
    PoincareMovingWindowForecaster
)
from Forecasting.lorentz_components import (
    LorentzForecaster,
    LorentzMovingWindowForecaster
)


class Model(nn.Module):
    """
    Hyperbolic Forecasting Model v4 - Unified Entry Point
    
    Supports:  
    - Euclidean forecasting
    - Poincare ball forecasting
    - Lorentz hyperboloid forecasting
    
    Modes:
    - Original: channel-dependent (use_moving_window=False)
    - Moving Window: channel-independent (use_moving_window=True)
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.manifold_type = configs.manifold_type
        self.use_moving_window = configs.use_moving_window
        
        # Common parameters
        common_params = {
            'lookback':  configs.seq_len,
            'pred_len': configs.pred_len,
            'n_features': configs.enc_in,
            'encode_dim': configs.encode_dim,
            'hidden_dim': configs.hidden_dim,
            'segment_length': configs.mstl_period,
            'dropout': 0.1,
            'use_truncated_bptt': True,
            'truncate_every': 4
        }
        
        # Select forecaster based on manifold type and mode
        if configs.manifold_type == "Euclidean":
            if configs.use_moving_window:
                self.forecaster = EuclideanMovingWindowForecaster(
                    **common_params,
                    window_size=getattr(configs, 'window_size', 5)
                )
            else:
                self.forecaster = EuclideanForecaster(**common_params)
                
        elif configs.manifold_type == "Poincare": 
            common_params['curvature'] = configs.curvature
            if configs.use_moving_window:
                self.forecaster = PoincareMovingWindowForecaster(
                    **common_params,
                    window_size=getattr(configs, 'window_size', 5)
                )
            else:
                self.forecaster = PoincareForecaster(**common_params)
                
        elif configs.manifold_type == "Lorentzian":
            common_params['curvature'] = configs.curvature
            if configs.use_moving_window:
                self.forecaster = LorentzMovingWindowForecaster(
                    **common_params,
                    window_size=getattr(configs, 'window_size', 5)
                )
            else:
                self.forecaster = LorentzForecaster(**common_params)
        else:
            raise ValueError(f"Unknown manifold type: {configs.manifold_type}")
        
        mode_str = "Moving Window (Channel-Independent)" if configs.use_moving_window else "Original (Channel-Dependent)"
        print(f"\n{'='*70}")
        print(f"Initialized {configs.manifold_type} Forecaster v4")
        print(f"Mode: {mode_str}")
        print(f"{'='*70}\n")
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Forward pass - delegates to selected forecaster
        
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len, n_features]
        
        Returns:
            x_hat: [B, pred_len, n_features] - combined predictions
            x_hyp: dict - hyperbolic states (empty for Euclidean)
        """
        forecasts = self.forecaster(trend, seasonal_coarse, seasonal_fine, residual)
        x_hat = forecasts["predictions"]
        x_hyp = forecasts.get("hyperbolic_states", {})
        return x_hat, x_hyp