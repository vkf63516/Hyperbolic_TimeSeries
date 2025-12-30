import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

class LearnableMultivariateDecomposition(nn.Module):
    """
    End-to-end learnable decomposition optimized for forecasting. 
    """
    def __init__(self, n_features, kernel_size, detected_periods=None):
        super().__init__()
        
        self.n_features = n_features
        
        # Depthwise convolutions (each feature gets its own filter)
        # Trend: long-term smoothing
        self.trend_filter = nn.Conv1d(
            n_features, n_features, kernel_size, 
            padding="same", groups=n_features
        )
        
        # Seasonal coarse: based on detected coarse period
        coarse_kernel = detected_periods[1] if detected_periods else kernel_size
        self.seasonal_coarse_filter = nn.Conv1d(
            n_features, n_features, coarse_kernel,
            padding="same", groups=n_features
        )
        
        # Seasonal fine: based on detected fine period
        fine_kernel = detected_periods[0] if detected_periods else kernel_size//2
        self.seasonal_fine_filter = nn.Conv1d(
            n_features, n_features, fine_kernel,
            padding="same", groups=n_features
        )
                
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, n_features]
        Returns:
            dict with trend, seasonal_coarse, seasonal_fine, residual
        """
        # Transpose to [B, n_features, seq_len] for Conv1d
        x = x.transpose(1, 2)
        
        # Decompose hierarchically (same as your orthogonalMSTL logic)
        trend = self.trend_filter(x)
        seasonal_coarse = self.seasonal_coarse_filter(x - trend)
        seasonal_fine = self.seasonal_fine_filter(x - trend - seasonal_coarse)
        residual = x - trend - seasonal_coarse - seasonal_fine
        # Transpose back to [B, seq_len, n_features]
        return {
            'trend': trend. transpose(1, 2),
            'seasonal_coarse': seasonal_coarse.transpose(1, 2),
            'seasonal_fine': seasonal_fine.transpose(1, 2),
            'residual': residual.transpose(1, 2)
        }
