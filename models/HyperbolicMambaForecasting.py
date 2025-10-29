import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from embed.mamba_embed_lorentz import ParallelLorentzBlock
from decoder.mamba_decoder_lorentz import HyperbolicMambaDecoder
from Lifting.reconstructor import HyperbolicReconstructionHead
import numpy as np
try:
    from statsmodels.tsa.seasonal import STL
except ImportError:
    STL = None

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # Extract configuration
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        
        # Model hyperparameters
        self.embed_dim = getattr(configs, 'embed_dim', 32)
        self.hidden_dim = getattr(configs, 'hidden_dim', 64)
        self.curvature = getattr(configs, 'curvature', 1.0)
        self.lookback = getattr(configs, 'lookback', self.seq_len)
        
        # MSTL decomposition period (auto-detected or default)
        self.period_len = getattr(configs, 'period_len', 24)
        
        # Initialize manifold
        self.manifold = geoopt.manifolds.Lorentz(k=self.curvature)
        
        # Encoder: Takes decomposed time series components and encodes to hyperbolic space
        self.encoder = ParallelLorentzBlock(
            lookback=self.lookback,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            curvature=self.curvature
        )
        
        # Decoder: Performs autoregressive forecasting on the manifold
        self.decoder = HyperbolicMambaDecoder(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            manifold=self.manifold,
            lookback=self.lookback
        )
        
        # Reconstruction head: Maps from hyperbolic space back to Euclidean predictions
        self.recon_head = HyperbolicReconstructionHead(
            embed_dim=self.embed_dim,
            output_dim=self.enc_in,
            manifold=self.manifold
        )
    
    def _decompose_series(self, x):
        """
        Decompose time series using STL decomposition with multiple seasonal periods.
        This properly extracts trend, daily seasonal, weekly seasonal, and residual components.
        
        Args:
            x: [B, T, C] - Input time series
            
        Returns:
            trend, daily, weekly, resid: Each [B, T, 1] using first channel
        """
        B, T, C = x.shape
        device = x.device
        
        # Use the first channel for decomposition
        series = x[:, :, 0].detach().cpu().numpy()  # [B, T]
        
        # Initialize components
        trend_all = np.zeros_like(series)
        seasonal_daily_all = np.zeros_like(series)
        seasonal_weekly_all = np.zeros_like(series)
        resid_all = np.zeros_like(series)
        
        # Decompose each sample in the batch
        for b in range(B):
            ts = series[b]  # [T]
            
            if STL is not None and T > 2 * self.period_len:
                # Use STL decomposition for the primary (daily) period
                try:
                    stl = STL(ts, seasonal=self.period_len, trend=None)
                    result = stl.fit()
                    trend_all[b] = result.trend
                    seasonal_daily_all[b] = result.seasonal
                    resid_all[b] = result.resid
                    
                    # For weekly component, decompose the residual if we have enough data
                    weekly_period = self.period_len * 7  # weekly period
                    if T > 2 * weekly_period:
                        try:
                            stl_weekly = STL(resid_all[b], seasonal=weekly_period, trend=None)
                            result_weekly = stl_weekly.fit()
                            seasonal_weekly_all[b] = result_weekly.seasonal
                            resid_all[b] = result_weekly.resid
                        except:
                            # If weekly decomposition fails, keep original residual
                            seasonal_weekly_all[b] = 0
                    else:
                        seasonal_weekly_all[b] = 0
                        
                except Exception as e:
                    # Fallback to simple moving average if STL fails
                    kernel_size = min(self.period_len, T // 4)
                    if kernel_size > 1:
                        trend_all[b] = np.convolve(ts, np.ones(kernel_size)/kernel_size, mode='same')
                    else:
                        trend_all[b] = ts
                    resid_all[b] = ts - trend_all[b]
                    seasonal_daily_all[b] = 0
                    seasonal_weekly_all[b] = 0
            else:
                # Fallback: simple moving average for trend
                kernel_size = min(7, T // 4)
                if kernel_size > 1:
                    trend_all[b] = np.convolve(ts, np.ones(kernel_size)/kernel_size, mode='same')
                else:
                    trend_all[b] = ts
                resid_all[b] = ts - trend_all[b]
                seasonal_daily_all[b] = 0
                seasonal_weekly_all[b] = 0
        
        # Convert back to torch tensors and add channel dimension
        trend = torch.from_numpy(trend_all).float().to(device).unsqueeze(-1)  # [B, T, 1]
        daily = torch.from_numpy(seasonal_daily_all).float().to(device).unsqueeze(-1)  # [B, T, 1]
        weekly = torch.from_numpy(seasonal_weekly_all).float().to(device).unsqueeze(-1)  # [B, T, 1]
        resid = torch.from_numpy(resid_all).float().to(device).unsqueeze(-1)  # [B, T, 1]
        
        return trend, daily, weekly, resid
    
    def forward(self, x):
        """
        Forward pass for HyperbolicMambaForecasting
        
        Args:
            x: [B, seq_len, enc_in] - Input time series
            
        Returns:
            predictions: [B, pred_len, enc_in] - Forecasted time series
        """
        B, T, C = x.shape
        
        # Decompose time series using STL/MSTL to get proper seasonal components
        # This uses the actual seasonal decomposition instead of just setting
        # weekly and daily to residual
        trend, daily, weekly, resid = self._decompose_series(x)
        
        # Encode to hyperbolic space
        enc_out = self.encoder(trend, weekly, daily, resid)
        
        # Get the combined hyperbolic representation
        z_current = enc_out["combined_h"]  # [B, embed_dim+1]
        
        # Autoregressive forecasting on the manifold with periodic detachment
        predictions = []
        K = 6  # Detachment period (from Forecaster.py)
        for t in range(self.pred_len):
            # Predict next step on manifold
            z_next, _ = self.decoder(z_current)
            
            # Reconstruct to Euclidean space
            x_pred = self.recon_head(z_next)  # [B, enc_in]
            predictions.append(x_pred.unsqueeze(1))
            
            # Update current state for next prediction
            z_current = z_next
            
            # Periodic detachment to prevent gradient explosion
            if (t + 1) % K == 0:
                z_current = z_current.detach()
        
        # Concatenate predictions
        output = torch.cat(predictions, dim=1)  # [B, pred_len, enc_in]
        
        return output
        