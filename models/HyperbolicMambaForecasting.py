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
        
    def forward(self, x):
        """
        Forward pass for HyperbolicMambaForecasting
        
        Args:
            x: [B, seq_len, enc_in] - Input time series
            
        Returns:
            predictions: [B, pred_len, enc_in] - Forecasted time series
        """
        B, T, C = x.shape
        
        # Simple decomposition approach: use moving average for trend,
        # and residuals for the other components
        # This is a lightweight alternative to full MSTL decomposition
        
        # Trend: simple moving average over a window
        kernel_size = min(7, T // 4)  # adaptive kernel size
        if kernel_size > 1:
            # Apply 1D average pooling for trend extraction
            x_padded = F.pad(x.transpose(1, 2), (kernel_size//2, kernel_size//2), mode='replicate')
            trend = F.avg_pool1d(x_padded, kernel_size=kernel_size, stride=1).transpose(1, 2)
        else:
            trend = x
        
        # Residual after trend removal
        residual = x - trend
        
        # For weekly and daily components, we use subsets of the residual
        # In a more sophisticated version, you'd use FFT or MSTL
        # Here we split the residual to provide diverse inputs to the encoder
        weekly = residual
        daily = residual
        
        # Take only the first channel for each component to match encoder expectations
        trend = trend[:, :, 0:1]      # [B, T, 1]
        weekly = weekly[:, :, 0:1]    # [B, T, 1]
        daily = daily[:, :, 0:1]      # [B, T, 1]
        resid = residual[:, :, 0:1]   # [B, T, 1]
        
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
        