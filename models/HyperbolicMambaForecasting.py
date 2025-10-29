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
        
        # Whether to use TimeBaseMSTL decomposition (handled externally)
        self.use_decomposition = getattr(configs, 'use_decomposition', False)
        
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
    
    def forward(self, x, decomposed_components=None):
        """
        Forward pass for HyperbolicMambaForecasting
        
        Args:
            x: [B, seq_len, enc_in] - Input time series (used if decomposed_components is None)
            decomposed_components: Optional dict with keys 'trend', 'daily', 'weekly', 'resid'
                                   Each should be [B, seq_len, 1]
            
        Returns:
            predictions: [B, pred_len, enc_in] - Forecasted time series
        """
        B = x.shape[0]
        
        # Use pre-decomposed components if provided (from TimeBaseMSTL preprocessing)
        if decomposed_components is not None:
            trend = decomposed_components['trend']      # [B, T, 1]
            daily = decomposed_components['daily']      # [B, T, 1]
            weekly = decomposed_components['weekly']    # [B, T, 1]
            resid = decomposed_components['resid']      # [B, T, 1]
        else:
            # Fallback: extract first channel and use as-is for all components
            # This allows the model to work with standard data loaders
            # In production, you should use TimeBaseMSTL preprocessing
            x_first = x[:, :, 0:1]  # [B, T, 1]
            trend = x_first
            daily = torch.zeros_like(x_first)
            weekly = torch.zeros_like(x_first)
            resid = torch.zeros_like(x_first)
        
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
        