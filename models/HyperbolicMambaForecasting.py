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
        
        # For this model, we assume the input has already been decomposed
        # or we can treat different channels as different components
        # For simplicity, we'll split channels into trend, weekly, daily, resid components
        # This is a simplified version - in practice, you'd use TimeBaseMSTL decomposition
        
        # Simple channel splitting (adjust based on your decomposition strategy)
        if C >= 4:
            trend = x[:, :, 0:1]
            weekly = x[:, :, 1:2]
            daily = x[:, :, 2:3]
            resid = x[:, :, 3:4]
        else:
            # If fewer channels, duplicate or use the available channels
            trend = x[:, :, 0:1]
            weekly = x[:, :, 0:1] if C < 2 else x[:, :, 1:2]
            daily = x[:, :, 0:1] if C < 3 else x[:, :, 2:3]
            resid = x[:, :, 0:1] if C < 4 else x[:, :, 3:4]
        
        # Encode to hyperbolic space
        enc_out = self.encoder(trend, weekly, daily, resid)
        
        # Get the combined hyperbolic representation
        z_current = enc_out["combined_h"]  # [B, embed_dim+1]
        
        # Autoregressive forecasting on the manifold
        predictions = []
        for t in range(self.pred_len):
            # Predict next step on manifold
            z_next, _ = self.decoder(z_current)
            
            # Reconstruct to Euclidean space
            x_pred = self.recon_head(z_next)  # [B, enc_in]
            predictions.append(x_pred.unsqueeze(1))
            
            # Update current state for next prediction
            z_current = z_next
        
        # Concatenate predictions
        output = torch.cat(predictions, dim=1)  # [B, pred_len, enc_in]
        
        return output
        