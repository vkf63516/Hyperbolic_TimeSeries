import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from embed.mamba_embed_lorentz import ParallelLorentzBlock
from embed.segment_mamba_embed_lorentz import SegmentParallelLorentzBlock
from Forecaster import HyperbolicSeqForecaster
from Segment_Forecaster import HyperbolicSegmentForecaster
from spec import safe_expmap0

class Model(nn.Module):
    """
    Hyperbolic Mamba Forecasting Model
    
    Architecture:
    1. Decompose time series into trend, seasonal_daily, seasonal_weekly, residual
    2. Encode each component to hyperbolic space via Mamba encoders
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
        self.use_hierarchy = configs.use_hierarchy
        self.hierarchy_scales = configs.hierarchy_scales
        # Model dimensions
        # Number of input features
        self.enc_in = getattr(configs, 'enc_in', 1)
        
        # Get segment length from configs or use default (24 for hourly data with daily period)
        self.seg_len = getattr(configs, 'seg_len', 24)
        # Keep mstl_period as alias for backward compatibility
        self.mstl_period = self.seg_len
        
        # Embedding: Maps decomposed components to hyperbolic space
        self.embedding = SegmentParallelLorentzBlock(
            lookback=self.seq_len,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            curvature=self.curvature
        )
        
        # Forecaster: Autoregressively predicts in hyperbolic space
        self.forecaster = HyperbolicSegmentForecaster(
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            seg_len=self.seg_len,
            manifold=self.embedding.manifold
        )
   
        
    # def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
    #     """
    #     Args:
    #         x_enc: [B, seq_len, enc_in] - input sequence
    #         x_mark_enc: optional time features for encoder
    #         x_dec: [B, label_len + pred_len, dec_in] - mvar input (for compatibility)
    #         x_mark_dec: optional time features for mvar
            
    #     Returns:
    #         predictions: [B, pred_len, output_dim]
    #     """
    #     B, T, C = x_enc.shape
        
    #     # For multivariate, process each feature separately or use the first feature
    #     # Here we assume decomposition expects univariate input
    #     # You should replace this with actual decomposed components from TimeBaseMSTL
        
    #     # Placeholder: Simple decomposition (replace with TimeBaseMSTL output)
    #     # In practice, you'd get trend, weekly, daily, resid from your decomposition module
    #     trend = x_enc.mean(dim=-1, keepdim=True)  # [B, T, 1] - simplified trend
    #     seasonal_weekly = torch.zeros(B, T, 1, device=x_enc.device)  # [B, T, 1]
    #     seasonal_daily = torch.zeros(B, T, 1, device=x_enc.device)   # [B, T, 1]
    #     resid = x_enc[..., :1] - trend  # [B, T, 1] - simplified residual
        
    #     # Encode components to hyperbolic space
    #     encoded = self.encoder(trend, seasonal_weekly, seasonal_daily, residual)
        
    #     # Get the combined hyperbolic representation
    #     z0 = encoded['combined_h']  # [B, embed_dim+1]
        
    #     # Forecast in hyperbolic space and reconstruct
    #     x_hat, z_pred = self.forecaster.forecast(
    #         pred_len=self.pred_len,
    #         z0=z0,
    #         teacher_forcing=False
    #     )
        
    #     # x_hat: [B, pred_len, output_dim]
    #     # Expand output to match input dimensions if needed
    #     if x_hat.shape[-1] != C:
    #         x_hat = x_hat.repeat(1, 1, C)
            
    #     return x_hat
    
    def forward_with_decomposition(self, trend, seasonal_weekly, seasonal_daily, residual):
        """
        Forward pass with explicit decomposed components.
        Use this when you have TimeBaseMSTL decomposition.
        
        Args:
            trend: [B, seq_len, 1]
            weekly: [B, seq_len, 1]
            daily: [B, seq_len, 1]
            resid: [B, seq_len, 1]
            
        Returns:
            predictions: [B, pred_len, output_dim]
        """
        # Encode components to hyperbolic space
        encoded = self.encoder(trend, seasonal_weekly, seasonal_daily, residual)
        
        # Get individual and combined hyperbolic representations
        trend_h = encoded['trend_h']
        seasonal_weekly_h = encoded['seasonal_weekly_h']
        seasonal_daily_h = encoded['seasonal_daily_h']
        residual_h = encoded['residual_h']
        
        # Forecast using combined representation
        x_hat, z_pred = self.forecaster.forecast(
            pred_len=self.pred_len,
            trend_z=trend_h,
            seasonal_weekly_z=seasonal_weekly_h,  
            seasonal_daily_z=seasonal_daily_h,
            residual_z=residual_h,
            teacher_forcing=False
        )
        
        return x_hat
        
        