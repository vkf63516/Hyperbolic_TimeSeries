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
        self.mstl_period = configs.mstl_period
        self.use_segments = configs.use_segments
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
            self.embedding = ParallelLorentzBlock(
                lookback=self.seq_len,
                input_dim=self.enc_in,
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                curvature=self.curvature,
                use_hierarchy=self.use_hierarchy
            )
            # Forecaster: Autoregressively predicts in hyperbolic space
            self.forecaster = HyperbolicSeqForecaster(
                embed_dim=self.embed_dim,
                hidden_dim=self.hidden_dim,
                output_dim=1,
                manifold=self.embedding.manifold
            )
   
    
    def forward(self, trend, seasonal_weekly, seasonal_daily, 
                                   residual, use_hierarchy=False, 
                                   teacher_forcing=False, z_true_seq=None):
        """
        Forward pass with explicit decomposed components.
        Use this when you have TimeBaseMSTL decomposition.
        
        Args:
            trend: [B, seq_len, C]
            weekly: [B, seq_len, C]
            daily: [B, seq_len, C]
            resid: [B, seq_len, C]
            
        Returns:
            predictions: [B, pred_len, output_dim]
        """
        # Encode components to hyperbolic space
        if self.use_segments:
            lookback_segment = self.seq_len // self.mstl_period
            embedded = self.embedding(trend, seasonal_weekly, seasonal_daily, residual)
            trend_h = embedded['trend_h']
            seasonal_weekly_h = embedded['seasonal_weekly_h']
            seasonal_daily_h = embedded['seasonal_daily_h']
            residual_h = embedded['residual_h']
            z0 = embedded["combined_h"] 
            x_hat, z_pred = self.forecaster(
                pred_len=self.pred_len,
                trend_z=trend_h,
                seasonal_weekly_z=seasonal_weekly_h,  
                seasonal_daily_z=seasonal_daily_h,
                residual_z=residual_h,
                z0=z0,
                teacher_forcing=teacher_forcing,
                z_true_seq=None
            )
        else:
            embedded = self.embedding(trend, seasonal_weekly,
                                 seasonal_daily, residual)
        
        # Get individual and combined hyperbolic representations
            trend_h = embedded['trend_h']
            seasonal_weekly_h = embedded['seasonal_weekly_h']
            seasonal_daily_h = embedded['seasonal_daily_h']
            residual_h = embedded['residual_h']
            z0 = embedded["combined_h"] 
            # Forecast using combined representation
            x_hat, z_pred = self.forecaster(
                pred_len=self.pred_len,
                trend_z=trend_h,
                seasonal_weekly_z=seasonal_weekly_h,  
                seasonal_daily_z=seasonal_daily_h,
                residual_z=residual_h,
                z0=z0,
                teacher_forcing=teacher_forcing,
                z_true_seq=None
            )
        
        return x_hat
