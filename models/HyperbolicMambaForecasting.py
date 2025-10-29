import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))

from embed.mamba_embed_lorentz import ParallelLorentzBlock
from Forecaster import HyperbolicSeqForecaster
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
        
        