import torch
import torch.nn as nn
import geoopt
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0

class MLPEmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=3, lookback=None, use_attention_pooling=False):
        super().__init__()
        self.lookback = lookback
        self.use_attention_pooling = use_attention_pooling
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.5)
            ) for _ in range(n_layer)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        if self.use_attention_pooling:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(self, x):
        """
        x: [B, seqlen, input_dim]
        returns: [B, output_dim]  (Euclidean latent, tangent vectors)
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        #Compresses a series to a vector.
        # STEP 5: Pool across time (attention or mean)
        if self.use_attention_pooling:
            attn_scores = self.attention(x)  # [32, 96, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)  # [32, 96, 1]
            x_pooled = (x * attn_weights).sum(dim=1)  # [32, 64]
        else:
            x_pooled = x.mean(dim=1)   # mean pooling would be good for point level forecasting
        return self.output_proj(x_pooled)

class ParallelEuclideanEmbed(nn.Module):
    def __init__(self, lookback, input_dim, embed_dim=32, hidden_dim=64,
                use_hierarchy=False, hierarchy_scales=[0.5,1.0,1.5,2.0], 
                n_layer=2, use_attention_pooling=False):
        super().__init__()
        # 5 parallel Mamba encoder blocks
        self.use_hierarchy = use_hierarchy
        if self.use_hierarchy:
            self.log_scales = nn.ParameterList([
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[0]))),  # trend
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[1]))),  # weekly
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[2]))),  # daily
                nn.Parameter(torch.log(torch.tensor(hierarchy_scales[3])))   # residual
            ])
        self.trend_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            n_layer=n_layer,
            use_attention_pooling=use_attention_pooling)
        self.daily_embed = MLPEmbed(
            input_dim=input_dim,
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            n_layer=n_layer,
            use_attention_pooling=use_attention_pooling)
        self.weekly_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            n_layer=n_layer,
            use_attention_pooling=use_attention_pooling)
        self.residual_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            n_layer=n_layer,
            use_attention_pooling=use_attention_pooling)

    def hierarchical_weighted_combine(self, e_trend, e_weekly, e_daily, e_residual):
        """
        Weighted combination in Euclidean space.
        Inverse weighting: trend (general) gets higher weight than residual (specific).
        
        Args:
            e_trend, e_weekly, e_daily, e_residual: [B, embed_dim]
        
        Returns:
            combined_e: [B, embed_dim] - weighted combination
        """
        # Get scales (learned during training)
        trend_scale = torch.exp(self.log_scales[0])      # e.g., 0.5
        weekly_scale = torch.exp(self.log_scales[1])     # e.g., 1.0
        daily_scale = torch.exp(self.log_scales[2])      # e.g., 1.5
        residual_scale = torch.exp(self.log_scales[3])   # e.g., 2.0
        
        # Inverse weights (trend = most important, residual = least)
        # Smaller scale → higher weight (trend is more general, gets more weight)
        trend_weight = 1.0 / trend_scale
        weekly_weight = 1.0 / weekly_scale
        daily_weight = 1.0 / daily_scale
        residual_weight = 1.0 / residual_scale
        
        # Normalize weights to sum to 1
        total_weight = trend_weight + weekly_weight + daily_weight + residual_weight
        
        # Weighted sum
        combined_e = (
            (trend_weight / total_weight) * e_trend +
            (weekly_weight / total_weight) * e_weekly +
            (daily_weight / total_weight) * e_daily +
            (residual_weight / total_weight) * e_residual
        )
        
        return combined_e

    def forward(self, trend, daily, weekly, residual):
        """
        Encode decomposed time series components to Euclidean space.
        
        Args:
            trend: [B, seq_len, input_dim] 
            daily: [B, seq_len, input_dim] 
            weekly: [B, seq_len, input_dim]
            residual: [B, seq_len, input_dim]
        
        Returns:
            dict with:
                - trend_e, seasonal_daily_e, seasonal_weekly_e, residual_e: [B, embed_dim]
                - combined_e: [B, embed_dim]
        """
        # Embed each branch to Euclidean latent vector
        e_trend = self.trend_embed(trend)
        e_daily = self.daily_embed(daily)
        e_weekly = self.weekly_embed(weekly)
        e_residual = self.residual_embed(residual)
        
        if self.use_hierarchy:
            # Hierarchical weighted combination
            # Trend gets highest weight (most general)
            # Residual gets lowest weight (most specific)
            combined_e = self.hierarchical_weighted_combine(e_trend, e_weekly, e_daily, e_residual)
        else:
            # Simple sum (no hierarchy, all components equally weighted)
            combined_e = e_trend + e_daily + e_weekly + e_residual

        return {
            "trend_e": e_trend,
            "seasonal_daily_e": e_daily,
            "seasonal_weekly_e": e_weekly,
            "residual_e": e_residual,
            "combined_e": combined_e
        }

