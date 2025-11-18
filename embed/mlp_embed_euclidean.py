import torch
import torch.nn as nn
import geoopt
from pathlib import Path
import sys 
sys.path.append(str(Path(__file__).resolve().parents[0]))
from spec import safe_expmap, safe_expmap0

class MLPEmbed(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=3, lookback=None, embed_dropout=0.1, use_attention_pooling=False):
        super().__init__()
        self.lookback = lookback
        self.use_attention_pooling = use_attention_pooling
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(embed_dropout)
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
                n_layer=2, embed_dropout=0.1, use_attention_pooling=False):
        super().__init__()
        # 5 parallel Mamba encoder blocks
        
        self.trend_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            n_layer=n_layer,
            embed_dropout=embed_dropout,
            use_attention_pooling=use_attention_pooling)
        self.fine_embed = MLPEmbed(
            input_dim=input_dim,
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            n_layer=n_layer,
            embed_dropout=embed_dropout,
            use_attention_pooling=use_attention_pooling)
        self.coarse_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            n_layer=n_layer,
            embed_dropout=embed_dropout,
            use_attention_pooling=use_attention_pooling)
        self.residual_embed = MLPEmbed(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            n_layer=n_layer,
            use_attention_pooling=use_attention_pooling)

    
    def forward(self, trend, fine, coarse, residual):
        """
        Encode decomposed time series components to Euclidean space.
        
        Args:
            trend: [B, seq_len, input_dim] 
            fine: [B, seq_len, input_dim] 
            coarse: [B, seq_len, input_dim]
            residual: [B, seq_len, input_dim]
        
        Returns:
            dict with:
                - trend_e, seasonal_fine_e, seasonal_coarse_e, residual_e: [B, embed_dim]
                - combined_e: [B, embed_dim]
        """
        # Embed each branch to Euclidean latent vector
        e_trend = self.trend_embed(trend)
        e_fine = self.fine_embed(fine)
        e_coarse = self.coarse_embed(coarse)
        e_residual = self.residual_embed(residual)
        
        # Simple sum (no hierarchy, all components equally weighted)
        combined_e = e_trend + e_fine + e_coarse + e_residual

        return {
            "trend_e": e_trend,
            "seasonal_fine_e": e_fine,
            "seasonal_coarse_e": e_coarse,
            "residual_e": e_residual,
            "combined_e": combined_e
        }

