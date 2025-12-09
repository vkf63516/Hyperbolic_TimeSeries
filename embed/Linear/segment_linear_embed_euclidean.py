import torch
import torch.nn as nn
import torch.nn.functional as f
import geoopt


class SegmentLinearEmbed(nn.Module):
    def __init__(self, input_dim, output_dim, segment_length, dropout=0.1,
                 lookback=None, use_segment_norm=True, share_feature_weights=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.segment_length = segment_length
        self.use_segment_norm = use_segment_norm
        self.share_feature_weights = share_feature_weights
        
        # Calculate segments
        if lookback is not None:
            self.num_segments = lookback // segment_length
            self.pad_len = 0
            if lookback % segment_length != 0:
                self.pad_len = (self.num_segments + 1) * segment_length - lookback
                self.num_segments += 1
        
        self.total_len = self.num_segments * segment_length
        
        if share_feature_weights:
            self.shared_linear = nn.Linear(self.total_len, output_dim)
            self.shared_linear.weight = nn.Parameter(
                (1 / self.total_len) * torch.ones([output_dim, self.total_len])
            )
            print(f"SegmentLinearEmbed (SHARED): {input_dim} features → {self.total_len * output_dim} params")
        else:
            self.feature_linears = nn.ModuleList([
                nn.Linear(self.total_len, output_dim) for _ in range(input_dim)
            ])
            # for linear in self.feature_linears:
            #     linear.weight = nn.Parameter(
            #         (1 / self.total_len) * torch.ones([output_dim, self.total_len])
            #     )
            print(f"SegmentLinearEmbed (PER-FEATURE): {input_dim} features → {input_dim * self.total_len * output_dim} params")
        self.dropout = nn.Dropout(dropout)
    
    def _normalize_segments(self, x):
        """Period normalization (from TimeBase)"""
        if self.use_segment_norm:
            # Per-period normalization (TimeBase style)
            period_mean = x.mean(dim=2, keepdim=True)
            x = x - period_mean
            return x, period_mean
        else:
            B, N_seg, seg_len, C = x.shape
            x_flat = x.reshape(B, -1, C)
            mean = x_flat.mean(dim=1, keepdim=True)
            x = x_flat - mean
            return x.reshape(B, N_seg, seg_len, C), mean
    def forward(self, x):
        """
        Forward with optional orthogonal loss.
        
        Args:
            x: [B, seq_len, C]
        
        Returns:
            output: [B, output_dim]
        """
        B, seq_len, C = x.shape
        
        # Padding (TimeBase style)
        if self.pad_len > 0:
            pad_start = max(0, seq_len - self.pad_len)
            pad_data = x[:, pad_start:pad_start + self.pad_len, :]
            x = torch.cat([x, pad_data], dim=1)
        
        # Segment
        x = x.reshape(B, self.num_segments, self.segment_length, C)
        
        # Normalize (TimeBase style)
        x, norm_stats = self._normalize_segments(x)
        
        # Flatten time: [B, total_len, C]
        x = x.reshape(B, self.total_len, C)
        
        # Transpose to [B, C, total_len] (TimeBase/DLinear style)
        x = x.permute(0, 2, 1)
        
        if self.share_feature_weights:
            x_flat = x.reshape(-1, self.total_len)
            output_flat = self.shared_linear(x_flat)
            output = output_flat.reshape(B, C, self.output_dim)
        # Instead of looping
        else:
            # Stack all linear weights: [C, output_dim, total_len]
            weights = torch.stack([self.feature_linears[i].weight for i in range(self.input_dim)], dim=0)
            biases = torch.stack([self.feature_linears[i].bias for i in range(self.input_dim)], dim=0)
            
            # Batch matrix multiply: [B, C, total_len] @ [C, total_len, output_dim] -> [B, C, output_dim]
            output = torch.einsum('bci,cio->bco', x, weights.transpose(1, 2)) + biases.unsqueeze(0)
        
        output = output.mean(dim=1) #[B, embed_dim]
        return output

class SegmentParallelEuclidean(nn.Module):
    def __init__(self, lookback, input_dim, embed_dim=32, share_feature_weights=False,
                segment_length=24, embed_dropout=0.1, use_segment_norm=False):
        super().__init__()
        # 5 parallel Mamba encoder blocks
        
        self.trend_embed = SegmentLinearEmbed(
            input_dim=input_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            segment_length=segment_length,
            dropout=embed_dropout,
            use_segment_norm=use_segment_norm,
            share_feature_weights=share_feature_weights
        )
        self.fine_embed = SegmentLinearEmbed(
            input_dim=input_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            segment_length=segment_length,
            dropout=embed_dropout,
            use_segment_norm=use_segment_norm,
            share_feature_weights=share_feature_weights
        )
        self.coarse_embed = SegmentLinearEmbed(
            input_dim=input_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            segment_length=segment_length,
            dropout=embed_dropout,
            use_segment_norm=use_segment_norm,
            share_feature_weights=share_feature_weights
        )
        self.residual_embed = SegmentLinearEmbed(
            input_dim=input_dim, 
            output_dim=embed_dim, 
            lookback=lookback, 
            segment_length=segment_length,
            dropout=embed_dropout,
            use_segment_norm=use_segment_norm,
            share_feature_weights=share_feature_weights
        )

    
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
