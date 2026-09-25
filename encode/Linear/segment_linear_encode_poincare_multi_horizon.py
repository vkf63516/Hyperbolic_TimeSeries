import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from spec import safe_expmap0
from DynamicsMvar.poincare_disk import poincareball_factory


class SegmentLinearencodeMultiHorizonCD(nn.Module):
    """
    Channel-Dependent version of SegmentLinearencodeMultiHorizon.

    Instead of processing one feature at a time (input [Bf, seq_len] with
    Bf = B*C channels collapsed into the batch axis), this version consumes
    ALL channels jointly per segment and pools across the channel dimension,
    exactly analogous to how SegmentLinearencodeMW pools across channels for
    the single-horizon channel-dependent encoder.

    Produces ONE encoding per segment (segment structure preserved), fused
    across channels.

    Input:  [B, seq_len, C]
    Output: [B, num_segments, encode_dim]  # One encoding per segment, channel-fused
    """

    def __init__(self, lookback, encode_dim, num_channels, segment_length=24,
                 dropout=0.1, share_feature_weights=False):
        super().__init__()

        self.encode_dim = encode_dim
        self.segment_length = segment_length
        self.lookback = lookback
        self.num_channels = num_channels
        self.share_feature_weights = share_feature_weights

        self.num_segments = lookback // segment_length
        self.pad_seq_len = 0
        if self.lookback > self.num_segments * self.segment_length:
            self.pad_seq_len = (self.num_segments + 1) * self.segment_length - self.lookback
            self.num_segments += 1

        if share_feature_weights:
            # One shared linear applied to every channel independently, then
            # pooled across channels (mean).
            self.shared_linear = nn.Linear(self.segment_length, encode_dim)
            print(f"SegmentLinearencodeMultiHorizonCD (SHARED): "
                  f"{num_channels} channels -> {self.segment_length * encode_dim} params")
        else:
            # Per-channel linear weights, batched via einsum (same trick as
            # SegmentLinearencodeMW's per-feature branch).
            self.feature_linears = nn.ModuleList([
                nn.Linear(self.segment_length, encode_dim) for _ in range(num_channels)
            ])
            print(f"SegmentLinearencodeMultiHorizonCD (PER-CHANNEL): "
                  f"{num_channels} channels -> {num_channels * self.segment_length * encode_dim} params")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, C] - full historical sequence, all channels

        Returns:
            z: [B, num_segments, encode_dim] - one channel-fused encoding per segment
        """
        B, seq_len, C = x.shape

        if self.pad_seq_len > 0:
            pad = torch.zeros(B, self.pad_seq_len, C, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        # Reshape into segments, keep channels: [B, num_segments, seg_len, C]
        x_seg = x.view(B, self.num_segments, self.segment_length, C)

        # Move channels next to batch for per-channel projection:
        # [B, num_segments, seg_len, C] -> [B, num_segments, C, seg_len]
        x_seg = x_seg.permute(0, 1, 3, 2)

        if self.share_feature_weights:
            x_flat = x_seg.reshape(-1, self.segment_length)          # [B*num_segments*C, seg_len]
            out_flat = self.shared_linear(x_flat)                    # [B*num_segments*C, encode_dim]
            out = out_flat.view(B, self.num_segments, C, self.encode_dim)
        else:
            # Stack per-channel weights: [C, encode_dim, seg_len]
            weights = torch.stack([lin.weight for lin in self.feature_linears], dim=0)
            biases = torch.stack([lin.bias for lin in self.feature_linears], dim=0)  # [C, encode_dim]

            # x_seg: [B, num_segments, C, seg_len]
            # weights.transpose(1,2): [C, seg_len, encode_dim]
            out = torch.einsum('bnci,cio->bnco', x_seg, weights.transpose(1, 2)) \
                  + biases.view(1, 1, C, self.encode_dim)
            # out: [B, num_segments, C, encode_dim]

        # Fuse channels: pool across C (channel-dependent mixing, like SegmentLinearencodeMW)
        seg_encode = out.mean(dim=2)  # [B, num_segments, encode_dim]

        seg_encode = self.dropout(seg_encode)

        return seg_encode


class SegmentedParallelPoincareMultiHorizonCD(nn.Module):
    """
    Channel-Dependent version of SegmentedParallelPoincareMultiHorizon.

    Encodes trend/seasonal_coarse/seasonal_fine/residual JOINTLY across all
    channels (features are fused inside each segment's encoding rather than
    processed independently and stacked), while still outputting
    [B, num_segments, encode_dim] so downstream moving-window/multi-horizon
    dynamics logic is unchanged.
    """

    def __init__(self, lookback, num_channels, encode_dim, curvature=1.0, segment_length=24,
                 encode_dropout=0.1, share_feature_weights=False):
        super().__init__()

        self.encode_dim = encode_dim
        self.num_channels = num_channels

        # Segment-aware, channel-dependent encoders (output per-segment,
        # channel-fused encodings)
        self.trend_encode = SegmentLinearencodeMultiHorizonCD(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels,
            segment_length=segment_length, dropout=encode_dropout,
            share_feature_weights=share_feature_weights
        )
        self.seasonal_coarse_encode = SegmentLinearencodeMultiHorizonCD(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels,
            segment_length=segment_length, dropout=encode_dropout,
            share_feature_weights=share_feature_weights
        )
        self.seasonal_fine_encode = SegmentLinearencodeMultiHorizonCD(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels,
            segment_length=segment_length, dropout=encode_dropout,
            share_feature_weights=share_feature_weights
        )
        self.residual_encode = SegmentLinearencodeMultiHorizonCD(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels,
            segment_length=segment_length, dropout=encode_dropout,
            share_feature_weights=share_feature_weights
        )

        # Poincaré ball manifold
        self.manifold = geoopt.manifolds.PoincareBall(c=curvature)

        # Scaling parameter
        self.effective_scale = nn.Parameter(torch.tensor(1.0))

        self.mobius_weights = nn.Parameter(torch.ones(4) * 0.25)

    def map_segments_to_hyperbolic(self, segment_encodes):
        """
        Map each segment encoding to hyperbolic space independently.

        Args:
            segment_encodes: [B, num_segments, encode_dim]

        Returns:
            hyperbolic_encodes: [B, num_segments, encode_dim]
        """
        B, N, D = segment_encodes.shape
        encodes_flat = segment_encodes.reshape(B * N, D)

        # Scale
        effective_scale = torch.tanh(self.effective_scale)
        scaled_encodes = encodes_flat * effective_scale

        # Map to hyperbolic space
        hyperbolic_flat = self.manifold.expmap0(scaled_encodes)  # [B*N, encode_dim]

        # Project to manifold
        hyperbolic_flat = self.manifold.projx(hyperbolic_flat)

        # Reshape back to sequence
        hyperbolic_encodes = hyperbolic_flat.view(B, N, D)  # [B, num_segments, encode_dim]

        return hyperbolic_encodes

    def mobius_fusion_segments(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """
        Fuse trend/coarse/fine/residual components for each segment
        independently using Möbius addition.

        Args:
            z_trend_h, z_coarse_h, z_fine_h, z_residual_h: [B, num_segments, encode_dim]

        Returns:
            combined_h: [B, num_segments, encode_dim]
        """
        B, N, D = z_trend_h.shape

        # Flatten for batch Möbius operations
        z_trend_flat = z_trend_h.reshape(B * N, D)
        z_coarse_flat = z_coarse_h.reshape(B * N, D)
        z_fine_flat = z_fine_h.reshape(B * N, D)
        z_residual_flat = z_residual_h.reshape(B * N, D)

        # Normalize weights
        weights = torch.softmax(self.mobius_weights, dim=0)

        # Sequential Möbius addition with weights
        combined_flat = self.manifold.mobius_scalar_mul(weights[0], z_trend_flat)

        scaled_coarse = self.manifold.mobius_scalar_mul(weights[1], z_coarse_flat)
        combined_flat = self.manifold.mobius_add(combined_flat, scaled_coarse)

        scaled_fine = self.manifold.mobius_scalar_mul(weights[2], z_fine_flat)
        combined_flat = self.manifold.mobius_add(combined_flat, scaled_fine)

        scaled_residual = self.manifold.mobius_scalar_mul(weights[3], z_residual_flat)
        combined_flat = self.manifold.mobius_add(combined_flat, scaled_residual)

        # Ensure numerical stability
        combined_flat = self.manifold.projx(combined_flat)

        # Reshape back to sequence
        combined_h = combined_flat.view(B, N, D)

        return combined_h

    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Encode with segment structure preserved, channels fused jointly.

        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len, C]

        Returns:
            dict with hyperbolic encodings [B, num_segments, encode_dim] for each component
        """
        # Encode to per-segment, channel-fused encodings:
        z_trend_segments = self.trend_encode(trend)
        z_coarse_segments = self.seasonal_coarse_encode(seasonal_coarse)
        z_fine_segments = self.seasonal_fine_encode(seasonal_fine)
        z_residual_segments = self.residual_encode(residual)

        # Map each segment to hyperbolic space: [B, num_segments, encode_dim]
        z_trend_h = self.map_segments_to_hyperbolic(z_trend_segments)
        z_coarse_h = self.map_segments_to_hyperbolic(z_coarse_segments)
        z_fine_h = self.map_segments_to_hyperbolic(z_fine_segments)
        z_residual_h = self.map_segments_to_hyperbolic(z_residual_segments)

        # Möbius fusion for each segment: [B, num_segments, encode_dim]
        combined_h = self.mobius_fusion_segments(z_trend_h, z_coarse_h, z_fine_h, z_residual_h)

        return {
            "trend_h": z_trend_h,
            "seasonal_coarse_h": z_coarse_h,
            "seasonal_fine_h": z_fine_h,
            "residual_h": z_residual_h,
            "combined_h": combined_h
        }