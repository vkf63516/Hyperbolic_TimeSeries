import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt

from DynamicsMvar.poincare_disk import poincareball_factory, PoincareBall


# ============================================================
# PoincareLinearToManifold
# ============================================================
class PoincareLinearToManifold(nn.Module):
    """
    Like DynamicsMvar.Poincare_Residual_Dynamics.PoincareLinear, EXCEPT it
    stops before the final logmap0 call. PoincareLinear is tangent-in /
    tangent-out (useful for velocity corrections); this variant is
    Euclidean-in / manifold-point-out -- i.e. it IS the encoder that lifts
    raw features onto the Poincare ball.

        x (Euclidean)
          -> ball.expmap0(x)            tangent-at-origin -> manifold point
          -> ball.fully_connected(...)  Mobius-aware linear transform
          -> ball.projx(y)              numerically safe manifold point
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ball: PoincareBall,
        bias: bool = True,
        id_init: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball
        self.has_bias = bias
        self.id_init = id_init

        self.z = nn.Parameter(torch.empty(in_features, out_features))
        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.id_init:
            with torch.no_grad():
                self.z.copy_(0.5 * torch.eye(self.in_features, self.out_features))
        else:
            nn.init.normal_(
                self.z, mean=0, std=(2 * self.in_features * self.out_features) ** -0.5
            )
        if self.has_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ball.expmap0(x, dim=-1)
        y = self.ball.fully_connected(x=x, z=self.z, bias=self.bias)
        return self.ball.projx(y, dim=-1)  # manifold point, NOT logmap0'd back


# ============================================================
# Segment encoder -- drop-in replacement for SegmentLinearencodeMultiHorizon
# Matches the ORIGINAL contract exactly: 2D in, 3D out. Each of
# trend / seasonal_coarse / seasonal_fine / residual gets its own instance
# of this module (that's how "channels" -- really decomposition components
# -- are handled in this architecture, not via a C axis inside the module).
# ============================================================
class SegmentPoincareEncodeMultiHorizon(nn.Module):
    """
    Encodes each segment into ONE point on a Poincare ball, using
    PoincareLinearToManifold instead of nn.Linear + a separate manual
    expmap0 step done later by the parent module.

    Input:  x [Bf, seq_len]
    Output: z [Bf, num_segments, encode_dim]   -- points on `ball`
    """

    def __init__(
        self,
        lookback: int,
        encode_dim: int,
        num_channels: int,        # stored for signature parity; unused here
        ball: PoincareBall,
        segment_length: int = 24,
        dropout: float = 0.1,
        individual: bool = False,  # stored for signature parity; unused here
    ):
        super().__init__()

        self.encode_dim = encode_dim
        self.segment_length = segment_length
        self.lookback = lookback
        self.num_channels = num_channels
        self.individual = individual
        self.ball = ball

        self.num_segments = lookback // segment_length
        self.pad_seq_len = 0
        if self.lookback > self.num_segments * self.segment_length:
            self.pad_seq_len = (self.num_segments + 1) * self.segment_length - self.lookback
            self.num_segments += 1

        self.segment_to_manifold = PoincareLinearToManifold(
            in_features=segment_length,
            out_features=encode_dim,
            ball=ball,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Bf, seq_len]
        Returns:
            z: [Bf, num_segments, encode_dim]  -- points on `ball`
        """
        Bf = x.shape[0]

        if self.pad_seq_len > 0:
            pad = torch.zeros(B, self.pad_seq_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        x_seg = x.view(Bf, self.num_segments, self.segment_length)  # [B, num_segments, segment_length]

        z = self.segment_to_manifold(x_seg)  # [Bf, num_segments, encode_dim] -- already on manifold
        # z = self.dropout(z)
        # NOTE: dropout here acts directly on manifold-point coordinates,
        # which isn't geometrically principled. Consider, if you see
        # instability:
        #   v = self.ball.logmap0(z); v = self.dropout(v)
        #   z = self.ball.projx(self.ball.expmap0(v))

        return z


# ============================================================
# Parent module -- updated to use the new encoder directly.
# map_segments_to_hyperbolic is REMOVED: the encoder now produces
# manifold points itself, so there's no separate lifting step left to do.
# ============================================================
class SegmentedParallelPoincareMultiHorizonPL(nn.Module):
    """
    Encode each decomposition component (trend / seasonal_coarse /
    seasonal_fine / residual) into [Bf, num_segments, encode_dim] points
    ON the Poincare manifold directly via SegmentPoincareEncodeMultiHorizon,
    then fuse per-segment with Mobius addition.
    """

    def __init__(self, lookback, num_channels, encode_dim, curvature=1.0, segment_length=24,
                 encode_dropout=0.1):
        super().__init__()

        self.encode_dim = encode_dim
        self.num_channels = num_channels

        # Shared manifold for the whole module: all four encoders are tied
        # to THIS curvature, so there's no cross-curvature mismatch between
        # the encoder and the rest of the pipeline (logmap/expmap/mobius_add
        # downstream all use self.manifold as well).
        self.manifold = geoopt.manifolds.PoincareBall(c=curvature)

        # A matching custom-math ball, at the SAME curvature, non-learnable,
        # so it doesn't drift away from self.manifold during training.
        # (poincareball_factory's `c` is passed through as-is when
        # learnable=False -- see PoincareBallCustomAutograd/StdGrad.c property.)
        self.encoder_ball = poincareball_factory(
            c=curvature, custom_autograd=False, learnable=True
        )

        self.trend_encode = SegmentPoincareEncodeMultiHorizon(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels,
            ball=self.encoder_ball, segment_length=segment_length, dropout=encode_dropout,
        )
        self.seasonal_coarse_encode = SegmentPoincareEncodeMultiHorizon(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels,
            ball=self.encoder_ball, segment_length=segment_length, dropout=encode_dropout,
        )
        self.seasonal_fine_encode = SegmentPoincareEncodeMultiHorizon(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels,
            ball=self.encoder_ball, segment_length=segment_length, dropout=encode_dropout,
        )
        self.residual_encode = SegmentPoincareEncodeMultiHorizon(
            encode_dim=encode_dim, lookback=lookback, num_channels=num_channels,
            ball=self.encoder_ball, segment_length=segment_length, dropout=encode_dropout,
        )

        self.mobius_weights = nn.Parameter(torch.ones(4) * 0.25)
        print("NO Encoder used")
        # NOTE: self.effective_scale is gone -- it used to scale Euclidean
        # segment_encodes before expmap0 inside map_segments_to_hyperbolic.
        # That step no longer exists since the encoder does its own
        # expmap0 internally. If you still want a learnable pre-lift scale,
        # it would need to live INSIDE SegmentPoincareEncodeMultiHorizon,
        # applied to x_seg before self.segment_to_manifold(x_seg).

    def mobius_fusion_segments(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """
        Fuse components for each segment independently using Mobius addition.

        Args:
            z_trend_h, z_coarse_h, z_fine_h, z_residual_h: [Bf, num_segments, encode_dim]

        Returns:
            combined_h: [Bf, num_segments, encode_dim]
        """
        Bf, N, D = z_trend_h.shape

        z_trend_flat = z_trend_h.reshape(Bf * N, D)
        z_coarse_flat = z_coarse_h.reshape(Bf * N, D)
        z_fine_flat = z_fine_h.reshape(Bf * N, D)
        z_residual_flat = z_residual_h.reshape(Bf * N, D)

        weights = torch.softmax(self.mobius_weights, dim=0)

        combined_flat = self.manifold.mobius_scalar_mul(weights[0], z_trend_flat)

        scaled_coarse = self.manifold.mobius_scalar_mul(weights[1], z_coarse_flat)
        combined_flat = self.manifold.mobius_add(combined_flat, scaled_coarse)

        scaled_fine = self.manifold.mobius_scalar_mul(weights[2], z_fine_flat)
        combined_flat = self.manifold.mobius_add(combined_flat, scaled_fine)

        scaled_residual = self.manifold.mobius_scalar_mul(weights[3], z_residual_flat)
        combined_flat = self.manifold.mobius_add(combined_flat, scaled_residual)

        combined_flat = self.manifold.projx(combined_flat)

        combined_h = combined_flat.view(Bf, N, D)

        return combined_h

    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [Bf, seq_len]

        Returns:
            dict with hyperbolic encodedings [Bf, num_segments, encode_dim] for each component
        """
        # Each encoder now returns points ALREADY on the manifold --
        # no separate map_segments_to_hyperbolic step needed.
        z_trend_h = self.trend_encode(trend)
        z_coarse_h = self.seasonal_coarse_encode(seasonal_coarse)
        z_fine_h = self.seasonal_fine_encode(seasonal_fine)
        z_residual_h = self.residual_encode(residual)

        combined_h = self.mobius_fusion_segments(z_trend_h, z_coarse_h, z_fine_h, z_residual_h)

        return {
            "trend_h": z_trend_h,
            "seasonal_coarse_h": z_coarse_h,
            "seasonal_fine_h": z_fine_h,
            "residual_h": z_residual_h,
            "combined_h": combined_h,
        }