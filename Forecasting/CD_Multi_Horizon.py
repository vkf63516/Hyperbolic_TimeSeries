import torch
import torch.nn.functional as F
import torch.nn as nn
from encode.Linear.segment_linear_encode_poincare_multi_horizon import SegmentedParallelPoincareMultiHorizonCD
from Lifting.horizon_channel_dependent_reconstructor import HorizonHyperbolicSegmentReconstructionHeadCD
from spec import RevIN, safe_expmap
from DynamicsMvar.poincare_disk import poincareball_factory
from DynamicsMvar.Poincare_Residual_Dynamics import PoincareLinear
from spec import safe_expmap, compute_hierarchical_loss_with_manifold_dist
from loss import hyperbolic_velocity_consistency_loss


class ParallelDirectPoincareDynamicsCD(nn.Module):
    """
    Parallel direct multi-horizon dynamics using geodesic evolution.
    Predicts ALL future segments in one batched forward pass.

    Identical logic to ParallelDirectPoincareDynamics (channel-independent
    version) — dynamics themselves don't care whether channels were fused
    upstream or not, since they just operate on [B, D] / [B, N, D] states.
    Kept as its own class (rather than reused) so channel-dependent and
    channel-independent forecasters stay fully decoupled and independently
    tunable.
    """

    def __init__(self, encode_dim, manifold, num_horizons):
        super().__init__()
        self.manifold = manifold
        self.encode_dim = encode_dim
        self.num_horizons = num_horizons

        self.ball = poincareball_factory(c=1.0, custom_autograd=False, learnable=True)

        # ===== Base velocity network =====
        self.velocity_net = PoincareLinear(
            encode_dim,
            encode_dim,
            self.ball
        )

        # ===== Learnable step size (scalar) =====
        self.step_sizes = nn.Parameter(torch.tensor(1.0))

        # ===== Horizon-specific curvature correction =====
        self.horizon_scales = nn.Parameter(torch.zeros(num_horizons))  # [T]
        self.register_buffer(
            'time_indices',
            torch.linspace(0, 1, num_horizons).reshape(-1, 1)
        )

    def compute_initial_velocity(self, z_history):
        """
        Compute initial velocity from historical trajectory.

        Args:
            z_history: [B, num_hist_segments, encode_dim]
        Returns:
            v_init: [B, encode_dim]
        """
        B, N, D = z_history.shape

        if N < 2:
            return torch.zeros(B, D, device=z_history.device)

        velocities = self.manifold.logmap(z_history[:, :-1, :], z_history[:, 1:, :])

        weights = torch.tensor([0.9 ** (N - 2 - i) for i in range(N - 1)], device=z_history.device)
        weights = weights / weights.sum()

        v_avg = (velocities * weights.view(1, -1, 1)).mean(dim=1)

        return v_avg

    def forward(self, z_0, v_init):
        """
        Parallel prediction of ALL future states in one batched expmap call.

        Args:
            z_0:    [B, encode_dim] - last historical state on manifold
            v_init: [B, encode_dim] - aggregated historical velocity

        Returns:
            z_all: [B, num_horizons, encode_dim]
        """
        B, D = z_0.shape
        T = self.num_horizons

        step   = torch.sigmoid(self.step_sizes)
        v_0t   = self.velocity_net(v_init)                                    # [B, D]
        t_vals = torch.arange(1, T + 1, device=z_0.device).float()  # [T]
        horizon_weights = torch.tanh(self.horizon_scales)     # [T] in (-1,1)
        v_base = step * horizon_weights.view(T, 1, 1) * t_vals.view(T, 1, 1) * v_0t.unsqueeze(0)
        v_all   = v_base.permute(1, 0, 2).reshape(B * T, D)                   # [B*T, D]
        z_0_exp = z_0.unsqueeze(1).expand(B, T, D).reshape(B * T, D)          # [B*T, D]

        z_all = self.manifold.expmap(z_0_exp, v_all)                         # [B*T, D]
        return z_all.view(B, T, D)


class DirectMultiHorizonHyperbolicForecasterCD(nn.Module):
    """
    Channel-Dependent parallel direct multi-horizon forecaster with geodesic dynamics.

    Difference vs DirectMultiHorizonHyperbolicForecaster (channel-independent):
    - No collapse of [B, N, F] -> [B*F, N]. All features are kept together
      and passed straight into SegmentedParallelPoincareMultiHorizonCD, which
      fuses channels internally per-segment (see encoder docstring).
    - Dynamics, fusion, and reconstruction therefore operate on a SINGLE
      channel-fused trajectory per batch element rather than one trajectory
      per (batch, feature) pair.
    - Reconstructor outputs all n_features at once (output_dim=n_features)
      instead of a single scalar channel replicated across the batch axis.
    """

    def __init__(self, lookback, pred_len, n_features, encode_dim,
                 hidden_dim, curvature, manifold_type, segment_length=24,
                 use_revin=True, encode_dropout=0.3, recon_dropout=0.2,
                 window_size=2, share_feature_weights=True):
        super().__init__()

        self.lookback = lookback
        self.pred_len = pred_len
        self.n_features = n_features
        self.segment_length = segment_length
        self.num_pred_segments = pred_len // segment_length
        self.encode_dim = encode_dim
        self.use_revin = use_revin
        self.window_size = window_size
        self.manifold_type = manifold_type

        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)

        # ===== Channel-Dependent Encoder =====
        self.encode_hyperbolic = SegmentedParallelPoincareMultiHorizonCD(
            lookback=lookback,
            num_channels=n_features,
            encode_dim=encode_dim,
            curvature=curvature,
            segment_length=segment_length,
            encode_dropout=encode_dropout,
            share_feature_weights=share_feature_weights
        )

        self.manifold = self.encode_hyperbolic.manifold
        self.num_input_segments = lookback // segment_length

        # ===== Parallel geodesic dynamics for each component =====
        # Only ONE trajectory per component now (channels already fused),
        # instead of one per (batch, feature) pair.
        self.dynamics_trend = ParallelDirectPoincareDynamicsCD(
            encode_dim, self.manifold, self.num_pred_segments
        )
        self.dynamics_coarse = ParallelDirectPoincareDynamicsCD(
            encode_dim, self.manifold, self.num_pred_segments
        )
        self.dynamics_fine = ParallelDirectPoincareDynamicsCD(
            encode_dim, self.manifold, self.num_pred_segments
        )
        self.dynamics_resid = ParallelDirectPoincareDynamicsCD(
            encode_dim, self.manifold, self.num_pred_segments
        )

        # ===== ONE decoder for all segments, all channels =====
        self.reconstructor = HorizonHyperbolicSegmentReconstructionHeadCD(
            encode_dim=encode_dim,
            output_dim=n_features,  # channel-dependent: reconstruct all features at once
            segment_length=self.segment_length,
            manifold=self.manifold,
            num_pred_segments=self.num_pred_segments,
            manifold_type=self.manifold_type,
            hidden_dim=hidden_dim,
            dropout=recon_dropout
        )

        # Möbius fusion weights
        self.mobius_weights = nn.Parameter(torch.ones(4) * 0.25)

        print(f"\n{'='*70}")
        print(f"🚀 PARALLEL Direct Multi-Horizon Hyperbolic Forecaster (Channel-Dependent)")
        print(f"{'='*70}")
        print(f"Features: {n_features} (channel-dependent, fused per segment)")
        print(f"Future segments: {self.num_pred_segments}")
        print(f"Method: Geodesic evolution with time-conditional velocity")
        print(f"All timesteps computed in ONE batched forward pass ✓")
        print(f"ONE decoder for all segments and all channels ✓")
        print(f"{'='*70}\n")

    def mobius_fusion(self, z_next_trend, z_next_coarse, z_next_fine, z_next_resid):
        """
        Fuse components for each segment independently using Möbius addition.

        Args:
            z_next_trend, z_next_coarse, z_next_fine, z_next_resid: [B, H, encode_dim]

        Returns:
            combined: [B, H, encode_dim]
        """
        weights = torch.softmax(self.mobius_weights, dim=0)

        combined = self.manifold.mobius_scalar_mul(weights[0], z_next_trend)

        scaled_coarse = self.manifold.mobius_scalar_mul(weights[1], z_next_coarse)
        combined = self.manifold.mobius_add(combined, scaled_coarse)

        scaled_fine = self.manifold.mobius_scalar_mul(weights[2], z_next_fine)
        combined = self.manifold.mobius_add(combined, scaled_fine)

        scaled_residual = self.manifold.mobius_scalar_mul(weights[3], z_next_resid)
        combined = self.manifold.mobius_add(combined, scaled_residual)

        combined = self.manifold.projx(combined)

        return combined

    def process_features(self, trend, coarse, fine, resid):
        """
        Process ALL features jointly through the entire pipeline (channels
        fused inside the encoder, not collapsed into the batch axis).

        Args:
            trend, coarse, fine, resid: [B, seq_len, n_features]

        Returns:
            dict with predictions: [B, pred_len, n_features]
        """
        B = trend.shape[0]

        # ===== Encode historical segments (channel-fused per segment) =====
        encode_h = self.encode_hyperbolic(trend, coarse, fine, resid)

        z_trend_hist = encode_h["trend_h"]  # [B, num_segments, encode_dim]
        z_coarse_hist = encode_h["seasonal_coarse_h"]
        z_fine_hist = encode_h["seasonal_fine_h"]
        z_resid_hist = encode_h["residual_h"]
        _, N_hist, D = z_trend_hist.shape

        z_trend_win = z_trend_hist
        z_coarse_win = z_coarse_hist
        z_fine_win = z_fine_hist
        z_resid_win = z_resid_hist
        v_trend_init = self.dynamics_trend.compute_initial_velocity(z_trend_win)
        v_coarse_init = self.dynamics_coarse.compute_initial_velocity(z_coarse_win)
        v_fine_init = self.dynamics_fine.compute_initial_velocity(z_fine_win)
        v_resid_init = self.dynamics_resid.compute_initial_velocity(z_resid_win)

        # Initial states
        z_trend_0 = z_trend_hist[:, -1, :]
        z_coarse_0 = z_coarse_hist[:, -1, :]
        z_fine_0 = z_fine_hist[:, -1, :]
        z_resid_0 = z_resid_hist[:, -1, :]

        # ===== PARALLEL DIRECT PREDICTION =====
        z_trend_future = self.dynamics_trend(z_trend_0, v_trend_init)  # [B, H, D]
        z_coarse_future = self.dynamics_coarse(z_coarse_0, v_coarse_init)
        z_fine_future = self.dynamics_fine(z_fine_0, v_fine_init)
        z_resid_future = self.dynamics_resid(z_resid_0, v_resid_init)

        encodings_dict = {
            "trend_h": z_trend_future,
            "seasonal_coarse_h": z_coarse_future,
            "seasonal_fine_h": z_fine_future,
            "residual_h": z_resid_future
        }
        hierarchy_loss = compute_hierarchical_loss_with_manifold_dist(
            encodings_dict,
            manifold=self.manifold,
            margin=0.1
        )

        # ===== Fuse and reconstruct all future segments, all channels =====
        z_fused = self.mobius_fusion(z_trend_future, z_coarse_future, z_fine_future, z_resid_future)
        prediction = self.reconstructor(z_fused)  # [B, segment_length, H, n_features] (channel-dependent decoder output)
        predictions = prediction.reshape(B, self.segment_length * self.num_pred_segments, self.n_features)

        return {
            'predictions': predictions,
            'latent_z': z_fused,
            'latent_trend': z_trend_future,
            'latent_coarse': z_coarse_future,
            'latent_fine': z_fine_future,
            'latent_resid': z_resid_future,
            'hierarchy_loss': hierarchy_loss
        }

    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Channel-dependent parallel direct multi-horizon forecasting.

        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len, n_features]

        Returns:
            dict with predictions: [B, pred_len, n_features]
        """
        B, N, F = trend.shape

        # ===== RevIN Normalization =====
        x_combined = trend + seasonal_coarse + seasonal_fine + residual

        if self.use_revin:
            self.revin(x_combined, mode='norm')
            trend = self._normalize_component(trend)
            seasonal_coarse = self._normalize_component(seasonal_coarse)
            seasonal_fine = self._normalize_component(seasonal_fine)
            residual = self._normalize_component(residual)

        # ===== No collapsing: all channels stay together =====
        batched_out = self.process_features(trend, seasonal_coarse, seasonal_fine, residual)
        predictions_norm = batched_out['predictions']  # [B, pred_len, n_features]

        # ===== RevIN Denormalization =====
        if self.use_revin:
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm

        return {
            'predictions': predictions,
            'hyperbolic_states': {
                'combined_h': batched_out['latent_z'],      # [B, H, D]
                'trend_h': batched_out['latent_trend'],      # [B, H, D]
                'coarse_h': batched_out['latent_coarse'],    # [B, H, D]
                'fine_h': batched_out['latent_fine'],        # [B, H, D]
                'resid_h': batched_out['latent_resid']       # [B, H, D]
            },
            'hierarchy_loss': batched_out["hierarchy_loss"]
        }

    def _normalize_component(self, component):
        """
        Normalize a single component using stored RevIN statistics.

        Args:
            component: [B, N, F]

        Returns:
            normalized component: [B, N, F]
        """
        x = (component - self.revin.mean) / self.revin.stdev

        if self.revin.affine:
            x = x * self.revin.affine_weight + self.revin.affine_bias

        return x