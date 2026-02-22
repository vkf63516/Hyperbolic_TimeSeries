import torch
import torch.nn as nn
from encode.Multi_Horizon.segment_encode_multi_horizon_poincare import SegmentedParallelPoincareMultiHorizon
from Lifting.horizon_hyperbolic_segment_reconstructor import HorizonHyperbolicSegmentReconstructionHead
from spec import RevIN, safe_expmap
from DynamicsMvar.poincare_disk import poincareball_factory
from DynamicsMvar.Poincare_Residual_Dynamics import PoincareLinear
from spec import compute_hierarchical_loss_with_manifold_dist

class ParallelDirectPoincareDynamics(nn.Module):
    """
    Parallel direct multi-horizon dynamics using geodesic evolution.
    Predicts ALL future segments in one batched forward pass.
    """
    
    def __init__(self, encode_dim, manifold, num_horizons):
        super().__init__()
        self.manifold = manifold
        self.encode_dim = encode_dim
        self.num_horizons = num_horizons
        
        self.ball = poincareball_factory(c=1.0, custom_autograd=False, learnable=True)
        
        # ===== Single time-conditional velocity network =====
        self.velocity_net = PoincareLinear(
            encode_dim,  
            encode_dim,
            self.ball
        )
        
        # Learnable step sizes per timestep
        self.step_sizes = nn.Parameter(torch.tensor(1.0))
    
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
        
        # Compute velocities between consecutive segments
        velocities = []
        for i in range(N - 1):
            v_i = self.manifold.logmap(z_history[:, i, :], z_history[:, i + 1, :])
            velocities.append(v_i)
        
        velocities = torch.stack(velocities, dim=1)  # [B, N-1, D]
        
        # Exponentially weighted average (recent segments weighted more)
        weights = torch.tensor(
            [0.9 ** (N - 2 - i) for i in range(N - 1)],
            device=z_history.device
        )
        weights = weights / weights.sum()
        
        v_avg = (velocities * weights.view(1, -1, 1)).mean(dim=1)
        
        return v_avg
    
    def forward(self, z_0, v_init):
        """
        Parallel prediction of ALL future states in one batched forward pass.
        
        Args:
            z_0: [B, encode_dim] - initial state on manifold
            v_init: [B, encode_dim] - initial velocity in tangent space
        
        Returns:
            z_all: [B, num_horizons, encode_dim] - ALL future states
        """
        B, D = z_0.shape
        T = self.num_horizons
        
        predictions = []
        
        step = torch.sigmoid(self.step_sizes)
        
        for t in range(1, T + 1):
            # Scale velocity by point ()
            v_0t = self.velocity_net(v_init)
            v_t = step * t * v_0t
            # Geodesic evolution
            z_t = self.manifold.expmap(z_0, v_t)
            predictions.append(z_t)
        
        return torch.stack(predictions, dim=1)  # [B, T, D]


class DirectMultiHorizonHyperbolicForecaster(nn.Module):
    """
    Parallel direct multi-horizon forecaster with geodesic dynamics.
    """
    
    def __init__(self, lookback, pred_len, n_features, encode_dim, 
                 hidden_dim, curvature, manifold_type, segment_length=24,
                 use_revin=True, encode_dropout=0.3, recon_dropout=0.2,
                 window_size=2):
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
        
        # ===== Encoder =====
        self.encode_hyperbolic = SegmentedParallelPoincareMultiHorizon(
            lookback=lookback,
            num_channels=n_features,
            encode_dim=encode_dim,
            curvature=curvature,
            segment_length=segment_length,
            encode_dropout=encode_dropout
        )
        
        self.manifold = self.encode_hyperbolic.manifold
        
        # ===== Parallel geodesic dynamics for each component =====
        self.dynamics_trend = ParallelDirectPoincareDynamics(
            encode_dim, self.manifold, self.num_pred_segments
        )
        self.dynamics_coarse = ParallelDirectPoincareDynamics(
            encode_dim, self.manifold, self.num_pred_segments
        )
        self.dynamics_fine = ParallelDirectPoincareDynamics(
            encode_dim, self.manifold, self.num_pred_segments
        )
        self.dynamics_resid = ParallelDirectPoincareDynamics(
            encode_dim, self.manifold, self.num_pred_segments
        )
        
        # ===== ONE decoder for all segments =====
        self.reconstructor = HorizonHyperbolicSegmentReconstructionHead(
            encode_dim=encode_dim,
            output_dim=1,
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
        print(f"🚀 PARALLEL Direct Multi-Horizon Hyperbolic Forecaster")
        print(f"{'='*70}")
        print(f"Features: {n_features} (channel-independent)")
        print(f"Future segments: {self.num_pred_segments}")
        print(f"Method: Geodesic evolution with time-conditional velocity")
        print(f"All timesteps computed in ONE batched forward pass ✓")
        print(f"ONE decoder for all segments ✓")
        print(f"{'='*70}\n")

    def mobius_fusion(self, z_next_trend, z_next_coarse, z_next_fine, z_next_resid):
        """
        Fuse components for each segment independently using Möbius addition.
        
        Args:
            z_next_trend, z_next_coarsr, z_next_fine, z_next_resid: [B, encode_dim]
        
        Returns:
            combined: [B, encode_dim]
        """
        # Normalize weights
        weights = torch.softmax(self.mobius_weights, dim=0)

        # Sequential Möbius addition with weights
        combined = self.manifold.mobius_scalar_mul(weights[0], z_next_trend)
        
        scaled_coarse = self.manifold.mobius_scalar_mul(weights[1], z_next_coarse)
        combined = self.manifold.mobius_add(combined, scaled_coarse)
        
        scaled_fine = self.manifold.mobius_scalar_mul(weights[2], z_next_fine)
        combined = self.manifold.mobius_add(combined, scaled_fine)
    
        scaled_residual = self.manifold.mobius_scalar_mul(weights[3], z_next_resid)
        combined = self.manifold.mobius_add(combined, scaled_residual)
        
        # Ensure numerical stability
        combined = self.manifold.projx(combined)
                
        return combined
    
    def process_batched_features(self, trend_f, coarse_f, fine_f, resid_f):
        """
        Process batched features through entire pipeline.
        
        Args:
            trend_f, coarse_f, fine_f, resid_f: [B, seq_len]
        
        Returns:
            dict with predictions: [B, pred_len]
        """
        B = trend_f.shape[0]
        
        # ===== Encode historical segments =====
        encode_h = self.encode_hyperbolic(trend_f, coarse_f, fine_f, resid_f)
        
        z_trend_hist = encode_h["trend_h"]  # [B, num_hist_segments, encode_dim]
        z_coarse_hist = encode_h["seasonal_coarse_h"]
        z_fine_hist = encode_h["seasonal_fine_h"]
        z_resid_hist = encode_h["residual_h"]
        # print(z_trend_hist.shape)
        _, N_hist, D = z_trend_hist.shape
        
        # ===== Take window for velocity computation =====
     
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
        z_trend_future = self.dynamics_trend(z_trend_0, v_trend_init)  # [B*F, T, D]
        z_coarse_future = self.dynamics_coarse(z_coarse_0, v_coarse_init)
        z_fine_future = self.dynamics_fine(z_fine_0, v_fine_init)
        z_resid_future = self.dynamics_resid(z_resid_0, v_resid_init)
        
        # ===== PARALLEL GEODESIC EVOLUTION =====
        # Each component: ALL future segments in ONE forward pass!
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
        predictions = []
        # ===== Fuse and reconstruct each future segment =====
            
        z_fused = self.mobius_fusion(z_trend_future, z_coarse_future, z_fine_future, z_resid_future)
        prediction = self.reconstructor(z_fused)  # [B, segment_length]
        predictions = prediction.reshape(B, self.segment_length * self.num_pred_segments)
        
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
        Channel-independent parallel direct multi-horizon forecasting.
        
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len, n_features]
        
        Returns:
            dict with predictions: [B, pred_len, n_features]
        """
        B, L, F = trend.shape
        
        # ===== RevIN Normalization =====
        x_combined = trend + seasonal_coarse + seasonal_fine + residual
        
        if self.use_revin:
            self.revin(x_combined, mode='norm')
            trend = self._normalize_component(trend)
            seasonal_coarse = self._normalize_component(seasonal_coarse)
            seasonal_fine = self._normalize_component(seasonal_fine)
            residual = self._normalize_component(residual)
        
        # ===== Collapse features: [B, L, F] → [B*F, L] =====
        def collapse(x):
            return x.permute(0, 2, 1).contiguous().view(B * F, L)
        
        trend_bf = collapse(trend)
        coarse_bf = collapse(seasonal_coarse)
        fine_bf = collapse(seasonal_fine)
        resid_bf = collapse(residual)

        def uncollapse_latent(tensor_b):
            # tensor_b: [B*F, num_pred_segments, D] -> [B, F, num_pred_segments, D]
            # print(tensor_b.shape)
            Bf, N, D = tensor_b.shape
            return tensor_b.view(B, F, N, D)
        # ===== Process all features in batch =====
        batched_out = self.process_batched_features(trend_bf, coarse_bf, fine_bf, resid_bf)
        predictions_norm = batched_out['predictions']  # [B*F, pred_len]
                
        # ===== RevIN Denormalization =====
        if self.use_revin:
            predictions_norm = predictions_norm.view(B, F, -1).permute(0, 2, 1).contiguous()
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm
        
        return {
            'predictions': predictions,
            'hyperbolic_states': {
                'combined_h': uncollapse_latent(batched_out['latent_z']),
                'trend_h': uncollapse_latent(batched_out['latent_trend']),
                'coarse_h': uncollapse_latent(batched_out['latent_coarse']),
                'fine_h': uncollapse_latent(batched_out['latent_fine']),
                'resid_h': uncollapse_latent(batched_out['latent_resid'])
            },
            'hierarchy_loss': batched_out["hierarchy_loss"]
        }

    def _normalize_component(self, component):
        """
        Normalize a single component using stored RevIN statistics.
        
        Args:
            component: [B, L, F] 
        
        Returns:
            normalized component: [B, L, F]
        """
        # Apply normalization: (x - mean) / std
        x = (component - self.revin.mean) / self.revin.stdev
        
        # Apply affine transformation if enabled
        if self.revin.affine:
            x = x * self.revin.affine_weight + self.revin.affine_bias
        
        return x