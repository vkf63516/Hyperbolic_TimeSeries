import torch
import torch.nn as nn
from Lifting.horizon_euclidean_segment_reconstructor import EuclideanHorizonSegmentReconstructionHead
from encode.Multi_Horizon.segment_encode_multi_horizon_euclidean import SegmentedParallelEuclideanMultiHorizon
from spec import RevIN, safe_expmap

class ParallelDirectEuclideanDynamics(nn.Module):
    def __init__(self, encode_dim, num_horizons):
        super().__init__()
        self.encode_dim = encode_dim
        self.num_horizons = num_horizons
        
        # Standard linear layer instead of PoincareLinear
        self.velocity_net = nn.Linear(encode_dim, encode_dim)
        self.step_sizes = nn.Parameter(torch.tensor(1.0))
    
    def compute_initial_velocity(self, z_history):
        B, N, D = z_history.shape
        if N < 2:
            return torch.zeros(B, D, device=z_history.device)
        
        # Simple finite difference — no logmap needed
        velocities = z_history[:, 1:, :] - z_history[:, :-1, :]  # [B, N-1, D]
        
        weights = torch.tensor(
            [0.9 ** (N - 2 - i) for i in range(N - 1)],
            device=z_history.device
        )
        weights = weights / weights.sum()
        
        v_avg = (velocities * weights.view(1, -1, 1)).mean(dim=1)
        return v_avg
    
    def forward(self, z_0, v_init):
        B, D = z_0.shape
        T = self.num_horizons
        
        step = torch.sigmoid(self.step_sizes)
        v_0t = self.velocity_net(v_init)  # standard linear transform
        
        # Euclidean geodesic = straight line
        # z_t = z_0 + step * t * v_0t
        t_vals = torch.arange(1, T + 1, device=z_0.device).float()
        v_all = (step * t_vals.view(T, 1, 1) * v_0t.unsqueeze(0))
        v_all = v_all.permute(1, 0, 2).reshape(B * T, D)
        z_0_exp = z_0.unsqueeze(1).expand(B, T, D).reshape(B * T, D)
        
        # Euclidean expmap = simple addition
        z_all = z_0_exp + v_all
        
        return z_all.view(B, T, D)

class EuclideanMultiHorizonHyperbolicForecaster(nn.Module):
    """
    Parallel direct multi-horizon forecaster with geodesic dynamics.
    """
    
    def __init__(self, lookback, pred_len, n_features, encode_dim, 
                 hidden_dim, manifold_type, segment_length=24,
                 use_revin=True, encode_dropout=0.3, recon_dropout=0.2):
        super().__init__()
        
        self.lookback = lookback
        self.pred_len = pred_len
        self.n_features = n_features
        self.segment_length = segment_length
        self.num_pred_segments = pred_len // segment_length
        self.encode_dim = encode_dim
        self.use_revin = use_revin
        self.manifold_type = manifold_type
        
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)
        
        # ===== Encoder =====
        self.encode_euclidean = SegmentedParallelEuclideanMultiHorizon(
            lookback=lookback,
            num_channels=n_features,
            encode_dim=encode_dim,
            segment_length=segment_length,
            encode_dropout=encode_dropout
        )
        
        
        # ===== Parallel geodesic dynamics for each component =====
        self.dynamics_trend = ParallelDirectEuclideanDynamics(encode_dim, self.num_pred_segments)
        self.dynamics_coarse = ParallelDirectEuclideanDynamics(encode_dim, self.num_pred_segments)
        self.dynamics_fine = ParallelDirectEuclideanDynamics(encode_dim, self.num_pred_segments)
        self.dynamics_resid = ParallelDirectEuclideanDynamics(encode_dim, self.num_pred_segments)
        
        # ===== ONE decoder for all segments =====
        self.reconstructor = EuclideanHorizonSegmentReconstructionHead(
            encode_dim=encode_dim,
            output_dim=1,
            segment_length=self.segment_length,
            num_pred_segments=self.num_pred_segments,
            hidden_dim=hidden_dim,
            dropout=recon_dropout
        )
        
        #  fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(4) * 0.25)
        
        print(f"\n{'='*70}")
        print(f"🚀 PARALLEL Direct Multi-Horizon Hyperbolic Forecaster")
        print(f"{'='*70}")
        print(f"Features: {n_features} (channel-independent)")
        print(f"Future segments: {self.num_pred_segments}")
        print(f"Method: Geodesic evolution with time-conditional velocity")
        print(f"All timesteps computed in ONE batched forward pass ✓")
        print(f"ONE decoder for all segments ✓")
        print(f"{'='*70}\n")

    def euclidean_fusion(self, z_trend, z_coarse, z_fine, z_resid):
        """
        Euclidean weighted sum — replaces Mobius fusion.
        
        Args:
            z_trend, z_coarse, z_fine, z_resid: [B, T, D]
        Returns:
            combined: [B, T, D]
        """
        weights = torch.softmax(self.fusion_weights, dim=0)
        return (weights[0] * z_trend  +
                weights[1] * z_coarse +
                weights[2] * z_fine   +
                weights[3] * z_resid)

    def process_batched_features(self, trend_f, coarse_f, fine_f, resid_f):
        """
        Euclidean equivalent of process_batched_features.
        
        Args:
            trend_f, coarse_f, fine_f, resid_f: [B, seq_len]
        Returns:
            dict with predictions: [B, pred_len]
        """
        B = trend_f.shape[0]
        
        # ===== Encode historical segments =====
        # Keep same encoder — isolates geometry contribution
        encode_e = self.encode_euclidean(trend_f, coarse_f, fine_f, resid_f)
        
        z_trend_hist  = encode_e["trend_e"]           # [B, N_h, D]
        z_coarse_hist = encode_e["seasonal_coarse_e"]
        z_fine_hist   = encode_e["seasonal_fine_e"]
        z_resid_hist  = encode_e["residual_e"]
        
        # ===== Compute initial velocities — Euclidean finite difference =====
        v_trend_init  = self.dynamics_trend.compute_initial_velocity(z_trend_hist)
        v_coarse_init = self.dynamics_coarse.compute_initial_velocity(z_coarse_hist)
        v_fine_init   = self.dynamics_fine.compute_initial_velocity(z_fine_hist)
        v_resid_init  = self.dynamics_resid.compute_initial_velocity(z_resid_hist)
        
        # ===== Initial states =====
        z_trend_0  = z_trend_hist[:, -1, :]
        z_coarse_0 = z_coarse_hist[:, -1, :]
        z_fine_0   = z_fine_hist[:, -1, :]
        z_resid_0  = z_resid_hist[:, -1, :]
        
        # ===== Euclidean straight line evolution =====
        z_trend_future  = self.dynamics_trend(z_trend_0, v_trend_init)   # [B, T, D]
        z_coarse_future = self.dynamics_coarse(z_coarse_0, v_coarse_init)
        z_fine_future   = self.dynamics_fine(z_fine_0, v_fine_init)
        z_resid_future  = self.dynamics_resid(z_resid_0, v_resid_init)
        
        # ===== Euclidean hierarchical loss =====
        encodings_dict = {
            "trend_e":           z_trend_future,
            "seasonal_coarse_e": z_coarse_future,
            "seasonal_fine_e":   z_fine_future,
            "residual_e":        z_resid_future
        }

        
        # ===== Euclidean fusion =====
        z_fused = self.euclidean_fusion(
            z_trend_future, z_coarse_future, z_fine_future, z_resid_future
        )  # [B, T, D]
        
        # ===== Reconstruct =====
        prediction  = self.reconstructor(z_fused)
        predictions = prediction.reshape(B, self.segment_length * self.num_pred_segments)
        
        
        return {
            'predictions':        predictions,
            'latent_z':           z_fused,
            'latent_trend':       z_trend_future,
            'latent_coarse':      z_coarse_future,
            'latent_fine':        z_fine_future,
            'latent_resid':       z_resid_future,
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
