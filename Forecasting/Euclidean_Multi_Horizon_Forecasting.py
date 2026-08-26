import torch
import torch.nn as nn
from spec import RevIN, safe_expmap

class SegmentLinearencodeMultiHorizon(nn.Module):
    """
    This done for each feature 
    Produces ONE encodeding per segment.
    Input:  [Bf, seq_len]
    Output: [Bf, num_segments, encode_dim]  # One encodeding per segment
    """
    
    def __init__(self, lookback, encode_dim, num_channels, segment_length=24, dropout=0.1, individual=False):
        super().__init__()
        
        self.encode_dim = encode_dim
        self.segment_length = segment_length
        self.lookback = lookback
        self.num_segments = lookback // segment_length
        self.pad_seq_len = 0
        self.num_channels = num_channels
        self.individual = individual
        
        if self.lookback > self.num_segments * self.segment_length:
            self.pad_seq_len = (self.num_segments + 1) * self.segment_length - self.lookback
            self.num_segments += 1

        self.temporal_linears = nn.Linear(self.segment_length, encode_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [Bf, 720] - full historical sequence
        Returns:
            z: [Bf, 30, 64] - each segment is a point on the mainifold
        """
        B = x.shape[0]
        
        if self.pad_seq_len > 0:
            pad = torch.zeros(B, self.pad_seq_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        # Reshape into segments
        x_seg = x.view(B, self.num_segments, self.segment_length)  # [B, num_segments, seg_len]
        
        # encode each segment (KEEP segment structure!)
    
        seg_encode = self.temporal_linears(x_seg)  # [B, num_segments, encode_dim]
        seg_encode = self.dropout(seg_encode)
        # print(seg_encode.shape)
        
        return seg_encode

class SegmentLinearEncodeNoDecomp(nn.Module):
    """
    This done for each feature 
    Produces ONE encodeding per segment.
    Input:  [Bf, seq_len]
    Output: [Bf, num_segments, encode_dim]  # One encodeding per segment
    """
    
    def __init__(self, lookback, encode_dim, num_channels, segment_length=24, dropout=0.1, individual=False):
        super().__init__()
        
        self.encode_dim = encode_dim
        self.segment_length = segment_length
        self.lookback = lookback
        self.num_segments = lookback // segment_length
        self.pad_seq_len = 0
        self.num_channels = num_channels
        self.individual = individual
        
        if self.lookback > self.num_segments * self.segment_length:
            self.pad_seq_len = (self.num_segments + 1) * self.segment_length - self.lookback
            self.num_segments += 1
        if self.individual:
            self.temporal_linears = nn.ModuleList()
            for _ in range(self.num_channels):
                self.temporal_linears.append(nn.Linear(segment_length, encode_dim))

        self.temporal_linears = nn.Linear(segment_length, encode_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len] - single feature
        Returns:
            z: [B, num_segments, encode_dim] - trajectory
        """
        B = x.shape[0]
        
        # Pad if necessary
        if self.pad_seq_len > 0:
            pad = torch.zeros(B, self.pad_seq_len, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        
        # Reshape into segments
        x_seg = x.view(B, self.num_segments, self.segment_length)  # [B, num_seg, seg_len]
        
        # encode each segment (KEEP segment structure!)
    
        seg_encode = self.temporal_linears(x_seg)  # [B, num_segments, encode_dim]
        seg_encode = self.dropout(seg_encode)
        
        return seg_encode

class ParallelDirectEuclideanDynamics(nn.Module):
    def __init__(self, encode_dim, num_horizons):
        super().__init__()
        self.num_horizons = num_horizons
        self.velocity_net = nn.Linear(encode_dim, encode_dim)
        self.step_sizes = nn.Parameter(torch.tensor(1.0))

    def compute_initial_velocity(self, z_history):
        B, N, S = z_history.shape
        if N < 2:
            return torch.zeros(B, S, device=z_history.device)
        velocities = z_history[:, 1:, :] - z_history[:, :-1, :]
        weights = torch.tensor(
            [0.9 ** (N - 2 - i) for i in range(N - 1)],
            device=z_history.device,
            dtype=z_history.dtype
        )
        weights = weights / weights.sum()
        return (velocities * weights.view(1, -1, 1)).sum(dim=1)

    def forward(self, z_0, v_init):
        B, D = z_0.shape
        T = self.num_horizons
        step = torch.sigmoid(self.step_sizes)
        v_0t = self.velocity_net(v_init)
        t_vals = torch.arange(1, T + 1, device=z_0.device).float()
        v_all = (step * t_vals.view(T, 1, 1) * v_0t.unsqueeze(0))
        v_all = v_all.permute(1, 0, 2).reshape(B * T, D)
        z_0_exp = z_0.unsqueeze(1).expand(B, T, D).reshape(B * T, D)
        return (z_0_exp + v_all).view(B, T, D)


class EuclideanMultiHorizonForecaster(nn.Module):
    def __init__(self, lookback, pred_len, n_features, encode_dim=64, segment_length=24,
                 use_revin=True, manifold_type="Euclidean", encode_dropout=0.3):
        super().__init__()

        self.lookback = lookback
        self.pred_len = pred_len
        self.n_features = n_features
        self.segment_length = segment_length
        self.num_input_segments = lookback // segment_length
        self.num_pred_segments = pred_len // segment_length
        self.use_revin = use_revin
        self.encode_dim = encode_dim
        self.use_no_decomposition = True

        num_segments = lookback // segment_length

        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)
        
        if self.use_no_decomposition:
            print("Euclidean No Decomposition ablation")
            self.filt = SegmentLinearEncodeNoDecomp(
                lookback=self.lookback,
                encode_dim=self.encode_dim,
                num_channels=self.n_features,
                segment_length=self.segment_length,
                dropout=encode_dropout
            )
            self.dynamics = ParallelDirectEuclideanDynamics(self.encode_dim)

        self.trend_filt = SegmentLinearencodeMultiHorizon(
            lookback=self.lookback,
            encode_dim=self.encode_dim,
            num_channels=self.n_features,
            segment_length=self.segment_length,
            dropout=encode_dropout
        )
        self.coarse_filt = SegmentLinearencodeMultiHorizon(
            lookback=self.lookback,
            encode_dim=self.encode_dim,
            num_channels=self.n_features,
            segment_length=self.segment_length,
            dropout=encode_dropout
        )
        self.fine_filt = SegmentLinearencodeMultiHorizon(
            lookback=self.lookback,
            encode_dim=self.encode_dim,
            num_channels=self.n_features,
            segment_length=self.segment_length,
            dropout=encode_dropout
        )
        self.resid_filt = SegmentLinearencodeMultiHorizon(
            lookback=self.lookback,
            encode_dim=self.encode_dim,
            num_channels=self.n_features,
            segment_length=self.segment_length,
            dropout=encode_dropout
        )
        
        # ===== Dynamics — one per component =====
        self.dynamics_trend  = ParallelDirectEuclideanDynamics(self.encode_dim, self.num_pred_segments)
        self.dynamics_coarse = ParallelDirectEuclideanDynamics(self.encode_dim, self.num_pred_segments)
        self.dynamics_fine   = ParallelDirectEuclideanDynamics(self.encode_dim, self.num_pred_segments)
        self.dynamics_resid  = ParallelDirectEuclideanDynamics(self.encode_dim, self.num_pred_segments)


        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(4) * 0.25)

        print(f"\n{'='*70}")
        print(f"PARALLEL Direct Multi-Horizon Euclidean Forecaster")
        print(f"{'='*70}")
        print(f"Features:        {n_features} (channel-independent)")
        print(f"Future segments: {self.num_pred_segments}")
        print(f"Evolution:       z_t = z_0 + step * t * v")
        print(f"Losses:          MSE only")
        print(f"{'='*70}\n")


    def weighted_sum_fusion(self, z_trend, z_coarse, z_fine, z_resid):
        weights = torch.softmax(self.fusion_weights, dim=0)
        return (weights[0] * z_trend  +
                weights[1] * z_coarse +
                weights[2] * z_fine   +
                weights[3] * z_resid)

    def process_batched_features(self, trend_f, coarse_f, fine_f, resid_f):
        B = trend_f.shape[0]
        trend_seg = self.trend_filt(trend_f)
        coarse_seg = self.coarse_filt(coarse_f)
        fine_seg = self.fine_filt(fine_f)
        resid_seg = self.resid_filt(resid_f)

        # ===== Velocities =====
        v_trend_init  = self.dynamics_trend.compute_initial_velocity(trend_seg)
        v_coarse_init = self.dynamics_coarse.compute_initial_velocity(coarse_seg)
        v_fine_init   = self.dynamics_fine.compute_initial_velocity(fine_seg)
        v_resid_init  = self.dynamics_resid.compute_initial_velocity(resid_seg)

        # ===== Initial states =====
        z_trend_0  = trend_seg[:, -1, :]
        z_coarse_0 = coarse_seg[:, -1, :]
        z_fine_0   = fine_seg[:, -1, :]
        z_resid_0  = resid_seg[:, -1, :]

        # ===== Evolution =====
        
        z_trend_future  = self.dynamics_trend(z_trend_0, v_trend_init)
        z_coarse_future = self.dynamics_coarse(z_coarse_0, v_coarse_init)
        z_fine_future   = self.dynamics_fine(z_fine_0, v_fine_init)
        z_resid_future  = self.dynamics_resid(z_resid_0, v_resid_init)
        # ===== Fusion =====
        z_fused = self.weighted_sum_fusion(
            z_trend_future, z_coarse_future, z_fine_future, z_resid_future
        )

        remade = nn.Linear(self.encode_dim, self.segment_length)

        predictions = remade(z_fused)
        # ===== Reconstruct — MLP directly, no logmap0 =====
        # B_full, T, D = z_fused.shape
        # z_flat = z_fused.reshape(B_full * T, D)
        # pred_flat = self.reconstructor(z_flat)
        # predictions = pred_flat.reshape(B_full, T * self.segment_length)

        return {
            'predictions':  predictions,
            'latent_z':     z_fused,
            'latent_trend': z_trend_future,
            'latent_coarse':z_coarse_future,
            'latent_fine':  z_fine_future,
            'latent_resid': z_resid_future,
        }

    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Channel-independent Euclidean multi-horizon forecasting.
 
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, L, F]
        Returns:
            dict with predictions: [B, pred_len, F]
        """
        B, L, F = trend.shape
 
        # ===== RevIN normalization =====
        x_combined = trend + seasonal_coarse + seasonal_fine + residual
 
        if self.use_revin:
            self.revin(x_combined, mode='norm')
            trend           = self._normalize_component(trend)
            seasonal_coarse = self._normalize_component(seasonal_coarse)
            seasonal_fine   = self._normalize_component(seasonal_fine)
            residual        = self._normalize_component(residual)
 
        # ===== Collapse features: [B, L, F] → [B*F, L] =====
        def collapse(x):
            return x.permute(0, 2, 1).contiguous().view(B * F, L)
 
        def uncollapse_latent(tensor_b):
            Bf, N, S = tensor_b.shape
            return tensor_b.view(B, F, N, S)
 
        # ===== Process =====
        batched_out = self.process_batched_features(
            collapse(trend),
            collapse(seasonal_coarse),
            collapse(seasonal_fine),
            collapse(residual)
        )
        predictions_norm = batched_out['predictions']  # [B*F, pred_len]
 
        # ===== RevIN denormalization =====
        if self.use_revin:
            predictions_norm = predictions_norm.view(B, F, -1).permute(0, 2, 1).contiguous()
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm
 
        return {
            'predictions': predictions,
            'euclidean_states': {
                'combined_e': uncollapse_latent(batched_out['latent_z']),
                'trend_e':    uncollapse_latent(batched_out['latent_trend']),
                'coarse_e':   uncollapse_latent(batched_out['latent_coarse']),
                'fine_e':     uncollapse_latent(batched_out['latent_fine']),
                'resid_e':    uncollapse_latent(batched_out['latent_resid'])
            },
        }
 
    def _normalize_component(self, component):
        """
        Normalize component using stored RevIN statistics.
        Identical to hyperbolic model.
 
        Args:
            component: [B, H, F]
        Returns:
            normalized: [B, H, F]
        """
        x = (component - self.revin.mean) / self.revin.stdev
        if self.revin.affine:
            x = x * self.revin.affine_weight + self.revin.affine_bias
        return x
