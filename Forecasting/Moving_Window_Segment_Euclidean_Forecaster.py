import torch
import torch.nn as nn
from encode.Moving_Window.moving_segment_linear_encode_euclidean import SegmentParallelEuclideanMovingWindow
from DynamicsMvar.Euclidean_Residual_Dynamics import ResidualDynamics
from Lifting.euclidean_segment_reconstructor import EuclideanSegmentReconstructionHead
from spec import RevIN


class MovingWindowEuclideanForecaster(nn.Module):
    """
    Channel-Independent Euclidean forecaster with moving window.
    
    Key features:
    1. Channel-independent processing (like DLinear, PatchTST)
    2. Each feature processed independently with shared parameters
    3. Moving window with velocity-based dynamics
    4. Euclidean space (no hyperbolic geometry)
    5. Adaptive window sizing (caps at 15 segments by default)
    """
    def __init__(self, lookback, pred_len, n_features, encode_dim, hidden_dim, 
                 manifold_type, segment_length=24, 
                 use_revin=False, recon_dropout=0.3, encode_dropout=0.5,
                 num_layers=2, use_segment_norm=True, window_size=None, 
                 use_truncated_bptt=True, truncate_every=4):
        """
        Args:
            window_size: int - number of segments to keep in moving window
                        If None, uses adaptive sizing (max 15 segments)
            share_feature_weights: MUST be True for channel independence
        """
        super().__init__()

        if pred_len % segment_length != 0:
            raise ValueError(f"pred_len ({pred_len}) must be divisible by segment_length ({segment_length})")

        self.lookback = lookback
        self.encode_dim = encode_dim
        self.pred_len = pred_len
        self.n_features = n_features
        self.segment_length = segment_length
        self.num_pred_segments = pred_len // segment_length
        self.use_revin = use_revin
        self.manifold_type = manifold_type
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        
        # Adaptive window sizing for efficiency
        num_input_segments = lookback // segment_length
        if window_size is None:
            self.window_size = min(num_input_segments, 15)  # Max 15 segments
        else:
            self.window_size = min(window_size, num_input_segments)
        
        print(f"\n{'='*70}")
        print(f"Channel-Independent Moving Window Euclidean Forecaster")
        print(f"{'='*70}")
        print(f"Features: {n_features} (processed independently)")
        print(f"Lookback: {lookback} → {num_input_segments} segments")
        print(f"Segment length: {segment_length}")
        print(f"encode dim: {encode_dim}")
        print(f"Window size: {self.window_size}")
        print(f"Strategy: channel independence + Euclidean dynamics")
        
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)
        
        # Encoder for SINGLE feature (input_dim=1, shared across all features)
        self.encode_euclidean = SegmentParallelEuclideanMovingWindow(
            lookback=lookback,
            encode_dim=encode_dim,
            segment_length=segment_length,
            encode_dropout=encode_dropout,
        )
        self.step_size = nn.Parameter(torch.tensor(0.1))

        # Dynamics
        self.dynamics = ResidualDynamics(
            encode_dim=encode_dim
        )
        
        # Reconstructor for SINGLE feature (output_dim=1)
        self.reconstructor = EuclideanSegmentReconstructionHead(
            encode_dim=encode_dim,
            output_dim=1,  # Single feature output
            segment_length=segment_length,
            hidden_dim=hidden_dim,
            dropout=recon_dropout
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Parameters per feature: {total_params:,} (shared!)")
        print(f"{'='*70}\n")


    def compute_combined_velocity(self, z_current, decay=0.9):
        """
        Exponentially-weighted velocity averaging (FIXED).
        
        Args:
            decay: 0.9 = moderate recency (recent 4x more weight)
                0.85 = strong recency (recent 8x more weight)
                0.95 = mild recency (recent 2x more weight)
                1.0 = uniform (no recency bias)
        """
        B, N, D = z_current.shape
        
        # # Optional window limiting
        # if self.window_size is not None and self.window_size < N:
        #     z_current = z_current[:, -self.window_size-1:, :]
        #     N = self.window_size + 1
        
        # Compute velocities
        z_start = z_current[:, :-1, :]
        z_end = z_current[:, 1:, :]
        
        z_start_flat = z_start.reshape(B * (N-1), D)
        z_end_flat = z_end.reshape(B * (N-1), D)
        
        velocities_flat = z_end_flat - z_start_flat
        velocities = velocities_flat.view(B, N-1, D)
        
        # === FIXED: Exponential averaging ===
        if decay == 1.0:
            avg_velocity = velocities.mean(dim=1)
        else:
            # Create weights on GPU (FIXED)
            indices = torch.arange(N-1, dtype=torch.float32, device=z_current.device)
            weights = decay ** (N - 2 - indices)  # Recent gets higher weight
            weights = weights / weights.sum()
            avg_velocity = (velocities * weights.view(1, -1, 1)).sum(dim=1)
        
        return avg_velocity

    
    def process_batched_features(self, trend_f, coarse_f, fine_f, resid_f):
        """
        Process a batch of features through the entire forecasting pipeline.
        
        Args:
            trend_f, coarse_f, fine_f, resid_f: [B, seq_len, 1] single feature
        
        Returns:
            dict with predictions for this feature: [B, pred_len]
        """
        B = trend_f.shape[0]
        
        # Encode to Euclidean segments: [B, num_segments, encode_dim]
        encode_e = self.encode_euclidean(trend_f, coarse_f, fine_f, resid_f)
        
        # Initialize moving windows with historical segments
        z_current = encode_e["combined_e"]  # [B, num_segments, encode_dim]
        z_current_trend = encode_e["trend_e"]
        z_current_coarse = encode_e["seasonal_coarse_e"]
        z_current_fine = encode_e["seasonal_fine_e"]
        z_current_resid = encode_e["residual_e"]
        
        # Take last window_size segments
        if z_current.shape[1] > self.window_size:
            z_current = z_current[:, -self.window_size:, :]
            z_current_trend = z_current_trend[:, -self.window_size:, :]
            z_current_coarse = z_current_coarse[:, -self.window_size:, :]
            z_current_fine = z_current_fine[:, -self.window_size:, :]
            z_current_resid = z_current_resid[:, -self.window_size:, :]
        
        # Storage for predictions
        predictions_norm = []
        trend_predictions = []
        coarse_predictions = []
        fine_predictions = []
        residual_predictions = []
        step_size = torch.sigmoid(self.step_size)

        # Autoregressive forecasting with moving window
        for seg_step in range(self.num_pred_segments):
            # Compute velocities for ALL components at once (vectorized)
            velocity = self.compute_combined_velocity(z_current)
            # === Combined signal ===
# For each component:
            z_last = z_current[:, -1, :]                                    # Get last state
            z_next_raw = self.dynamics(z_last, velocity)  # Compute raw next
            z_next_interp = z_last + step_size * (z_next_raw - z_last)    # Interpolate
            pred_seg = self.reconstructor(z_next_interp)                        # Reconstruct 
            z_current = torch.cat([z_current[:, 1:], z_next_interp.unsqueeze(1)], dim=1)  # Update 
            predictions_norm.append(pred_seg)
          
            # Truncated BPTT: Detach AFTER using current states
            if self.use_truncated_bptt and (seg_step + 1) % self.truncate_every == 0 and seg_step < self.num_pred_segments - 1:
                z_current = z_current.detach()

        # Stack predictions
        predictions_norm = torch.stack(predictions_norm, dim=1)  # [B, num_seg, seg_len, 1]


        # Flatten to [B, pred_len] - SQUEEZE first to remove feature dimension!
        predictions_norm = predictions_norm.squeeze(-1).reshape(B, self.num_pred_segments * self.segment_length)
        
        return {
            'predictions': predictions_norm
        }
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Channel-independent forecasting with moving window trajectory modeling.
        
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len, n_features]
        
        Returns:
            dict with predictions for ALL features: [B, pred_len, n_features]
        """
        
        # Store RevIN stats
        x_combined = trend + seasonal_coarse + seasonal_fine + residual
        B, L, F = x_combined.shape
        if self.use_revin:
            self.revin(x_combined, mode='norm')
        
        # Collapse features into batch axis: (B, L, F) -> (B*F, L, 1)
        def collapse(x):
            # x: [B, L, F] -> [B*F, L, 1]
            x = x.permute(0, 2, 1).contiguous()  # [B, F, L]
            x = x.view(B * F, L).unsqueeze(-1)   # [B*F, L, 1]
            return x
        
        trend_b = collapse(trend)
        coarse_b = collapse(seasonal_coarse)
        fine_b = collapse(seasonal_fine)
        resid_b = collapse(residual)
        
        # Process all features at once
        batched_out = self.process_batched_features(trend_b, coarse_b, fine_b, resid_b)
        predictions_norm = batched_out['predictions']  # [B*F, pred_len]

        # Apply RevIN denormalization
        if self.use_revin:
            predictions_norm = predictions_norm.view(B, F, -1).permute(0, 2, 1).contiguous()  # [B, pred_len, F]
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm.view(B, F, -1).permute(0, 2, 1).contiguous()
        
        # Uncollapse component predictions
        def uncollapse_comp_predict(component_b):
            return component_b.view(B, F, -1).permute(0, 2, 1).contiguous()

        return {
            'predictions': predictions
        }