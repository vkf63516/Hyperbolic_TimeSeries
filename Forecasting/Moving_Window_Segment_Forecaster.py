import torch
import torch.nn as nn
from encode.Moving_Window.moving_segment_linear_encode_poincare import SegmentedParallelPoincareMovingWindow
from encode.Moving_Window.moving_segment_linear_encode_lorentz import SegmentedParallelLorentzMovingWindow
from encode.Linear.segment_linear_encode_lorentz import SegmentedParallelLorentz
from DynamicsMvar.Poincare_Residual_Dynamics import HyperbolicPoincareDynamics
from DynamicsMvar.Lorentz_Residual_Dynamics import HyperbolicLorentzDynamics
from Lifting.hyperbolic_segment_reconstructor import HyperbolicSegmentReconstructionHead
from spec import RevIN


class MovingWindowHyperbolicForecaster(nn.Module):
    """
    Channel-Independent segment-aware hyperbolic forecaster with moving window.
    
    Key features:
    1. Channel-independent processing (like DLinear, PatchTST)
    2. Each feature processed independently with shared parameters
    3. Vectorized velocity computation per feature
    4. Adaptive window sizing (caps at 15 segments by default)
    """
    def __init__(self, lookback, pred_len, n_features, encode_dim, hidden_dim, 
                 curvature, manifold_type, segment_length=24, num_basis=8,
                 use_revin=False, dynamic_dropout=0.3, encode_dropout=0.5, recon_dropout=0.1,
                 num_layers=2, window_size=None, 
                 use_truncated_bptt=True, truncate_every=4):
        """
        Args:
            window_size: int - number of segments to keep in moving window
                        If None, uses adaptive sizing (max 15 segments)
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
        self.recon_dropout = recon_dropout
        self.num_basis = num_basis
        
        # Adaptive window sizing for efficiency
        num_input_segments = lookback // segment_length
        if window_size is not None:
            self.window_size = min(window_size, num_input_segments)
        
        print(f"\n{'='*70}")
        print(f"Channel-Independent Moving Window Hyperbolic Forecaster")
        print(f"{'='*70}")
        print(f"Features: {n_features} (processed independently)")
        print(f"Lookback: {lookback} → {num_input_segments} segments")
        print(f"Segment length: {segment_length}")
        print(f"encode dim: {encode_dim}")
        print(f"Window size: {self.window_size}")
        print(f"Strategy: channel independence + hyperbolic dynamics")
        
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)
        
        # Encoder for SINGLE feature (input_dim=1, shared across all features)
        if manifold_type == "Poincare":
            self.encode_hyperbolic = SegmentedParallelPoincareMovingWindow(
                lookback=lookback,
                encode_dim=encode_dim,
                curvature=curvature,
                segment_length=segment_length,
                encode_dropout=encode_dropout,
            )
        else:
            self.encode_hyperbolic = SegmentedParallelLorentzMovingWindow(
                lookback=lookback,
                encode_dim=encode_dim,
                curvature=curvature,
                segment_length=segment_length,
                encode_dropout=encode_dropout
            )

        
        self.manifold = self.encode_hyperbolic.manifold
        
        # Dynamics that accepts avg_velocity
        if manifold_type == "Poincare":
            self.dynamics = HyperbolicPoincareDynamics(
                encode_dim=encode_dim,
                segment_length=segment_length,
                manifold=self.manifold
            )
        if manifold_type == "Lorentzian":
            self.dynamics = HyperbolicLorentzDynamics(
                encode_dim=encode_dim,
                segment_length=segment_length,
                manifold=self.manifold
            )
    
        self.reconstructor = HyperbolicSegmentReconstructionHead(
            encode_dim=encode_dim,
            output_dim=1,
            segment_length=segment_length,  # NEW
            manifold=self.manifold,
            hidden_dim=hidden_dim,
            dropout=recon_dropout,
        )


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
        if self.window_size == None:
            return None
        z_start = z_current[:, :-1, :]
        z_end = z_current[:, 1:, :]
        
        z_start_flat = z_start.reshape(B * (N-1), D)
        z_end_flat = z_end.reshape(B * (N-1), D)
        
        velocities_flat = self.manifold.logmap(z_start_flat, z_end_flat)
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
            trend_f, coarse_f, fine_f, resid_f: [B, seq_len] single feature
        
        Returns:
            dict with predictions for this feature: [B, pred_len]
        """
        B = trend_f.shape[0]
        
        # Add feature dimension: [B, seq_len] → [B, seq_len, 1]
        trend_f = trend_f.unsqueeze(-1)
        coarse_f = coarse_f.unsqueeze(-1)
        fine_f = fine_f.unsqueeze(-1)
        resid_f = resid_f.unsqueeze(-1)
        
        # Encode to hyperbolic segments: [B, num_segments, encode_dim]
        encode_h = self.encode_hyperbolic(trend_f, coarse_f, fine_f, resid_f)
        
        # Initialize moving windows with historical segments
        z_current = encode_h["combined_h"]  # [B, num_segments, encode_dim]
        z_current_trend = encode_h["trend_h"]
        z_current_coarse = encode_h["seasonal_coarse_h"]
        z_current_fine = encode_h["seasonal_fine_h"]
        z_current_resid = encode_h["residual_h"]
        
        # Take last window_size segments
        if z_current.shape[1] > self.window_size:
            z_current = z_current[:, -self.window_size:, :]
        
        # Storage for predicted latents
        latent_z = []
        predictions_norm = []


        # Autoregressive forecasting with moving window
        for seg_step in range(self.num_pred_segments):
            # Compute velocities for ALL components at once (vectorized)
            if (seg_step + 1) % self.truncate_every == 0 and seg_step < self.num_pred_segments - 1:
                z_current = z_current.detach()
       
            # === Combined signal ===
            z_last = z_current[:, -1, :]
            z_prev = z_current[:, -2, :] if z_current.shape[1] > 1 else None
            average_velocity = self.compute_combined_velocity(z_current)
            z_next, _ = self.dynamics(z_last, z_prev, average_velocity)

            latent_z.append(z_next)
            
            # Update window: drop oldest, add newest
            z_current = torch.cat([z_current[:, 1:, :], z_next.unsqueeze(1)], dim=1)
            x_pred_norm_seg = self.reconstructor(z_next)
            predictions_norm.append(x_pred_norm_seg)

        # Stack predicted segments
        # Stack latent states
        def stack_latents(lst):
            return torch.stack(lst, dim=1)  # [B, num_segments, encode_dim]


        # Batch decode all segments (output is [B, num_pred_segments, segment_length, 1])
        predictions_norm = torch.stack(predictions_norm, dim=1) 

        # Flatten to [B, pred_len]
        predictions_norm = predictions_norm.reshape(B, self.num_pred_segments * self.segment_length)
        
        return {
            'predictions': predictions_norm,
            'latent_z': stack_latents(latent_z)
        }
    
    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """
        Channel-independent forecasting with moving window trajectory modeling.
        
        Args:
            trend, seasonal_coarse, seasonal_fine, residual: [B, seq_len, n_features]
        
        Returns:
            dict with predictions for ALL features: [B, pred_len, n_features]
        """
        
        # Step 1: Combine to get normalization statistics
        x_combined = trend + seasonal_coarse + seasonal_fine + residual
        B, L, F = x_combined.shape
        # Step 2: Get normalization stats and normalize each component
        if self.use_revin:
            # Compute stats from combined signal (stores mean/stdev in self.revin)
            _ = self.revin(x_combined, mode='norm')
            
            # Apply same normalization to each component
            trend = self._normalize_component(trend)
            seasonal_coarse = self._normalize_component(seasonal_coarse)
            seasonal_fine = self._normalize_component(seasonal_fine)
            residual = self._normalize_component(residual)
            # print(f"After norm - trend: mean={trend.mean():.4f}, std={trend.std():.4f}")
        
        # Step 3: Collapse features into batch axis: (B, L, F) -> (B*F, L)
        def collapse(x):
            # x: [B, L, F] -> [B*F, L]
            x = x.permute(0, 2, 1).contiguous()  # [B, F, L]
            x = x.view(B * F, L)   # [B*F, L]
            return x
        trend_b = collapse(trend)
        coarse_b = collapse(seasonal_coarse)
        fine_b = collapse(seasonal_fine)
        resid_b = collapse(residual)
        
        # Step 4: Process through hyperbolic model
        batched_out = self.process_batched_features(trend_b, coarse_b, fine_b, resid_b)
        predictions_norm = batched_out['predictions']  # [B*F, pred_len]

        # Apply RevIN denormalization
        if self.use_revin:
            predictions_norm = predictions_norm.view(B, F, -1).permute(0, 2, 1).contiguous()  # [B, pred_len, F]
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm.view(B, F, -1).permute(0, 2, 1).contiguous()
        
        # Repackage other outputs (latent states) to shapes similar to your original output
        # latent_z: [B*F, num_pred_segments, encode_dim] -> [B, F, num_pred_segments, encode_dim]
        def uncollapse_latent(tensor_b):
            # tensor_b: [B*F, num_pred_segments, D] -> [B, F, num_pred_segments, D]
            Bf, S, D = tensor_b.shape
            return tensor_b.view(B, F, S, D)

        return {
            'predictions': predictions,
            'hyperbolic_states': {
                "combined_h": uncollapse_latent(batched_out["latent_z"]),
            }
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
