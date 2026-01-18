import torch
import torch.nn as nn
from encode.Moving_Window.moving_segment_linear_encode_poincare import SegmentedParallelPoincareMovingWindow
from encode.Moving_Window.moving_segment_linear_encode_lorentz import SegmentedParallelLorentzMovingWindow
from encode.Linear.segment_linear_encode_lorentz import SegmentedParallelLorentz
from DynamicsMvar.Poincare_Residual_Dynamics import HyperbolicPoincareDynamics
from DynamicsMvar.Lorentz_Residual_Dynamics import HyperbolicLorentzDynamics
from Lifting.hyperbolic_segment_reconstructor import HyperbolicSegmentReconstructionHead
from spec import RevIN, safe_expmap, safe_expmap0


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
                 curvature, manifold_type, segment_length=24,
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
                num_channels=self.n_features
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
                manifold=self.manifold
            )
        if manifold_type == "Lorentzian":
            self.dynamics = HyperbolicLorentzDynamics(
                encode_dim=encode_dim,
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
        self.mobius_weights = nn.Parameter(torch.ones(4) * 0.25)
        self.lorentz_weights = nn.Parameter(torch.ones(4) * 0.25)



    def mobius_fusion_segments(self, z_next_trend, z_next_coarse, z_next_fine, z_next_resid):
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

    def weighted_lorentz_mean(self, points, weights):
        """
        Compute weighted Einstein midpoint (Fréchet mean) in Lorentz manifold.
        This is the proper way to combine multiple points in hyperbolic space.
        
        Uses iterative algorithm to find the point that minimizes weighted distances.
        
        Args:
            points: list of [B, encode_dim+1] - points on Lorentz manifold
            weights: [num_points] - normalized weights
        
        Returns:
            mean_point: [B, encode_dim+1] - weighted mean on manifold
        """
        # Initialize at weighted tangent space mean
        tangents = [self.manifold.logmap0(p) for p in points]
        weighted_tangent = sum(w * t for w, t in zip(weights, tangents))
        current_mean = safe_expmap0(self.manifold, weighted_tangent)
        current_mean = self.manifold.projx(current_mean)
        
        # Iterative refinement (Karcher flow)
        # Usually converges in 5-10 iterations
        for _ in range(10):
            # Compute tangent vectors from current mean to each point
            tangent_vecs = [self.manifold.logmap(current_mean, p) for p in points]
            
            # Weighted sum in tangent space at current mean
            weighted_vec = sum(w * v for w, v in zip(weights, tangent_vecs))
            
            # Check convergence
            if torch.norm(weighted_vec, dim=-1).max() < 1e-5:
                break
            
            # Move along weighted direction
            current_mean = safe_expmap(self.manifold, current_mean, 0.5 * weighted_vec)
            current_mean = self.manifold.projx(current_mean)
        
        return current_mean

    def lorentz_fusion(self, z_next_trend, z_next_coarse, z_next_fine, z_next_resid):
        """
        Non-hierarchical fusion using weighted Einstein midpoint in Lorentz space.
        This is the geometrically correct way to combine points in hyperbolic space.
        
        Args:
            z_next_trend, z_next_coarse, z_next_fine, z_next_resid: [B, encode_dim+1]
        
        Returns:
            combined_h: [B, encode_dim+1] - combined point on manifold
            combined_tangent: [B, encode_dim] - tangent vector representation at origin
        """
        # Normalize weights to sum to 1
        weights = torch.softmax(self.lorentz_weights, dim=0)
        
        # Collect all points
        points = [z_next_trend, z_next_coarse, z_next_fine, z_next_resid]
        
        # Compute weighted Einstein midpoint
        combined_h = self.weighted_lorentz_mean(points, weights)
                
        return combined_h


    def compute_combined_velocity_incremental(self, z_current, cached_velocities=None, decay=0.9):
        """
        Incremental velocity computation for moving window.
        
        Returns:
            avg_velocity: [B, D]
            new_cached_velocities: [B, N-1, D] to pass to next step
        """
        B, N, D = z_current.shape
        
        if cached_velocities is None: 
            # First call:  compute all velocities
            z_start = z_current[:, :-1, :]
            z_end = z_current[: , 1:, :]
            z_start_flat = z_start.reshape(B * (N-1), D)
            z_end_flat = z_end.reshape(B * (N-1), D)
            velocities_flat = self.manifold.logmap(z_start_flat, z_end_flat)
            velocities = velocities_flat.view(B, N-1, D)
        else:
            # Incremental:  drop oldest, add newest
            # cached_velocities:  [B, N-1, D] from previous window
            # Only compute NEW velocity:  z_current[-2] → z_current[-1]
            z_prev = z_current[:, -2, :]  # [B, D]
            z_last = z_current[:, -1, :]  # [B, D]
            new_velocity = self.manifold.logmap(z_prev, z_last)  # [B, D]
            
            # Move window: remove oldest velocity (index 0), append new
            velocities = torch.cat([
                cached_velocities[:, 1:, : ],  # Drop first velocity
                new_velocity.unsqueeze(1)      # Add new velocity
            ], dim=1)  # [B, N-1, D]
        
        # Compute weighted average (same as before)
        if decay == 1.0:
            avg_velocity = velocities.mean(dim=1)
        else:
            indices = torch.arange(N-1, dtype=torch.float32, device=z_current.device)
            weights = decay ** (N - 2 - indices)  # Recent gets higher weight
            weights = weights / weights.sum()
            avg_velocity = (velocities * weights.view(1, -1, 1)).sum(dim=1)
        
        return avg_velocity, velocities
        
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
        # [B, num_segments, encode_dim]
        z_current_trend = encode_h["trend_h"]
        z_current_coarse = encode_h["seasonal_coarse_h"]
        z_current_fine = encode_h["seasonal_fine_h"]
        z_current_resid = encode_h["residual_h"]
        
        # Take last window_size segments
        if z_current_trend.shape[1] > self.window_size:
            z_current_trend = z_current_trend[:, -self.window_size:, :]
        if z_current_coarse.shape[1] > self.window_size:
            z_current_coarse = z_current_coarse[:, -self.window_size:, :]
        if z_current_fine.shape[1] > self.window_size:
            z_current_fine = z_current_fine[:, -self.window_size:, :]
        if z_current_resid.shape[1] > self.window_size:
            z_current_resid = z_current_resid[:, -self.window_size:, :]
        
        # Storage for predicted latents
        latent_z = []
        predictions_norm = []
        cached_velocities_trend = None
        cached_velocities_coarse = None
        cached_velocities_fine = None 
        cached_velocities_resid = None
        # Autoregressive forecasting with moving window
        for seg_step in range(self.num_pred_segments):
            # Compute velocities for ALL components at once (vectorized)
            if (seg_step + 1) % self.truncate_every == 0 and seg_step < self.num_pred_segments - 1:
                z_current_trend = z_current_trend.detach()
                z_current_coarse = z_current_coarse.detach()
                z_current_fine = z_current_fine.detach()
                z_current_resid = z_current_resid.detach()
                cached_velocities_trend = None
                cached_velocities_coarse = None
                cached_velocities_fine = None
                cached_velocities_resid = None
       
            # === Trend signal ===
            z_last_trend = z_current_trend[:, -1, :]
            z_prev_trend = z_current_trend[:, -2, :] if z_current_trend.shape[1] > 1 else None
            average_velocity_trend, cached_velocities_trend = self.compute_combined_velocity_incremental(
                z_current_trend, cached_velocities_trend
            )
            z_next_trend, _ = self.dynamics(z_last_trend, z_prev_trend, average_velocity_trend)
            z_current_trend = torch.cat([z_current_trend[:, 1:, :], z_next_trend.unsqueeze(1)], dim=1)

            z_last_coarse = z_current_coarse[:, -1, :]
            z_prev_coarse = z_current_coarse[:, -2, :] if z_current_coarse.shape[1] > 1 else None
            average_velocity_coarse, cached_velocities_coarse = self.compute_combined_velocity_incremental(
                z_current_coarse, cached_velocities_coarse
            )
            z_next_coarse, _ = self.dynamics(z_last_coarse, z_prev_coarse, average_velocity_coarse)
            z_current_coarse = torch.cat([z_current_coarse[:, 1:, :], z_next_coarse.unsqueeze(1)], dim=1)

            z_last_fine = z_current_fine[:, -1, :]
            z_prev_fine = z_current_fine[:, -2, :] if z_current_fine.shape[1] > 1 else None
            average_velocity_fine, cached_velocities_fine = self.compute_combined_velocity_incremental(
                z_current_fine, cached_velocities_fine
            )
            z_next_fine, _ = self.dynamics(z_last_fine, z_prev_fine, average_velocity_fine)
            z_current_fine = torch.cat([z_current_fine[:, 1:, :], z_next_fine.unsqueeze(1)], dim=1)

            z_last_resid = z_current_resid[:, -1, :]
            z_prev_resid = z_current_resid[:, -2, :] if z_current_resid.shape[1] > 1 else None
            average_velocity_resid, cached_velocities_resid = self.compute_combined_velocity_incremental(
                z_current_resid, cached_velocities_resid
            )
            z_next_resid, _ = self.dynamics(z_last_resid, z_prev_resid, average_velocity_resid)
            z_current_resid = torch.cat([z_current_resid[:, 1:, :], z_next_resid.unsqueeze(1)], dim=1)
            if self.manifold_type == "Poincare":
                z_fused = self.mobius_fusion_segments(z_next_trend, z_next_coarse, z_next_fine, z_next_resid)
            else:
                z_fused = self.lorentz_fusion(z_next_trend, z_next_coarse, z_next_fine, z_next_resid)

            latent_z.append(z_fused)
        latent_z_stacked = torch.stack(latent_z, dim=1)  # [B*F, num_pred_segments, encode_dim]

        #  Flatten segments into batch dimension
        BF, N, D = latent_z_stacked.shape
        latent_flat = latent_z_stacked.reshape(BF * N, D)  # [B*F*num_pred_segments, encode_dim]

        #  Call reconstructor ONCE
        predictions_flat = self.reconstructor(latent_flat)  # [B*F*N, segment_length]

        #  Reshape back
        predictions_norm = predictions_flat.reshape(BF, N * self.segment_length)  # [B*F, pred_len]
            # Update window: drop oldest, add newest

        # Stack predicted segments
        # Stack latent states
        def stack_latents(lst):
            return torch.stack(lst, dim=1)  # [B, num_segments, encode_dim]


        # Batch decode all segments (output is [B, num_pred_segments, segment_length, 1])
        # predictions_norm = torch.stack(predictions_norm, dim=1) 

        # Flatten to [B, pred_len]
        # predictions_norm = predictions_norm.reshape(B, self.num_pred_segments * self.segment_length)
        
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
