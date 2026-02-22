import torch
import torch.nn as nn
from encode.Moving_Window.moving_segment_linear_encode_poincare import SegmentedParallelPoincareMovingWindow
from encode.Moving_Window.moving_segment_linear_encode_lorentz import SegmentedParallelLorentzMovingWindow
from encode.Linear.segment_linear_encode_lorentz import SegmentedParallelLorentz
from DynamicsMvar.Poincare_Residual_Dynamics import HyperbolicPoincareDynamics
from DynamicsMvar.Lorentz_Residual_Dynamics import HyperbolicLorentzDynamics
from Lifting.hyperbolic_segment_reconstructor import HyperbolicSegmentReconstructionHead
from spec import RevIN, safe_expmap, safe_expmap0, compute_hierarchical_loss_with_manifold_dist


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
            manifold_type=self.manifold_type,
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

    def weighted_lorentz_mean_segments(self, points, weights):
        """
        ? FIXED: Conservative fusion with multiple fallback strategies.
        """
        B, D_plus_1 = points[0].shape
        
        # Compute weighted tangent
        tangents = [self.manifold.logmap0(p) for p in points]
        weighted_tangent = sum(w * t for w, t in zip(weights, tangents))
        
        # ? FIX 5: Check for NaN/Inf in tangents BEFORE clamping
        if torch.isnan(weighted_tangent).any() or torch.isinf(weighted_tangent).any():
            print(f"?? Invalid tangent at segment {points[0]}, using first point")
            return points[0]
        
        # ? FIX 6: More conservative clamping (5.0 ? 2.0)
        tangent_norm = torch.norm(weighted_tangent, dim=-1, keepdim=True)
        max_norm = 2.0  # Was 5.0
        if (tangent_norm > max_norm).any():
            weighted_tangent = weighted_tangent / tangent_norm * torch.clamp(tangent_norm, max=max_norm)
        
        # Try to map to manifold
        current_mean = safe_expmap0(self.manifold, weighted_tangent)
        current_mean = self.manifold.projx(current_mean)
        
        # ? FIX 7: Check for NaN BEFORE iteration
        if torch.isnan(current_mean).any() or torch.isinf(current_mean).any():
            print(f"?? NaN in initial mean at {points[0]}, using first point")
            return point[0]
        
        # ? FIX 8: Reduced iterations (10 ? 5) with smaller steps
        for iteration in range(5):
            tangent_vecs = [self.manifold.logmap(current_mean, p) for p in points]
            weighted_vec = sum(w * v for w, v in zip(weights, tangent_vecs))
            
            # Early convergence
            if torch.norm(weighted_vec, dim=-1).max() < 1e-5:
                break
            
            # Conservative update
            vec_norm = torch.norm(weighted_vec, dim=-1, keepdim=True)
            if (vec_norm > max_norm).any():
                weighted_vec = weighted_vec / vec_norm * torch.clamp(vec_norm, max=max_norm)
            
            # ? FIX 9: Smaller step size (0.3 ? 0.1)
            next_mean = safe_expmap(self.manifold, current_mean, 0.1 * weighted_vec)
            next_mean = self.manifold.projx(next_mean)
            
            if torch.isnan(next_mean).any():
                break  # Keep current_mean
            
            current_mean = next_mean
        
        return current_mean
    
    def lorentz_fusion(self, z_trend_h, z_coarse_h, z_fine_h, z_residual_h):
        """Fusion across all segment positions."""
        weights = torch.softmax(self.lorentz_weights, dim=0)
        points = [z_trend_h, z_coarse_h, z_fine_h, z_residual_h]
        combined_h = self.weighted_lorentz_mean_segments(points, weights)
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
            z_end = z_current[:, 1:, :]
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
            avg_velocity = (velocities * weights.view(1, -1, 1)).mean(dim=1)
        
        return avg_velocity, velocities
        
    def process_batched_features(self, trend_f, coarse_f, fine_f, resid_f):
        """
        Process a batch of features through the entire forecasting pipeline.
        
        Args:
            trend_f, coarse_f, fine_f, resid_f: [B, seq_len] single feature
        
        Returns:
            dict with predictions for this feature:  [B, pred_len]
        """
        B = trend_f.shape[0]
        
        # Add feature dimension:  [B, seq_len] → [B, seq_len, 1]
        trend_f = trend_f.unsqueeze(-1)
        coarse_f = coarse_f.unsqueeze(-1)
        fine_f = fine_f.unsqueeze(-1)
        resid_f = resid_f.unsqueeze(-1)
        
        # Encode to hyperbolic segments:  [B, num_segments, encode_dim]
        encode_h = self.encode_hyperbolic(trend_f, coarse_f, fine_f, resid_f)
        
        # Initialize moving windows: Stack all components [4, B, num_segments, encode_dim]
        z_components = torch.stack([
            encode_h["trend_h"],
            encode_h["seasonal_coarse_h"],
            encode_h["seasonal_fine_h"],
            encode_h["residual_h"]
        ], dim=0)
        
        # Take last window_size segments
        if z_components.shape[2] > self.window_size:
            z_components = z_components[:, :, -self.window_size:, :]
        
        # Storage for predicted latents
        latent_z = []
        latent_trend = []
        latent_coarse = []
        latent_fine = []
        latent_resid = []
        hierarchy_losses = []
        cached_velocities = None  # Single cache for all components (batched)
        
        # Autoregressive forecasting with moving window
        for seg_step in range(self.num_pred_segments):
            # Truncated BPTT
            if (seg_step + 1) % self.truncate_every == 0 and seg_step < self.num_pred_segments - 1:
                z_components = z_components.detach()
                cached_velocities = None
            
            num_comp, B, N, D = z_components.shape
            
            # === BATCHED Velocity Computation ===
            # Flatten:  [4, B, N, D] → [4*B, N, D]
            z_comp_flat = z_components.reshape(num_comp * B, N, D)
            
            # Flatten cached velocities if they exist
            if cached_velocities is not None:
                cache_flat = cached_velocities.reshape(num_comp * B, N-1, D)
            else:
                cache_flat = None
            
            # Single velocity call for ALL components
            avg_velocity_flat, new_cache_flat = self.compute_combined_velocity_incremental(
                z_comp_flat, cache_flat
            )  # [4*B, D], [4*B, N-1, D]
            
            # Reshape cached velocities back:  [4*B, N-1, D] → [4, B, N-1, D]
            cached_velocities = new_cache_flat.reshape(num_comp, B, N-1, D)
            
            # === BATCHED Dynamics ===
            # Extract last and previous states (already flat from z_comp_flat)
            z_last_flat = z_comp_flat[:, -1, :]  # [4*B, D]
            z_prev_flat = z_comp_flat[:, -2, :] if N > 1 else None  # [4*B, D]
            
            # SINGLE dynamics call for all components
            z_next_flat, _ = self.dynamics(z_last_flat, z_prev_flat, avg_velocity_flat)
            # [4*B, D]
            
            # === Update Windows ===
            # Concatenate new state:  [4*B, N, D] 
            z_comp_flat = torch.cat([
                z_comp_flat[:, 1:, :],  # Drop oldest
                z_next_flat.unsqueeze(1)  # Add newest
            ], dim=1)
            
            # Reshape back:  [4*B, N, D] → [4, B, N, D]
            z_components = z_comp_flat.reshape(num_comp, B, N, D)
            
            # === Unstack for Fusion ===
            # Reshape:  [4*B, D] → [4, B, D]
            z_next_all = z_next_flat.reshape(num_comp, B, D)
            
            z_next_trend = z_next_all[0]    # [B, D]
            z_next_coarse = z_next_all[1]   # [B, D]
            z_next_fine = z_next_all[2]     # [B, D]
            z_next_resid = z_next_all[3]    # [B, D]
            encodings_dict = {
                "trend_h": z_next_trend,
                "seasonal_coarse_h": z_next_coarse,
                "seasonal_fine_h": z_next_fine,
                "residual_h": z_next_resid
            }
            step_hierarchy_loss = compute_hierarchical_loss_with_manifold_dist(
                encodings_dict, 
                manifold=self.manifold,
                margin=0.1
            )
            hierarchy_losses.append(step_hierarchy_loss)
            
            # === Möbius/Lorentz Fusion ===
            if self.manifold_type == "Poincare":
                z_fused = self.mobius_fusion_segments(
                    z_next_trend, z_next_coarse, z_next_fine, z_next_resid
                )
            else:
                z_fused = self.lorentz_fusion(
                    z_next_trend, z_next_coarse, z_next_fine, z_next_resid
                )
            
            latent_z.append(z_fused)  # [B, D]
        
        # === Batched Reconstruction ===
        # Stack all latents: [B, num_pred_segments, encode_dim]
        latent_z_stacked = torch.stack(latent_z, dim=1)
 
        avg_hierarchy_loss = torch.mean(torch.stack(hierarchy_losses)) if hierarchy_losses else torch.tensor(0.0)
        
        # Flatten segments into batch dimension
        BF, N, D = latent_z_stacked.shape
        latent_flat = latent_z_stacked.reshape(BF * N, D)  # [B*num_pred_segments, encode_dim]
        
        # Call reconstructor ONCE
        predictions_flat = self.reconstructor(latent_flat)  # [B*N, segment_length]
        
        # Reshape back
        predictions_norm = predictions_flat.reshape(BF, N * self.segment_length)  # [B, pred_len]
        
        return {
            'predictions': predictions_norm,
            'latent_z': latent_z_stacked,  # Already stacked! 
            'hierarchy_loss': avg_hierarchy_loss
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
            },
            'hierarchy_loss': batched_out['hierarchy_loss']
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
