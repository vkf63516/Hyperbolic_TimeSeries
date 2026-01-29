import sys
import torch
import torch.nn as nn
import geoopt
from encode.Linear.segment_linear_encode_poincare import SegmentedParallelPoincare
from encode.Linear.segment_linear_encode_lorentz import SegmentedParallelLorentz
from DynamicsMvar.Poincare_Residual_Dynamics import HyperbolicPoincareDynamics
from DynamicsMvar.Lorentz_Residual_Dynamics import HyperbolicLorentzDynamics
from Lifting.hyperbolic_segment_reconstructor import HyperbolicSegmentReconstructionHead  # NEW
from spec import RevIN, safe_expmap, compute_hierarchical_loss_with_manifold_dist


class SegmentedHyperbolicForecaster(nn.Module):
    """
    Segment-aware hyperbolic forecaster.
    
    Key improvements over point-level:
    1. Encodes sequences as segments (e.g., daily patterns in hourly data)
    2. Forecasts entire segments at once (not individual points)
    3. Maintains segment structure throughout encoding → dynamics → reconstruction
    """
    def __init__(self, lookback, pred_len, n_features, encode_dim, hidden_dim, 
                 curvature, manifold_type, segment_length=24, 
                 use_attention_pooling=False, use_revin=False,
                 use_truncated_bptt=False, truncate_every=4, window_size=2,  # Truncate every N segments
                 dynamic_dropout=0.3, encode_dropout=0.5, recon_dropout=0.2, 
                 num_layers=2, share_feature_weights=True):
        """
        Args:
            lookback: int - lookback window (should be divisible by segment_length)
            pred_len: int - prediction horizon (should be divisible by segment_length)
            n_features: int - number of input features
            encode_dim: int - hyperbolic encodeding dimension
            hidden_dim: int - hidden dimension for MLPs
            curvature: float - manifold curvature
            manifold_type: str - "Poincare" or "Lorentzian"
            segment_length: int - length of each segment (e.g., 24 for daily in hourly data)
            use_attention_pooling: bool - attention pooling over segments during encoding
            use_revin: bool - use reversible instance normalization
            use_truncated_bptt: bool - truncated backprop through time
            truncate_every: int - truncate gradient every N SEGMENTS (not steps!)
            dynamic_dropout: float - dropout in dynamics
            encode_dropout: float - dropout in encodeder
            recon_dropout: float - dropout in reconstructor
            num_layers: int - number of layers
            use_segment_norm: bool - normalize each segment independently
        """
        super().__init__()

        # Validate that lookback and pred_len are divisible by segment_length
        if lookback % segment_length != 0:
            print(f"Warning: lookback ({lookback}) not divisible by segment_length ({segment_length}). "
                  f"Will pad to {(lookback // segment_length + 1) * segment_length}")
        if pred_len % segment_length != 0:
            raise ValueError(f"pred_len ({pred_len}) must be divisible by segment_length ({segment_length})")

        self.lookback = lookback
        self.encode_dim = encode_dim
        self.pred_len = pred_len
        self.n_features = n_features
        self.segment_length = segment_length
        self.num_pred_segments = pred_len // segment_length  # NEW: forecast in segments
        self.use_revin = use_revin
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        self.hidden_dim = hidden_dim
        self.manifold_type = manifold_type
        self.encode_dropout = encode_dropout
        self.dynamic_dropout = dynamic_dropout
        self.num_layers=num_layers
        num_input_segments = lookback // segment_length
        if window_size is not None:
            self.window_size = min(window_size, num_input_segments)
        else:
        # Adaptive:  cap at 15 segments for efficiency
            self.window_size = min(15, num_input_segments)
        print(f"Window size: {self.window_size} segments")
        self.share_feature_weights = share_feature_weights
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)
        
        # Segmented encoder
        if manifold_type == "Poincare":
            self.encode_hyperbolic = SegmentedParallelPoincare(
                lookback=lookback,
                input_dim=n_features,
                encode_dim=encode_dim,
                curvature=curvature,
                segment_length=segment_length,
                encode_dropout=self.encode_dropout,
                share_feature_weights=self.share_feature_weights,
            )
        elif manifold_type == "Lorentzian":  # Lorentzian
            self.encode_hyperbolic = SegmentedParallelLorentz(
                lookback=lookback,
                input_dim=n_features,
                encode_dim=encode_dim,
                curvature=curvature,
                segment_length=segment_length,
                encode_dropout=self.encode_dropout,
                share_feature_weights=self.share_feature_weights
            )
        self.manifold = self.encode_hyperbolic.manifold
        self.dynamics = self._create_dynamics()
    
        self.reconstructor = HyperbolicSegmentReconstructionHead(
            encode_dim=encode_dim,
            output_dim=n_features,
            segment_length=segment_length,  # NEW
            manifold=self.manifold,
            manifold_type=self.manifold_type,
            hidden_dim=hidden_dim,
            dropout=recon_dropout
        )
        self.mobius_weights = nn.Parameter(torch.ones(4) * 0.25)


    def _create_dynamics(self):
        if self.manifold_type == "Poincare":
            return HyperbolicPoincareDynamics(
                encode_dim=self.encode_dim,
                manifold=self.manifold
            )
        if self.manifold_type == "Lorentzian":
            return HyperbolicLorentzDynamics(
                encode_dim=self.encode_dim,
                manifold=self.manifold
            )

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

        
    def compute_combined_velocity_incremental(self, z_current, cached_velocities=None, decay=0.9):
        """
        Incremental velocity computation for moving window. 
        
        Args:
            z_current: [B, N, D] - current window of segments
            cached_velocities: [B, N-1, D] - velocities from previous step
            decay:  float - exponential decay for weighting recent velocities
        
        Returns: 
            avg_velocity: [B, D] - weighted average velocity
            new_cached_velocities: [B, N-1, D] - updated cache
        """
        B, N, D = z_current.shape
        
        if N < 2:
            avg_velocity = torch.zeros(B, D, device=z_current.device, dtype=z_current.dtype)
            cached_velocities_out = torch.zeros(B, 0, D, device=z_current.device, dtype=z_current.dtype)
            return avg_velocity, cached_velocities_out
        
        if cached_velocities is None: 
            # First call:  compute all velocities
            z_start = z_current[:, :-1, :]  # [B, N-1, D]
            z_end = z_current[:, 1:, :]     # [B, N-1, D]
            z_start_flat = z_start.reshape(B * (N-1), D)
            z_end_flat = z_end.reshape(B * (N-1), D)
            velocities_flat = self.manifold.logmap(z_start_flat, z_end_flat)
            velocities = velocities_flat.view(B, N-1, D)
        else:
            # Incremental:  drop oldest, add newest
            z_prev = z_current[:, -2, :]  # [B, D]
            z_last = z_current[:, -1, :]  # [B, D]
            new_velocity = self.manifold.logmap(z_prev, z_last)  # [B, D]
            
            # Move window:  remove oldest velocity, append new
            velocities = torch.cat([
                cached_velocities[:, 1:, :],  # Drop first
                new_velocity.unsqueeze(1)      # Add new
            ], dim=1)
        
        # Compute weighted average
        if decay == 1.0:
            avg_velocity = velocities.mean(dim=1)
        else:
            indices = torch.arange(N-1, dtype=torch.float32, device=z_current.device)
            weights = decay ** (N - 2 - indices)  # Recent gets higher weight
            weights = weights / weights.sum()
            avg_velocity = (velocities * weights.view(1, -1, 1)).sum(dim=1)
        
        return avg_velocity, velocities


    def forward(self, trend, seasonal_coarse, seasonal_fine, residual):
        """Segment-level forecasting with moving window."""
        
        # Encode (same as before)
        x_combined = trend + seasonal_coarse + seasonal_fine + residual
        
        if self.use_revin:
            self.revin(x_combined, mode='norm')
        
        encode_h = self.encode_hyperbolic(trend, seasonal_coarse, seasonal_fine, residual)
        
        # NEW: Stack components for batched processing [5, B, num_segments, encode_dim]
        z_components = torch.stack([
            encode_h["trend_h"].unsqueeze(1),
            encode_h["seasonal_coarse_h"].unsqueeze(1),
            encode_h["seasonal_fine_h"].unsqueeze(1),
            encode_h["residual_h"].unsqueeze(1)
        ], dim=0)
        # NEW: Initialize moving window (take last window_size segments)
        if z_components.shape[2] > self.window_size:
            z_components = z_components[:, :, -self.window_size:, :]
        
        # Storage
        predictions_norm = []
        trend_predictions = []
        coarse_predictions = []
        fine_predictions = []
        residual_predictions = []
        
        latent_z = []
        latent_trend = []
        latent_coarse = []
        latent_fine = []
        latent_resid = []
        hierarchy_losses = []

        cached_velocities = None  # For incremental velocity computation
        
        # NEW:  Autoregressive rollout with moving window
        for seg_step in range(self.num_pred_segments):
            # Truncated BPTT
            if (seg_step + 1) % self.truncate_every == 0 and seg_step < self.num_pred_segments - 1:
                z_components = z_components.detach()
                cached_velocities = None
            
            num_comp, B, N, D = z_components.shape
            
            # Flatten for batched processing:  [5, B, N, D] → [5*B, N, D]
            z_comp_flat = z_components.reshape(num_comp * B, N, D)
            
            # Flatten cached velocities
            if cached_velocities is not None:
                cache_flat = cached_velocities.reshape(num_comp * B, N-1, D)
            else:
                cache_flat = None
            
            # Compute velocities for ALL components in one call
            avg_velocity_flat, new_cache_flat = self.compute_combined_velocity_incremental(
                z_comp_flat, cache_flat
            )
            
            # Reshape cache back
            cached_velocities = new_cache_flat.reshape(num_comp, B, N-1, D)
            
            # Extract states
            z_last_flat = z_comp_flat[:, -1, :]   # [5*B, D]
            z_prev_flat = z_comp_flat[:, -2, :] if N > 1 else None
            
            # Dynamics with velocity
            z_next_flat, _ = self.dynamics(z_last_flat, z_prev_flat, avg_velocity_flat)
            
            # Update window:  drop oldest, add newest
            z_comp_flat = torch.cat([
                z_comp_flat[:, 1:, :],       # Drop first segment
                z_next_flat.unsqueeze(1)     # Add new prediction
            ], dim=1)
            
            # Reshape back:  [5*B, N, D] → [5, B, N, D]
            z_components = z_comp_flat.reshape(num_comp, B, N, D)
            
            # Unstack for reconstruction
            z_next_all = z_next_flat.reshape(num_comp, B, D)
            z_current_trend = z_next_all[0]
            z_current_coarse = z_next_all[1]
            z_current_fine = z_next_all[2]
            z_current_resid = z_next_all[3]
            encodings_dict = {
                "trend_h": z_current_trend,
                "seasonal_coarse_h": z_current_coarse,
                "seasonal_fine_h": z_current_fine,
                "residual_h": z_current_resid
            }
            hierarchy_loss = compute_hierarchical_loss_with_manifold_dist(encodings_dict, self.manifold)
            hierarchy_losses.append(hierarchy_loss)
            z_fused = self.mobius_fusion_segments(z_current_trend, z_current_coarse, z_current_fine, z_current_resid)
            
            # Store latents
            # latent_z.append(z_fused)
            # latent_trend.append(z_current_trend)
            # latent_coarse.append(z_current_coarse)
            # latent_fine.append(z_current_fine)
            # latent_resid.append(z_current_resid)
            
            # Reconstruct segments
            predictions_norm.append(self.reconstructor(z_fused))
           
        # Stack and reshape (same as before)
        predictions_norm = torch.stack(predictions_norm, dim=1).reshape(-1, self.pred_len, self.n_features)
      
        
        if self.use_revin:
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm
        
        average_hierarchy_loss = torch.mean(torch.stack(hierarchy_losses)) if hierarchy_losses else torch.tensor(0.0)
        # ---- NEW: Stack hyperbolic latent states ----
        def stack_latents(lst):
            segs = torch.stack(lst, dim=1)  # [B, num_segments, encode_dim+1]
            return segs

        hyperbolic_states = {
            "combined_h": stack_latents(latent_z),
            "trend_h":    stack_latents(latent_trend),
            "coarse_h":   stack_latents(latent_coarse),
            "fine_h":     stack_latents(latent_fine),
            "resid_h":    stack_latents(latent_resid),
        }


        return {
            'predictions': predictions,  # [B, pred_len, n_features]
            'hyperbolic_states': hyperbolic_states,
            'hierarchy_loss': average_hierarchy_loss
        }