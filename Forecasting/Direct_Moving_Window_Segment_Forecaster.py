import torch
import torch.nn as nn
from encode.Moving_Window.moving_segment_linear_encode_poincare_direct import DirectPoincareMovingWindow
from DynamicsMvar. Poincare_Residual_Dynamics import HyperbolicPoincareDynamics
from Lifting. hyperbolic_segment_reconstructor import HyperbolicSegmentReconstructionHead
from spec import RevIN


class DirectHyperbolicForecaster(nn.Module):
    """
    ABLATION: No decomposition baseline. 
    Raw signal → hyperbolic encoding → dynamics → reconstruction
    """
    
    def __init__(self, lookback, pred_len, n_features, encode_dim, hidden_dim,
                 curvature, manifold_type, segment_length=24,
                 use_revin=False, encode_dropout=0.5, 
                 recon_dropout=0.1, window_size=None, 
                 use_truncated_bptt=True, truncate_every=4):
        super().__init__()
        
        if pred_len % segment_length != 0:
            raise ValueError(f"pred_len must be divisible by segment_length")
        
        self.lookback = lookback
        self. encode_dim = encode_dim
        self.pred_len = pred_len
        self.n_features = n_features
        self.segment_length = segment_length
        self.num_pred_segments = pred_len // segment_length
        self.use_revin = use_revin
        self.manifold_type = manifold_type
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        self.window_size = window_size
        num_input_segments = lookback // segment_length
        print(f"\n{'='*70}")
        print(f"ABLATION: Direct Hyperbolic Forecaster (No Decomposition)")
        print(f"{'='*70}")
        print(f"Features: {n_features}")
        print(f"Lookback:  {lookback}")
        print(f"Segment length: {segment_length}")
        print(f"Encode dim: {encode_dim}")
        print(f"Strategy: Raw signal → Hyperbolic encoding")
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
        
        # SINGLE encoder (no decomposition)
        self.encode_hyperbolic = DirectPoincareMovingWindow(
            lookback=lookback,
            encode_dim=encode_dim,
            curvature=curvature,
            segment_length=segment_length,
            encode_dropout=encode_dropout,
            num_channels=n_features
        )
        
        self.manifold = self.encode_hyperbolic.manifold
        
        # Same dynamics as decomposed version
        self.dynamics = HyperbolicPoincareDynamics(
            encode_dim=encode_dim,
            manifold=self.manifold
        )
        
        # Same reconstructor
        self.reconstructor = HyperbolicSegmentReconstructionHead(
            encode_dim=encode_dim,
            output_dim=1,
            segment_length=segment_length,
            manifold_type=self.manifold_type,
            manifold=self.manifold,
            hidden_dim=hidden_dim,
            dropout=recon_dropout,
        )
    
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
            z_end_flat = z_end. reshape(B * (N-1), D)
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
    def process_batched_features(self, x_raw):
        """
        Process raw signal (no decomposition).
        
        Args:
            x_raw: [B, seq_len] - raw signal
        
        Returns:
            dict with predictions:  [B, pred_len]
        """
        B = x_raw.shape[0]
        
        # Add feature dimension:  [B, seq_len] → [B, seq_len, 1]
        x_raw = x_raw.unsqueeze(-1)
        
        # Encode directly (no decomposition)
        encode_h = self.encode_hyperbolic(x_raw)  # [B, num_segments, encode_dim]
        
        # Initialize moving window
        z_current = encode_h["combined_h"]
        
        # Take last window_size segments
        if self.window_size and z_current.shape[1] > self.window_size:
            z_current = z_current[: , -self.window_size:, :]
        
        latent_z = []
        predictions_norm = []
        cached_velocities = None

        
        # Same autoregressive forecasting loop
        for seg_step in range(self.num_pred_segments):
            if (seg_step + 1) % self.truncate_every == 0 and seg_step < self.num_pred_segments - 1:
                z_current = z_current.detach()
                cached_velocities = None
            
            z_last = z_current[:, -1, :]
            z_prev = z_current[:, -2, : ] if z_current.shape[1] > 1 else None
            average_velocity, cached_velocities = self.compute_combined_velocity_incremental(
                z_current, cached_velocities
            )            
            z_next, _ = self.dynamics(z_last, z_prev, average_velocity)
            
            latent_z.append(z_next)
            z_current = torch.cat([z_current[: , 1:, :], z_next.unsqueeze(1)], dim=1)
            x_pred_norm_seg = self.reconstructor(z_next)
            predictions_norm.append(x_pred_norm_seg)
        
        predictions_norm = torch.stack(predictions_norm, dim=1)
        predictions_norm = predictions_norm.reshape(B, self.num_pred_segments * self.segment_length)
        
        return {
            'predictions': predictions_norm,
            'latent_z': torch.stack(latent_z, dim=1)
        }
    
    def forward(self, x):
        """
        Args: 
            x: [B, seq_len, n_features] - raw input (NO decomposition)
        
        Returns:
            dict with predictions:  [B, pred_len, n_features]
        """
        B, L, F = x.shape
        
        # RevIN normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        # Collapse features into batch:  [B, L, F] → [B*F, L]
        x_b = x.permute(0, 2, 1).contiguous().view(B * F, L)
        
        # Process through hyperbolic model (no decomposition)
        batched_out = self.process_batched_features(x_b)
        predictions_norm = batched_out['predictions']  # [B*F, pred_len]
        
        # Denormalize
        if self.use_revin:
            predictions_norm = predictions_norm.view(B, F, -1).permute(0, 2, 1).contiguous()
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm.view(B, F, -1).permute(0, 2, 1).contiguous()
        
        # Uncollapse latent states
        def uncollapse_latent(tensor_b):
            Bf, S, D = tensor_b.shape
            return tensor_b.view(B, F, S, D)
        
        return {
            'predictions': predictions,
            'hyperbolic_states': {
                "combined_h": uncollapse_latent(batched_out["latent_z"]),
            }
        }
