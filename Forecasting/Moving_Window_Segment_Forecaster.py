import torch
import torch.nn as nn
from embed.Moving_Window.moving_segment_linear_embed_poincare import SegmentedParallelPoincareMovingWindow
from embed.Linear.segment_linear_embed_lorentz import SegmentedParallelLorentz
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
    def __init__(self, lookback, pred_len, n_features, embed_dim, hidden_dim, 
                 curvature, manifold_type, segment_length=24, 
                 use_revin=False, dynamic_dropout=0.3, embed_dropout=0.5,
                 num_layers=2, use_segment_norm=True, window_size=None, 
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
        self.embed_dim = embed_dim
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
            # Cap window size for efficiency while preserving performance
            self.window_size = min(num_input_segments, 15)  # Max 15 segments
        else:
            self.window_size = min(window_size, num_input_segments)
        
        print(f"\n{'='*70}")
        print(f"Channel-Independent Moving Window Hyperbolic Forecaster")
        print(f"{'='*70}")
        print(f"Features: {n_features} (processed independently)")
        print(f"Lookback: {lookback} → {num_input_segments} segments")
        print(f"Segment length: {segment_length}")
        print(f"Embed dim: {embed_dim}")
        print(f"Window size: {self.window_size}")
        print(f"Strategy: channel independence + hyperbolic dynamics")
        
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)
        
        # Encoder for SINGLE feature (input_dim=1, shared across all features)
        if manifold_type == "Poincare":
            self.embed_hyperbolic = SegmentedParallelPoincareMovingWindow(
                lookback=lookback,
                embed_dim=embed_dim,
                curvature=curvature,
                segment_length=segment_length,
                use_segment_norm=use_segment_norm,
                embed_dropout=embed_dropout,
            )
        
        self.manifold = self.embed_hyperbolic.manifold
        
        # Dynamics that accepts avg_velocity
        if manifold_type == "Poincare":
            self.dynamics = HyperbolicPoincareDynamics(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                manifold=self.manifold,
                dropout=dynamic_dropout,
                n_layers=num_layers
            )
        # elif manifold_type == "Lorentzian":
        #     self.dynamics = HyperbolicLorentzDynamics(
        #         embed_dim=embed_dim,
        #         manifold=self.manifold,
        #         dropout=dynamic_dropout,
        #         n_layers=num_layers
        #     )
        self.reconstructor = HyperbolicSegmentReconstructionHead(
            embed_dim=embed_dim,
            output_dim=1,
            segment_length=segment_length,  # NEW
            manifold=self.manifold,
            hidden_dim=hidden_dim,
            dropout=0.2
        )
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Parameters per feature: {total_params:,} (shared!)")
        print(f"{'='*70}\n")
    
    def compute_all_component_velocities(self, z_current, z_trend, z_coarse, z_fine, z_resid, decay=0.9):
        """
        Compute velocities for all components for a SINGLE feature (OPTIMIZED).
        
        Args:
            All args: [B, window_size, embed_dim] (single feature)
        
        Returns:
            dict with velocities for each component
        """
        B, N, D = z_current.shape
        
        # Stack all components: [B, 5, N, D]
        z_all = torch.stack([z_current, z_trend, z_coarse, z_fine, z_resid], dim=1)
        
        # Get all consecutive pairs: [B, 5, N-1, D]
        z_all_start = z_all[:, :, :-1, :]
        z_all_end = z_all[:, :, 1:, :]
        
        # Flatten for batch logmap: [B*5*(N-1), D]
        num_components = 5
        z_all_start_flat = z_all_start.reshape(B * num_components * (N-1), D)
        z_all_end_flat = z_all_end.reshape(B * num_components * (N-1), D)
        
        # Single batched logmap for ALL components at once!
        velocities_flat = self.manifold.logmap(z_all_start_flat, z_all_end_flat)
        
        # Reshape: [B, 5, N-1, D]
        velocities = velocities_flat.view(B, num_components, N-1, D)

        # Exponential Moving Average
        weights = torch.tensor([decay**i for i in range(N-2, -1, -1)], device=velocities.device)
        weights = weights / weights.sum()
        # Average across trajectory for each component: [B, 5, D]
        avg_velocities = (velocities * weights.view(1, 1, -1, 1)).sum(dim=2)
        
        # Return as dict
        return {
            'combined': avg_velocities[:, 0, :],
            'trend': avg_velocities[:, 1, :],
            'coarse': avg_velocities[:, 2, :],
            'fine': avg_velocities[:, 3, :],
            'resid': avg_velocities[:, 4, :],
        }
    
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
        
        # Encode to hyperbolic segments: [B, num_segments, embed_dim]
        embed_h = self.embed_hyperbolic(trend_f, coarse_f, fine_f, resid_f)
        
        # Initialize moving windows with historical segments
        z_current = embed_h["combined_h"]  # [B, num_segments, embed_dim]
        z_current_trend = embed_h["trend_h"]
        z_current_coarse = embed_h["seasonal_coarse_h"]
        z_current_fine = embed_h["seasonal_fine_h"]
        z_current_resid = embed_h["residual_h"]
        
        # Take last window_size segments
        if z_current.shape[1] > self.window_size:
            z_current = z_current[:, -self.window_size:, :]
            z_current_trend = z_current_trend[:, -self.window_size:, :]
            z_current_coarse = z_current_coarse[:, -self.window_size:, :]
            z_current_fine = z_current_fine[:, -self.window_size:, :]
            z_current_resid = z_current_resid[:, -self.window_size:, :]
        
        # Storage for predicted latents
        latent_z = []
        latent_trend = []
        latent_coarse = []
        latent_fine = []
        latent_resid = []
        trend_predictions = []
        coarse_predictions = []
        fine_predictions = []
        residual_predictions = []
        predictions_norm = []


        # Autoregressive forecasting with moving window
        for seg_step in range(self.num_pred_segments):
            # Compute velocities for ALL components at once (vectorized)
            velocities = self.compute_all_component_velocities(
                z_current, z_current_trend, z_current_coarse, 
                z_current_fine, z_current_resid
            )
            if (seg_step + 1) % self.truncate_every == 0 and seg_step < self.num_pred_segments - 1:
                z_current = z_current.detach()
                z_current_trend = z_current_trend.detach()
                z_current_coarse = z_current_coarse.detach()
                z_current_fine = z_current_fine.detach()
                z_current_resid = z_current_resid.detach()
            
       
            # === Combined signal ===
            z_last = z_current[:, -1, :]
            z_prev = z_current[:, -2, :] if z_current.shape[1] > 1 else None
            z_next, _ = self.dynamics(z_last, z_prev, velocities['combined'])
            latent_z.append(z_next)
            
            # Update window: drop oldest, add newest
            z_current = torch.cat([z_current[:, 1:, :], z_next.unsqueeze(1)], dim=1)
            x_pred_norm_seg = self.reconstructor(z_next)
            predictions_norm.append(x_pred_norm_seg)

            # === Trend ===
            z_last_trend = z_current_trend[:, -1, :]
            z_prev_trend = z_current_trend[:, -2, :] if z_current_trend.shape[1] > 1 else None
            z_next_trend, _ = self.dynamics(z_last_trend, z_prev_trend, velocities['trend'])
            latent_trend.append(z_next_trend)
            z_current_trend = torch.cat([z_current_trend[:, 1:, :], z_next_trend.unsqueeze(1)], dim=1)
            trend_pred_seg = self.reconstructor(z_next_trend)
            trend_predictions.append(trend_pred_seg)

            # === Seasonal Coarse ===
            z_last_coarse = z_current_coarse[:, -1, :]
            z_prev_coarse = z_current_coarse[:, -2, :] if z_current_coarse.shape[1] > 1 else None
            z_next_coarse, _ = self.dynamics(z_last_coarse, z_prev_coarse, velocities['coarse'])
            latent_coarse.append(z_next_coarse)
            z_current_coarse = torch.cat([z_current_coarse[:, 1:, :], z_next_coarse.unsqueeze(1)], dim=1)
            coarse_pred_seg = self.reconstructor(z_next_coarse)
            coarse_predictions.append(coarse_pred_seg)

            # === Seasonal Fine ===
            z_last_fine = z_current_fine[:, -1, :]
            z_prev_fine = z_current_fine[:, -2, :] if z_current_fine.shape[1] > 1 else None
            z_next_fine, _ = self.dynamics(z_last_fine, z_prev_fine, velocities['fine'])
            latent_fine.append(z_next_fine)
            z_current_fine = torch.cat([z_current_fine[:, 1:, :], z_next_fine.unsqueeze(1)], dim=1)
            fine_pred_seg = self.reconstructor(z_next_fine)
            fine_predictions.append(fine_pred_seg)

            # === Residual ===
            z_last_resid = z_current_resid[:, -1, :]
            z_prev_resid = z_current_resid[:, -2, :] if z_current_resid.shape[1] > 1 else None
            z_next_resid, _ = self.dynamics(z_last_resid, z_prev_resid, velocities['resid'])
            latent_resid.append(z_next_resid)
            z_current_resid = torch.cat([z_current_resid[:, 1:, :], z_next_resid.unsqueeze(1)], dim=1)
            residual_pred_seg = self.reconstructor(z_next_resid)
            residual_predictions.append(residual_pred_seg)


        # Stack predicted segments
        # Stack latent states
        def stack_latents(lst):
            return torch.stack(lst, dim=1)  # [B, num_segments, embed_dim]


        # Batch decode all segments (output is [B, num_pred_segments, segment_length, 1])
        predictions_norm = torch.stack(predictions_norm, dim=1) 
        trend_predictions = torch.stack(trend_predictions, dim=1) 
        coarse_predictions = torch.stack(coarse_predictions, dim=1) 
        fine_predictions = torch.stack(fine_predictions, dim=1) 
        residual_predictions = torch.stack(residual_predictions, dim=1) 

        # Flatten to [B, pred_len]
        predictions_norm = predictions_norm.reshape(B, self.num_pred_segments * self.segment_length)
        trend_predictions = trend_predictions.reshape(B, self.num_pred_segments * self.segment_length)
        coarse_predictions = coarse_predictions.reshape(B, self.num_pred_segments * self.segment_length)
        fine_predictions = fine_predictions.reshape(B, self.num_pred_segments * self.segment_length)
        residual_predictions = residual_predictions.reshape(B, self.num_pred_segments * self.segment_length)
        
        return {
            'predictions': predictions_norm,
            'trend': trend_predictions,
            'coarse': coarse_predictions,
            'fine': fine_predictions,
            'residual': residual_predictions,
            'latent_z': stack_latents(latent_z),
            'latent_trend': stack_latents(latent_trend),
            'latent_coarse': stack_latents(latent_coarse),
            'latent_fine': stack_latents(latent_fine),
            'latent_resid': stack_latents(latent_resid)
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
        # Storage for all features
        batched_out = self.process_batched_features(trend_b, coarse_b, fine_b, resid_b)
        predictions_norm = batched_out['predictions']  # [B*F, pred_len]

        # Apply RevIN denormalization
        if self.use_revin:
            predictions_norm = predictions_norm.view(B, F, -1).permute(0, 2, 1).contiguous()  # [B, pred_len, F]
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm.view(B, F, -1).permute(0, 2, 1).contiguous()
        
        # Repackage other outputs (latent states) to shapes similar to your original output
        # latent_z: [B*F, num_pred_segments, embed_dim] -> [B, F, num_pred_segments, embed_dim]
        def uncollapse_latent(tensor_b):
            # tensor_b: [B*F, num_pred_segments, D] -> [B, F, num_pred_segments, D]
            Bf, S, D = tensor_b.shape
            return tensor_b.view(B, F, S, D)
        
        def uncollapse_comp_predict(component_b):
            return component_b.view(B, F, -1).permute(0, 2, 1).contiguous()

        return {
            'predictions': predictions,
            'trend_predictions': uncollapse_comp_predict(batched_out["trend"]),
            'coarse_predictions': uncollapse_comp_predict(batched_out["coarse"]),
            'fine_predictions': uncollapse_comp_predict(batched_out["fine"]),
            'residual_predictions': uncollapse_comp_predict(batched_out["residual"]),
            'hyperbolic_states': {
                "combined_h": uncollapse_latent(batched_out["latent_z"]),
                "trend_h": uncollapse_latent(batched_out["latent_trend"]),
                "coarse_h": uncollapse_latent(batched_out["latent_coarse"]),
                "fine_h": uncollapse_latent(batched_out["latent_fine"]),
                "resid_h": uncollapse_latent(batched_out["latent_resid"]),
            }
        }
