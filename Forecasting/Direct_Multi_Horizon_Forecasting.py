import torch
import torch.nn as nn
from encode.Multi_Horizon.segment_encode_no_decomp_poincare import DirectPoincareNoDecomp
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
        velocities = self.manifold.logmap(z_history[:, :-1, :], z_history[:, 1:, :])  # [B, N-1, D]
        
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
        v_0t = self.velocity_net(v_init)        
        t_vals = torch.arange(1, T + 1, device=z_0.device).float()  # [T]
        v_all = (step * t_vals.view(T, 1, 1) * v_0t.unsqueeze(0))   # [T, B, D]
        v_all = v_all.permute(1, 0, 2).reshape(B * T, D)             # [B*T, D]
        z_0_exp = z_0.unsqueeze(1).expand(B, T, D).reshape(B * T, D) # [B*T, D]
    
        z_all = self.manifold.expmap(z_0_exp, v_all)           # [B*T, D]
        return z_all.view(B, T, D)                                    # [B, T, D]

class DirectNoDecompHyperbolicForecaster(nn.Module):
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
        self.encode_hyperbolic = DirectPoincareNoDecomp(
            lookback=lookback,
            num_channels=n_features,
            encode_dim=encode_dim,
            curvature=curvature,
            segment_length=segment_length,
            encode_dropout=encode_dropout
        )
        
        self.manifold = self.encode_hyperbolic.manifold
        
        # ===== Parallel geodesic dynamics for each component =====
        self.dynamics = ParallelDirectPoincareDynamics(
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
        print(f"Ablation No Decomposition Direct Multi-Horizon Hyperbolic Forecaster")
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
    
    def process_batched_features(self, x_raw):
        """
        Process batched features through entire pipeline.
        
        Args:
            trend_f, coarse_f, fine_f, resid_f: [B, seq_len]
        
        Returns:
            dict with predictions: [B, pred_len]
        """
        B = x_raw.shape[0]
        
        # ===== Encode historical segments =====
        encode_h = self.encode_hyperbolic(x_raw)
        
        z_hist = encode_h["combined_h"]  # [B, num_hist_segments, encode_dim]
        # print(z_trend_hist.shape)
        _, N_hist, D = z_hist.shape
        
        # ===== Take window for velocity computation =====
     
        z_win = z_hist
        v_init = self.dynamics.compute_initial_velocity(z_win)
        
        # Initial states
        z_0 = z_hist[:, -1, :]
        
        # ===== PARALLEL DIRECT PREDICTION =====
        z_future = self.dynamics(z_0, v_init)  # [B*F, T, D]
        
        predictions = []
            
        prediction = self.reconstructor(z_future)  # [B, segment_length]
        predictions = prediction.reshape(B, self.segment_length * self.num_pred_segments)
        
        return {
            'predictions': predictions,
            'latent_z': z_future,
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
