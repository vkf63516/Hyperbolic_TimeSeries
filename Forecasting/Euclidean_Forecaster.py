import torch
import torch.nn as nn
from embed.mlp_embed_euclidean import ParallelEuclideanEmbed
from Lifting.euclidean_reconstructor import EuclideanReconstructor
from spec import RevIN

class ResidualDynamics(nn.Module):
    """
    Residual block for dynamics prediction.
    Outputs: z_next = z + weighted_residual
    This prevents drift by keeping updates close to identity mapping.
    """
    def __init__(self, embed_dim, hidden_dim, dropout=0.3, n_layers=2):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(embed_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer (maps back to embed_dim)
        layers.append(nn.Linear(hidden_dim, embed_dim))
        
        self.residual_net = nn.Sequential(*layers)
        
        # Learnable residual weight (initialized near 0 for stability)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, z):
        """
        Args:
            z: [B, embed_dim] - current state
        Returns:
            z_next: [B, embed_dim] - next state with residual connection
        """
        residual = self.residual_net(z)
        # Residual connection: z_next = z + α * f(z)
        # where α is learnable and starts small
        return z + self.residual_weight * residual


class PointForecastEuclidean(nn.Module):
    """
    Point-level Euclidean forecasting
    NO manifold operations - pure Euclidean arithmetic
    """
    def __init__(self, lookback, n_features, pred_len, embed_dim=32, hidden_dim=64,
                 use_hierarchy=False, hierarchy_scales=[0.5, 1.0, 1.0, 1.5], 
                 use_attention_pooling=False, use_revin=False, 
                 use_truncated_bptt=False, truncate_every=16,
                 dynamic_dropout=0.3, embed_dropout=0.5, recon_dropout=0.2, num_layers=2):
        super().__init__()
        self.lookback = lookback
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.use_revin = use_revin
        self.use_truncated_bptt = use_truncated_bptt
        self.truncate_every = truncate_every
        if self.use_revin:
            self.revin = RevIN(num_features=n_features, eps=1e-5, affine=True)        # Encoder: 4 parallel Mamba blocks
        self.embed = ParallelEuclideanEmbed(
            lookback=lookback,
            input_dim=n_features,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            use_hierarchy=use_hierarchy,
            n_layer=num_layers,
            embed_dropout=embed_dropout,
            hierarchy_scales=hierarchy_scales,
            use_attention_pooling=use_attention_pooling
        )
        self.dynamics = ResidualDynamics(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dynamic_dropout,
            n_layers=num_layers
        )
        self.step_size = nn.Parameter(torch.tensor(0.1))
        # Reconstructor: simple MLP (no logmap/expmap needed!)
        self.reconstructor = EuclideanReconstructor(
            embed_dim=embed_dim,
            output_dim=n_features,
            hidden_dim=hidden_dim,
            dropout=recon_dropout
        )


    def forward(self, trend, coarse, fine, residual):
        """
        Forecast each component separately, combine at the end.
        """
        # Step 1: Normalize each component SEPARATELY
        x_combined = trend + coarse + fine + residual  # [B, lookback, n_features]
        
        # Step 2: Store RevIN stats from combined signal (NO transformation applied!)
        if self.use_revin:
            # This ONLY stores mean and std, doesn't transform the data
            # We pass mode='norm' but it's just to compute and store stats
            self.revin(x_combined, mode='norm')
        
        # Step 3: Encode RAW components (they're already normalized by StandardScaler!)
        # We DON'T normalize them again!
        embed_output = self.embed(trend, coarse, fine, residual)
        z_current = embed_output['combined_e']  # [B, embed_dim]
        
        # Step 4: Autoregressive forecasting (in normalized space)
        predictions_norm = []
        
        for step in range(self.pred_len):
            # Reconstruct in normalized space
            x_pred_norm = self.reconstructor(z_current)  # [B, n_features]
            predictions_norm.append(x_pred_norm)
            
            # Update latent state
            z_next = self.dynamics(z_current)
            step_size = torch.sigmoid(self.step_size)
            z_current = z_current + step_size * (z_next - z_current)
            
            # Optional: truncated BPTT
            if self.use_truncated_bptt and (step + 1) % self.truncate_every == 0 and step < self.pred_len - 1:
                z_current = z_current.detach()
        
        predictions_norm = torch.stack(predictions_norm, dim=1)  # [B, pred_len, n_features]
        
        # Step 5: Denormalize predictions using stored stats
        if self.use_revin:
            predictions = self.revin(predictions_norm, mode='denorm')
        else:
            predictions = predictions_norm
        
        return predictions
