import torch
import torch.nn as nn
import torch.nn.functional as F
from HyperCore.hypercore.manifolds.poincare import PoincareBall
from HyperCore.hypercore.nn.linear.poincare_linear import PoincareLinear
def cal_hyperbolic_orthogonal_loss(basis, manifold):
    """
    Orthogonal loss in hyperbolic space.
    
    Instead of Euclidean Gram matrix, we use hyperbolic distances.
    We want basis functions to be far apart (orthogonal).
    
    Args:
        basis: [B, P, K] tensor on hyperbolic manifold
        manifold: geoopt manifold object
    
    Returns:
        loss: scalar
    """
    B, P, K = basis.shape
    
    # Compute pairwise distances for each timestep
    total_loss = 0
    count = 0
    
    for t in range(P):
        basis_t = basis[:, t, :]  # [B, K]
        
        # Compute pairwise hyperbolic distances
        for i in range(K):
            for j in range(i + 1, K):
                basis_i = basis_t[:, i:i+1]  # [B, 1]
                basis_j = basis_t[:, j:j+1]  # [B, 1]
                
                # Hyperbolic distance
                dist = manifold.dist(basis_i, basis_j)  # [B]
                
                # Want large distances (orthogonal basis)
                # Penalize if distance < threshold
                loss = torch.relu(1.0 - dist)
                total_loss += loss.mean()
                count += 1
    
    return total_loss / count if count > 0 else torch.tensor(0.0)


class Model(nn.Module):
    """
    Minimal hyperbolic enhancement of TimeBase (ICML 2025).
    
    Changes from original TimeBase:
    1. Replace nn.Linear with HyperbolicLinear
    2. Add hyperbolic orthogonal loss
    3. Operate in Poincaré ball manifold
    
    NO embedder - works directly on segment values like TimeBase!
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        
        # TimeBase parameters (same as original)
        self.use_segment_norm = True
        self.use_orthogonal = configs.use_orthogonal
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = 24
        self.basis_num = configs.num_basis
        self.manifold_type = configs.manifold_type
        
        # Compute number of segments
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        
        self.pad_seq_len = 0
        if self.seq_len > self.seg_num_x * self.period_len:
            self.pad_seq_len = (self.seg_num_x + 1) * self.period_len - self.seq_len
            self.seg_num_x += 1
        if self.pred_len > self.seg_num_y * self.period_len:
            self.seg_num_y += 1
        
        # NOVEL: Hyperbolic manifold
        self.manifold = PoincareBall(c=1.0)
        
        # NOVEL: Learnable curvature (optional)
        self.learnable_curvature = False
        if self.learnable_curvature:
            self.curvature = nn.Parameter(torch.tensor(1.0))
        
        self.individual = False
        
        # NOVEL: Replace nn.Linear with HyperbolicLinear
        if self.individual:
            self.ts2basis = nn.ModuleList()
            self.basis2ts = nn.ModuleList()
            for i in range(self.enc_in):
                self.ts2basis.append(
                    PoincareLinear(manifold=self.manifold, in_features=self.seg_num_x, c=1.0, out_features=self.basis_num)
                )
                self.basis2ts.append(
                    PoincareLinear(manifold=self.manifold, in_features=self.basis_num, c=1.0, out_features=self.seg_num_y)
                )
        else:
            self.ts2basis = PoincareLinear(manifold=self.manifold, c=1.0, in_features=self.seg_num_x, out_features=self.basis_num)

            self.basis2ts = PoincareLinear(manifold=self.manifold, c=1.0, in_features=self.basis_num, out_features=self.seg_num_y)

        
        print(f"\n{'='*70}")
        print(f"Poincare TimeBase Minimal")
        print(f"{'='*70}")
        print(f"Input segments (x): {self.seg_num_x}")
        print(f"Output segments (y): {self.seg_num_y}")
        print(f"Basis functions: {self.basis_num}")
        print(f"Period length: {self.period_len}")
        print(f"Features: {self.enc_in}")
        print(f"Individual: {self.individual}")
        print(f"Manifold: Poincaré Ball")
        print(f"{'='*70}\n")
    
    def _normalize_input(self, x, b, c):
        """
        Normalize input (same as TimeBase).
        
        Args:
            x: [bc, p, n] where p=period_len, n=seg_num_x
            b: batch size
            c: number of channels
        
        Returns:
            x_normalized: [bc, p, n]
            norm_stats: dict with normalization stats
        """
        if self.use_segment_norm:
            # Normalize each period independently
            period_mean = torch.mean(x, dim=-1, keepdim=True)
            x = x - period_mean
            return x, {'period_mean': period_mean}
        else:
            # Global normalization
            x = x.reshape(b, c, -1)
            mean = torch.mean(x, dim=-1, keepdim=True)
            x = x - mean
            x = x.reshape(-1, self.period_len, self.seg_num_x)
            return x, {'mean': mean}
    
    def _denormalize_output(self, x, norm_stats, b, c):
        """
        Denormalize output (same as TimeBase).
        
        Args:
            x: [bc, p, n] where p=period_len, n=seg_num_y
            norm_stats: dict with normalization stats
            b: batch size
            c: number of channels
        
        Returns:
            x_denorm: [bc, p, n]
        """
        if self.use_segment_norm:
            x = x + norm_stats['period_mean']
        else:
            x = x.reshape(b, c, -1)
            x = x + norm_stats['mean']
            x = x.reshape(-1, self.period_len, self.seg_num_y)
        return x
    
    def forward(self, x):
        """
        Forward pass with hyperbolic geometry.
        
        Args:
            x: [B, T, C] input time series
        
        Returns:
            If use_orthogonal=True: (predictions, orthogonal_loss)
            Else: predictions
            
            predictions: [B, pred_len, C]
        """
        b, t, c = x.shape
        batch_size = b
        x = x.permute(0, 2, 1)  # [B, C, T]
        
        # Padding (same as TimeBase)
        if self.pad_seq_len > 0:
            pad_start = (self.seg_num_x - 1) * self.period_len
            x = torch.cat([x, x[:, :, pad_start - self.pad_seq_len:pad_start]], dim=-1)
        
        # Reshape into segments
        x = x.reshape(batch_size, self.enc_in, self.seg_num_x, self.period_len)
        x = x.permute(0, 1, 3, 2)  # [B, C, period_len, seg_num_x]
        x = x.reshape(-1, self.period_len, self.seg_num_x)  # [B*C, period_len, seg_num_x]
        
        # Normalize
        x, norm_stats = self._normalize_input(x, b, c)
        
        # === NOVEL: Hyperbolic Basis Transformation ===
        
        # Map to hyperbolic space
        x_hyp = self.manifold.expmap0(x)  # [B*C, period_len, seg_num_x]
        
        if self.individual:
            # Process each channel separately
            x_hyp = x_hyp.reshape(b, c, self.period_len, self.seg_num_x)
            x_pred_list = []
            x_basis_list = []
            
            for i in range(self.enc_in):
                x_i_hyp = x_hyp[:, i, :, :]  # [B, period_len, seg_num_x]
                
                # Hyperbolic basis transformation
                x_basis_i_hyp = self.ts2basis[i](x_i_hyp)  # [B, period_len, basis_num]
                x_basis_list.append(x_basis_i_hyp)
                
                # Hyperbolic forecasting
                x_pred_i_hyp = self.basis2ts[i](x_basis_i_hyp)  # [B, period_len, seg_num_y]
                x_pred_list.append(x_pred_i_hyp)
            
            # Stack results
            x_basis_hyp = torch.stack(x_basis_list, dim=1)  # [B, C, period_len, basis_num]
            x_pred_hyp = torch.stack(x_pred_list, dim=1)  # [B, C, period_len, seg_num_y]
            
            # Reshape for denormalization
            x_basis_hyp = x_basis_hyp.reshape(-1, self.period_len, self.basis_num)
            x_pred_hyp = x_pred_hyp.reshape(-1, self.period_len, self.seg_num_y)
        else:
            # Shared weights across channels
            x_basis_hyp = self.ts2basis(x_hyp)  # [B*C, period_len, basis_num]
            x_pred_hyp = self.basis2ts(x_basis_hyp)  # [B*C, period_len, seg_num_y]
        
        # Map back to Euclidean space
        x_pred = self.manifold.logmap0(x_pred_hyp)
        
        # Denormalize
        x_pred = self._denormalize_output(x_pred, norm_stats, b, c)
        
        # Reshape back to [B, T, C]
        x_pred = x_pred.reshape(batch_size, self.enc_in, self.period_len, self.seg_num_y)
        x_pred = x_pred.permute(0, 1, 3, 2)  # [B, C, seg_num_y, period_len]
        x_pred = x_pred.reshape(batch_size, self.enc_in, -1)
        x_pred = x_pred.permute(0, 2, 1)  # [B, T, C]
        
        # Return predictions (trim to pred_len)
        x_pred = x_pred[:, :self.pred_len, :]
        
        # NOVEL: Hyperbolic orthogonal loss
        if self.use_orthogonal:
            orthogonal_loss = cal_hyperbolic_orthogonal_loss(x_basis_hyp, self.manifold)
            return x_pred, orthogonal_loss
        else:
            return x_pred