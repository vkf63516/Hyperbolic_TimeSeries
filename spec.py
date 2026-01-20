import torch
import torch.nn as nn
import geoopt
import numpy as np 
from sklearn.preprocessing import StandardScaler
import math
# --------------------------
# Clamping with safe Exponential map
# --------------------------
import torch
import torch.nn as nn
import geoopt
import numpy as np 
from sklearn.preprocessing import StandardScaler
import math

# --------------------------
# Clamping with safe Exponential map - supports both 2D and 3D inputs
# --------------------------

def compute_hierarchical_loss_with_manifold_dist(encodedings_dict, manifold, margin=0.1):
    """
    Enforce hierarchy using actual hyperbolic distances from origin.
    
    Hierarchy: trend < coarse < fine < residual (distance from origin)
    """
    trend_h = encodedings_dict["trend_h"]
    coarse_h = encodedings_dict["seasonal_coarse_h"]
    fine_h = encodedings_dict["seasonal_fine_h"]
    residual_h = encodedings_dict["residual_h"]
    
    # Origin on Lorentz manifold
    origin = manifold.origin(trend_h.shape[-1], device=trend_h.device)  # [1, 0, 0, ..., 0]
    
    # Hyperbolic distances from origin (encodes depth)
    trend_dist_from_origin = manifold.dist(trend_h, origin)
    coarse_dist_from_origin = manifold.dist(coarse_h, origin)
    fine_dist_from_origin = manifold.dist(fine_h, origin)
    residual_dist_from_origin = manifold.dist(residual_h, origin)
    
    # Part 1: Distance-based hierarchy (norms)
    # Enforce: trend_dist < coarse_dist < fine_dist < residual_dist
    hierarchy_loss = (
        torch.relu(trend_dist_from_origin + margin - coarse_dist_from_origin) +
        torch.relu(coarse_dist_from_origin + margin - fine_dist_from_origin) +
        torch.relu(fine_dist_from_origin + margin - residual_dist_from_origin)
    ).mean()
    
    # Part 2: Entailment (parent-child proximity)
    # Parents and children should be geodesically close
    trend_to_coarse = manifold.dist(trend_h, coarse_h)
    coarse_to_fine = manifold.dist(coarse_h, fine_h)
    fine_to_residual = manifold.dist(fine_h, residual_h)
    
    entailment_loss = (
        trend_to_coarse +
        coarse_to_fine +
        fine_to_residual
    ).mean()
    
    total_loss = hierarchy_loss + 0.5 * entailment_loss
    return total_loss


def safe_expmap0_lorentz(manifold, v, eps=1e-8, initial_scale=0.1):
    """
    Safe expmap0 for Lorentz manifold based on MiT implementation. 
    Uses tanh scaling to prevent overflow in acosh. 
    """
    # Apply tanh scaling like MiT
    scale_factor = torch.tensor(initial_scale, device=v.device, dtype=v.dtype)
    effective_scale = torch.tanh(scale_factor)  # Maps to (-1, 1)
    
    scaled_v = v * effective_scale
    
    # Use native expmap0 (geoopt handles the math correctly)
    result = manifold.expmap0(scaled_v)
    
    # Project to ensure manifold constraint
    result = manifold.projx(result)
    
    return result


def safe_expmap(manifold, base_point, v, eps=1e-15, max_norm=7.0):
    """Similar to safe_expmap0 but for non-origin base points"""
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    # print(f"V_norm ")
    scale_factor = max_norm * torch.tanh(v_norm / max_norm)
    # print(f"Scale Factor {scale_factor}")
    v_safe = v / v_norm * scale_factor
        
    # Expmap
    result = manifold.expmap(base_point, v_safe)
    
    # ADD THIS: Check result
    if torch.isnan(result).any():
        nan_mask = torch.isnan(result).any(dim=-1)
        result = torch.where(nan_mask.unsqueeze(-1), base_point, result)
    
    return result
    

def safe_expmap0(manifold, v, eps=1e-15, max_norm=7.0):
    """
    Safely map tangent vector v to manifold point.
    
    For Lorentz: ensures norm(v) < max_norm to avoid overflow
    For Poincaré: ensures ||v|| < max_norm for stability
    
    Args:
        manifold: geoopt.Lorentz or geoopt.PoincareBall
        v: tangent vector [B, D]
        eps: numerical stability epsilon
        max_norm: maximum allowed norm (default 0.99 keeps safe margin)
    
    Returns:
        Point on manifold [B, D+1] for Lorentz, [B, D] for Poincaré
    """
    # Clip norm to safe region
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    v_safe = v / v_norm * torch.clamp(v_norm, max=max_norm)
    
    return manifold.expmap0(v_safe)

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss