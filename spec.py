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

def compute_hierarchical_loss_with_manifold_dist(embeddings_dict, manifold, margin=0.1):
    """
    Enforce hierarchy using actual hyperbolic distances from origin.
    
    Hierarchy: trend < weekly < daily < residual (distance from origin)
    """
    trend_h = embeddings_dict["trend_h"]
    weekly_h = embeddings_dict["seasonal_weekly_h"]
    daily_h = embeddings_dict["seasonal_daily_h"]
    residual_h = embeddings_dict["residual_h"]
    
    # Origin on Lorentz manifold
    origin = manifold.origin  # [1, 0, 0, ..., 0]
    
    # Hyperbolic distances from origin (encodes depth)
    trend_dist_from_origin = manifold.dist(trend_h, origin)
    weekly_dist_from_origin = manifold.dist(weekly_h, origin)
    daily_dist_from_origin = manifold.dist(daily_h, origin)
    residual_dist_from_origin = manifold.dist(residual_h, origin)
    
    # Part 1: Distance-based hierarchy (norms)
    # Enforce: trend_dist < weekly_dist < daily_dist < residual_dist
    hierarchy_loss = (
        torch.relu(trend_dist_from_origin + margin - weekly_dist_from_origin) +
        torch.relu(weekly_dist_from_origin + margin - daily_dist_from_origin) +
        torch.relu(daily_dist_from_origin + margin - residual_dist_from_origin)
    ).mean()
    
    # Part 2: Entailment (parent-child proximity)
    # Parents and children should be geodesically close
    trend_to_weekly = manifold.dist(trend_h, weekly_h)
    weekly_to_daily = manifold.dist(weekly_h, daily_h)
    
    entailment_loss = (
        trend_to_weekly +
        weekly_to_daily
    ).mean()
    
    total_loss = hierarchy_loss + 0.5 * entailment_loss
    return total_loss


def segment_safe_expmap0(manifold, u, max_norm=10.0, eps=1e-6):
    original_shape = u.shape
    B, N, D = original_shape
    u = u.reshape(B * N, D)
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.minimum(torch.ones_like(norm), max_norm) / norm
    # clamp = torch.clamp(scale, max=max_norm) / norm
    u = u * scale
    # Exponential map
    x = manifold.expmap0(u)
    x = manifold.projx(x)
    manifold_dim = x.shape[-1]  # embed_dim + 1 for Lorentz
    x = x.reshape(B, N, manifold_dim)
    return x


def safe_expmap(manifold, base_point, v, eps=1e-15, max_norm=0.99):
    """Similar to safe_expmap0 but for non-origin base points"""
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    v_safe = v / v_norm * torch.clamp(v_norm, max=max_norm)
    
    return manifold.expmap(base_point, v_safe)

def safe_expmap0(manifold, v, eps=1e-15, max_norm=0.99):
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
    def __init__(self, num_channels: int, affine: bool = True, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_channels))
            self.beta  = nn.Parameter(torch.zeros(1, 1, num_channels))

    @torch.no_grad()
    def _stats(self, x):
        mu = x.mean(dim=1, keepdim=True)                  # [B,1,C]
        var = x.var(dim=1, unbiased=False, keepdim=True)  # [B,1,C]
        sigma = torch.sqrt(var + self.eps)
        return mu, sigma

    def normalize(self, x):
        mu, sigma = self._stats(x)
        x_n = (x - mu) / sigma
        if self.affine:
            x_n = x_n * self.gamma + self.beta
        return x_n, (mu, sigma)

    def denormalize(self, x, stats):
        mu, sigma = stats
        if self.affine:
            gamma = torch.clamp(self.gamma, min=1e-6)
            x = (x - self.beta) / gamma
        return x * sigma + mu

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

    
# 

# def prepare_timebase_data_with_mstl(train_dict, val_dict, test_dict, 
#                                      mstl_period, input_len, pred_len, device="cuda"):
#     """
#     Complete pipeline: normalize → segment with MSTL period.
    
#     Args:
#         stride: 'overlap' for stride=1 (standard sliding window)
#                 'period' for stride=mstl_period (period-aligned samples)
#                 int for custom stride
#     """
    
#     # Choose segmentation strategy
#         # Standard overlapping sliding windows
#     segment_fn = lambda d: Create_Segments_With_MSTL_Period(
#         d, input_len, pred_len, mstl_period, device
#     )
    
#     train_seg = segment_fn(train_dict)
#     val_seg = segment_fn(val_dict)
#     test_seg = segment_fn(test_dict)
    
#     return train_seg, val_seg, test_seg
