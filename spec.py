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
def safe_expmap(manifold, u, max_norm=5.0, eps=1e-8):
    """
    Safe exponential map supporting both 2D [B, D] and 3D [B, N, D] inputs.
    """
    original_shape = u.shape
    is_3d = len(original_shape) == 3
    
    # Flatten 3D to 2D if needed
    if is_3d:
        B, N, D = original_shape
        u = u.reshape(B * N, D)
    
    # Apply clamping
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(norm, max=max_norm) / norm
    u = u * scale
    
    # Exponential map
    x = manifold.expmap(u)
    x = manifold.projx(x)
    
    # Reshape back to 3D if needed
    if is_3d:
        manifold_dim = x.shape[-1]  # embed_dim + 1 for Lorentz
        x = x.reshape(B, N, manifold_dim)
    
    return x

def safe_expmap0(manifold, u, max_norm=10.0, eps=1e-8):
    """
    Safe exponential map from origin supporting both 2D [B, D] and 3D [B, N, D] inputs.
    
    Args:
        manifold: geoopt manifold (e.g., Lorentz)
        u: tangent vectors at origin
           - 2D: [B, D] for point-level
           - 3D: [B, num_seg, D] for segment-level
        max_norm: maximum norm for clamping
        eps: small constant for numerical stability
    
    Returns:
        manifold points with same shape structure as input
        - 2D input: [B, manifold_dim]
        - 3D input: [B, num_seg, manifold_dim]
    """
    original_shape = u.shape
    is_3d = len(original_shape) == 3
    
    # Flatten 3D to 2D if needed: [B, N, D] -> [B*N, D]
    if is_3d:
        B, N, D = original_shape
        u = u.reshape(B * N, D)
    
    # Apply norm clamping
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(norm, max=max_norm) / norm
    u = u * scale
    
    # Exponential map from origin
    x = manifold.expmap0(u)
    x = manifold.projx(x)
    
    # Reshape back to 3D if input was 3D
    if is_3d:
        manifold_dim = x.shape[-1]  # embed_dim + 1 for Lorentz manifold
        x = x.reshape(B, N, manifold_dim)
    
    return x

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

    
def Create_Segments_With_MSTL_Period(tensors_dict, input_len, pred_len, 
                                      mstl_period=24, device="cuda"):
    """
    Create sliding window samples using MSTL-detected period as segment length.
    
    Args:
        tensors_dict: { feature: { component: [1, T, C] } }
        input_len: lookback window length T
        pred_len: forecast horizon L
        mstl_period: Period detected by MSTL (e.g., 24 for daily with hour frequency)
        device: computation device
    
    Returns:
        segments_dict with sliding windows split into periodic segments
    """
    segment_length = mstl_period  # Use MSTL period directly (daily for all)
    segments = {}
    
    for feat, comps in tensors_dict.items():
        X, Y = {}, {}
        
        for comp_name, comp_tensor in comps.items():
            data = comp_tensor.squeeze(0).to(device)  # [T, C]
            T, C = data.shape
            
            # Sliding window parameters
            num_samples = T - input_len - pred_len + 1
            
            # Calculate how many complete periods fit in windows
            num_input_segments = input_len // segment_length
            num_pred_segments = (pred_len + segment_length - 1) // segment_length
            
            # Adjust to exact multiples of period
            effective_input_len = num_input_segments * segment_length
            effective_pred_len = num_pred_segments * segment_length
            
            x_segments = []
            y_segments = []
            
            # Create sliding windows (overlapping)
            for i in range(num_samples):
                # Input window: [i : i+effective_input_len]
                x_window = data[i:i+effective_input_len]  # [effective_input_len, C]
                
                # Reshape into periodic segments
                x_seg = x_window.reshape(num_input_segments, segment_length, C)
                x_segments.append(x_seg)
                
                # Output window: [i+input_len : i+input_len+effective_pred_len]
                y_start = i + input_len
                y_window = data[y_start:y_start+effective_pred_len]
                
                # Handle edge case at end of series
                if y_window.shape[0] < effective_pred_len:
                    pad_len = effective_pred_len - y_window.shape[0]
                    padding = torch.zeros(pad_len, C, device=device)
                    y_window = torch.cat([y_window, padding], dim=0)
                
                y_seg = y_window.reshape(num_pred_segments, segment_length, C)
                y_segments.append(y_seg)
            
            X[comp_name] = torch.stack(x_segments, dim=0)  # [N_samples, N_seg, P, C]
            Y[comp_name] = torch.stack(y_segments, dim=0)  # [N_samples, N'_seg, P, C]
        
        segments[feat] = {"X": X, "Y": Y}
    
    return segments


def prepare_timebase_data_with_mstl(train_dict, val_dict, test_dict, 
                                     mstl_period, input_len, pred_len, device="cuda"):
    """
    Complete pipeline: normalize → segment with MSTL period.
    
    Args:
        stride: 'overlap' for stride=1 (standard sliding window)
                'period' for stride=mstl_period (period-aligned samples)
                int for custom stride
    """
    
    # Choose segmentation strategy
        # Standard overlapping sliding windows
    segment_fn = lambda d: Create_Segments_With_MSTL_Period(
        d, input_len, pred_len, mstl_period, device
    )
    
    train_seg = segment_fn(train_dict)
    val_seg = segment_fn(val_dict)
    test_seg = segment_fn(test_dict)
    
    return train_seg, val_seg, test_seg
