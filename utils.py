import torch
import torch.nn as nn
import geoopt
import numpy as np 
# --------------------------
# Clamping with safe Exponential map
# --------------------------
def safe_expmap0(manifold, u, max_norm=10.0, eps=1e-8):
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = torch.clamp(norm, max=max_norm) / norm
    u = u * scale
    x = manifold.expmap0(u)
    return manifold.projx(x)

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

def Create_Segmented_Tensors(tensors_dict, input_len=720, pred_len=96, stride=None):
    """
    Create fixed-length (X, Y) segments from decomposed tensors for each feature.
    Equivalent to TSLib/TimeBase dataloader segmenting.

    Parameters
    ----------
    tensors_dict : dict
        {feature_name: {"trend": tensor[T,C], "seasonal": tensor[T,C], "residual": tensor[T,C]}}
    input_len : int
        Length of past window used as model input.
    pred_len : int
        Length of future window to predict.
    stride : int
        Step size between windows (default: pred_len).
        Smaller stride increases overlap (TimeMixer-style).
    """
    segmented = {}
    if stride is None:
        stride = pred_len

    for feat, comps in tensors_dict.items():
        X_trend, Y_trend = [], []
        X_seas,  Y_seas  = [], []
        X_res,   Y_res   = [], []

        trend = comps["trend"]
        seas = comps["seasonal"]
        resid = comps["residual"]
        T = trend.shape[0]

        for i in range(0, T - input_len - pred_len + 1, stride):
            X_trend.append(trend[i : i + input_len])
            Y_trend.append(trend[i + input_len : i + input_len + pred_len])

            X_seas.append(seas[i : i + input_len])
            Y_seas.append(seas[i + input_len : i + input_len + pred_len])

            X_res.append(resid[i : i + input_len])
            Y_res.append(resid[i + input_len : i + input_len + pred_len])

        segmented[feat] = {
            "X": {
                "trend": torch.stack(X_trend),
                "seasonal": torch.stack(X_seas),
                "residual": torch.stack(X_res),
            },
            "Y": {
                "trend": torch.stack(Y_trend),
                "seasonal": torch.stack(Y_seas),
                "residual": torch.stack(Y_res),
            }
        }
    return segmented
