import torch
import torch.nn as nn
import geoopt
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
        self.val_loss_min = np.Inf
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

