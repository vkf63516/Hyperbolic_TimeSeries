import torch
import random
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pandas as pd
import geoopt
from pathlib import Path
import sys
import numpy as np
import warnings
import matplotlib.pyplot as plt 
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# --------------------------------------------------------------------
# Setup paths (so local packages can be imported easily)
# --------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[0]))

from encoder.mamba_encoders_lorentz import ParallelLorentzEncoder
from Decomposition.TimeBase_Series_Trend_Decomposition import TimeBaseMSTL
from Decomposition.tensor_utils import build_decomposition_tensors
from Decomposition.visualization_utils import plot_component_grid, plot_variance_contribution, plot_component_correlation_maps
from Forecaster import HyperbolicSeqForecaster

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

set_seed(42)

# ---- NORMALIZATION HELPERS ----
COMP_KEYS = ["trend", "seasonal_hourly", "seasonal_daily", "seasonal_weekly", "residual"]

def compute_norm_stats(components_dict):
    """
    components_dict: { series_name: {component_key: 1D array of length T} }
    returns: { series_name: {component_key: (mean, std)} }
    """
    stats = {}
    for name, comp in components_dict.items():
        stats[name] = {}
        for k in COMP_KEYS:
            x = np.asarray(comp[k], dtype=np.float64)
            m = float(np.nanmean(x))
            s = float(np.nanstd(x))
            if not np.isfinite(s) or s < 1e-6:
                s = 1.0  # avoid division by zero; keep scale ~1
            stats[name][k] = (m, s)
    return stats

def apply_norm(components_dict, stats):
    """
    Apply (x - mean)/std using per-series, per-component stats.
    """
    out = {}
    for name, comp in components_dict.items():
        out[name] = {}
        for k in COMP_KEYS:
            x = np.asarray(comp[k], dtype=np.float64)
            m, s = stats[name][k] if name in stats else (0.0, 1.0)
            y = (x - m) / s
            # sanitize any NaN/Inf that might sneak in
            # y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            out[name][k] = y
    return out


# -------------------------------------------------------------
# 1. Device setup
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------
# 2. Load dataset
# -------------------------------------------------------------
df = pd.read_csv("time-series-dataset/dataset/weather/weather.csv", parse_dates=["date"], index_col="date")
df = df.select_dtypes(include=[np.number])

train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)


# -------------------------------------------------------------
# 4. Compute adaptive periods and window lengths
# -------------------------------------------------------------
timebase = TimeBaseMSTL(n_basis_components=5, orthogonal_lr=1e-3, orthogonal_iters=300)

# Automatically infers hourly, daily, weekly steps
steps_per_period = timebase.timesteps_from_index(df)
hourly, daily, weekly = steps_per_period

lookback = weekly
pred_len = int(daily)

print(f"Timesteps per hour/day/week: {hourly}, {daily}, {weekly}")
print(f"lookback window={lookback}, pred_len={pred_len}")

# -------------------------------------------------------------
# 5. Decompose training and validation data
# -------------------------------------------------------------
def check_nan_tensor(name, t):
    if torch.isnan(t).any():
        print(f"[NaN DETECTED] in {name}")
    if torch.isinf(t).any():
        print(f"[Inf DETECTED] in {name}")
    if torch.isnan(t).any() or torch.isinf(t).any():
        print(f"{name} stats:", t.min().item(), t.max().item(), t.mean().item())

def check_tensor_values(tensors_dict, name="Train"):
    """
    Check decomposition tensors before training to ensure there are no NaNs, infs,
    or degenerate values in trend / seasonal / residual components.
    """
    print(f"\nChecking tensor values before training ({name} set):")
    for feat, comps in tensors_dict.items():
        for key in ["trend", "seasonal", "residual"]:
            t = comps[key]
            if torch.is_tensor(t):
                t = t.detach().cpu()
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            vmin = t.min().item()
            vmax = t.max().item()
            mean = t.mean().item()
            std = t.std().item()
            print(
                f"{feat:<10s} {key:<10s} "
                f"NaN? {has_nan:<5} | Inf? {has_inf:<5} | "
                f"min={vmin:.4f} | max={vmax:.4f} | mean={mean:.4f} | std={std:.4f}"
            )

# Example: after building train_tensors_dict and val_tensors_dict

print("Performing TimeBaseMSTL decomposition...")

train_components = timebase.fit_transform(train_df)
val_components   = timebase.fit_transform(val_df)
# -------------------------------------------------------------
# 3. Normalize using training statistics only
# -------------------------------------------------------------
# for name, comp in train_components.items():
#     plot_component_grid(name, train_components)
# plot_variance_contribution(train_components)
# plot_component_correlation_maps(train_components)

# Normalize & tensors
train_stats = compute_norm_stats(train_components)
train_norm  = apply_norm(train_components, train_stats)
val_norm    = apply_norm(val_components, train_stats)


def build_timebase_tensors(decomp_dict):
    tensors = {}
    for name, comp in decomp_dict.items():
        tensors[name] = build_decomposition_tensors(comp)
    return tensors

train_tensors_dict = build_timebase_tensors(train_norm)
val_tensors_dict   = build_timebase_tensors(val_norm)
check_tensor_values(train_tensors_dict, "Train")
check_tensor_values(val_tensors_dict, "Validation")

# Convert each feature’s tensors to batch format
def to_batch_feature_dict(feature_dict):
    return {k: v.unsqueeze(0).float().to(device) for k, v in feature_dict.items()}

# --------------------------------------------------------------------
# 6. Initialize models
# --------------------------------------------------------------------
embed_dim = 32
hidden_dim = 64
output_dim = 1
pred_len_96 = 96
pred_len_96 = 192
pred_len_96 = 336
pred_len_96 = 720
encoder = ParallelLorentzEncoder(lookback=lookback, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
manifold = encoder.manifold
forecaster = HyperbolicSeqForecaster(embed_dim=embed_dim, hidden_dim=hidden_dim, output_dim=output_dim, manifold=manifold).to(device)

params = list({p: None for p in list(encoder.parameters()) + list(forecaster.parameters())}.keys())
optimizer = geoopt.optim.RiemannianAdam(params, lr=1e-4)
num_epochs = 200
training_and_validation(encoder=encoder, forecaster=forecaster, optimizer=optimizer, params=params, num_epochs=num_epochs, pred_len=pred_len_96)
#torch.autograd.set_detect_anomaly(True)
# --------------------------------------------------------------------
# 67. Hyperbolic loss helper
# --------------------------------------------------------------------
def hyperbolic_mse(manifold, z_pred, eps=1e-6):
    # z_pred: [B, H, D+1]
    z_pred = manifold.projx(z_pred)

    z1 = z_pred[:, 1:, :].reshape(-1, z_pred.size(-1))
    z0 = z_pred[:, :-1, :].reshape(-1, z_pred.size(-1))
    if torch.isnan(z_pred).any() or torch.isinf(z_pred).any():
        print("[NaN DETECTED] in z_pred passed to hyperbolic_mse")
        print("z_pred stats:", torch.nanmean(z_pred))
    # Compute Lorentz inner product manually and clamp it
    inner = -manifold.inner(None, z1, z0)
    if torch.isnan(inner).any() or torch.isinf(inner).any():
        print("[NaN DETECTED] before clamp in hyperbolic_mse")

    inner = torch.clamp(inner, min=1.0 + eps)
    dist = torch.acosh(inner)
    return (dist ** 2).mean()
# --------------------------------------------------------------------
# 8. Training loop
# --------------------------------------------------------------------
def training_and_validation(encoder, forecaster, optimizer, params, num_epochs, pred_len=96)

    best_val_loss = np.inf
    save_path = Path("checkpoints/best_model.pt")
    save_path.parent.mkdir(exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        encoder.train()
        forecaster.train()
        optimizer.zero_grad()

        train_losses = []
        for feat, tensors_f in train_tensors_dict.items():
        #print("i = ", i)
            batch_f = to_batch_feature_dict(tensors_f)

            enc_out = encoder(batch_f["trend"], batch_f["seasonal"], batch_f["residual"])
            check_nan_tensor("trend_h", enc_out["trend_h"])
            check_nan_tensor("season_h", enc_out["season_h"])
            check_nan_tensor("resid_h", enc_out["resid_h"])

            zt, zs, zr = enc_out["trend_h"], enc_out["season_h"], enc_out["resid_h"]

            x_hat, z_pred = forecaster.forecast(pred_len, trend_z=zt, seasonal_z=zs, resid_z=zr)
            x_true = (
                batch_f["trend"][:, -pred_len:, :] +
                batch_f["seasonal"][:, -pred_len:, :] + 
                batch_f["residual"][:, -pred_len:, :]
            )
            loss_rec = F.mse_loss(x_hat, x_true)
            loss_geo = hyperbolic_mse(forecaster.manifold, z_pred)
            loss = loss_rec + 0.1 * loss_geo
            writer.add_scalar("Loss/train_total", loss.item(), epoch)
            writer.add_scalar("Loss/train_reconstruction", loss_rec.item(), epoch)
            writer.add_scalar("Loss/train_hyperbolic", loss_geo.item(), epoch)
            loss.backward()
            train_losses.append(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        
        optimizer.step()
        train_loss = float(np.mean(train_losses))

    # ---------------- Validation ----------------
        encoder.eval()
        forecaster.eval()
        val_losses = []
        with torch.no_grad():
            for feat, tensors_f in val_tensors_dict.items():
                batch_f = to_batch_feature_dict(tensors_f)
                enc_val = encoder(batch_f["trend"], batch_f["seasonal"], batch_f["residual"])
                zv_t, zv_s, zv_r = enc_val["trend_h"], enc_val["season_h"], enc_val["resid_h"]
                x_val_hat, z_val_pred = forecaster.forecast(pred_len, trend_z=zv_t, seasonal_z=zv_s, resid_z=zv_r)
                x_val_true = (
                    batch_f["trend"][:, -pred_len:, :] +
                    batch_f["seasonal"][:, -pred_len:, :] +
                    batch_f["residual"][:, -pred_len:, :]
                )
                val_loss_rec = F.mse_loss(x_val_hat, x_val_true)
                val_loss_geo = hyperbolic_mse(forecaster.manifold, z_val_pred)
                val_loss = val_loss_rec + 0.1 * val_loss_geo
                writer.add_scalar("Loss/train_total", val_loss.item(), epoch)
                writer.add_scalar("Loss/train_reconstruction", val_loss_rec.item(), epoch)
                writer.add_scalar("Loss/train_hyperbolic", val_loss_geo.item(), epoch)
                val_losses.append(val_loss.detach().item())
    
        val_loss = float(np.mean(val_losses))
        print(f"Epoch [{epoch}/{num_epochs}] - Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "forecaster": forecaster.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, save_path)
            print(f"Saved best model (Val MSE: {val_loss:.6f})")
    print("Training complete.")
