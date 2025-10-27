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
import time
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
from utils import RevIN, EarlyStopping
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

set_seed(42)
@torch.no_grad()
def compute_global_train_stats(train_tensors_dict, device="cpu"):
    sum_x = None
    sum_x2 = None
    n = 0
    for _, tensors_f in train_tensors_dict.items():
        x = (tensors_f["trend"] + tensors_f["seasonal"] + tensors_f["residual"]).float().to(device)
        if x.dim() == 2:  # [T,C] → [1,T,C]
            x = x.unsqueeze(0)
        B, T, C = x.shape
        x = x.view(B * T, C)  # flatten time
        s  = x.sum(dim=0, keepdim=True)
        s2 = (x * x).sum(dim=0, keepdim=True)
        cnt = x.shape[0]
        sum_x  = s if sum_x  is None else sum_x  + s
        sum_x2 = s2 if sum_x2 is None else sum_x2 + s2
        n += cnt
    mu = sum_x / n
    var = sum_x2 / n - mu * mu
    sigma = torch.sqrt(torch.clamp(var, min=1e-8))
    return mu.unsqueeze(0).unsqueeze(0), sigma.unsqueeze(0).unsqueeze(0)  # [1,1,C]

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
def training_and_validation(encoder, forecaster, optimizer,
                            params, num_epochs, train_tensors_dict, val_tensors_dict, 
                            revin_trend, revin_seasonal, revin_resid, 
                            revin_target, global_mu, global_sigma, pred_len=96):

    early_stopper = EarlyStopping(patience=10, verbose=True, delta=0.0)
    best_val_loss = np.inf
    save_path = Path("checkpoints/best_model.pt")
    save_path.parent.mkdir(exist_ok=True)
    save_dir = save_path.parent
    
    for epoch in range(1, num_epochs + 1):
        epoch_time = time.time()
        encoder.train()
        forecaster.train()
        train_losses = []
        
        # ---------------- Training ----------------
        for feat, tensors_f in train_tensors_dict.items():
            optimizer.zero_grad()

            batch_f = to_batch_feature_dict(tensors_f)  # dict of [B,T,C]

            # RevIN on inputs (ignore stats for inputs)
            trend_n, _ = revin_trend.normalize(batch_f["trend"])
            seasonal_n, _ = revin_seasonal.normalize(batch_f["seasonal"])
            resid_n, _ = revin_resid.normalize(batch_f["residual"])

            # Encode normalized inputs
            enc_out = encoder(trend_n, seasonal_n, resid_n)
            check_nan_tensor("trend_h", enc_out["trend_h"])
            check_nan_tensor("season_h", enc_out["season_h"])
            check_nan_tensor("resid_h", enc_out["resid_h"])

            zt, zs, zr = enc_out["trend_h"], enc_out["season_h"], enc_out["resid_h"]

            # Forecast in target-normalized space
            x_hat_n, z_pred = forecaster.forecast(pred_len, trend_z=zt, seasonal_z=zs, resid_z=zr)

            # Build target and normalize with HISTORY-ONLY stats
            x_true = batch_f["trend"] + batch_f["seasonal"] + batch_f["residual"]  # [B,T,C]
            x_hist = x_true[:, :-pred_len, :]  # only historical data
            mu, sigma = revin_target._stats(x_hist)
            sigma = torch.clamp(sigma, min=1e-6)
            x_true_n = (x_true[:, -pred_len:, :] - mu) / sigma

            # Losses
            loss_rec = F.mse_loss(x_hat_n, x_true_n)
            loss_geo = hyperbolic_mse(forecaster.manifold, z_pred)
            loss = loss_rec + 0.1 * loss_geo

            writer.add_scalar("Loss/train_total", loss.item(), epoch)
            writer.add_scalar("Loss/train_reconstruction", loss_rec.item(), epoch)
            writer.add_scalar("Loss/train_hyperbolic", loss_geo.item(), epoch)

            loss.backward()
            train_losses.append(float(loss.detach().item()))
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            
        train_loss = float(np.mean(train_losses))

        # ---------------- Validation ----------------
        encoder.eval()
        forecaster.eval()
        val_losses = []
        val_denorm_losses = []
        val_paper_losses = []
        
        with torch.no_grad():
            for feat, tensors_f in val_tensors_dict.items():
                batch_f = to_batch_feature_dict(tensors_f)
                
                # Normalize input components
                trend_n, _ = revin_trend.normalize(batch_f["trend"])
                seasonal_n, _ = revin_seasonal.normalize(batch_f["seasonal"])
                resid_n, _ = revin_resid.normalize(batch_f["residual"])

                # Encode and forecast
                enc_val = encoder(trend_n, seasonal_n, resid_n)
                zv_t, zv_s, zv_r = enc_val["trend_h"], enc_val["season_h"], enc_val["resid_h"]
                x_val_hat_n, z_val_pred = forecaster.forecast(
                    pred_len, trend_z=zv_t, seasonal_z=zv_s, resid_z=zv_r
                )

                # Get true values and split history/future
                x_val_true = batch_f["trend"] + batch_f["seasonal"] + batch_f["residual"]
                x_hist = x_val_true[:, :-pred_len, :]  # history only
                x_fut = x_val_true[:, -pred_len:, :]   # future (ground truth)
                
                # Compute normalization stats from HISTORY ONLY (no leakage)
                mu, sigma = revin_target._stats(x_hist)
                sigma = torch.clamp(sigma, min=1e-6)
                
                # Normalize target using history-only stats
                x_val_true_n = (x_fut - mu) / sigma
                
                # Compute normalized-space losses
                val_loss_rec = F.mse_loss(x_val_hat_n, x_val_true_n)
                val_loss_geo = hyperbolic_mse(forecaster.manifold, z_val_pred)
                val_loss = val_loss_rec + 0.1 * val_loss_geo
                
                writer.add_scalar("Loss/val_total", val_loss.item(), epoch)
                writer.add_scalar("Loss/val_reconstruction", val_loss_rec.item(), epoch)
                writer.add_scalar("Loss/val_hyperbolic", val_loss_geo.item(), epoch)
                val_losses.append(val_loss.detach().item())
                
                # Denormalize predictions for real-scale metrics
                x_val_hat_den = x_val_hat_n * sigma + mu
                val_denorm_mse = F.mse_loss(x_val_hat_den, x_fut).item()
                val_denorm_mae = F.l1_loss(x_val_hat_den, x_fut).item()
                val_denorm_losses.append(val_denorm_mse)
                
                # Paper-scale metrics (global StandardScaler)
                mu_g = global_mu.to(x_val_hat_den.device)
                sigma_g = torch.clamp(global_sigma.to(x_val_hat_den.device), min=1e-6)
                x_hat_paper = (x_val_hat_den - mu_g) / sigma_g
                x_fut_paper = (x_fut - mu_g) / sigma_g
                val_mse_paper = F.mse_loss(x_hat_paper, x_fut_paper).item()
                val_paper_losses.append(val_mse_paper)
                
        val_loss = float(np.mean(val_losses))
        
        # Logging
        log_msg = f"Epoch [{epoch}/{num_epochs}] - Train: {train_loss:.6f} | Val: {val_loss:.6f}"
        if val_paper_losses:
            val_paper_mse = float(np.mean(val_paper_losses))
            log_msg += f" | Val (paper): {val_paper_mse:.6f}"
            writer.add_scalar("Loss/val_paper_mse", val_paper_mse, epoch)
        print(log_msg)
        print("Epoch: {} cost time: {}".format(epoch, time.time() - epoch_time))
        
        # Early stopping and checkpoint saving
        early_stopper(val_loss, forecaster, str(save_dir))
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch}.")
            break
            
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

def test_evaluation(encoder, forecaster, revin_trend, revin_seasonal, revin_resid, revin_target,
                    test_tensors_dict, global_mu, global_sigma, pred_len=96, device="cuda", 
                    ckpt_path="checkpoints/best_model.pt"):
    """
    Evaluate the trained model on the held-out test set.
    Computes reconstruction, hyperbolic, and denormalized MSE/MAE losses.
    """
    print("\n=== Starting Test Evaluation ===")
    
    # -------------------------------------------------
    # Load best saved model
    # -------------------------------------------------
    if Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        forecaster.load_state_dict(ckpt["forecaster"])
        print(f"Loaded best model checkpoint from epoch {ckpt['epoch']} (Val MSE: {ckpt['val_loss']:.6f})")
    else:
        print("Warning: checkpoint not found — using current weights.")

    encoder.eval()
    forecaster.eval()

    test_mse_n, test_geo, test_mse_den, test_mae_den = [], [], [], []
    test_mse_paper = []

    with torch.no_grad():
        for feat, tensors_f in test_tensors_dict.items():
            batch_f = {k: v.unsqueeze(0).float().to(device) for k, v in tensors_f.items()}

            # Normalize input components
            trend_n, _ = revin_trend.normalize(batch_f["trend"])
            seasonal_n, _ = revin_seasonal.normalize(batch_f["seasonal"])
            resid_n, _ = revin_resid.normalize(batch_f["residual"])

            # Encode
            enc_out = encoder(trend_n, seasonal_n, resid_n)
            zt, zs, zr = enc_out["trend_h"], enc_out["season_h"], enc_out["resid_h"]

            # Forecast
            x_hat_n, z_pred = forecaster.forecast(pred_len, zt, zs, zr)

            # True target - split into history and future
            x_true = batch_f["trend"] + batch_f["seasonal"] + batch_f["residual"]
            x_hist = x_true[:, :-pred_len, :]  # history only
            x_fut = x_true[:, -pred_len:, :]   # future (ground truth)

            # --- Normalized-space losses (no leakage) ---
            mu, sigma = revin_target._stats(x_hist)  # stats from history only
            sigma = torch.clamp(sigma, min=1e-6)
            x_true_n = (x_fut - mu) / sigma
            
            rec_loss = F.mse_loss(x_hat_n, x_true_n).item()
            geo_loss = hyperbolic_mse(forecaster.manifold, z_pred).item()
            test_mse_n.append(rec_loss)
            test_geo.append(geo_loss)

            # --- Denormalized-space losses (for report) ---
            x_hat_den = x_hat_n * sigma + mu
            test_mse_den.append(F.mse_loss(x_hat_den, x_fut).item())
            test_mae_den.append(F.l1_loss(x_hat_den, x_fut).item())

            # --- Paper-scale (global stats) ---
            mu_g = global_mu.to(x_hat_den.device)
            sigma_g = torch.clamp(global_sigma.to(x_hat_den.device), min=1e-6)
            x_hat_paper = (x_hat_den - mu_g) / sigma_g
            x_fut_paper = (x_fut - mu_g) / sigma_g
            test_mse_paper.append(F.mse_loss(x_hat_paper, x_fut_paper).item())

    # -------------------------------------------------
    # Final reporting
    # -------------------------------------------------
    mse_n, geo, mse_den, mae_den, mse_paper = (
        np.mean(test_mse_n), np.mean(test_geo),
        np.mean(test_mse_den), np.mean(test_mae_den),
        np.mean(test_mse_paper)
    )

    print(f"=== TEST RESULTS ===")
    print(f"Normalized-space MSE: {mse_n:.6f}")
    print(f"Hyperbolic loss:      {geo:.6f}")
    print(f"Denormalized MSE:     {mse_den:.6f}")
    print(f"Denormalized MAE:     {mae_den:.6f}")
    print(f"Paper-scale MSE:      {mse_paper:.6f}")
    print("====================\n")

    return {
        "mse_norm": mse_n,
        "geo_loss": geo,
        "mse_denorm": mse_den,
        "mae_denorm": mae_den,
        "mse_paper": mse_paper
    }

# -------------------------------------------------------------
# 1. Device setup
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    # Enable TensorFloat-32 (TF32) on Ampere+ GPUs (e.g., A100)
    torch.backends.cuda.matmul.allow_tf32 = True   # speeds up Linear/MatMul
    torch.backends.cudnn.allow_tf32 = True         # speeds up Conv/related ops
    
print(f"Using device: {device}")

# -------------------------------------------------------------
# 2. Load dataset
# -------------------------------------------------------------
df = pd.read_csv("time-series-dataset/dataset/weather/weather.csv", parse_dates=["date"], index_col="date")
df = df.select_dtypes(include=[np.number])

train_val_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
train_df, val_df = train_test_split(train_val_df, test_size=0.125, shuffle=False)
def split_ratios(train_df, val_df, test_df, total):
    return len(train_df)/total, len(val_df)/total, len(test_df)/total

tr, va, te = split_ratios(train_df, val_df, test_df, len(df))
print(f"Ratios -> train:{tr:.3f}, val:{va:.3f}, test:{te:.3f}")

# -------------------------------------------------------------
# 4. Compute adaptive periods and window lengths
# -------------------------------------------------------------
timebase = TimeBaseMSTL(n_basis_components=5, orthogonal_lr=1e-3, orthogonal_iters=300)

# Automatically infers hourly, daily, weekly steps
steps_per_period = timebase.timesteps_from_index(df)
hourly, daily, weekly = steps_per_period

lookback = weekly
pred_len_96 = 96
pred_len_192 = 192
pred_len_336 = 336
pred_len_720 = 720

print(f"Timesteps per day/week: {daily}, {weekly}")
print(f"lookback window={lookback}, pred_len={pred_len_96}")

# -------------------------------------------------------------
# 5. Decompose training and validation data
# -------------------------------------------------------------

timebase.fit(train_df)
train_components = timebase.transform(train_df)
val_components   = timebase.transform(val_df)
test_components = timebase.transform(test_df)
# -------------------------------------------------------------
# 3. Normalize using training statistics only
# -------------------------------------------------------------
# for name, comp in train_components.items():
#     plot_component_grid(name, train_components)
# plot_variance_contribution(train_components)
# plot_component_correlation_maps(train_components)
