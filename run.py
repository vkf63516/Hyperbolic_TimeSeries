import torch
import random
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pandas as pd
import geoopt
from pathlib import Path
import sys
import numpy as np

# --------------------------------------------------------------------
# Setup paths (so local packages can be imported easily)
# --------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[0]))

from encoder.mamba_encoders_lorentz import ParallelLorentzEncoder
from Decomposition.Series_Trend_Decomposition import trend_seasonal_decomposition_parallel, timesteps_based_on_frequency
from Decomposition.tensor_utils import build_decomposition_tensors
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

# -------------------------------------------------------------
# 1. Device setup
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------------------------------------
# 2. Load dataset
# -------------------------------------------------------------
df = pd.read_csv("time-series-dataset/dataset/weather/weather.csv", parse_dates=["date"], index_col="date")
df = df[["OT"]].dropna()

train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)

# -------------------------------------------------------------
# 3. Compute adaptive periods and window lengths
# -------------------------------------------------------------
freq = pd.infer_freq(train_df.index)
hourly, daily, weekly = timesteps_based_on_frequency(freq, train_df.index)

# Define adaptive windows
seq_len = weekly          # one full weekly seasonal cycle
window_size = 2 * weekly  # two weeks of data per training window
pred_len = int(daily)     # predict one day ahead

print(f"Frequency: {freq}")
print(f"Timesteps per hour/day/week: {hourly}, {daily}, {weekly}")
print(f"seq_len={seq_len}, window_size={window_size}, pred_len={pred_len}")

# -------------------------------------------------------------
# 4. Decompose training and validation data
# -------------------------------------------------------------
train_components = trend_seasonal_decomposition_parallel(train_df, window_size=window_size)
val_components   = trend_seasonal_decomposition_parallel(val_df, window_size=window_size)

def select_last_window(components_dict):
    """
    Selects the latest decomposition window for each feature and
    builds PyTorch tensors from its components.
    """
    out = {}
    for col, df_decomp in components_dict.items():
        last_win = df_decomp.loc[max(df_decomp.index.get_level_values('window'))]
        out[col] = build_decomposition_tensors(last_win)
    return out

train_tensors_dict = select_last_window(train_components)
val_tensors_dict   = select_last_window(val_components)

# Convert each feature’s tensors to batch format
def to_batch_feature_dict(feature_dict):
    return {k: v.unsqueeze(0).to(device) for k, v in feature_dict.items()}

# --------------------------------------------------------------------
# 5. Initialize models
# --------------------------------------------------------------------
embed_dim = 32
hidden_dim = 64
output_dim = 1

encoder = ParallelLorentzEncoder(seq_len=seq_len, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
manifold = encoder.manifold
forecaster = HyperbolicSeqForecaster(embed_dim=embed_dim, hidden_dim=hidden_dim, output_dim=output_dim, manifold=manifold).to(device)

optimizer = geoopt.optim.RiemannianAdam(
    list(encoder.parameters()) + list(forecaster.parameters()), lr=1e-3
)

# --------------------------------------------------------------------
# 6. Hyperbolic loss helper
# --------------------------------------------------------------------
def hyperbolic_mse(manifold, z_pred):
    dist = manifold.dist(z_pred[:, 1:], z_pred[:, :-1])
    return (dist ** 2).mean()

# --------------------------------------------------------------------
# 7. Training loop
# --------------------------------------------------------------------
num_epochs = 200
best_val_loss = np.inf
save_path = Path("checkpoints/best_model.pt")
save_path.parent.mkdir(exist_ok=True)

for epoch in range(1, num_epochs + 1):
    encoder.train()
    forecaster.train()
    optimizer.zero_grad()

    train_losses = []
    for feat, tensors_f in train_tensors_dict.items():
        batch_f = to_batch_feature_dict(tensors_f)

        enc_out = encoder(batch_f["trend"], batch_f["seasonal"], batch_f["residual"])
        zt, zs, zr = enc_out["trend_h"], enc_out["season_h"], enc_out["resid_h"]

        x_hat, z_pred = forecaster.forecast(pred_len, trend_z=zt, seasonal_z=zs, resid_z=zr)
        x_true = batch_f["trend"][:, -pred_len:, :]

        loss_rec = F.mse_loss(x_hat, x_true)
        loss_geo = hyperbolic_mse(forecaster.manifold, z_pred)
        loss = loss_rec + 0.1 * loss_geo

        loss.backward()
        train_losses.append(loss_rec.detach().item())

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
            x_val_hat, _ = forecaster.forecast(pred_len, trend_z=zv_t, seasonal_z=zv_s, resid_z=zv_r)
            x_val_true = batch_f["trend"][:, -pred_len:, :]
            val_losses.append(F.mse_loss(x_val_hat, x_val_true).item())

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
        print(f"✅ Saved best model (Val MSE: {val_loss:.6f})")

print("Training complete.")