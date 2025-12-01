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

# --------------------------------------------------------------------
# Setup paths (so local packages can be imported easily)
# --------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[0]))

from Decomposition.Orthogonal_Series_Trend_Decomposition import orthogonalMSTL
from Decomposition.visualization_utils import plot_variance_contribution, plot_component_correlation_maps
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True   # speeds up Linear/MatMul
    torch.backends.cudnn.allow_tf32 = True         # speeds up Conv/related ops
    
print(f"Using device: {device}")

# -------------------------------------------------------------
# 2. Load dataset
# -------------------------------------------------------------
df = pd.read_csv("time-series-dataset/dataset/traffic.csv", parse_dates=["date"], index_col="date")
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
orthogonal = orthogonalMSTL(n_basis_components=10, orthogonal_lr=1e-3, orthogonal_iters=300)

# Automatically infers fine, coarse steps
steps_per_period = orthogonal.detect_periods(df)
fine, coarse = steps_per_period

lookback = coarse
pred_len_96 = 96
pred_len_192 = 192
pred_len_336 = 336
pred_len_720 = 720

print(f"Timesteps per day/week: {fine}, {coarse}")
print(f"lookback window={lookback}, pred_len={pred_len_96}")

# -------------------------------------------------------------
# 5. Decompose training and validation data
# -------------------------------------------------------------

orthogonal.fit(train_df)
train_components = orthogonal.transform(train_df)
# -------------------------------------------------------------
# 3. Normalize using training statistics only
# -------------------------------------------------------------
plot_variance_contribution(train_components)
plot_component_correlation_maps(train_components)
