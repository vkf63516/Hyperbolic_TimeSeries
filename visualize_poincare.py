import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from numpy.linalg import norm
from math import acosh
from sklearn.decomposition import PCA

from geoopt.manifolds import PoincareBall

# Load encodedings
H = torch.tensor(np.load('results/ETTh1_96_96_Poincare_type_exp1_Segment_HyperbolicForecasting_ETTh1_decomposition_ETTh1.csv_ftM_sl96_pl96_exp_ebfixed_Poincare_0_seed2023/hyper.npy'))     # (100, 3, 384)
untrained_encodedings = torch.tensor(np.load('results/ETTh1_96_96_Poincare_type_exp1_Segment_HyperbolicForecasting_ETTh1_decomposition_ETTh1.csv_ftM_sl96_pl96_exp_ebfixed_Poincare_0_seed2023/true.npy')) # (100, 3, 384)
def rescale_to_radius(x, target=0.9):
    # x: [N, d] Poincaré coordinates
    norm = torch.norm(x, dim=-1, keepdim=True) + 1e-9
    scale = (target / norm).clamp(max=10.0)   # avoid massive blow-ups
    return x * scale

# Setup manifold
manifold = PoincareBall(c=1.0)
poincare_points = H     # shape (2785, 672)
poincare_scaled = rescale_to_radius(poincare_points, target=0.9)
u = poincare_scaled[:, :2].cpu().numpy()

plt.scatter(u[:,0], u[:,1], s=6)
plt.gca().add_patch(plt.Circle((0,0), 1, fill=False))
plt.gca().set_aspect("equal")
plt.show()
# import umap
# import matplotlib.pyplot as plt

# # Flatten across batch if needed
# def logmap0_poincare(x, eps=1e-9):
#     normx = np.linalg.norm(x, axis=-1, keepdims=True)
#     return 2 * x / (1 - normx**2 + eps)


# # ---------- Hyperbolic Exp Map at Origin ----------
# def expmap0_poincare(v, eps=1e-9):
#     normv = np.linalg.norm(v, axis=-1, keepdims=True)
#     return np.tanh(normv / 2) * (v / (normv + eps))


# x = H[0]                  # one trajectory, shape (T, 32)


# # ---------- Step 1: Map from ℍ³² → tangent space T₀ℍ³² ----------
# x_tangent = logmap0_poincare(x)


# # ---------- Step 2: PCA to reduce 32D tangent → 2D ----------
# pca = PCA(n_components=2)
# x_2d_tangent = pca.fit_transform(x_tangent)


# # ---------- Step 3: Map back into the Poincaré disk ℍ² ----------
# x_2d_poincare = expmap0_poincare(x_2d_tangent)


# # ---------- Step 4: Plot in the true 2D Poincaré disk ----------
# plt.figure(figsize=(6,6))
# plt.scatter(x_2d_poincare[:,0], x_2d_poincare[:,1], s=5)

# # Draw unit circle (disk boundary)
# circle = plt.Circle((0,0), 1, fill=False, color='black', linewidth=2)
# plt.gca().add_patch(circle)

# plt.title("2D Hyperbolic Visualization via Log → PCA → Exp")
# plt.gca().set_aspect('equal')
# plt.show()
