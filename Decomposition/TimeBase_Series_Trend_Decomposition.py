import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL

class TimeBaseMSTL:
    """
    TimeBase-inspired MSTL decomposition using learned orthogonal basis functions
    instead of .
    """

    def __init__(self, n_basis_components=10, orthogonal_lr=1e-3, orthogonal_iters=500):
        """
        Parameters
        ----------
        n_basis_components : int
            Number of basis functions to learn.
        orthogonal_lr : float
            Learning rate for gradient-based orthogonal basis fitting.
        orthogonal_iters : int
            Number of optimization iterations.
        """
        self.n_basis_components = n_basis_components
        self.orthogonal_lr = orthogonal_lr
        self.orthogonal_iters = orthogonal_iters
        self.basis_components = {}
        self.series_coefficients = {}

    # -------------------------------
    # Period Detection
    # -------------------------------
    def timesteps_from_index(self, df):
        """Compute seasonal periods (hourly, daily, weekly) in timesteps."""
        index = df.index
        freq = pd.infer_freq(index)
        if freq is None:
            step = index.to_series().diff().median()
            step_seconds = step.total_seconds()
        else:
            step_seconds = pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).total_seconds()
        if step_seconds == 0:
            raise ValueError("Could not determine valid time step.")
        periods_seconds = [3600, 24*3600, 7*24*3600]
        steps = [max(1, int(round(p / step_seconds))) for p in periods_seconds]
        return steps

    # -------------------------------
    # Segment Extraction
    # -------------------------------
    def extract_periodic_segments(self, df, steps_per_period):
        """Extract repeating segments from all series."""
        n_series = len(df.columns)
        n_points = len(df)
        segments_dict = {}

        for period in steps_per_period:
            n_segments = n_points // period
            segments = np.zeros((n_series, n_segments, period))
            for i, col in enumerate(df.columns):
                for seg in range(n_segments):
                    start_idx = seg * period
                    end_idx = start_idx + period
                    if end_idx <= n_points:
                        segments[i, seg, :] = df[col].iloc[start_idx:end_idx].values
            segments_dict[period] = segments
        return segments_dict

    # -------------------------------
    # Orthogonal Basis Learning (TimeBase-style)
    # -------------------------------
    def learn_orthogonal_basis(self, segments):
        """
        Learn orthogonal basis functions from all valid segments.
        Instead of PCA, this uses a gradient-based orthogonalization process.
        """
        n_series, n_segments, seg_len = segments.shape
        flattened = segments.reshape(-1, seg_len)
        valid_mask = np.any(flattened != 0, axis=1)
        valid_segments = flattened[valid_mask]

        if len(valid_segments) == 0:
            return np.zeros((self.n_basis_components, seg_len)), np.zeros((n_series, self.n_basis_components))

        # Initialize random basis (orthogonalized)
        rng = np.random.default_rng(42)
        basis = rng.normal(size=(self.n_basis_components, seg_len))
        basis = self._orthogonalize(basis)

        # Gradient descent to minimize reconstruction error + orthogonality
        for _ in range(self.orthogonal_iters):
            # Reconstruction
            coeffs = valid_segments @ basis.T  # [num_segments, n_basis]
            recon = coeffs @ basis             # [num_segments, seg_len]
            error = valid_segments - recon

            # Gradient: minimize MSE + orthogonality penalty
            grad = -2 * (error.T @ coeffs).T / len(valid_segments)
            gram = basis @ basis.T
            off_diag = gram - np.diag(np.diag(gram))
            grad += 0.01 * (off_diag @ basis)  # orthogonality regularizer

            # Update
            basis -= self.orthogonal_lr * grad
            basis = self._orthogonalize(basis)

        # Compute mean coefficients per series
        series_coeffs = np.zeros((n_series, self.n_basis_components))
        for i in range(n_series):
            valid_series = segments[i][np.any(segments[i] != 0, axis=1)]
            if len(valid_series) > 0:
                coeffs = valid_series @ basis.T
                series_coeffs[i] = coeffs.mean(axis=0)

        return basis, series_coeffs

    def _orthogonalize(self, matrix):
        """Orthonormalize a set of basis vectors using Gram-Schmidt."""
        Q, _ = np.linalg.qr(matrix.T)
        return Q.T

    # -------------------------------
    # MSTL Decomposition
    # -------------------------------
    def decompose_basis_components(self, basis_components, period):
        """Decompose each orthogonal basis using MSTL."""
        decomposed_basis = {}
        for i, component in enumerate(basis_components):
            try:
                repeated = np.tile(component, 10)
                mstl = MSTL(repeated, periods=[period])
                res = mstl.fit()
                decomposed_basis[f'basis_{i}'] = {
                    'trend': res.trend[:period],
                    'seasonal': res.seasonal[:period] if res.seasonal.ndim == 1 else res.seasonal[:period, 0],
                    'resid': res.resid[:period],
                }
            except Exception as e:
                print(f"Failed to decompose basis {i}: {e}")
                decomposed_basis[f'basis_{i}'] = {
                    'trend': np.mean(component) * np.ones_like(component),
                    'seasonal': component - np.mean(component),
                    'resid': np.zeros_like(component),
                }
        return decomposed_basis

    # -------------------------------
    # Reconstruct per-series signals
    # -------------------------------
    def reconstruct_series_decomposition(self, df, decompositions, coeffs_dict):
        """Reconstruct each series from decomposed basis."""
        series_decompositions = {}
        n_points = len(df)

        for i, name in enumerate(df.columns):
            total_trend = np.zeros(n_points)
            total_seasonal = np.zeros(n_points)

            for period, basis_decomp in decompositions.items():
                coeffs = coeffs_dict[period][i]
                for j, (bname, comp) in enumerate(basis_decomp.items()):
                    coeff = coeffs[j] if j < len(coeffs) else 0
                    repeats = (n_points + period - 1) // period
                    for key in ["trend", "seasonal"]:
                        tiled = np.tile(comp[key], repeats)[:n_points]
                        if key == "trend":
                            total_trend += coeff * tiled
                        else:
                            total_seasonal += coeff * tiled

            residual_actual = df[name].values - (total_trend + total_seasonal)
            series_decompositions[name] = {
                "trend": total_trend,
                "seasonal": total_seasonal,
                "residual": residual_actual,
            }
        return series_decompositions

    # -------------------------------
    # Full Pipeline
    # -------------------------------
    def fit_transform(self, df):
        print(f"Decomposing {df.shape[1]} series using orthogonal TimeBase-MSTL...")
        steps_per_period = self.timesteps_from_index(df)
        print(f"⏱ Inferred periods: {steps_per_period} timesteps")

        segments_dict = self.extract_periodic_segments(df, steps_per_period)
        decompositions, coeffs_dict = {}, {}

        for period, segments in segments_dict.items():
            print(f"\nPeriod {period} → segments {segments.shape[1]}")
            basis, coeffs = self.learn_orthogonal_basis(segments)
            decomposed_basis = self.decompose_basis_components(basis, period)
            decompositions[period] = decomposed_basis
            coeffs_dict[period] = coeffs

        return self.reconstruct_series_decomposition(df, decompositions, coeffs_dict)
