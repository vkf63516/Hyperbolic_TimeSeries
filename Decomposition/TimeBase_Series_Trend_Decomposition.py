import numpy as np
import pandas as pd
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))
from Decomposition.BatchedMSTL import BatchedMSTL

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
        self.feature_names = None

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
            print(
            f"Extracted {n_segments} segments (period={period}, "
            f"length={period}, total_features={n_series})"
        )
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
        print(n_series)
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

    def decompose_basis_components(self, basis_components, periods, seasonal_type="seasonal_hourly"):
        """
        Decompose all orthogonal basis components together using BatchedMSTL.
        Each basis is treated as a separate time series in the batch.
        """
        decomposed_basis = {}
        print(f"⏱ Decomposing {len(basis_components)} basis components using BatchedMSTL...")

        try:
            # --- Convert all basis components into a 2D array ---
            basis_matrix = np.stack(basis_components, axis=0)  # [n_basis, n_timesteps]
            n_basis, n_points = basis_matrix.shape

            # --- Filter valid periods for this basis length ---
            valid_periods = [p for p in periods if p < n_points / 2]
            if not valid_periods:
                valid_periods = [min(periods)]
            print(f"Valid MSTL periods: {valid_periods}")

            # --- Run Batched MSTL ---
            mstl = BatchedMSTL(basis_matrix, periods=valid_periods)
            batch_results = mstl.fit()  # list of DecomposeResult per basis

            # --- Convert outputs into dictionary form ---
            for i, res in enumerate(batch_results):
                trend = res.trend.values if isinstance(res.trend, pd.Series) else res.trend
                resid = res.resid.values if isinstance(res.resid, pd.Series) else res.resid

                # Handle multiple seasonal columns
                if isinstance(res.seasonal, pd.DataFrame):
                    seasonal_total = res.seasonal.sum(axis=1).values
                else:
                    seasonal_total = res.seasonal.values

                decomposed_basis[f"basis_{i}"] = {
                    "trend": trend,
                    seasonal_type: seasonal_total,
                    "resid": resid,
                }

        except Exception as e:
            print(f" Batched MSTL failed: {e}")
        # --- Safe fallback for entire batch ---
            for i, component in enumerate(basis_components):
                base = np.mean(component) * np.ones_like(component)
                seas = component - np.mean(component)
                decomposed_basis[f"basis_{i}"] = {
                    "trend": base,
                    "seasonal_hourly": seas if seasonal_type == "seasonal_hourly" else np.zeros_like(base),
                    "seasonal_daily": seas if seasonal_type == "seasonal_daily" else np.zeros_like(base),
                    "seasonal_weekly": seas if seasonal_type == "seasonal_weekly" else np.zeros_like(base),
                    "resid": np.zeros_like(component),
                }

        return decomposed_basis


    # -------------------------------
    # Reconstruct per-series signals
    # -------------------------------
    """
    Rebuilds (Trend + Seasonal + Residual) from segment level decompositions
    """
    def reconstruct_series_decomposition(self, df, decompositions, coeffs_dict, smooth_window_ratio=0.1):
        """Reconstruct each series from decomposed basis."""
        series_decompositions = {}
        n_points = len(df)
        window = max(5, int(len(df) * smooth_window_ratio))

        for i, name in enumerate(df.columns):
            trend_global = (
                pd.Series(df[name].values)
                .rolling(window=window, min_periods=1, center=True)
                .mean()
                .values
            )

            total_seasonal_hourly = np.zeros(n_points)
            total_seasonal_daily = np.zeros(n_points)
            total_seasonal_weekly = np.zeros(n_points)

            for period, basis_decomp in decompositions.items():
                coeffs = coeffs_dict[period][i] # learned coefficients for this feature, telling how much each basis contributes.
                # looping through each basis function (trend, seasonal, residual pattern).
                for j, (bname, comp) in enumerate(basis_decomp.items()):
                    coeff = coeffs[j] if j < len(coeffs) else 0
                    for key in ["seasonal_hourly", "seasonal_daily", "seasonal_weekly"]:
                        if key not in comp:
                            continue
                        pattern = comp[key]
                        if pattern is None or len(pattern) == 0:
                            continue
                        # Repeat pattern efficiently using modulo indexing
                        idx = np.arange(n_points) % len(pattern)
                        repeated = pattern[idx]
                       
                        if key == "seasonal_hourly":
                            total_seasonal_hourly += coeff * repeated
                        if key == "seasonal_daily":
                            total_seasonal_daily += coeff * repeated 
                        if key == "seasonal_weekly":
                            total_seasonal_weekly += coeff * repeated 

            residual_actual = df[name].values - (trend_global + (
                                                                total_seasonal_weekly + 
                                                                total_seasonal_daily +
                                                                total_seasonal_hourly
                                                                ))
            series_decompositions[name] = {
                "trend": trend_global,
                "seasonal_weekly": total_seasonal_weekly,
                "seasonal_daily": total_seasonal_daily,
                "seasonal_hourly": total_seasonal_hourly,
                "residual": residual_actual
            }
        return series_decompositions

    # -------------------------------
    # Full Pipeline
    # -------------------------------
    def fit(self, df):
        """
        Learn bases and coefficients from the training data,
        but don't reconstruct or return anything yet.
        """
        print(f"Learning orthogonal bases from {df.shape[1]} series...")
        steps_per_period = self.timesteps_from_index(df)
        self.feature_names = list(df.columns)
        self.steps_per_period = steps_per_period

        segments_dict = self.extract_periodic_segments(df, steps_per_period)
        self.basis_components = {}
        self.series_coefficients = {}

        period_lst = [int(key) for key in segments_dict.keys()]
        for period, segments in segments_dict.items():
            basis, coeffs = self.learn_orthogonal_basis(segments)
            self.basis_components[period] = basis
            self.series_coefficients[period] = coeffs
        return self

    def transform(self, df):
        """
        Use the learned bases and coefficients from `fit()`
        to decompose and reconstruct new data (val/test sets).
        """
        if not self.basis_components or not self.series_coefficients:
            raise RuntimeError("Call fit(train_df) before transform().")

        print(f"Reusing learned bases to decompose {df.shape[1]} series...")
        period_lst = list(self.basis_components.keys())
        decompositions = {}
        for period, basis in self.basis_components.items():
            if period == min(period_lst):
                seasonal_type = "seasonal_hourly"
            elif period == max(period_lst):
                seasonal_type = "seasonal_weekly"
            else:
                seasonal_type = "seasonal_daily"
            decomposed_basis = self.decompose_basis_components(basis, period_lst, seasonal_type)
            decompositions[period] = decomposed_basis
        return self.reconstruct_series_decomposition(df, decompositions, self.series_coefficients)

    # def fit_transform(self, df):
    #     print(f"Decomposing {df.shape[1]} series using orthogonal TimeBase-MSTL...")
    #     steps_per_period = self.timesteps_from_index(df)
    #     print(f"⏱ Inferred periods: {steps_per_period} timesteps")

    #     segments_dict = self.extract_periodic_segments(df, steps_per_period)
    #     decompositions = {}
    #     coeffs_dict = {}
    #     period_lst = [int(key) for key in segments_dict.keys()]
    #     for period, segments in segments_dict.items():
    #         print(f"\nPeriod {period} → segments {segments.shape[1]}")
    #         basis, coeffs = self.learn_orthogonal_basis(segments)
    #         if period == min(period_lst):
    #             seasonal_type = "seasonal_hourly"
    #         elif period == max(period_lst):
    #             seasonal_type = "seasonal_weekly"
    #         else:
    #             seasonal_type = "seasonal_daily"
    #         decomposed_basis = self.decompose_basis_components(basis, period_lst, seasonal_type)
    #         decompositions[period] = decomposed_basis
    #         coeffs_dict[period] = coeffs

    #     return self.reconstruct_series_decomposition(df, decompositions, coeffs_dict)
