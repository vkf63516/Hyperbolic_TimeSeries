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
        """Compute seasonal periods (daily, weekly) in timesteps."""
        index = df.index
        freq = pd.infer_freq(index)
        if freq is None:
            step = index.to_series().diff().median()
            step_seconds = step.total_seconds()
        else:
            step_seconds = pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).total_seconds()
        if step_seconds == 0:
            raise ValueError("Could not determine valid time step.")
        periods_seconds = [24*3600, 7*24*3600]
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
            segment_len = period * 2
            n_segments = n_points // segment_len #Extract a specific number of segments for each seasonality type
            segments = np.zeros((n_series, n_segments, segment_len))
            for i, col in enumerate(df.columns):
                for seg in range(n_segments):
                    start_idx = seg * segment_len
                    end_idx = start_idx + segment_len
                    if end_idx <= n_points:
                        segments[i, seg, :] = df[col].iloc[start_idx:end_idx].values
            segments_dict[period] = segments
            print(
                f"Extracted {n_segments} segments (period={period}, "
                f"length={segment_len}, total_features={n_series})"
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
        print(f"Learning from {n_series} series with {n_segments} segments each")
        flattened = segments.reshape(-1, seg_len)
        valid_mask = np.any(flattened != 0, axis=1)
        valid_segments = flattened[valid_mask]
        all_segment_coeffs = np.zeros((n_series, n_segments, self.n_basis_components))
        if len(valid_segments) == 0:
            return (
                np.zeros((self.n_basis_components, seg_len)), 
                np.zeros((n_series, n_segments, self.n_basis_components))
            )


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

            # Keep ALL segment-level coefficients (don't average!)
        all_segment_coeffs = np.zeros((n_series, n_segments, self.n_basis_components))
    
        for i in range(n_series):
            series_segments = segments[i]
            valid_mask_series = np.any(series_segments != 0, axis=1)
            valid_series = series_segments[valid_mask_series]
        
            if len(valid_series) > 0:
                coeffs = valid_series @ basis.T  # (n_valid_segments, n_basis)
                all_segment_coeffs[i, :len(coeffs), :] = coeffs
            # ↑ Keep ALL segment coefficients, DON'T average!
        return basis, all_segment_coeffs  # Shape: (n_series, n_segments, n_basis) ✓

    def _orthogonalize(self, matrix):
        """Orthonormalize a set of basis vectors using Gram-Schmidt."""
        Q, _ = np.linalg.qr(matrix.T)
        return Q.T

    # -------------------------------
    # MSTL Decomposition
    # -------------------------------

    def decompose_basis_components(self, basis_components, periods, seasonal_type="seasonal_daily"):
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
                valid_periods = periods
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
    def reconstruct_series_decomposition(self, df, decompositions, coeffs_dict, 
                                     smooth_window_ratio=0.1, use_segment_level=True):
        """
        Reconstruct each series from decomposed basis.
    
        Parameters
        ----------
        use_segment_level : bool
            If True, uses segment-level coefficients (temporal)
            If False, uses averaged coefficients (static)
        """
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
        
            total_seasonal_daily = np.zeros(n_points)
            total_seasonal_weekly = np.zeros(n_points)
        
            for period, basis_decomp in decompositions.items():
                if use_segment_level:
                # NEW: Use segment-level coefficients
                    all_segment_coeffs = coeffs_dict[period][i]  # (n_segments, n_basis)
                    segment_length = period * 2  # Assuming 2× multiplier
                
                # Reconstruct each segment and concatenate
                    for seg_idx, seg_coeffs in enumerate(all_segment_coeffs):
                        start_idx = seg_idx * segment_length
                        end_idx = min(start_idx + segment_length, n_points)
                        seg_len = end_idx - start_idx
                    
                        if seg_len == 0:
                            continue
                    
                    # Reconstruct this segment's seasonal
                        for j, (bname, comp) in enumerate(basis_decomp.items()):
                            coeff = seg_coeffs[j] if j < len(seg_coeffs) else 0
                        
                            for key in ["seasonal_daily", "seasonal_weekly"]:
                                if key not in comp or comp[key] is None:
                                    continue
                            
                                pattern = comp[key]
                                idx = np.arange(seg_len) % len(pattern)
                                repeated = pattern[idx]
                            
                                if key == "seasonal_daily":
                                    total_seasonal_daily[start_idx:end_idx] += coeff * repeated
                                if key == "seasonal_weekly":
                                    total_seasonal_weekly[start_idx:end_idx] += coeff * repeated
            
        
            residual_actual = df[name].values - (trend_global + 
                                                total_seasonal_weekly + 
                                                total_seasonal_daily)
        
            series_decompositions[name] = {
                "trend": trend_global,
                "seasonal_weekly": total_seasonal_weekly,
                "seasonal_daily": total_seasonal_daily,
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
        self.series_coefficients = {}  # Now stores (n_series, n_segments, n_basis)

        for period, segments in segments_dict.items():
            basis, all_coeffs = self.learn_orthogonal_basis(segments)
            self.basis_components[period] = basis
            self.series_coefficients[period] = all_coeffs  # ✓ Segment-level!
        
            print(f"Learned basis for period {period}: basis shape {basis.shape}, coeffs shape {all_coeffs.shape}")
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
                seasonal_type = "seasonal_daily"
            else:
                seasonal_type = "seasonal_weekly"
            decomposed_basis = self.decompose_basis_components(basis, [int(period/2)], seasonal_type)
            decompositions[period] = decomposed_basis
        return self.reconstruct_series_decomposition(df, decompositions, self.series_coefficients)

