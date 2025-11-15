import numpy as np
import pandas as pd
import sys 
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))
from statsmodels.tsa.seasonal import MSTL
from Decomposition.Series_Trend_Decomposition import trend_seasonal_decomposition_parallel

class TimeBaseMSTL:
    """
    TimeBase-inspired MSTL decomposition using learned orthogonal basis functions
    instead of .
    """

    def __init__(self, n_basis_components=10, orthogonal_lr=1e-3, orthogonal_iters=500, seq_len=96):
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
        self.seq_len = seq_len
        self.basis_components = {}
        self.series_coefficients = {}
        self.feature_names = None


    def _should_use_original_mstl(self, df, threshold=15):
        """
        Decide whether to use original MSTL or TimeBase MSTL.
    
        Parameters
        ----------
        df : pd.DataFrame
            Input data
        threshold : int
            Number of features below which to use original MSTL for accuracy
    
        Returns
        -------
        bool
            True if should use original MSTL, False for TimeBase MSTL
        """
        n_features = df.shape[1]
    
        if n_features <= threshold:
            print(f"Dataset has {n_features} features (≤ {threshold})")
            print(f"Using original MSTL for accurate decomposition\n")
            return True
        else:
            print(f"Dataset has {n_features} features (> {threshold})")
            print(f"Using TimeBase MSTL (basis decomposition)\n")
            return False

    # -------------------------------
    # Period Detection
    # -------------------------------
    def detect_periods(self, df):
        """
        Data-driven period detection.
    
        Computes periods relative to seq_len (if provided) or dataset length.
        Uses fixed ratios to extract fine and coarse temporal patterns.
    
        Parameters
        ----------
        df : pd.DataFrame
            Input time series data
    
        Returns
        -------
        list of int
            [fine_period, coarse_period] in timesteps
        """
        reference_length = self.seq_len 
        n_points = len(df)
    
        print(f"\n{'='*70}")
        print(f"PERIOD DETECTION")
        print(f"{'='*70}")
        print(f"Dataset length: {n_points} points")
        print(f"Reference length (seq_len): {reference_length} points")
    
        # Compute periods as fractions of reference length
        FINE_RATIO = 4      # Fine period: 25% of reference
        COARSE_RATIO = 2    # Coarse period: 50% of reference
    
        period_fine = max(3, reference_length // FINE_RATIO)
        period_coarse = max(5, reference_length // COARSE_RATIO)
        # Ensure coarse > fine (at least 1.5x)
        if period_coarse < period_fine * 2.5:
            period_coarse = int(period_fine * 4)
    
        # Check how many segments we can extract from dataset
        segment_len_fine = period_fine * 2
        segment_len_coarse = period_coarse * 2
        n_segments_fine = n_points // segment_len_fine
        n_segments_coarse = n_points // segment_len_coarse
    
        print(f"\nDetected periods:")
        print(f"  Fine:   {period_fine:5d} steps ({period_fine/reference_length*100:5.1f}% of reference)")
        print(f"          → segment_len={segment_len_fine}, yields {n_segments_fine} segments")
        print(f"  Coarse: {period_coarse:5d} steps ({period_coarse/reference_length*100:5.1f}% of reference)")
        print(f"          → segment_len={segment_len_coarse}, yields {n_segments_coarse} segments")
    
        # Validate: need at least 3 segments
        min_segments = 3
        if n_segments_fine < min_segments or n_segments_coarse < min_segments:
            print(f"\n  Insufficient segments, adjusting to use dataset length...")
            period_fine = max(3, n_points // 20)
            period_coarse = max(5, n_points // 8)
            n_segments_fine = n_points // (period_fine * 2)
            n_segments_coarse = n_points // (period_coarse * 2)
            print(f"  Adjusted fine:   {period_fine} → {n_segments_fine} segments")
            print(f"  Adjusted coarse: {period_coarse} → {n_segments_coarse} segments")
        else:
            print(f"\n Periods valid!")
    
        print(f"{'='*70}\n")
        return [period_fine, period_coarse]
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

    def decompose_basis_components(self, basis_components, periods, seasonal_type="seasonal_fine"):
        """
        Decompose each orthogonal basis component individually using statsmodels MSTL.
        Each basis is treated as a 1D time series.
    
        Parameters
        ----------
        basis_components : np.ndarray
            Shape (n_basis, n_timesteps) - e.g., (10, 288)
        periods : list of int
            Seasonal periods for MSTL - e.g., [72]
        seasonal_type : str
            Either "seasonal_fine" or "seasonal_coarse"
    
        Returns
        -------
        dict
            Decomposition for each basis: {f"basis_{i}": {"trend": ..., seasonal_type: ..., "residual": ...}}
        """
        decomposed_basis = {}
        n_basis = len(basis_components)
        print(f"⏱ Decomposing {n_basis} basis components using statsmodels MSTL...")

        for i in range(n_basis):
        # Extract 1D basis component
            basis_1d = basis_components[i]  # Shape: (n_timesteps,) e.g., (288,)
            n_points = len(basis_1d)
        
            try:
            # Filter valid periods for this basis length
                valid_periods = [p for p in periods if p < n_points / 2]
                if not valid_periods:
                    print(f"⚠ Warning: No valid periods for basis_{i} (length={n_points}), using original periods")
                    valid_periods = periods
            
                print(f"  Basis {i}: length={n_points}, periods={valid_periods}")
            
            # Run statsmodels MSTL on this 1D basis
                mstl = MSTL(basis_1d, periods=valid_periods)
                result = mstl.fit()
            
            # Extract components
                trend = result.trend
                residual = result.resid
                seasonal_total = result.seasonal.sum(axis=1)
            
            # # Handle seasonal component(s)
            #     if isinstance(result.seasonal, pd.DataFrame):
            #     # Multiple seasonal components - sum them
            #         seasonal_total = result.seasonal.sum(axis=1).values
            #     else:
            #     # Single seasonal component
            #         seasonal_total = result.seasonal.sum(axis=1)
            
                decomposed_basis[f"basis_{i}"] = {
                    "trend": trend,
                    seasonal_type: seasonal_total,
                    "residual": residual,
                }
            
            except Exception as e:
                print(f"⚠ MSTL failed for basis_{i}: {e}")
                # Fallback: simple mean-centering
                base = np.mean(basis_1d) * np.ones_like(basis_1d)
                seas = basis_1d - base
                decomposed_basis[f"basis_{i}"] = {
                    "trend": base,
                    seasonal_type: seas,
                    "residual": np.zeros_like(basis_1d),
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
        The segments are used internally for reconstruction.
        Segments to series
    
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
                .rolling(window=window, min_periods=1, center=False)
                .mean()
                .values
            )
        
            total_seasonal_fine = np.zeros(n_points)
            total_seasonal_coarse = np.zeros(n_points)
        
            for period, basis_decomp in decompositions.items():
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
                    
                        for key in ["seasonal_fine", "seasonal_coarse"]:
                            if key not in comp or comp[key] is None:
                                continue
                        
                            pattern = comp[key]
                            idx = np.arange(seg_len) % len(pattern)
                            repeated = pattern[idx]
                        
                            if key == "seasonal_fine":
                                total_seasonal_fine[start_idx:end_idx] += coeff * repeated
                            if key == "seasonal_coarse":
                                total_seasonal_coarse[start_idx:end_idx] += coeff * repeated
        
        
            residual_actual = df[name].values - (trend_global + 
                                                total_seasonal_coarse + 
                                                total_seasonal_fine)
        
            series_decompositions[name] = {
                "trend": trend_global,
                "seasonal_coarse": total_seasonal_coarse,
                "seasonal_fine": total_seasonal_fine,
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
        steps_per_period = self.detect_periods(df)
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
        if self._should_use_original_mstl(df):
            return self._transform_with_original_mstl(df)
        else:
            return self._transform_with_orthogonal_mstl(df)

    def _transform_with_original_mstl(self, df):
        """
        Use original MSTL for decomposition (accurate, for low-dim data).
        """
        print(f"Reusing original MSTL to decompose {df.shape[1]} series...")
    
        # Call your original MSTL
        decomposed_dict = trend_seasonal_decomposition_parallel(df, self.steps_per_period)
        
        return decomposed_dict

    def _transform_with_orthogonal_mstl(self, df):
        """
        Use the learned bases and coefficients from `fit()`
        to decompose and reconstruct new data (val/test sets).
        """

        print(f"Reusing learned bases to decompose {df.shape[1]} series...")
        period_lst = list(self.basis_components.keys())
        decompositions = {}
    
        for period, basis in self.basis_components.items():
            sub_periods = [int(period/4), int(period/3), int(period/2)]
            if period == min(period_lst):
                seasonal_type = "seasonal_fine"
            else:
                seasonal_type = "seasonal_coarse"
            decomposed_basis = self.decompose_basis_components(basis, sub_periods, seasonal_type)
            decompositions[period] = decomposed_basis
        return self.reconstruct_series_decomposition(df, decompositions, self._compute_new_coefficients(df))
   
    def _compute_new_coefficients(self, df):
        """Compute coefficients by projecting new data onto learned basis."""
        new_segments_dict = self.extract_periodic_segments(df, self.steps_per_period)
        new_coefficients = {}
    
        for period, new_segments in new_segments_dict.items():
            learned_basis = self.basis_components[period]
            n_series, n_segments, seg_len = new_segments.shape
            all_segment_coeffs = np.zeros((n_series, n_segments, self.n_basis_components))

            for i in range(n_series):
                valid_series = new_segments[i][np.any(new_segments[i] != 0, axis=1)]
                if len(valid_series) > 0:
                    all_segment_coeffs[i, :len(valid_series), :] = valid_series @ learned_basis.T

            new_coefficients[period] = all_segment_coeffs
    
        return new_coefficients

