import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL
from sklearn.decomposition import PCA

class TimeBaseMSTL:
    """
    TimeBase-inspired approach for efficient multivariate MSTL decomposition
    with dynamic period detection using pandas Timedelta.
    """

    def __init__(self, n_basis_components=10):
        """
        Parameters
        ----------
        n_basis_components : int
            Number of PCA basis components to extract.
        """
        self.n_basis_components = n_basis_components
        self.basis_components = {}
        self.series_coefficients = {}
        
    def timesteps_from_index(df):
        """
        Flexible version combining both TimeBase and calendar logic.
        Computes user-defined periods dynamically.
        """
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
        steps = [int(round(p / step_seconds)) for p in periods_seconds]
        return steps

    
    def extract_periodic_segments(self, df, steps_per_period):
        """Extract segments for each seasonal period dynamically."""
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
    
    def extract_basis_components(self, segments, segment_type="periodic"):
        """Extract shared basis components across all series for a given segment type."""
        n_series, n_segments, segment_length = segments.shape
        
        # Flatten and remove invalid segments
        flattened_segments = segments.reshape(-1, segment_length)
        valid_mask = np.any(flattened_segments != 0, axis=1)
        valid_segments = flattened_segments[valid_mask]
        
        if len(valid_segments) == 0:
            return np.zeros((self.n_basis_components, segment_length)), np.zeros((n_series, self.n_basis_components))
        
        # PCA to find shared basis
        n_comp = min(self.n_basis_components, len(valid_segments), segment_length)
        pca = PCA(n_components=n_comp)
        pca.fit(valid_segments)
        
        basis_components = pca.components_
        series_coefficients = np.zeros((n_series, pca.n_components_))
        
        for i in range(n_series):
            series_segments = segments[i]
            valid_series_segments = series_segments[np.any(series_segments != 0, axis=1)]
            if len(valid_series_segments) > 0:
                coeffs = pca.transform(valid_series_segments)
                series_coefficients[i] = np.mean(coeffs, axis=0)
        
        return basis_components, series_coefficients, pca
    
    def decompose_basis_components(self, basis_components, period):
        """Decompose shared basis components using MSTL."""
        decomposed_basis = {}
        
        for i, component in enumerate(basis_components):
            try:
                # Repeat to create synthetic long time series for MSTL
                repeated = np.tile(component, 10)
                mstl = MSTL(repeated, periods=[period])
                res = mstl.fit()
                
                decomposed_basis[f'basis_{i}'] = {
                    'trend': res.trend[:period],
                    'seasonal': res.seasonal[:period] if res.seasonal.ndim == 1 else res.seasonal[:period, 0],
                    'resid': res.resid[:period]
                }
            except Exception as e:
                print(f"Failed to decompose basis {i}: {e}")
                decomposed_basis[f'basis_{i}'] = {
                    'trend': np.mean(component) * np.ones_like(component),
                    'seasonal': component - np.mean(component),
                    'resid': np.zeros_like(component)
                }
        
        return decomposed_basis
    
    def reconstruct_series_decomposition(self, df, decompositions, coeffs_dict):
        """Reconstruct each series from decomposed basis and learned coefficients."""
        series_decompositions = {}
        n_points = len(df)
        
        for i, name in enumerate(df.columns):
            total_trend = np.zeros(n_points)
            total_seasonal = np.zeros(n_points)
            total_resid = np.zeros(n_points)
            
            for period, basis_decomp in decompositions.items():
                coeffs = coeffs_dict[period][i]
                for j, (bname, comp) in enumerate(basis_decomp.items()):
                    coeff = coeffs[j] if j < len(coeffs) else 0
                    repeats = (n_points + period - 1) // period
                    for key in ["trend", "seasonal", "resid"]:
                        tiled = np.tile(comp[key], repeats)[:n_points]
                        if key == "trend": 
                            total_trend += coeff * tiled
                        elif key == "seasonal":
                            total_seasonal += coeff * tiled
                        else:
                            total_resid += coeff * tiled
            
            residual_actual = df[name].values - (total_trend + total_seasonal)
            series_decompositions[name] = {
                "trend": total_trend,
                "seasonal": total_seasonal,
                "residual": residual_actual
            }
        
        return series_decompositions
    
    def fit_transform(self, df):
        """Run full pipeline: extract segments → basis → MSTL → reconstruct."""
        print(f"Decomposing {df.shape[1]} series with dynamic TimeBase-MSTL...")
        steps_per_period = self.timesteps_from_index(df)
        print(f"⏱ Periods inferred as {steps_per_period} timesteps")
        
        # Extract segments
        segments_dict = self.extract_periodic_segments(df, steps_per_period)
        
        decompositions = {}
        coeffs_dict = {}
        
        for period, segments in segments_dict.items():
            print(f"\nPeriod {period} timesteps → {segments.shape[1]} segments")
            basis, coeffs, pca = self.extract_basis_components(segments)
            decomposed_basis = self.decompose_basis_components(basis, period)
            decompositions[period] = decomposed_basis
            coeffs_dict[period] = coeffs
        
        return self.reconstruct_series_decomposition(df, decompositions, coeffs_dict)
