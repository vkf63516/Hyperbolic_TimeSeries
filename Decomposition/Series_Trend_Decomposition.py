import pandas as pd  
from statsmodels.tsa.seasonal import MSTL
from joblib import Parallel, delayed
import multiprocessing
from pandas.tseries.frequencies import to_offset
import warnings


def mstl_decomposition_for_window(series_window, valid_periods):
    """
    Decompose a single series using MSTL.
    
    Parameters
    ----------
    series_window : pd.Series
        Time series to decompose
    valid_periods : list of int
        Seasonal periods for MSTL
    
    Returns
    -------
    pd.DataFrame
        Columns: ['trend', 'seasonal_coarse', 'seasonal_fine', 'residual']
    """
    res_mstl = MSTL(series_window, periods=valid_periods).fit()
    
    trend_df = pd.DataFrame({"trend": res_mstl.trend})
    
    # MSTL returns seasonal components in order of periods
    # periods = [fine, coarse] → seasonal components = [fine, coarse]
    seasonal_cols = ["seasonal_fine", "seasonal_coarse"]
    seasonal_fine_df = pd.DataFrame({seasonal_cols[0]:res_mstl.seasonal[f'seasonal_{valid_periods[0]}']})
    seasonal_coarse_df = pd.DataFrame({seasonal_cols[1]:res_mstl.seasonal[f'seasonal_{valid_periods[1]}']})
    residual_df = pd.DataFrame({"residual": res_mstl.resid})

    return pd.concat([
        trend_df,
        seasonal_coarse_df,
        seasonal_fine_df,
        residual_df
    ], axis=1)


def simple_decomposition(series_window):
    """
    Fallback decomposition when not enough data for MSTL.
    
    Returns
    -------
    pd.DataFrame
        Columns: ['trend', 'seasonal_coarse', 'seasonal_fine', 'residual']
    """
    warnings.warn(
        f"Not enough data points ({len(series_window)}) for MSTL. "
        "Using simple mean-based decomposition."
    )
    
    trend = pd.Series(series_window.mean(), index=series_window.index, name="trend")
    seasonal_coarse = pd.Series(0, index=series_window.index, name="seasonal_coarse")
    seasonal_fine = pd.Series(0, index=series_window.index, name="seasonal_fine")
    residual = pd.Series(series_window - series_window.mean(), index=series_window.index, name="residual")
    
    return pd.concat([trend, seasonal_coarse, seasonal_fine, residual], axis=1)


def decompose_series(series, periods):
    """
    Decompose a single time series.
    
    Parameters
    ----------
    series : pd.Series
        Time series to decompose
    periods : list of int
        Seasonal periods [fine, coarse]
    
    Returns
    -------
    pd.DataFrame
        Decomposed components
    """
    series = series.dropna()
    
    # Check if we have enough data for MSTL
    # MSTL requires at least 2 full cycles of the longest period
    valid_periods = [p for p in periods if p * 2 <= len(series)]
    
    if len(valid_periods) < 2:
        # Not enough data, use simple decomposition
        return simple_decomposition(series)
    
    return mstl_decomposition_for_window(series, valid_periods)


def trend_seasonal_decomposition_parallel(time_dataset, periods, n_jobs=None):
    """
    Parallel MSTL decomposition for multi-feature time series.
    
    Parameters
    ----------
    time_dataset : pd.DataFrame
        Time series data with DatetimeIndex
    n_jobs : int, optional
        Number of parallel jobs (default: number of CPUs)
    
    Returns
    -------
    dict
        {feature_name: decomposed_df}
        Each decomposed_df has columns:
        ['trend', 'seasonal_coarse', 'seasonal_fine', 'residual']
    
    Examples
    --------
    >>> decomposed = trend_seasonal_decomposition_parallel(train_df)
    >>> print(decomposed['HUFL'].columns)
    Index(['trend', 'seasonal_coarse', 'seasonal_fine', 'residual'], dtype='object')
    """
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    # Detect periods from frequency
    
    print(f"Detected periods: {periods} (fine={periods[0]}, coarse={periods[1]})")
    
    # Parallelize across features
    results = {}
    for col in time_dataset.columns:
        results[col] = decompose_series(time_dataset[col], periods)
    
    return results