import torch 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot
from statsmodels.tsa.seasonal import MSTL
from joblib import Parallel, delayed
import multiprocessing

"""
Decompose time series into trend, seasonality, and residual
seasonality(daily, weekly, monthly)
"""
def trend_seasonal_decomposition(time_dataset):
    
    daily_seasonal = []
    weekly_seasonal = []
    monthly_seasonal = []
    trend = []
    residual = []
    mstl_periods = []
    freq = pd.infer_freq(time_dataset.index)
    inx = time_dataset.index
    mstl_periods = timesteps_based_on_frequency(freq, time_dataset.index)
    pass


def timesteps_based_on_frequency(freq, index):
    """
    Compute the number of timesteps corresponding to daily, weekly, and monthly
    seasonal cycles based on the frequency of the time series.

    Parameters
    ----------
    freq : str or None
        Frequency string inferred from pd.infer_freq(index), e.g., 'H', '15T', 'D'.
        If None, the function will compute the median timestep from the index.
    index : pd.DatetimeIndex
        Datetime index of the time series.

    Returns
    -------
    list of int
        Number of timesteps per day, week, and month: [daily, weekly, monthly]
    """
    
    if freq is None:
        # Fallback: infer step size from median difference between timestamps
        step = index.to_series().diff().median()
        step_seconds = step.total_seconds()
    else:
        # Convert frequency string to timedelta in seconds
        step_seconds = pd.tseries.frequencies.to_offset(freq).delta.total_seconds()
    
    if step_seconds == 0:
        raise ValueError("Could not determine valid time step.")

    # Compute number of timesteps per seasonal cycle
    steps_per_day = 86400 / step_seconds          # 86400 seconds in a day
    steps_per_week = steps_per_day * 7            # 7 days per week
    steps_per_month = steps_per_day * 30          # approximate 30 days per month

    # Round to nearest integer and return as a list
    return [int(round(steps_per_day)), int(round(steps_per_week)), int(round(steps_per_month))]

def decompose_window(series_window, periods):
    res = MSTL(series_window, periods=periods).fit()
    return pd.DataFrame({
        "trend": res.trend,
        "seasonal": res.seasonal,
        "resid": res.resid
    }, index=series_window.index)

# --- Helper to decompose a single column ---
def decompose_column(series, periods, window_size=None, n_jobs=1):
    series = series.dropna()
    
    if window_size is None or window_size >= len(series):
        # Full series decomposition
        return decompose_window(series, periods)
    else:
        # Sliding windows, parallelized
        windows = [series.iloc[i:i+window_size] for i in range(len(series)-window_size+1)]
        dfs = Parallel(n_jobs=n_jobs)(
            delayed(decompose_window)(w, periods) for w in windows
        )
        return pd.concat(dfs, keys=range(len(dfs)), names=["window", "time"])

# --- Main function ---
def trend_seasonal_decomposition_parallel(df, window_size=None, n_jobs=None):
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    periods = infer_mstl_periods_from_index(df.index)
    
    # Parallelize across columns
    results = Parallel(n_jobs=n_jobs)(
        delayed(decompose_column)(df[col], periods, window_size, n_jobs=1)  # n_jobs=1 inside each column
        for col in df.columns
    )
    
    return {col: res for col, res in zip(df.columns, results)}