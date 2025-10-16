import pandas as pd  
from statsmodels.tsa.seasonal import MSTL
from joblib import Parallel, delayed
import multiprocessing
from pandas.tseries.frequencies import to_offset
import warnings


def mstl_decomposition_for_window(series_window, valid_periods):
    res_mstl = MSTL(series_window, periods=valid_periods).fit()
    trend_df = pd.DataFrame({"trend": res_mstl.trend})
    seasonal_cols = ["seasonal_hourly", "seasonal_daily", "seasonal_weekly"]
    seasonal_df = pd.DataFrame(res_mstl.seasonal, columns=seasonal_cols, index=series_window.index)
    seasonal_hourly = pd.DataFrame(seasonal_df[seasonal_cols[0]])
    seasonal_daily = pd.DataFrame(seasonal_df[seasonal_cols[1]])
    seasonal_weekly = pd.DataFrame(seasonal_df[seasonal_cols[-1]])
    seasonal_group = pd.concat([seasonal_weekly, seasonal_daily, seasonal_hourly], axis=1)
    residual_df = pd.DataFrame({"residual": res_mstl.resid})

    return pd.concat([
        trend_df,
        seasonal_group,
        residual_df
    ], axis=1)

def decomposition_for_window_with_hd(series_window, hd_periods):
    res_hd = MSTL(series_window, periods=hd_periods).fit()
    trend_hd = pd.DataFrame({"trend": res_hd.trend})
    seasonal_cols = ["seasonal_hourly", "seasonal_daily"]
    seasonal_df = pd.DataFrame(res_hd.seasonal, columns=seasonal_cols, index=series_window.index)
    seasonal_hourly = pd.DataFrame(seasonal_df[seasonal_cols[0]])
    seasonal_daily = pd.DataFrame(seasonal_df[seasonal_cols[-1]])
    seasonal_group_hd = pd.concat([seasonal_daily, seasonal_hourly], axis=1)
    residual_hd = pd.DataFrame({"residual": res_hd.resid})
    return pd.concat([trend_hd, seasonal_group_hd, residual_hd], axis=1) 


def reorder_decomposition_for_window(series_window, valid_periods):
    if len(valid_periods) > 1:
        mstl_dw = decomposition_for_window_with_hd(series_window, valid_periods[:2])
        return mstl_dw 
    else:
        warnings.warn(f"Not enough valid periods for window of size {len(series_window)}. Returning simple mean decomposition.")
        # Simple decomposition: trend = mean, seasonality = 0, residual = deviations
        trend = pd.DataFrame(pd.Series(series_window.mean(), index=series_window.index, name="trend"))
        seasonal_hourly = pd.DataFrame(pd.Series(0, index=series_window.index, name="seasonal_hourly"))
        residual = pd.DataFrame(pd.Series(series_window - series_window.mean(), index=series_window.index, name="residual"))
        return pd.concat([trend, seasonal_hourly, residual], axis=1)

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
        Number of timesteps per hour, day, and week: [hourly, daily, weekly]
    """
    
    if freq is None:
        # Fallback: infer step size from median difference between timestamps
        step = index.to_series().diff().median()
        step_seconds = step.total_seconds()
    else:
        # Convert frequency string to timedelta in seconds
        step_seconds = pd.Timedelta(to_offset(freq)).total_seconds()
    
    if step_seconds == 0:
        raise ValueError("Could not determine valid time step.")

    # Compute number of timesteps per seasonal cycle
    steps_per_hour = 3600 / step_seconds          # 3600 seconds in an hour
    steps_per_day = steps_per_hour * 24      #86400 / step_seconds # 86400 seconds in a day
    steps_per_week = steps_per_day * 7            # 7 days per week
    # steps_per_month = steps_per_day * 30          # approximate 30 days per month

    # Round to nearest integer and return as a list
    return [int(round(steps_per_hour)), int(round(steps_per_day)), int(round(steps_per_week))]

"""
Gathers the dataFrames together for an input window 
"""
def decompose_window(series_window, periods):
    valid_periods = [p for p in periods if p * 2 <= len(series_window)]
    if len(valid_periods) < 3:
        return reorder_decomposition_for_window(series_window, valid_periods)
    
    return mstl_decomposition_for_window(series_window, valid_periods)

# --- Helper to decompose a single column ---
def decompose_column(series, periods, window_size=None, n_jobs=1):
    series = series.dropna()
    
    if window_size is None or window_size >= len(series):
        # Full series decomposition
        return decompose_window(series, periods)
    else:
        # Sliding windows, parallelized. Splits the data into windows
        num_windows = len(series) - window_size + 1
        if num_windows <= 0:
            # Window size is larger than series, fall back to full decomposition
            return decompose_window(series, periods)
        windows = [series.iloc[i:i+window_size] for i in range(num_windows)]
        dfs = Parallel(n_jobs=n_jobs)(
            delayed(decompose_window)(w, periods) for w in windows
        )
        return pd.concat(dfs, keys=range(len(dfs)), names=["window", "time"])

# --- Main function ---
def trend_seasonal_decomposition_parallel(time_dataset, window_size=None, n_jobs=None):
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    freq = pd.infer_freq(time_dataset.index)
    periods = timesteps_based_on_frequency(freq, time_dataset.index)
    
    # Parallelize across columns
    results = Parallel(n_jobs=n_jobs)(
        delayed(decompose_column)(time_dataset[col], periods, window_size, n_jobs=1)  # n_jobs=1 inside each column
        for col in time_dataset.columns
    )
    
    return {col: res for col, res in zip(time_dataset.columns, results)}
