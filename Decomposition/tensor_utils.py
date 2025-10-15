import torch
import pandas as pd

def build_decomposition_tensors(df_components):
    """
    Convert MSTL decomposition DataFrame into PyTorch tensors for parallel encoding.

    Parameters
    ----------
    df_components : pd.DataFrame
        Must contain columns:
            'trend', 'residual',
            'seasonal_hourly', 'seasonal_daily', 'seasonal_weekly'

    Returns
    -------
    dict of torch.Tensor
        'trend'    : [T, 1]
        'seasonal' : [T, 3] (hourly, daily, weekly)
        'residual' : [T, 1]
    """

    trend_tensor = torch.tensor(df_components["trend"].values, dtype=torch.float32).unsqueeze(-1)
    residual_tensor = torch.tensor(df_components["residual"].values, dtype=torch.float32).unsqueeze(-1)
    seasonal_df = pd.concat(
        {
            "hourly": df_components["seasonal_hourly"],
            "daily": df_components["seasonal_daily"],
            "weekly": df_components["seasonal_weekly"]
        },
        axis=1,
        names=['component', 'period']
    )
    seasonal_tensor = torch.tensor(seasonal_df.values, dtype=torch.float32)

    return {
        "trend": trend_tensor,
        "seasonal": seasonal_tensor,
        "residual": residual_tensor
    }
