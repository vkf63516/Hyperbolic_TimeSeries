import torch
import pandas as pd

# def build_decomposition_tensors(df_components):
#     """
#     Convert MSTL decomposition DataFrame into PyTorch tensors for parallel encoding.

#     Parameters
#     ----------
#     df_components : pd.DataFrame
#         Must contain columns:
#             'trend', 'residual',
#             'seasonal_hourly', 'seasonal_daily', 'seasonal_weekly'

#     Returns
#     -------
#     dict of torch.Tensor
#         'trend'    : [T, 1]
#         'seasonal' : [T, 3] (hourly, daily, weekly)
#         'residual' : [T, 1]
#     """

#     trend_tensor = torch.tensor(df_components["trend"].values, dtype=torch.float32).unsqueeze(-1)
#     residual_tensor = torch.tensor(df_components["residual"].values, dtype=torch.float32).unsqueeze(-1)
#     seasonal_df = pd.concat(
#         {
#             ("seasonal", "hourly"): df_components["seasonal_hourly"],
#             ("seasonal", "daily"): df_components["seasonal_daily"],
#             ("seasonal", "weekly"): df_components["seasonal_weekly"]
#         },
#         axis=1,
#         names=['component', 'period']
#     )
#     seasonal_tensor = torch.tensor(seasonal_df.values, dtype=torch.float32)

#     return {
#         "trend": trend_tensor,
#         "seasonal": seasonal_tensor,
#         "residual": residual_tensor
#     }

def build_decomposition_tensors(df_components_or_dict):
    """
    Convert MSTL or TimeBaseMSTL decomposition to torch tensors.
    Accepts either a pandas DataFrame or a per-series decomposition dict.
    """
    # Case 1: standard MSTL DataFrame 
    if isinstance(df_components_or_dict, pd.DataFrame):
        df = df_components_or_dict

    # Case 2: TimeBaseMSTL single-series dictionary
    elif isinstance(df_components_or_dict, dict) and "trend" in df_components_or_dict:
        df = pd.DataFrame({
            "trend": df_components_or_dict["trend"],
            "seasonal_hourly": df_components_or_dict["seasonal"],
            "seasonal_daily": df_components_or_dict["seasonal"],
            "seasonal_weekly": df_components_or_dict["seasonal"],
            "residual": df_components_or_dict["residual"],
        })
    else:
        raise ValueError("Input must be a pandas DataFrame or single-series decomposition dict")

    # Convert to tensors
    trend_tensor = torch.tensor(df["trend"].values, dtype=torch.float32).unsqueeze(-1)
    residual_tensor = torch.tensor(df["residual"].values, dtype=torch.float32).unsqueeze(-1)
    seasonal_tensor = torch.tensor(
        df[["seasonal_hourly", "seasonal_daily", "seasonal_weekly"]].values,
        dtype=torch.float32
    )
    # Aggregate seasonalities
    seasonal_tensor_comb = seasonal_tensor.sum(-1, keepdims=True)

    return {
        "trend": trend_tensor,
        "seasonal": seasonal_tensor_comb,
        "residual": residual_tensor
    }

def build_mamba_input_tensors(results_dict):
    """
    Convert TimeBaseMSTL output into Mamba encoder-ready tensors.
    Returns a tensor of shape [num_series, T, 3].
    """
    tensors = []
    for series_name, series_decomp in results_dict.items():
        df = pd.DataFrame({
            "trend": series_decomp["trend"],
            "seasonal": series_decomp["seasonal"],
            "residual": series_decomp["residual"]
        }).fillna(0)
        series_tensor = torch.from_numpy(df.to_numpy(dtype=np.float32))  # [T, 3]
        tensors.append(series_tensor)

    return torch.stack(tensors, dim=0)  # [num_series, T, 3]

