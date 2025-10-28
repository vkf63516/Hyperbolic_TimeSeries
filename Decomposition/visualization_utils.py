import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
my_path = os.path.dirname(os.path.abspath(__file__))
def plot_component_grid(feature_name, decompositions):
    comp = decompositions[feature_name]
    fig, axs = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    axs[0].plot(comp["trend"]) 
    axs[0].set_title("Trend")
    axs[1].plot(comp["seasonal_weekly"])
    axs[1].set_title("Seasonal_Weekly")
    axs[2].plot(comp["seasonal_daily"])
    axs[2].set_title("Seasonal_Daily")
    axs[4].plot(comp["residual"])
    axs[4].set_title("Residual")
    plt.suptitle(f"Decomposition Components — {feature_name}")
    safe_name = feature_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(f"{my_path}/images/{safe_name}_component_grid.svg", bbox_inches="tight", dpi=300)
    # plt.show()

def plot_variance_contribution(decompositions):
    trend_var = {k: np.var(v["trend"]) for k, v in decompositions.items()}
    seasonal_daily = {k: np.var(v["seasonal_daily"]) for k, v in decompositions.items()}
    seasonal_weekly = {k: np.var(v["seasonal_weekly"]) for k, v in decompositions.items()}
    resid_var = {k: np.var(v["residual"]) for k, v in decompositions.items()}

    df = pd.DataFrame({
        "Trend": trend_var,
        "Seasonal_Weekly": seasonal_weekly,
        "Seasonal_Daily": seasonal_daily,
        "Residual": resid_var
    }).T
    (df / df.sum()).T.plot.bar(stacked=True, figsize=(12, 6))
    plt.title("Variance Contribution by Component")
    plt.ylabel("Fraction of Total Variance")
    plt.savefig(f"{my_path}/images/Variance_Contribution.svg", bbox_inches='tight', dpi=300)
    # plt.show()

def plot_component_correlation_maps(decompositions, features=None, figsize=(16, 10)):
    """
    Plot heatmaps showing correlation across features for each component
    (Trend, Seasonal, Residual).

    Parameters
    ----------
    decompositions : dict
        Output dictionary from TimeBaseMSTL.reconstruct_series_decomposition.
        Format:
            {
                'feature_name': {
                    'trend': np.array,
                    'seasonal_daily': np.array,
                    'seasonal_weekly': np.array,
                    'residual': np.array
                },
                ...
            }
    features : list, optional
        Subset of feature names to include. If None, uses all.
    figsize : tuple
        Size of the full grid of correlation plots.
    """
    # --- Collect all features ---
    if features is None:
        features = list(decompositions.keys())

    # --- Build component DataFrames ---
    trend_df = pd.DataFrame({
        f: decompositions[f]["trend"] 
        for f in features})
    seasonal_weekly_df = pd.DataFrame({
        f: decompositions[f]["seasonal_weekly"]
        for f in features
    })
    seasonal_daily_df = pd.DataFrame({
        f: decompositions[f].["seasonal_daily"]
        for f in features
    })
    residual_df = pd.DataFrame({
        f: decompositions[f]["residual"] for f in features})

    # --- Compute correlation matrices ---
    corr_trend = trend_df.corr()
    corr_weekly = seasonal_weekly_df.corr()
    corr_daily = seasonal_daily_df.corr()
    corr_residual = residual_df.corr()

    # --- Plot heatmaps ---
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    sns.heatmap(corr_trend, cmap="vlag", center=0, ax=axes[0])
    axes[0].set_title("Trend Correlation")

    sns.heatmap(corr_weekly, cmap="vlag", center=0, ax=axes[1])
    axes[1].set_title("Weekly Correlation")

    sns.heatmap(corr_daily, cmap="vlag", center=0, ax=axes[2])
    axes[2].set_title("Daily Correlation")

    sns.heatmap(corr_residual, cmap="vlag", center=0, ax=axes[3])
    axes[3].set_title("Residual Correlation")

    plt.suptitle("Cross-Feature Correlation Maps by Component", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{my_path}/images/component_correlation.svg", bbox_inches="tight", dpi=300)
