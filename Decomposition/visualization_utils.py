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
    axs[1].plot(comp["seasonal_coarse"])
    axs[1].set_title("seasonal_coarse")
    axs[2].plot(comp["seasonal_fine"])
    axs[2].set_title("Seasonal_fine")
    axs[4].plot(comp["residual"])
    axs[4].set_title("Residual")
    plt.suptitle(f"Decomposition Components — {feature_name}")
    safe_name = feature_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(f"{my_path}/images/{safe_name}_component_grid.svg", bbox_inches="tight", dpi=300)
    # plt.show()

def plot_variance_contribution(decompositions):
    trend_var = {k: np.var(v["trend"]) for k, v in decompositions.items()}
    seasonal_fine = {k: np.var(v["seasonal_fine"]) for k, v in decompositions.items()}
    seasonal_coarse = {k: np.var(v["seasonal_coarse"]) for k, v in decompositions.items()}
    residual_var = {k: np.var(v["residual"]) for k, v in decompositions.items()}

    df = pd.DataFrame({
        "Trend": trend_var,
        "seasonal_coarse": seasonal_coarse,
        "Seasonal_fine": seasonal_fine,
        "Residual": residual_var
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
                    'seasonal_fine': np.array,
                    'seasonal_coarse': np.array,
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
    seasonal_coarse_df = pd.DataFrame({
        f: decompositions[f]["seasonal_coarse"]
        for f in features
    })
    seasonal_fine_df = pd.DataFrame({
        f: decompositions[f]["seasonal_fine"]
        for f in features
    })
    residual_df = pd.DataFrame({
        f: decompositions[f]["residual"] for f in features})

    # --- Compute correlation matrices ---
    corr_trend = trend_df.corr()
    corr_coarse = seasonal_coarse_df.corr()
    corr_fine = seasonal_fine_df.corr()
    corr_residual = residual_df.corr()

    # --- Plot heatmaps ---
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    sns.heatmap(corr_trend, cmap="vlag", center=0, ax=axes[0])
    axes[0].set_title("Trend Correlation")

    sns.heatmap(corr_coarse, cmap="vlag", center=0, ax=axes[1])
    axes[1].set_title("coarse Correlation")

    sns.heatmap(corr_fine, cmap="vlag", center=0, ax=axes[2])
    axes[2].set_title("fine Correlation")

    sns.heatmap(corr_residual, cmap="vlag", center=0, ax=axes[3])
    axes[3].set_title("Residual Correlation")

    plt.suptitle("Cross-Feature Correlation Maps by Component", fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{my_path}/images/component_correlation.svg", bbox_inches="tight", dpi=300)
