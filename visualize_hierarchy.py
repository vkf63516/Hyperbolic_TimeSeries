import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from data_provider.data_factory import data_provider
import argparse

def plot_frequency_spectrum(component_data, component_name, ax, sampling_rate=1.0):
    """
    Plot frequency spectrum (FFT) to show dominant frequencies.
    
    Args:
        component_data: 1D array of time series
        component_name: Name for the plot
        ax: Matplotlib axis
        sampling_rate: Samples per hour (1.0 for hourly data)
    """
    # Compute FFT
    fft = np.fft.fft(component_data)
    freq = np.fft.fftfreq(len(component_data), d=1/sampling_rate)
    
    # Only positive frequencies
    pos_mask = freq > 0
    freq = freq[pos_mask]
    magnitude = np.abs(fft[pos_mask])
    
    # Plot
    ax.plot(freq, magnitude, linewidth=1.5)
    ax.set_ylabel('Magnitude')
    ax.set_title(f'{component_name} - Frequency Spectrum', pad=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)  # Focus on low frequencies
    
    # Add markers for known periods
    if sampling_rate == 1.0:  # hourly data
        fine_freq = 1/24
        coarse_freq = 1/168
        # ax.axvline(fine_freq, color='orange', linestyle='--', alpha=0.5, label='fine (24h)')
        # ax.axvline(coarse_freq, color='green', linestyle='--', alpha=0.5, label='coarse (168h)')
        # ax.legend()


def plot_window_hierarchy(X_dict, Y_dict, window_idx=0, feature_idx=0, save_dir='./plots/hierarchy'):
    """
    Visualize the frequency hierarchy for a single window.
    Shows time domain + frequency domain for each component.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x
    
    # Extract data for the specified window and feature
    components = {}
    for key in ['trend', 'seasonal_coarse', 'seasonal_fine', 'residual']:
        x_data = to_numpy(X_dict[key])
        
        # Handle batch dimension
        if x_data.ndim == 3:
            x_data = x_data[window_idx]
        
        # Handle feature dimension
        if x_data.ndim == 2:
            x_data = x_data[:, feature_idx]
        
        components[key] = x_data
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    time_steps = np.arange(len(components['trend']))
    
    # Row 1: TREND
    ax_trend_time = fig.add_subplot(gs[0, 0])
    ax_trend_time.plot(time_steps, components['trend'], color='blue', linewidth=2)
    ax_trend_time.set_ylabel('Value')
    ax_trend_time.set_title('TREND (Slowest - Long-term pattern)')
    ax_trend_time.grid(True, alpha=0.3)
    
    ax_trend_freq = fig.add_subplot(gs[0, 1])
    plot_frequency_spectrum(components['trend'], 'Trend', ax_trend_freq)
    
    ax_trend_stats = fig.add_subplot(gs[0, 2])
    ax_trend_stats.text(0.1, 0.7, f"Mean: {components['trend'].mean():.3f}", fontsize=12)
    ax_trend_stats.text(0.1, 0.5, f"Std: {components['trend'].std():.3f}", fontsize=12)
    ax_trend_stats.text(0.1, 0.3, f"Range: [{components['trend'].min():.3f}, {components['trend'].max():.3f}]", fontsize=12)
    ax_trend_stats.text(0.1, 0.1, f"Variation: {np.diff(components['trend']).std():.4f}", fontsize=12)
    ax_trend_stats.set_title('Statistics')
    ax_trend_stats.axis('off')
    
    # Row 2: SEASONAL coarse
    ax_coarse_time = fig.add_subplot(gs[1, 0])
    ax_coarse_time.plot(time_steps, components['seasonal_coarse'], color='green', linewidth=2)
    ax_coarse_time.set_ylabel('Value')
    ax_coarse_time.set_title('SEASONAL coarse (Medium - 7-day cycles)')
    ax_coarse_time.grid(True, alpha=0.3)
    
    ax_coarse_freq = fig.add_subplot(gs[1, 1])
    plot_frequency_spectrum(components['seasonal_coarse'], 'coarse', ax_coarse_freq)
    
    ax_coarse_stats = fig.add_subplot(gs[1, 2])
    ax_coarse_stats.text(0.1, 0.7, f"Mean: {components['seasonal_coarse'].mean():.3f}", fontsize=12)
    ax_coarse_stats.text(0.1, 0.5, f"Std: {components['seasonal_coarse'].std():.3f}", fontsize=12)
    ax_coarse_stats.text(0.1, 0.3, f"Range: [{components['seasonal_coarse'].min():.3f}, {components['seasonal_coarse'].max():.3f}]", fontsize=12)
    ax_coarse_stats.text(0.1, 0.1, f"Variation: {np.diff(components['seasonal_coarse']).std():.4f}", fontsize=12)
    ax_coarse_stats.set_title('Statistics')
    ax_coarse_stats.axis('off')
    
    # Row 3: SEASONAL fine
    ax_fine_time = fig.add_subplot(gs[2, 0])
    ax_fine_time.plot(time_steps, components['seasonal_fine'], color='orange', linewidth=2)
    ax_fine_time.set_ylabel('Value')
    ax_fine_time.set_title('SEASONAL fine (Medium-Fast - 24h cycles)')
    ax_fine_time.grid(True, alpha=0.3)
    
    ax_fine_freq = fig.add_subplot(gs[2, 1])
    plot_frequency_spectrum(components['seasonal_fine'], 'fine', ax_fine_freq)
    
    ax_fine_stats = fig.add_subplot(gs[2, 2])
    ax_fine_stats.text(0.1, 0.7, f"Mean: {components['seasonal_fine'].mean():.3f}", fontsize=12)
    ax_fine_stats.text(0.1, 0.5, f"Std: {components['seasonal_fine'].std():.3f}", fontsize=12)
    ax_fine_stats.text(0.1, 0.3, f"Range: [{components['seasonal_fine'].min():.3f}, {components['seasonal_fine'].max():.3f}]", fontsize=12)
    ax_fine_stats.text(0.1, 0.1, f"Variation: {np.diff(components['seasonal_fine']).std():.4f}", fontsize=12)
    ax_fine_stats.set_title('Statistics')
    ax_fine_stats.axis('off')
    
    # Row 4: RESIDUAL
    ax_resid_time = fig.add_subplot(gs[3, 0])
    ax_resid_time.plot(time_steps, components['residual'], color='purple', linewidth=1.5)
    ax_resid_time.set_xlabel('Time Steps')
    ax_resid_time.set_ylabel('Value')
    ax_resid_time.set_title('RESIDUAL (Fastest - Noise & Irregularities)')
    ax_resid_time.grid(True, alpha=0.3)
    
    ax_resid_freq = fig.add_subplot(gs[3, 1])
    plot_frequency_spectrum(components['residual'], 'Residual', ax_resid_freq)
    
    ax_resid_stats = fig.add_subplot(gs[3, 2])
    ax_resid_stats.text(0.1, 0.7, f"Mean: {components['residual'].mean():.3f}", fontsize=12)
    ax_resid_stats.text(0.1, 0.5, f"Std: {components['residual'].std():.3f}", fontsize=12)
    ax_resid_stats.text(0.1, 0.3, f"Range: [{components['residual'].min():.3f}, {components['residual'].max():.3f}]", fontsize=12)
    ax_resid_stats.text(0.1, 0.1, f"Variation: {np.diff(components['residual']).std():.4f}", fontsize=12)
    ax_resid_stats.set_title('Statistics')
    ax_resid_stats.axis('off')
    
    plt.suptitle(f'Frequency Hierarchy Analysis - Window {window_idx}, Feature {feature_idx}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f'{save_dir}/hierarchy_window_{window_idx}_feature_{feature_idx}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved hierarchy plot to {save_dir}/hierarchy_window_{window_idx}_feature_{feature_idx}.png")


def plot_stacked_components(X_dict, window_idx=0, feature_idx=0, save_dir='./plots/hierarchy'):
    """
    Create a stacked plot showing how components add up to the original signal.
    This visualizes the hierarchy clearly.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy
    def to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x
    
    # Extract data
    components = {}
    for key in ['trend', 'seasonal_coarse', 'seasonal_fine', 'residual']:
        x_data = to_numpy(X_dict[key])
        if x_data.ndim == 3:
            x_data = x_data[window_idx]
        if x_data.ndim == 2:
            x_data = x_data[:, feature_idx]
        components[key] = x_data
    
    time_steps = np.arange(len(components['trend']))
    
    # Create stacked visualization
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.set_tight_layout(True)
    # Cumulative sums to show hierarchy
    cum_trend = components['trend']
    cum_coarse = cum_trend + components['seasonal_coarse']
    cum_fine = cum_coarse + components['seasonal_fine']
    cum_all = cum_fine + components['residual']
    
    # Plot with fill between
    ax.fill_between(time_steps, 0, cum_trend, alpha=0.3, color='blue', label='Trend (Base)')
    ax.fill_between(time_steps, cum_trend, cum_coarse, alpha=0.3, color='green', label='+ coarse Seasonality')
    ax.fill_between(time_steps, cum_coarse, cum_fine, alpha=0.3, color='orange', label='+ fine Seasonality')
    ax.fill_between(time_steps, cum_fine, cum_all, alpha=0.3, color='purple', label='+ Residual')
    
    # Plot lines
    ax.plot(time_steps, cum_trend, 'b-', linewidth=2, label='Trend')
    ax.plot(time_steps, cum_coarse, 'g-', linewidth=2, label='Trend + coarse')
    ax.plot(time_steps, cum_fine, color='orange', linewidth=2, label='Trend + coarse + fine')
    ax.plot(time_steps, cum_all, 'k-', linewidth=2.5, label='Full Signal (All Components)')
    
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Cumulative Value', fontsize=12)
    ax.set_title(f'Hierarchical Composition - Window {window_idx}, Feature {feature_idx}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    plt.savefig(f'{save_dir}/stacked_hierarchy_window_{window_idx}_feature_{feature_idx}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved stacked plot to {save_dir}/stacked_hierarchy_window_{window_idx}_feature_{feature_idx}.png")


# Main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='custom_decomposition')
    parser.add_argument('--root_path', type=str, default='./time-series-dataset/dataset/weather/')
    parser.add_argument('--data_path', type=str, default='weather.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=21)
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--num_basis', type=int, default=10)
    parser.add_argument('--orthogonal_lr', type=float, default=1e-3)
    parser.add_argument('--orthogonal_iters', type=int, default=100)
    parser.add_argument('--use_segments', action='store_true', default=False)
    parser.add_argument('--mstl_period', type=int, default=24)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()
    
    print("Loading data...")
    train_data, train_loader = data_provider(args, flag='train')
    
    # Get first batch
    for X_dict, Y_dict, _, _ in train_loader:
        print(f"\nData shapes:")
        print(f"  Trend: {X_dict['trend'].shape}")
        print(f"  coarse: {X_dict['seasonal_coarse'].shape}")
        print(f"  fine: {X_dict['seasonal_fine'].shape}")
        print(f"  Residual: {X_dict['residual'].shape}")
        
        print("\nGenerating hierarchy visualizations...")
        
        # Plot first 2 windows, first feature
        for window_idx in range(min(2, X_dict['trend'].shape[0])):
            print(f"\nProcessing window {window_idx}...")
            plot_window_hierarchy(X_dict, Y_dict, window_idx=window_idx, feature_idx=0)
            plot_stacked_components(X_dict, window_idx=window_idx, feature_idx=0)
        
        break
    
    print("\n✓ Done! Check ./plots/hierarchy/ for visualizations")