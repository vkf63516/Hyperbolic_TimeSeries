import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, leaves_list
from scipy.stats import ttest_ind
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import warnings
import random
import torch
warnings.filterwarnings('ignore')

# Import learnable decomposition
from Decomposition.Learnable_Decomposition import LearnableMultivariateDecomposition


# ============================================
# SEED SETTING FUNCTION
# ============================================
def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value (default: 42)
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Set Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Random seed set to {seed} for reproducibility")


def detect_periods_acf(data, max_lag=500):
    """
    Detect dominant periods using autocorrelation.
    
    Args:
        data: [T, n_features] time series
        max_lag: Maximum lag to consider
    
    Returns: 
        List of detected periods [fine_period, coarse_period]
    """
    # Use first feature for period detection
    ts = data[:, 0] if data.ndim > 1 else data
    
    # Compute ACF
    autocorr = acf(ts, nlags=min(max_lag, len(ts)//2), fft=True)
    
    # Find peaks in ACF
    peaks, properties = find_peaks(autocorr[1:], prominence=0.1)
    peaks = peaks + 1  # Adjust for slicing
    
    if len(peaks) >= 2:
        # Return two strongest periods
        peak_heights = autocorr[peaks]
        top_indices = np.argsort(peak_heights)[-2:][::-1]
        periods = peaks[top_indices]
        fine_period = min(periods)
        coarse_period = max(periods)
    elif len(peaks) == 1:
        fine_period = peaks[0]
        coarse_period = fine_period * 7  # Default multiplier
    else:
        # Default to daily and weekly for hourly data
        fine_period = 24
        coarse_period = 168
    
    print(f"Detected periods: fine={fine_period}, coarse={coarse_period}")
    return [fine_period, coarse_period]


def apply_learnable_decomposition_to_dataset(args, use_pretrained=False, model_path=None):
    """
    Load raw data and apply learnable decomposition to entire dataset.
    
    Args:
        args: Arguments with data parameters
        use_pretrained: Whether to load pretrained weights
        model_path: Path to pretrained model
    
    Returns:
        data_dict: Dictionary with full decomposed components
    """
    print("\n" + "=" * 80)
    print("APPLYING LEARNABLE DECOMPOSITION TO DATASET")
    print("=" * 80)
    
    # Load raw data
    from data_provider.data_factory import data_provider
    
    args_raw = argparse.Namespace(**vars(args))
    args_raw.data = 'custom'  # Use regular dataset loader
    
    train_data, train_loader = data_provider(args_raw, flag='train')
    
    print(f"Dataset size: {len(train_data)}")
    
    # Initialize decomposition model
    # Get sample to detect periods and dimensions
    sample_batch = next(iter(train_loader))
    batch_x = sample_batch[0]  # [B, seq_len, n_features]
    
    _, seq_len, n_features = batch_x.shape
    print(f"Sequence length: {seq_len}, Features: {n_features}")
    
    # Detect periods from sample
    sample_data = batch_x[0].cpu().numpy()
    detected_periods = detect_periods_acf(sample_data, max_lag=min(500, seq_len//2))
    
    # Initialize model
    kernel_size = max(detected_periods)
    model = LearnableMultivariateDecomposition(
        n_features=n_features,
        kernel_size=kernel_size,
        detected_periods=detected_periods
    )
    
    # Load pretrained weights if specified
    if use_pretrained and model_path and os.path.exists(model_path):
        print(f"Loading pretrained weights from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['decomposition_state_dict'])
        print("✓ Pretrained weights loaded")
    else:
        print("⚠ Using randomly initialized decomposition")
    
    model.eval()
    
    # Apply decomposition to entire dataset
    print("\nApplying decomposition to all batches...")
    
    all_components = {
        'trend': [],
        'coarse':  [],
        'fine': [],
        'residual':  []
    }
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, _, _) in enumerate(train_loader):
            # Decompose
            decomposed = model(batch_x)
            
            # Collect components (only use input part, not labels)
            all_components['trend'].append(decomposed['trend'])
            all_components['coarse'].append(decomposed['seasonal_coarse'])
            all_components['fine'].append(decomposed['seasonal_fine'])
            all_components['residual'].append(decomposed['residual'])
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(train_loader)} batches")
    
    # Concatenate all batches and reshape to [T, n_features]
    print("\nConcatenating components...")
    
    data_dict = {}
    for key in ['trend', 'coarse', 'fine', 'residual']:
        # Concatenate batches:  List of [B, seq_len, n_features] -> [total_samples, seq_len, n_features]
        concatenated = torch.cat(all_components[key], dim=0)
        
        # Reshape to [T, n_features] by treating all sequences as continuous
        # [N_batches, seq_len, n_features] -> [N_batches * seq_len, n_features]
        reshaped = concatenated.reshape(-1, n_features)
        
        # Convert to numpy
        data_dict[key] = reshaped.cpu().numpy()
        
        mem_mb = data_dict[key].nbytes / (1024**2)
        print(f"  {key: 8s}: {data_dict[key].shape} ({mem_mb:.1f} MB)")
    
    print("\n✓ Learnable decomposition applied to entire dataset")
    
    return data_dict


# Keep all other functions from original hierarchical_analysis.py
# (analyze_hierarchy_raw_timeseries, multiscale_hierarchy_test, 
#  plot_frequency_spectrum, visualize_component_characteristics,
#  test_spherical_structure, compute_euclidean_score,
#  experiment_2_hierarchical_structure)
# ...  [rest of the original functions] ... 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    
    # Data parameters
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./time-series-dataset/dataset/')
    parser.add_argument('--data_path', type=str, default='weather.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=21)
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--encode', type=str, default='timeF')
    
    # Loader parameters
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Learnable decomposition parameters
    parser.add_argument('--use_pretrained', action='store_true', default=False,
                       help='Load pretrained decomposition weights')
    parser.add_argument('--model_path', type=str, default='./checkpoints/decomposition_model.pth',
                       help='Path to pretrained model')
    
    # Analysis parameters
    parser.add_argument('--save_dir', type=str, default='./plots/hierarchy')
    parser.add_argument('--max_patterns', type=int, default=5000)
    parser.add_argument('--window_fine', type=int, default=144)
    parser.add_argument('--window_coarse', type=int, default=1008)
    parser.add_argument('--window_trend', type=int, default=4320)
    
    # Seed parameter
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed FIRST
    set_seed(args.seed)
    
    print("=" * 80)
    print("HIERARCHICAL STRUCTURE ANALYSIS - LEARNABLE DECOMPOSITION")
    print("=" * 80)
    print(f"Dataset: {args.data_path}")
    print(f"Max patterns per component: {args.max_patterns:,}")
    print(f"Random seed: {args.seed}")
    print(f"Using pretrained:  {args.use_pretrained}")
    
    # Apply learnable decomposition
    data_dict = apply_learnable_decomposition_to_dataset(
        args,
        use_pretrained=args.use_pretrained,
        model_path=args.model_path
    )
    
    # Component overview
    print("\nCreating component overview...")
    visualize_component_characteristics(data_dict, save_dir=args.save_dir)
    
    # Analyze each component
    results = {}
    window_sizes = {
        'fine': args.window_fine,
        'coarse': args.window_coarse,
        'trend': args.window_trend,
        'residual': args.window_fine
    }
    
    # Run complete analysis on all components
    for component in ['trend', 'coarse', 'fine', 'residual']:
        results[component] = experiment_2_hierarchical_structure(
            data_dict,
            component=component,
            window_size=window_sizes[component],
            max_patterns=args.max_patterns,
            save_dir=args.save_dir
        )
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY - ALL COMPONENTS (LEARNABLE DECOMPOSITION)")
    print("=" * 80)
    
    for comp, res in results.items():
        if res:
            print(f"\n{comp.upper()}:")
            print(f"  Hierarchical:  {res['hierarchy_score']}/5 (cophenetic: {res['cophenetic_correlation']:.4f})")
            print(f"  Spherical:     {res['spherical_score']}/5")
            print(f"  Euclidean:     {res['euclidean_score']}/5")
            print(f"  → PRIMARY: {res['primary_geometry']} (confidence: {res['confidence']})")
    
    print(f"\n✓ All analyses saved to: {args.save_dir}/")