import os
import numpy as np
import matplotlib.pyplot as plt
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
import gc
warnings.filterwarnings('ignore')

from Decomposition.Learnable_Decomposition import LearnableMultivariateDecomposition


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Random seed set to {seed}")


def detect_periods_acf(data, max_lag=500):
    """Detect dominant periods using autocorrelation."""
    ts = data[: , 0] if data.ndim > 1 else data
    autocorr = acf(ts, nlags=min(max_lag, len(ts)//2), fft=True)
    peaks, properties = find_peaks(autocorr[1:], prominence=0.1)
    peaks = peaks + 1
    
    if len(peaks) >= 2:
        peak_heights = autocorr[peaks]
        top_indices = np.argsort(peak_heights)[-2:][::-1]
        periods = peaks[top_indices]
        fine_period = min(periods)
        coarse_period = max(periods)
    elif len(peaks) == 1:
        fine_period = peaks[0]
        coarse_period = fine_period * 7
    else:
        fine_period = 24
        coarse_period = 168
    
    print(f"Detected periods: fine={fine_period}, coarse={coarse_period}")
    return [fine_period, coarse_period]


def apply_learnable_decomposition_streaming(
    args, 
    component='fine',
    window_size=144,
    max_patterns=5000,
    use_pretrained=False, 
    model_path=None,
    downsample_factor=1
):
    """
    Memory-efficient streaming decomposition and pattern extraction.
    
    Only keeps patterns in memory, not full decomposed dataset.
    
    Args:
        args: Data arguments
        component: Which component to extract ('trend', 'coarse', 'fine', 'residual')
        window_size: Window size for pattern extraction
        max_patterns: Maximum patterns to extract
        use_pretrained: Load pretrained weights
        model_path: Path to model
        downsample_factor:  Downsample time series by this factor (1=no downsampling)
    
    Returns:
        patterns: [n_patterns, window_size * n_features] array
        n_total_patterns: Total patterns available
    """
    print("\n" + "=" * 80)
    print(f"STREAMING DECOMPOSITION:  {component.upper()}")
    print("=" * 80)
    
    from data_provider.data_factory import data_provider
    
    # Load data
    args_raw = argparse.Namespace(**vars(args))
    args_raw.data = 'custom'
    
    train_data, train_loader = data_provider(args_raw, flag='train')
    
    # Initialize model from first batch
    sample_batch = next(iter(train_loader))
    batch_x = sample_batch[0].float()
    
    _, seq_len, n_features = batch_x.shape
    print(f"Sequence length: {seq_len}, Features: {n_features}")
    
    # Detect periods
    sample_data = batch_x[0].cpu().numpy()
    detected_periods = detect_periods_acf(sample_data, max_lag=min(500, seq_len//2))
    
    # Initialize model
    kernel_size = max(detected_periods)
    model = LearnableMultivariateDecomposition(
        n_features=n_features,
        kernel_size=kernel_size,
        detected_periods=detected_periods
    )
    
    if use_pretrained and model_path and os.path.exists(model_path):
        print(f"Loading pretrained weights from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['decomposition_state_dict'])
        print("✓ Pretrained weights loaded")
    else:
        print("⚠ Using randomly initialized decomposition")
    
    model.eval()
    
    # Component key mapping
    component_key_map = {
        'trend': 'trend',
        'coarse': 'seasonal_coarse',
        'fine': 'seasonal_fine',
        'residual': 'residual'
    }
    component_key = component_key_map[component]
    
    # Pattern extraction with streaming
    print(f"\nExtracting patterns for {component} (window={window_size})...")
    
    stride = max(1, window_size // 2)
    patterns = []
    n_total_patterns = 0
    
    # Calculate sampling rate to stay under max_patterns
    # Estimate total patterns first
    estimated_total = (len(train_loader) * seq_len) // stride
    sampling_rate = min(1.0, max_patterns / estimated_total) if estimated_total > 0 else 1.0
    
    print(f"Estimated total patterns: {estimated_total: ,}")
    print(f"Sampling rate: {sampling_rate:.3f}")
    print(f"Downsample factor: {downsample_factor}")
    
    with torch.no_grad():
        for batch_idx, (batch_x, _, _, _) in enumerate(train_loader):
            # Convert to float32
            batch_x = batch_x.float()
            
            # Decompose this batch only
            decomposed = model(batch_x)
            component_data = decomposed[component_key].cpu().numpy()  # [B, seq_len, n_features]
            
            # Extract patterns from this batch
            for b in range(component_data.shape[0]):
                ts = component_data[b]  # [seq_len, n_features]
                
                # Downsample if requested
                if downsample_factor > 1:
                    ts = ts[::downsample_factor, :]
                
                T = len(ts)
                
                # Sliding window
                for i in range(0, T - window_size + 1, stride):
                    n_total_patterns += 1
                    
                    # Sample with probability
                    if np.random.rand() < sampling_rate:
                        pattern = ts[i:i+window_size, : ].flatten()
                        patterns.append(pattern)
                        
                        # Stop if we've collected enough
                        if len(patterns) >= max_patterns:
                            break
                
                if len(patterns) >= max_patterns:
                    break
            
            # Progress and early stopping
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}: {len(patterns):,} patterns collected")
            
            if len(patterns) >= max_patterns:
                print(f"  Reached max_patterns limit, stopping early")
                break
            
            # Free memory
            del decomposed, component_data
            gc.collect()
    
    patterns = np.array(patterns)
    
    print(f"\n✓ Extracted {len(patterns):,} patterns from ~{n_total_patterns:,} total")
    print(f"  Pattern shape: {patterns.shape}")
    print(f"  Memory usage: {patterns.nbytes / (1024**2):.1f} MB")
    
    return patterns, n_total_patterns


def test_spherical_structure(patterns_normalized, component='fine'):
    """Minimal spherical structure test."""
    
    patterns_sphere = normalize(patterns_normalized, norm='l2', axis=1)
    n_patterns = len(patterns_sphere)
    
    # Compute angular distances
    cos_sim = cosine_similarity(patterns_sphere)
    angular_dist = np.arccos(np.clip(cos_sim, -1, 1))
    
    # Compare uniformity
    euclidean_dists = pdist(patterns_normalized, metric='euclidean')
    angular_dists_flat = angular_dist[np.triu_indices_from(angular_dist, k=1)]
    
    cv_euclidean = np.std(euclidean_dists) / np.mean(euclidean_dists) if np.mean(euclidean_dists) > 0 else 1.0
    cv_angular = np.std(angular_dists_flat) / np.mean(angular_dists_flat) if np.mean(angular_dists_flat) > 0 else 1.0
    
    spherical_fit = cv_euclidean / cv_angular if cv_angular > 0 else 1.0
    
    # Spherical clustering
    max_k = min(20, n_patterns // 2)
    spherical_silhouettes = []
    
    print(f"Computing spherical clustering quality...")
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(patterns_sphere)
        
        try:
            sil = silhouette_score(patterns_sphere, labels, metric='cosine')
            spherical_silhouettes.append(sil)
        except:
            spherical_silhouettes.append(0.0)
        
        if k in [2, 3, 5, 7, 10, max_k]:
            print(f"  Spherical k={k: 2d}: silhouette={spherical_silhouettes[-1]:.4f}")
    
    if spherical_silhouettes:
        best_k_sphere = np.argmax(spherical_silhouettes) + 2
        best_sil_sphere = spherical_silhouettes[best_k_sphere - 2]
        print(f"→ Best spherical k={best_k_sphere} (silhouette={best_sil_sphere:.4f})")
    else:
        best_k_sphere = 2
        best_sil_sphere = 0.0
    
    # Scoring
    score_sphere = 0
    
    if spherical_fit > 1.5:
        score_sphere += 2
    elif spherical_fit > 1.2:
        score_sphere += 1
    
    if best_sil_sphere > 0.3:
        score_sphere += 2
    elif best_sil_sphere > 0.2:
        score_sphere += 1
    
    if spherical_fit > 1.3 and best_sil_sphere > 0.25:
        score_sphere += 1
    
    print(f"Spherical structure score: {score_sphere}/5")
    
    # Free memory
    del patterns_sphere, cos_sim, angular_dist
    gc.collect()
    
    return {
        'spherical_fit': spherical_fit,
        'spherical_score': score_sphere,
        'best_silhouette_sphere': best_sil_sphere,
        'best_k_sphere': best_k_sphere,
        'spherical_silhouettes': spherical_silhouettes,
        'cv_euclidean': cv_euclidean,
        'cv_angular': cv_angular
    }


def compute_euclidean_score(hierarchy_score, spherical_score,
                           cophenetic, spherical_fit, distance_matrix):
    """Compute Euclidean structure score (0-5)."""
    score = 0
    breakdown = {}
    
    max_other = max(hierarchy_score, spherical_score)
    
    if max_other <= 1:
        score += 2
        breakdown['other_structures'] = '+2 (no special structure detected)'
    elif max_other <= 2:
        score += 1
        breakdown['other_structures'] = '+1 (weak special structure)'
    else:
        breakdown['other_structures'] = '+0 (strong special structure)'
    
    if cophenetic < 0.5:
        score += 1
        breakdown['anti_hierarchical'] = f'+1 (flat structure, cophenetic={cophenetic:.3f})'
    else:
        breakdown['anti_hierarchical'] = f'+0 (hierarchical, cophenetic={cophenetic:.3f})'
    
    if spherical_fit < 1.0:
        score += 1
        breakdown['anti_spherical'] = f'+1 (not spherical, fit={spherical_fit:.3f})'
    else:
        breakdown['anti_spherical'] = f'+0 (spherical fit={spherical_fit:.3f})'
    
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    cv = np.std(distances) / (np.mean(distances) + 1e-8)
    
    if cv < 0.4:
        score += 1
        breakdown['uniformity'] = f'+1 (uniform distances, CV={cv:.3f})'
    else:
        breakdown['uniformity'] = f'+0 (varied distances, CV={cv:.3f})'
    
    breakdown['total'] = score
    
    return score, breakdown


def experiment_hierarchical_structure_efficient(
    args,
    component='fine',
    window_size=144, 
    max_patterns=5000,
    use_pretrained=False,
    model_path=None,
    save_dir='./plots/hierarchy',
    downsample_factor=1
):
    """
    Memory-efficient complete structure analysis.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"MEMORY-EFFICIENT STRUCTURE ANALYSIS:  {component.upper()}")
    print("=" * 80)
    
    # Extract patterns with streaming (memory-efficient)
    patterns, n_total_patterns = apply_learnable_decomposition_streaming(
        args,
        component=component,
        window_size=window_size,
        max_patterns=max_patterns,
        use_pretrained=use_pretrained,
        model_path=model_path,
        downsample_factor=downsample_factor
    )
    
    n_patterns = len(patterns)
    print(f"\nNormalizing {n_patterns:,} patterns...")
    
    scaler = StandardScaler()
    patterns_normalized = scaler.fit_transform(patterns)
    
    # Free original patterns
    del patterns
    gc.collect()
    
    # ==========================================
    # HIERARCHICAL ANALYSIS
    # ==========================================
    print(f"\n{'='*80}")
    print(f"PART 1: HIERARCHICAL STRUCTURE ANALYSIS")
    print(f"{'='*80}")
    
    print(f"Computing pairwise distances...")
    distances = pdist(patterns_normalized, metric='euclidean')
    distance_matrix = squareform(distances)
    
    print(f"Building hierarchical clustering...")
    linkage_matrix = linkage(distances, method='ward')
    
    c, _ = cophenet(linkage_matrix, distances)
    print(f"Cophenetic correlation: {c:.4f}")
    
    # Test cluster quality
    max_k = min(20, n_patterns // 2)
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clustering.fit_predict(patterns_normalized)
        sil = silhouette_score(patterns_normalized, labels)
        silhouette_scores.append(sil)
        
        if k in [2, 3, 5, 7, 10, max_k]: 
            print(f"  k={k:2d}: silhouette={sil:.4f}")
    
    best_k = np.argmax(silhouette_scores) + 2
    best_sil = silhouette_scores[best_k - 2]
    print(f"→ Best k={best_k} (silhouette={best_sil:.4f})")
    
    # Branching factor
    if n_patterns > 1:
        depth = np.log2(n_patterns)
        branching = n_patterns ** (1 / depth)
    else:
        branching = 1.0
    
    print(f"Branching factor:  {branching:.2f}")
    
    # Cluster separation
    best_clustering = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
    labels = best_clustering.fit_predict(patterns_normalized)
    
    within_dists = []
    between_dists = []
    
    for i in range(n_patterns):
        for j in range(i+1, n_patterns):
            if labels[i] == labels[j]:
                within_dists.append(distance_matrix[i, j])
            else:
                between_dists.append(distance_matrix[i, j])
    
    if within_dists and between_dists:
        t_stat, p_value = ttest_ind(within_dists, between_dists)
        separation_significant = p_value < 0.001
    else:
        p_value = 1.0
        separation_significant = False
    
    # Hierarchy score
    hierarchy_score = 0
    if c > 0.7:
        hierarchy_score += 2
    elif c > 0.4:
        hierarchy_score += 1
    
    if branching > 3:
        hierarchy_score += 2
    elif branching > 2:
        hierarchy_score += 1
    
    if separation_significant:
        hierarchy_score += 1
    
    print(f"\nHierarchical structure score:  {hierarchy_score}/5")
    
    # ==========================================
    # SPHERICAL ANALYSIS
    # ==========================================
    print(f"\n{'='*80}")
    print(f"PART 2: SPHERICAL STRUCTURE ANALYSIS")
    print(f"{'='*80}")
    
    spherical_results = test_spherical_structure(patterns_normalized, component=component)
    
    # ==========================================
    # EUCLIDEAN ANALYSIS
    # ==========================================
    print(f"\n{'='*80}")
    print(f"PART 3: EUCLIDEAN STRUCTURE ANALYSIS")
    print(f"{'='*80}")
    
    euclidean_score, euclidean_breakdown = compute_euclidean_score(
        hierarchy_score,
        spherical_results['spherical_score'],
        c,
        spherical_results['spherical_fit'],
        distance_matrix
    )
    
    print(f"\nEuclidean Structure Analysis:")
    for key, value in euclidean_breakdown.items():
        if key != 'total':
            print(f"  {value}")
    
    print(f"\nEuclidean score: {euclidean_score}/5")
    
    # ==========================================
    # RECOMMENDATION
    # ==========================================
    print(f"\n{'='*80}")
    print(f"COMBINED GEOMETRY RECOMMENDATION")
    print(f"{'='*80}")
    
    spherical_score = spherical_results['spherical_score']
    
    print(f"\nStructure Scores:")
    print(f"  Hierarchical (tree-like):      {hierarchy_score}/5")
    print(f"  Spherical (pattern geometry):  {spherical_score}/5")
    print(f"  Euclidean (flat/unstructured): {euclidean_score}/5")
    
    scores = {
        'HYPERBOLIC': hierarchy_score,
        'SPHERICAL': spherical_score,
        'EUCLIDEAN': euclidean_score
    }
    
    scores_sorted = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    winner = scores_sorted[0][0]
    max_score = scores_sorted[0][1]
    runner_up_name = scores_sorted[1][0]
    runner_up_score = scores_sorted[1][1]
    
    gap = max_score - runner_up_score
    
    if gap >= 2:
        confidence = 'HIGH'
        symbol = '✓✓✓'
    elif gap >= 1:
        confidence = 'MODERATE'
        symbol = '✓✓'
    else: 
        confidence = 'LOW'
        symbol = '⚠⚠'
    
    recommendations = [{
        'geometry': winner,
        'priority': 'PRIMARY',
        'score': max_score,
        'confidence': confidence,
        'reason': f"{winner} scores highest ({max_score}/5 vs {runner_up_score}/5)"
    }]
    
    if gap < 2:
        recommendations.append({
            'geometry': runner_up_name,
            'priority': 'TEST',
            'score': runner_up_score,
            'confidence': 'SECONDARY',
            'reason': f"Close alternative ({runner_up_score}/5)"
        })
    
    print(f"\n{symbol} PRIMARY:  {winner} ({max_score}/5)")
    print(f"   Confidence: {confidence}")
    print(f"   Runner-up: {runner_up_name} ({runner_up_score}/5)")
    print(f"   Gap: {gap} points")
    
    # ==========================================
    # VISUALIZATION (simplified to save memory)
    # ==========================================
    print(f"\n{'='*80}")
    print(f"Generating visualizations...")
    print(f"{'='*80}")
    
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: Dendrogram
    ax1 = plt.subplot(2, 3, 1)
    truncate = 'lastp' if n_patterns > 50 else None
    p_val = 30 if n_patterns > 50 else n_patterns
    dendrogram(linkage_matrix, ax=ax1, truncate_mode=truncate, p=p_val)
    ax1.set_title(f'Dendrogram:  {component.title()}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 2D Projection (use PCA for large datasets)
    ax2 = plt.subplot(2, 3, 2)
    pca = PCA(n_components=2, random_state=42)
    patterns_2d = pca.fit_transform(patterns_normalized)
    scatter = ax2.scatter(patterns_2d[:, 0], patterns_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=30)
    ax2.set_title(f'PCA Projection (k={best_k})', fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance Matrix (subsample for visualization)
    ax3 = plt.subplot(2, 3, 3)
    order = leaves_list(linkage_matrix)
    dm_ordered = distance_matrix[order, :][: , order]
    
    if n_patterns > 200:
        subsample = np.linspace(0, n_patterns-1, 200, dtype=int)
        dm_plot = dm_ordered[subsample, :][:, subsample]
    else:
        dm_plot = dm_ordered
    
    im = ax3.imshow(dm_plot, cmap='viridis', aspect='auto')
    ax3.set_title('Distance Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax3)
    
    # Plot 4: Silhouette
    ax4 = plt.subplot(2, 3, 4)
    k_range = range(2, len(silhouette_scores) + 2)
    ax4.plot(k_range, silhouette_scores, 'o-', linewidth=2)
    ax4.axvline(best_k, color='red', linestyle='--', linewidth=2)
    ax4.scatter([best_k], [best_sil], color='red', s=200, marker='*', zorder=5)
    ax4.set_xlabel('Number of Clusters')
    ax4.set_ylabel('Silhouette Score')
    ax4.set_title('Clustering Quality', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Distance Distribution
    ax5 = plt.subplot(2, 3, 5)
    if within_dists and between_dists:
        bins = 50
        ax5.hist(within_dists, bins=bins, alpha=0.6, label='Within', color='blue')
        ax5.hist(between_dists, bins=bins, alpha=0.6, label='Between', color='orange')
        ax5.set_xlabel('Distance')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distance Distribution', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = f"""
STRUCTURE ANALYSIS (Memory-Efficient)
{'='*40}

Component: {component.upper()}
Patterns:   {n_patterns: ,} / {n_total_patterns:,}
Window:    {window_size}

HIERARCHICAL:
  Cophenetic:    {c:.4f}
  Branching:    {branching:.2f}
  Best k:       {best_k}
  Silhouette:   {best_sil:.4f}
  Score:        {hierarchy_score}/5

SPHERICAL:
  Fit ratio:    {spherical_results.get('spherical_fit', 0):.4f}
  Best sil:     {spherical_results.get('best_silhouette_sphere', 0):.4f}
  Score:        {spherical_score}/5

EUCLIDEAN: 
  Score:        {euclidean_score}/5

{'='*40}
RECOMMENDATION: 

{symbol} {winner} ({max_score}/5)

Confidence: {confidence}
Gap:  {gap} points
"""
    
    ax6.text(0.05, 0.5, summary, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Structure Analysis: {component.upper()} (Memory-Efficient)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f'{save_dir}/efficient_analysis_{component}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved analysis to {filename}")
    
    # Clean up
    del patterns_normalized, distance_matrix, linkage_matrix
    gc.collect()
    
    return {
        'component': component,
        'n_patterns': n_patterns,
        'n_total_patterns':  n_total_patterns,
        'cophenetic_correlation': c,
        'branching_factor': branching,
        'best_k': best_k,
        'hierarchy_score': hierarchy_score,
        'spherical_fit': spherical_results.get('spherical_fit', 0),
        'spherical_score': spherical_score,
        'euclidean_score': euclidean_score,
        'euclidean_breakdown': euclidean_breakdown,
        'recommendations': recommendations,
        'primary_geometry': winner,
        'runner_up_geometry': runner_up_name,
        'confidence': confidence,
        'score_gap': gap,
        'all_scores': scores
    }


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
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Decomposition parameters
    parser.add_argument('--use_pretrained', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, default='./checkpoints/decomposition_model.pth')
    
    # Analysis parameters
    parser.add_argument('--save_dir', type=str, default='./plots/hierarchy')
    parser.add_argument('--max_patterns', type=int, default=3000,
                       help='Max patterns to extract (lower = less memory)')
    parser.add_argument('--window_fine', type=int, default=144)
    parser.add_argument('--window_coarse', type=int, default=1008)
    parser.add_argument('--window_trend', type=int, default=2160)  # Reduced from 4320
    parser.add_argument('--downsample_factor', type=int, default=1,
                       help='Downsample time series (2=half resolution, saves memory)')
    parser.add_argument('--seed', type=int, default=42)
    
    # Component selection
    parser.add_argument('--components', type=str, default='trend,coarse,fine,residual',
                       help='Comma-separated list:  trend,coarse,fine,residual')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print("=" * 80)
    print("MEMORY-EFFICIENT HIERARCHICAL ANALYSIS - LEARNABLE DECOMPOSITION")
    print("=" * 80)
    print(f"Dataset: {args.data_path}")
    print(f"Max patterns: {args.max_patterns: ,}")
    print(f"Downsample:  {args.downsample_factor}x")
    print(f"Random seed: {args.seed}")
    
    # Parse components to analyze
    components_to_analyze = args.components.split(',')
    
    window_sizes = {
        'fine': args.window_fine,
        'coarse': args.window_coarse,
        'trend': args.window_trend,
        'residual': args.window_fine
    }
    
    results = {}
    
    for component in components_to_analyze:
        component = component.strip()
        if component not in window_sizes:
            print(f"⚠ Skipping unknown component: {component}")
            continue
        
        results[component] = experiment_hierarchical_structure_efficient(
            args,
            component=component,
            window_size=window_sizes[component],
            max_patterns=args.max_patterns,
            use_pretrained=args.use_pretrained,
            model_path=args.model_path,
            save_dir=args.save_dir,
            downsample_factor=args.downsample_factor
        )
        
        # Force garbage collection between components
        gc.collect()
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    for comp, res in results.items():
        if res: 
            print(f"\n{comp.upper()}:")
            print(f"  Patterns:       {res['n_patterns']:,} / {res['n_total_patterns']:,}")
            print(f"  Hierarchical:  {res['hierarchy_score']}/5")
            print(f"  Spherical:     {res['spherical_score']}/5")
            print(f"  Euclidean:     {res['euclidean_score']}/5")
            print(f"  → PRIMARY:  {res['primary_geometry']} ({res['confidence']})")
    
    print(f"\n✓ All analyses saved to:  {args.save_dir}/")