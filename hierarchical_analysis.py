import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet, leaves_list
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import warnings
warnings.filterwarnings('ignore')


def extract_mstl_components_from_dataset(dataset):
    """
    Extract MSTL components directly from dataset.
    Memory efficient - no batch iteration.
    """
    print("\nExtracting MSTL components from dataset...")
    
    # Direct access to stored components
    data_dict = {
        'trend': dataset.decomposed_components['trend'],
        'weekly': dataset.decomposed_components['seasonal_weekly'],
        'daily': dataset.decomposed_components['seasonal_daily'],
        'residual': dataset.decomposed_components['residual']
    }
    
    print(f"✓ Components extracted:")
    for name, data in data_dict.items():
        print(f"  {name:8s}: {data.shape}")
        # Memory check
        mem_mb = data.nbytes / (1024**2)
        print(f"            {mem_mb:.1f} MB")
    
    return data_dict


def plot_frequency_spectrum(component_data, component_name, ax, sampling_rate=1.0):
    """Plot FFT frequency spectrum."""
    fft = np.fft.fft(component_data)
    freq = np.fft.fftfreq(len(component_data), d=1/sampling_rate)
    
    pos_mask = freq > 0
    freq = freq[pos_mask]
    magnitude = np.abs(fft[pos_mask])
    
    ax.plot(freq, magnitude, linewidth=1.5)
    ax.set_ylabel('Magnitude')
    ax.set_title(f'{component_name} - Frequency Spectrum', pad=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)


def visualize_component_characteristics(data_dict, save_dir='./plots/hierarchy'):
    """Create overview of component frequency characteristics."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    
    components = [
        ('trend', 'blue', 'TREND'),
        ('weekly', 'green', 'WEEKLY'),
        ('daily', 'orange', 'DAILY'),
        ('residual', 'purple', 'RESIDUAL')
    ]
    
    for idx, (comp_name, color, label) in enumerate(components):
        data = data_dict[comp_name][:, 0]  # First feature
        
        # Time domain
        ax_time = axes[idx, 0]
        time_steps = np.arange(min(2000, len(data)))
        ax_time.plot(time_steps, data[:len(time_steps)], color=color, linewidth=1.5)
        ax_time.set_ylabel('Value')
        ax_time.set_title(f'{label} - Time Domain')
        ax_time.grid(True, alpha=0.3)
        
        # Frequency domain
        ax_freq = axes[idx, 1]
        plot_frequency_spectrum(data, label, ax_freq)
    
    axes[-1, 0].set_xlabel('Time Steps')
    axes[-1, 1].set_xlabel('Frequency (cycles/hour)')
    
    plt.suptitle('Component Characteristics: Time & Frequency Domain', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/component_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved component overview to {save_dir}/component_overview.png")

def test_spherical_structure(patterns_normalized, component='daily'):
    """Minimal spherical structure test."""
    
    # Project to unit sphere
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
    
    # ================================================
    # NEW: Actually compute spherical silhouettes!
    # ================================================
    max_k = min(20, n_patterns // 2)
    spherical_silhouettes = []
    
    print(f"Computing spherical clustering quality...")
    
    for k in range(2, max_k + 1):
        # Spherical k-means (uses cosine distance)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(patterns_sphere)
        
        # Compute silhouette with cosine metric
        try:
            sil = silhouette_score(patterns_sphere, labels, metric='cosine')
            spherical_silhouettes.append(sil)
        except:
            spherical_silhouettes.append(0.0)
        
        if k in [2, 3, 5, 7, 10, max_k]:
            print(f"  Spherical k={k:2d}: silhouette={spherical_silhouettes[-1]:.4f}")
    
    # Find best k for spherical
    if spherical_silhouettes:
        best_k_sphere = np.argmax(spherical_silhouettes) + 2
        best_sil_sphere = spherical_silhouettes[best_k_sphere - 2]
        print(f"→ Best spherical k={best_k_sphere} (silhouette={best_sil_sphere:.4f})")
    else:
        best_k_sphere = 2
        best_sil_sphere = 0.0
    
    # ================================================
    # Scoring (with actual silhouette now)
    # ================================================
    score_sphere = 0
    
    # Metric 1: Spherical fit ratio
    if spherical_fit > 1.5:
        score_sphere += 2
    elif spherical_fit > 1.2:
        score_sphere += 1
    
    # Metric 2: Spherical silhouette quality
    if best_sil_sphere > 0.3:
        score_sphere += 2
    elif best_sil_sphere > 0.2:
        score_sphere += 1
    
    # Metric 3: Comparison bonus (if spherical clearly better)
    if spherical_fit > 1.3 and best_sil_sphere > 0.25:
        score_sphere += 1
    
    print(f"Spherical structure score: {score_sphere}/5")
    
    
    return {
        'spherical_fit': spherical_fit,
        'spherical_score': score_sphere,
        'best_silhouette_sphere': best_sil_sphere,
        'best_k_sphere': best_k_sphere,
        'spherical_silhouettes': spherical_silhouettes,
        'cv_euclidean': cv_euclidean,
        'cv_angular': cv_angular
    }

def experiment_2_hierarchical_structure(
    data_dict,
    window_size, 
    component='daily',
    max_patterns=5000,  # Safety limit
    save_dir='./plots/hierarchy'
):
    """
    Complete structure analysis: hierarchical + spherical.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    component_data = data_dict[component]
    T, d_features = component_data.shape
    
    print("\n" + "=" * 80)
    print(f"COMPLETE STRUCTURE ANALYSIS: {component.upper()}")
    print("=" * 80)
    print(f"Data shape: T={T:,}, features={d_features}")
    
    stride = max(1, window_size // 2)
    print(f"Window size: {window_size}, Stride: {stride}")
    
    # ==========================================
    # PATTERN EXTRACTION (Same as before)
    # ==========================================
    print(f"\nExtracting patterns...")
    patterns = []
    for i in range(0, T - window_size + 1, stride):
        patterns.append(component_data[i:i+window_size, :].flatten())

        # Progress indicator
        if (i // stride) % 500 == 0:
            print(f"  Progress: {i}/{T} ({100*i/T:.1f}%)")
    
    patterns = np.array(patterns)
    n_patterns_total = len(patterns)

    print(f" Extracted {n_patterns_total:,} patterns")
    
    # Memory estimate
    pattern_mem_mb = patterns.nbytes / (1024**2)
    print(f"  Pattern memory: {pattern_mem_mb:.1f} MB")
    
    # Subsample if needed
    if n_patterns_total > max_patterns:
        print(f"Subsampling to {max_patterns:,} patterns")
        indices = np.random.choice(n_patterns_total, max_patterns, replace=False)
        indices.sort()
        patterns = patterns[indices]
        n_patterns = max_patterns
    else:
        n_patterns = n_patterns_total
    
    # Normalize
    print(f"Normalizing patterns...")
    scaler = StandardScaler()
    patterns_normalized = scaler.fit_transform(patterns)
    
    # ==========================================
    # PART 1: HIERARCHICAL STRUCTURE TEST
    # ==========================================
    print(f"\n{'='*80}")
    print(f"PART 1: HIERARCHICAL STRUCTURE ANALYSIS")
    print(f"{'='*80}")
    
    # Compute distances
    print(f"Computing pairwise distances...")
    import time
    start = time.time()

    distances = pdist(patterns_normalized, metric='euclidean')
    distance_matrix = squareform(distances)

    print(f"Distances computed in {time.time()-start:.1f}s")
    
    # Hierarchical clustering
    print(f"Building hierarchical clustering...")

    linkage_matrix = linkage(distances, method='ward')
    print(f"Clustering built in {time.time()-start:.1f}s")
    
    # Cophenetic correlation
    c, _ = cophenet(linkage_matrix, distances)

    print(f"Cophenetic correlation: {c:.4f}")
    
    # Test different k
    print(f"Testing cluster quality...")
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

    print(f"Branching factor: {branching:.2f}")
    
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
    
    within_mean = np.mean(within_dists) if within_dists else 0
    between_mean = np.mean(between_dists) if between_dists else 0
    ratio = between_mean / within_mean if within_mean > 0 else np.nan
    
    if within_dists and between_dists:
        t_stat, p_value = ttest_ind(within_dists, between_dists)
        separation_significant = p_value < 0.001
    else:
        p_value = 1.0
        separation_significant = False
    
    # Compute hierarchical score
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
    
    print(f"\nHierarchical structure score: {hierarchy_score}/5")
    
    # ==========================================
    # PART 2: SPHERICAL STRUCTURE TEST
    # ==========================================
    print(f"\n{'='*80}")
    print(f"PART 2: SPHERICAL STRUCTURE ANALYSIS")
    print(f"{'='*80}")
    
    spherical_results = test_spherical_structure(
        patterns_normalized, 
        component=component
    )

    # ==========================================
    # PART 4: COMBINED RECOMMENDATION
    # ==========================================
    print(f"\n{'='*80}")
    print(f"COMBINED GEOMETRY RECOMMENDATION")
    print(f"{'='*80}")
    
    spherical_score = spherical_results['spherical_score']
    
    print(f"\nStructure Scores:")
    print(f"  Hierarchical (tree-like):      {hierarchy_score}/5")
    print(f"  Spherical (pattern geometry):  {spherical_score}/5")
    
    # Decision logic
    recommendations = []
    
    # Strong hierarchy → Hyperbolic
    if hierarchy_score >= 4:
        recommendations.append({
            'geometry': 'HYPERBOLIC',
            'priority': 'PRIMARY',
            'score': hierarchy_score,
            'reason': 'Strong hierarchical structure detected'
        })
    elif hierarchy_score >= 2:
        recommendations.append({
            'geometry': 'HYPERBOLIC',
            'priority': 'TEST',
            'score': hierarchy_score,
            'reason': 'Moderate hierarchical structure'
        })
    
    # Strong spherical/periodic → Spherical
    spherical = spherical_score
    if spherical >= 4:
        recommendations.append({
            'geometry': 'SPHERICAL',
            'priority': 'PRIMARY',
            'score': spherical_score,
            'reason': f'Strong spherical structure (sph:{spherical_score})'
        })
    elif spherical >= 2:
        recommendations.append({
            'geometry': 'SPHERICAL',
            'priority': 'TEST',
            'score': spherical,
            'reason': f'Moderate spherical structure'
        })
    
    # Fallback → Euclidean
    if not recommendations:
        recommendations.append({
            'geometry': 'EUCLIDEAN',
            'priority': 'PRIMARY',
            'score': 0,
            'reason': 'No strong structure detected'
        })
    
    # Sort by score
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"FINAL RECOMMENDATIONS:")
    print(f"{'='*80}")
    
    for rank, rec in enumerate(recommendations, 1):
        if rec['priority'] == 'PRIMARY':
            symbol = "✓✓✓"
        elif rec['priority'] == 'TEST':
            symbol = "⚠⚠"
        else:
            symbol = "→"
        
        print(f"{rank}. {symbol} {rec['geometry']:12s} (score: {rec['score']}/5)")
        print(f"   Reason: {rec['reason']}")
    
    # ==========================================
    # VISUALIZATIONS (Enhanced)
    # ==========================================
    print(f"\n{'='*80}")
    print(f"Generating visualizations...")
    print(f"{'='*80}")
    # ==========================================
    # VISUALIZATIONS (2x3 grid)
    # ==========================================
    print(f"\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Dendrogram
    ax1 = plt.subplot(2, 3, 1)
    truncate = 'lastp' if n_patterns > 50 else None
    p_val = 30 if n_patterns > 50 else n_patterns
    dendrogram(linkage_matrix, ax=ax1, truncate_mode=truncate, p=p_val)
    ax1.set_title(f'Dendrogram: {component.title()}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Projection
    ax2 = plt.subplot(2, 3, 2)

    if n_patterns > 3000:
        pca = PCA(n_components=2, random_state=42)
        patterns_2d = pca.fit_transform(patterns_normalized)
        title_str = f'PCA (k={best_k})'
    else:
        perplexity = min(30, n_patterns - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        patterns_2d = tsne.fit_transform(patterns_normalized)
        title_str = f't-SNE (k={best_k})'
    
    scatter = ax2.scatter(patterns_2d[:, 0], patterns_2d[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=30)
    ax2.set_title(title_str, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Cluster')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance Matrix
    ax3 = plt.subplot(2, 3, 3)
    order = leaves_list(linkage_matrix)
    dm_ordered = distance_matrix[order, :][:, order]
    
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
        ax5.axvline(within_mean, color='blue', linestyle='--', linewidth=2)
        ax5.axvline(between_mean, color='orange', linestyle='--', linewidth=2)
        ax5.set_xlabel('Distance')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distance Distribution', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    # Build recommendation text
    primary_rec = recommendations[0]
    rec_symbol = "✓✓✓" if primary_rec['priority'] == 'PRIMARY' else "⚠⚠"
    rec_text = f"{rec_symbol} {primary_rec['geometry']}"
    
    summary = f"""
COMPLETE STRUCTURE ANALYSIS
{'='*40}

Component: {component.upper()}
Patterns:  {n_patterns:,}

HIERARCHICAL:
  Cophenetic:   {c:.4f}
  Branching:    {branching:.2f}
  Best k:       {best_k}
  Silhouette:   {best_sil:.4f}
  Score:        {hierarchy_score}/5

SPHERICAL:
  Fit ratio:    {spherical_results.get('spherical_fit', 0):.4f}
  Best sil:     {spherical_results.get('best_silhouette_sphere', 0):.4f}
  Score:        {spherical_score}/5


{'='*40}
RECOMMENDATION:

{rec_text}

{primary_rec['reason']}
"""
    
    if len(recommendations) > 1:
        secondary = recommendations[1]
        summary += f"\n\nSecondary: {secondary['geometry']}"
    
    ax6.text(0.05, 0.5, summary, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'Complete Structure Analysis: {component.upper()}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = f'{save_dir}/complete_analysis_{component}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved complete analysis to {filename}")
    
    # ==========================================
    # RETURN RESULTS
    # ==========================================
    return {
        'component': component,
        'n_patterns': n_patterns,
        # Hierarchical metrics
        'cophenetic_correlation': c,
        'branching_factor': branching,
        'best_k': best_k,
        'hierarchy_score': hierarchy_score,
        # Spherical metrics
        'spherical_fit': spherical_results.get('spherical_fit', 0),
        'spherical_score': spherical_score,
        # Combined recommendation
        'recommendations': recommendations,
        'primary_geometry': recommendations[0]['geometry'] if recommendations else 'EUCLIDEAN'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data parameters
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
    
    # MSTL parametersz
    parser.add_argument('--num_basis', type=int, default=10)
    parser.add_argument('--orthogonal_lr', type=float, default=1e-3)
    parser.add_argument('--orthogonal_iters', type=int, default=300)
    parser.add_argument('--use_segments', action='store_true', default=False)
    parser.add_argument('--mstl_period', type=int, default=144)
    
    # Loader parameters
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # Analysis parameters
    parser.add_argument('--save_dir', type=str, default='./plots/hierarchy')
    parser.add_argument('--max_patterns', type=int, default=5000)
    parser.add_argument('--window_daily', type=int, default=144)
    parser.add_argument('--window_weekly', type=int, default=1008)
    parser.add_argument('--window_trend', type=int, default=4320)
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("HIERARCHICAL STRUCTURE ANALYSIS FOR MSTL COMPONENTS")
    print("=" * 80)
    print(f"Dataset: {args.data_path}")
    print(f"Max patterns per component: {args.max_patterns:,}")
    
    # Load data
    from data_provider.data_factory import data_provider
    
    print("\nLoading training data...")
    train_data, train_loader = data_provider(args, flag='train')
    
    # Extract components (direct from dataset, not loader!)
    data_dict = extract_mstl_components_from_dataset(train_data)
    
    # Component overview
    print("\nCreating component overview...")
    visualize_component_characteristics(data_dict, save_dir=args.save_dir)
    
    # Analyze each component WITH ALL THREE TESTS
    results = {}
    window_sizes = {
        'daily': args.window_daily,
        'weekly': args.window_weekly,
        'trend': args.window_trend,
        'residual': args.window_daily  # Same as daily
    }
    
    # Run complete analysis on all components
    for component in ['trend', 'weekly', 'daily', 'residual']:
        results[component] = experiment_2_hierarchical_structure(
            data_dict,
            component=component,
            window_size=window_sizes[component],
            max_patterns=args.max_patterns,
            save_dir=args.save_dir
        )
    
    # Final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY - ALL COMPONENTS")
    print("=" * 80)
    
    for comp, res in results.items():
        if res:
            print(f"\n{comp.upper()}:")
            print(f"  Hierarchical:  {res['hierarchy_score']}/5 (cophenetic: {res['cophenetic_correlation']:.4f})")
            print(f"  Spherical:     {res['spherical_score']}/5")
            print(f"  → PRIMARY: {res['primary_geometry']}")
    
    print(f"\n All analyses saved to: {args.save_dir}/")