"""
Poincare Ball Visualization for Hyperbolic Time Series Embeddings

Correctly visualizes COMBINED (Mobius-fused) embeddings with temporal hierarchy.
Uses direct hyperbolic geometry to position nodes - NO UMAP crushing.

Temporal Hierarchy: Earlier segments ? Origin | Later segments ? Boundary
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
import os
from pathlib import Path
import geoopt
from sklearn.preprocessing import StandardScaler
from adjustText import adjust_text  # pip install adjustText


class PoincareVisualizer:
    """Visualizer for Poincare ball embeddings with proper geodesic grids"""
    
    def __init__(self, curvature=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.manifold_poincare = geoopt.manifolds.PoincareBall(c=curvature)
        
        # Use Stereographic manifold for geodesic grid drawing
        self.manifold_stereo = geoopt.manifolds.Stereographic(-curvature)
        
        # Single color scheme since we only have COMBINED scale
        self.hierarchy_cmap = 'RdYlBu_r'  # Red (late) to Blue (early)
        
    def add_geodesic_grid(self, ax, line_width=0.3):
        """
        Add geodesic grid to the plot (STRICTLY following viz_poincare.py).
        Uses Stereographic manifold for proper geodesic computation.
        """
        # Define geodesic grid parameters
        N_EVALS_PER_GEODESIC = 10000
        STYLE = "--"
        COLOR = "gray"
        LINE_WIDTH = line_width
        
        # Get manifold properties
        K = self.manifold_stereo.k.item()
        R = self.manifold_stereo.radius.item()
        
        # Get maximal numerical distance to origin
        r = torch.tensor((R, 0.0), dtype=self.manifold_stereo.dtype, device=self.device)
        r = self.manifold_stereo.projx(r)
        max_dist_0 = self.manifold_stereo.dist0(r).item()
        
        # Adjust line interval for spherical geometry
        circumference = 2 * np.pi * R
        n_geodesics_per_circumference = 4 * 6
        n_geodesics_per_quadrant = n_geodesics_per_circumference // 2
        grid_interval_size = circumference / n_geodesics_per_circumference
        
        if K < 0:
            n_geodesics_per_quadrant = int(max_dist_0 / grid_interval_size)
        
        # Create time evaluation array
        min_t = -1.2 * max_dist_0 if K < 0 else -circumference / 2.0
        t = torch.linspace(min_t, -min_t, N_EVALS_PER_GEODESIC, device=self.device)[:, None]
        
        def plot_geodesic(gv):
            ax.plot(*gv.cpu().t().numpy(), STYLE, color=COLOR, linewidth=LINE_WIDTH, alpha=0.6)
        
        # Define geodesic directions
        o = torch.zeros(2, device=self.device)
        u_x = torch.tensor((0.0, 1.0), device=self.device)
        u_y = torch.tensor((1.0, 0.0), device=self.device)
        
        if K < 0:
            x_geodesic = self.manifold_stereo.geodesic_unit(t, o, u_x)
            y_geodesic = self.manifold_stereo.geodesic_unit(t, o, u_y)
            plot_geodesic(x_geodesic)
            plot_geodesic(y_geodesic)
        
        for i in range(1, n_geodesics_per_quadrant):
            i_tensor = torch.as_tensor(float(i), device=self.device)
            x = self.manifold_stereo.geodesic_unit(i_tensor * grid_interval_size, o, u_y)
            y = self.manifold_stereo.geodesic_unit(i_tensor * grid_interval_size, o, u_x)
            
            x_geodesic = self.manifold_stereo.geodesic_unit(t, x, u_x)
            y_geodesic = self.manifold_stereo.geodesic_unit(t, y, u_y)
            
            plot_geodesic(x_geodesic)
            plot_geodesic(y_geodesic)
            if K < 0:
                plot_geodesic(-x_geodesic)
                plot_geodesic(-y_geodesic)
    
    def compute_hierarchy_score(self, embeddings, labels):
        """
        Compute temporal hierarchy quality score.
        
        Positive score = good temporal hierarchy (earlier segments closer to origin)
        """
        # Compute distances from origin in POINCARE space
        dists = self.manifold_poincare.dist0(embeddings).cpu().numpy()
        
        # Compute correlation: segment index vs distance from origin
        segments = labels['segment']
        
        if len(np.unique(segments)) <= 1:
            return 0.0
        
        correlation = np.corrcoef(segments, dists)[0, 1]
        
        return correlation
    
    def sample_hierarchical_nodes(self, embeddings, labels, max_nodes=30):
        """
        Sample nodes with ENFORCED feature diversity.
        Strategy: Round-robin over features, then segments.
        """
        if len(embeddings) <= max_nodes:
            return embeddings, labels
        
        segments = labels['segment']
        features = labels['feature']
        
        unique_segments = np.unique(segments)
        unique_features = np.unique(features)
        
        # NEW: Round-robin sampling
        selected_indices = []
        
        # Calculate how many nodes per feature
        nodes_per_feature = max(1, max_nodes // len(unique_features))
        
        for feat in unique_features:
            feat_mask = features == feat
            feat_indices = np.where(feat_mask)[0]
            
            if len(feat_indices) == 0:
                continue
            
            # Within this feature, sample across segments
            feat_segments = segments[feat_mask]
            unique_feat_segments = np.unique(feat_segments)
            
            # Distribute budget evenly across segments
            nodes_per_seg = max(1, nodes_per_feature // len(unique_feat_segments))
            
            for seg in unique_feat_segments:
                seg_feat_mask = (features == feat) & (segments == seg)
                seg_feat_indices = np.where(seg_feat_mask)[0]
                
                if len(seg_feat_indices) > 0:
                    # Take first node from this (feature, segment) pair
                    selected_indices.append(seg_feat_indices[0])
                    
                    if len(selected_indices) >= max_nodes:
                        break
            
            if len(selected_indices) >= max_nodes:
                break
        
        # Ensure we don't exceed max_nodes
        selected_indices = selected_indices[:max_nodes]
        selected_indices = np.array(selected_indices)
        
        sampled_embeddings = embeddings[selected_indices]
        sampled_labels = {k: v[selected_indices] for k, v in labels.items()}
        
        print(f"  Sampled {len(sampled_embeddings)}/{len(embeddings)} nodes")
        print(f"  Features in sample: {np.unique(sampled_labels['feature'])}")
        
        return sampled_embeddings, sampled_labels
    
    def select_best_subgraphs(self, embeddings, labels, num_subgraphs=5, 
                             min_points=20, max_nodes_per_subgraph=30):
        """
        Select best temporal hierarchy subgraphs.
        
        NOW: Each subgraph = ALL features x ALL segments for a SINGLE batch
        This gives meaningful multi-feature, multi-temporal hierarchy!
        """
        unique_batches = np.unique(labels['batch'])
        
        candidates = []
        
        print(f"\nComputing temporal hierarchy scores for batches...")
        
        for batch in unique_batches:
            # Extract ALL features x ALL segments for this batch
            mask = labels['batch'] == batch
            
            if mask.sum() < min_points:
                continue
            
            sub_embeddings = embeddings[mask]
            sub_labels = {k: v[mask] for k, v in labels.items()}
            
            # Compute temporal hierarchy score
            score = self.compute_hierarchy_score(sub_embeddings, sub_labels)
            
            candidates.append({
                'embeddings': sub_embeddings,
                'labels': sub_labels,
                'score': score,
                'batch': batch,
                'num_points': mask.sum()
            })
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n{'='*70}")
        print(f"SUBGRAPH SELECTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total candidates: {len(candidates)}")
        
        if len(candidates) > 0:
            print(f"Best score: {candidates[0]['score']:.4f}")
            print(f"Worst score: {candidates[-1]['score']:.4f}")
            
            # Show distribution
            all_scores = [c['score'] for c in candidates]
            positive_count = sum(1 for s in all_scores if s > 0)
            negative_count = sum(1 for s in all_scores if s < 0)
            
            print(f"Positive (good temporal hierarchy): {positive_count}")
            print(f"Negative (inverted hierarchy): {negative_count}")
        else:
            print("? WARNING: No valid subgraphs found!")
            return []
        
        # Select top N
        selected = candidates[:num_subgraphs]
        
        print(f"\nSelected top {len(selected)} subgraphs:")
        
        # Sample nodes for each subgraph
        for i, sg in enumerate(selected):
            sampled_emb, sampled_labels = self.sample_hierarchical_nodes(
                sg['embeddings'], 
                sg['labels'], 
                max_nodes=max_nodes_per_subgraph
            )
            sg['embeddings'] = sampled_emb
            sg['labels'] = sampled_labels
            sg['num_points'] = len(sampled_emb)
            
            hierarchy_quality = "GOOD ?" if sg['score'] > 0 else "INVERTED ?"
            print(f"Rank {i+1}: Batch {sg['batch']}, "
                  f"Score {sg['score']:.4f} ({hierarchy_quality}), Points {sg['num_points']}")
        
        return selected
    
    def scale_to_visible_range(self, embeddings, target_max=0.75):
        """
        Scale embeddings to [0, target_max] for better visibility.
        Uses HARD constraint to ensure spread.
        """
        # Compute current distances from origin
        dists = self.manifold_poincare.dist0(embeddings).cpu().numpy()
        current_max = dists.max()
        current_min = dists.min()
        
        if current_max - current_min < 1e-6:
            # All points at same distance - expand arbitrarily
            return embeddings
        
        # Map distances linearly: [current_min, current_max] ? [0.1, target_max]
        # This GUARANTEES spread from near-origin to boundary
        new_dists = 0.1 + (dists - current_min) / (current_max - current_min) * (target_max - 0.1)
        
        # Convert back to Poincare coordinates
        tangent = self.manifold_poincare.logmap0(embeddings)
        tangent_np = tangent.cpu().numpy()
        
        # Compute current tangent norms
        tangent_norms = np.linalg.norm(tangent_np, axis=1, keepdims=True)
        tangent_norms = np.clip(tangent_norms, 1e-8, None)  # Avoid division by zero
        
        # Compute target tangent norms from new distances
        # For Poincare: tanh(d/2) = r, so d = 2*arctanh(r)
        # But we work in tangent space: ||v|| = d for small d
        # For better control, use: ||v|| = artanh(r) / sqrt(c)
        target_tangent_norms = np.arctanh(np.clip(new_dists, 0, 0.99)).reshape(-1, 1)
        
        # Scale tangent vectors
        scaled_tangent = tangent_np * (target_tangent_norms / tangent_norms)
        
        # Map back to manifold
        scaled_tangent_torch = torch.tensor(scaled_tangent, dtype=torch.float32, device=self.device)
        scaled_embeddings = self.manifold_poincare.expmap0(scaled_tangent_torch)
        scaled_embeddings = self.manifold_poincare.projx(scaled_embeddings)
        
        return scaled_embeddings
    
    def project_with_polar_layout(self, embeddings, labels):
        """
        NEW: Use POLAR layout based on temporal hierarchy.
        
        Strategy:
        1.Radius = hyperbolic distance (temporal order)
        2.Angle = distributed by feature to avoid crowding
        3.poiNO UMAP - direct geometric layout
        """
        # Compute radii from hyperbolic distances
        dists = self.manifold_poincare.dist0(embeddings).cpu().numpy()
        
        # Normalize to [0.1, 0.75] range
        dists_min = dists.min()
        dists_max = dists.max()
        
        if dists_max - dists_min < 1e-6:
            radii = np.ones_like(dists) * 0.4
        else:
            radii = 0.1 + (dists - dists_min) / (dists_max - dists_min) * 0.65
        
        # Assign angles based on FEATURE to spread points
        segments = labels['segment']
        features = labels['feature']
        
        unique_features = np.unique(features)
        num_features = len(unique_features)
        
        # Base angle per feature
        base_angles = np.linspace(0, 2*np.pi, num_features, endpoint=False)
        
        # Assign angles
        angles = np.zeros(len(embeddings))
        for i, (seg, feat) in enumerate(zip(segments, features)):
            # Base angle from feature
            feat_idx = np.where(unique_features == feat)[0][0]
            base_angle = base_angles[feat_idx]
            
            # Add small offset based on segment to avoid exact overlap
            seg_offset = (seg / (segments.max() + 1)) * (2*np.pi / num_features) * 0.5
            
            angles[i] = base_angle + seg_offset
        
        # Convert polar to Cartesian
        coords_2d = np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        
        return coords_2d
    
    def get_feature_name(self, feature_idx, dataset_name):
        """Get human-readable feature name"""
        # ? Use the dataset_name parameter directly (already passed from visualize_subgraph)
        
        if "AQShunyi" in dataset_name or "AQWan" in dataset_name:
            feature_names = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 
                           'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
            return feature_names[feature_idx] if feature_idx < len(feature_names) else f"F{feature_idx}"
        elif "PM2" in dataset_name or "pm2_5" in dataset_name:
            return f"Station_{feature_idx}"
        elif "Electricity" in dataset_name or "electricity" in dataset_name:
            return f"Client_{feature_idx}"
        else:
            return f"F{feature_idx}"
    
    def visualize_subgraph(self, embeddings, labels, ax, title="", dataset_name="", rank=1):
        """Visualize a single subgraph with proper labels and spacing"""
        
        # Scale embeddings to visible range
        embeddings = self.scale_to_visible_range(embeddings, target_max=0.75)
        
        # Use POLAR layout instead of UMAP
        coords_2d = self.project_with_polar_layout(embeddings, labels)
        
        # Extract metadata
        segments = labels['segment']
        features = labels['feature']
        
        # Color by temporal progression
        segment_min = segments.min()
        segment_max = segments.max()
        
        if segment_max > segment_min:
            segment_norm = (segments - segment_min) / (segment_max - segment_min)
        else:
            segment_norm = np.zeros_like(segments, dtype=float)
        
        # Plot points
        scatter = ax.scatter(
            coords_2d[:, 0], coords_2d[:, 1],
            c=segment_norm,
            cmap=self.hierarchy_cmap,
            s=200,
            alpha=0.85,
            edgecolors='black',
            linewidth=1.2,
            zorder=5
        )
        
        # Add colorbar only for first subplot
        if rank == 1:
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Temporal Progress\n(Early --> Late)', fontsize=9, rotation=270, labelpad=20)
        
        # Add labels with SMART positioning to avoid overlap
        texts = []
        for idx in range(len(coords_2d)):
            seg_idx = int(segments[idx])
            feat_idx = int(features[idx])
            
            feat_name = self.get_feature_name(feat_idx, dataset_name)
            label = f"t{seg_idx}\n{feat_name}"
            
            # Create text object
            #txt = ax.annotate(
            #    label,
            #    (coords_2d[idx, 0], coords_2d[idx, 1]),
            #    fontsize=7,
            #    alpha=0.95,
            #    ha='center',
            #    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.6, edgecolor='black', linewidth=0.5)
            #)
            #texts.append(txt)
        
            ax.annotate(
                label,
                (coords_2d[idx, 0], coords_2d[idx, 1]),
                fontsize=7,
                alpha=0.95,
                ha='center',
                va='bottom',  # Align bottom of text to point (pushes text up)
                xytext=(0, 10),  # 8 points above the node
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.6, edgecolor='black', linewidth=0.5)
            )
                        
        # Use adjust_text to prevent overlap (requires pip install adjustText)
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, 
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                       expand_points=(1.5, 1.5),
                       force_points=0.5,
                       force_text=0.5)
        except ImportError:
            print("? Install adjustText for better label placement: pip install adjustText")
        
        # Draw Poincare disk boundary
        circle = Circle((0, 0), 1.0, fill=False, edgecolor='black', linewidth=2.5, linestyle='-', zorder=10)
        ax.add_patch(circle)
        
        # Origin marker
        ax.scatter(0, 0, color='black', s=400, marker='*', label='Origin', zorder=15, edgecolors='white', linewidths=1.5)
        
        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlim([-1.15, 1.15])
        ax.set_ylim([-1.15, 1.15])
        ax.set_aspect('equal')
        ax.axis('off')
    
    def visualize_best_subgraphs(self, embeddings_path, save_dir, num_subgraphs=5, 
                                max_nodes_per_subgraph=30):
        """Main visualization function"""
        
        # Load embeddings
        print(f"\n{'='*70}")
        print(f"LOADING EMBEDDINGS")
        print(f"{'='*70}")
        print(f"Path: {embeddings_path}")
        
        if not os.path.exists(embeddings_path):
            print(f"? ERROR: Embeddings file not found!")
            return
        
        data = torch.load(embeddings_path, map_location=self.device, weights_only=False)
        
        embeddings = data['embeddings']  # [N, encode_dim]
        labels = data['labels']  # dict
        
        print(f"? Loaded {len(embeddings)} embeddings")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Unique batches: {len(np.unique(labels['batch']))}")
        print(f"  Unique features: {len(np.unique(labels['feature']))}")
        print(f"  Unique segments: {len(np.unique(labels['segment']))}")
        print(f"  Scales: {np.unique(labels['scale'])}")
        
        # Verify all are "combined"
        if not all(labels['scale'] == 'combined'):
            print("? WARNING: Expected all embeddings to be 'combined' scale!")
        
        # Infer dataset name
        dataset_name = ""
        if "AQShunyi" in embeddings_path:
            dataset_name = "AQShunyi"
        elif "AQWan" in embeddings_path:
            dataset_name = "AQWan"
        elif "PM2" in embeddings_path or "pm2_5" in embeddings_path:
            dataset_name = "PM2.5"
        
        # Select best subgraphs
        print(f"\nSelecting top {num_subgraphs} subgraphs...")
        best_subgraphs = self.select_best_subgraphs(
            embeddings, labels, 
            num_subgraphs=num_subgraphs,
            max_nodes_per_subgraph=max_nodes_per_subgraph
        )
        
        if len(best_subgraphs) == 0:
            print("? ERROR: No valid subgraphs found!")
            return
        
        # Create combined visualization
        fig = plt.figure(figsize=(20, 12))
        
        os.makedirs(save_dir, exist_ok=True)
        
        for i, subgraph in enumerate(best_subgraphs):
            ax = fig.add_subplot(2, 3, i + 1)
            
            # Add geodesic grid
            self.add_geodesic_grid(ax, line_width=0.4)
            
            hierarchy_indicator = "?" if subgraph['score'] > 0 else "?"
            
            #title = (f"Rank #{i+1} {hierarchy_indicator} | Batch {subgraph['batch']}\n"
            #        f"Hierarchy Score: {subgraph['score']:.3f} | Nodes: {subgraph['num_points']}")
            
            self.visualize_subgraph(
                subgraph['embeddings'],
                subgraph['labels'],
                ax,
                title="",
                dataset_name=dataset_name,
                rank=i+1
            )
            
            if i == 0:
                ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            
            # Save individual subgraph at 200 DPI
            fig_single = plt.figure(figsize=(8, 8))
            ax_single = fig_single.add_subplot(111)
            self.add_geodesic_grid(ax_single, line_width=0.4)
            self.visualize_subgraph(
                subgraph['embeddings'],
                subgraph['labels'],
                ax_single,
                title="",
                dataset_name=dataset_name,
                rank=i+1
            )
            if i == 0:
                ax_single.legend(loc='upper left', fontsize=10, framealpha=0.9)
            
            single_path = os.path.join(save_dir, f'{dataset_name}_subgraph_rank{i+1}.png')
            plt.tight_layout()
            fig_single.savefig(single_path, dpi=200, bbox_inches='tight')
            plt.close(fig_single)
            print(f"? Saved individual subgraph to {single_path}")
        
        # Overall title
        fig.suptitle(
            f"{dataset_name} - Top {len(best_subgraphs)} Temporal Hierarchies (Mobius-Fused Embeddings)\n"
            "Temporal Hierarchy: Earlier time segments (blue) ? Origin | Later time segments (red) ? Boundary\n"
            "Labels: t{X} = time segment index | Feature name from dataset schema",
            fontsize=13,
            fontweight='bold'
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save combined visualization
        combined_path = os.path.join(save_dir, f'{dataset_name}_all_subgraphs.png')
        plt.savefig(combined_path, dpi=200, bbox_inches='tight')
        print(f"\n? Saved combined visualization to {combined_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize Poincare embeddings')
    parser.add_argument('--embeddings_path', type=str, required=True,
                       help='Path to embeddings .pt file')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--num_subgraphs', type=int, default=5,
                       help='Number of best subgraphs to visualize')
    parser.add_argument('--max_nodes', type=int, default=30,
                       help='Maximum nodes per subgraph')
    parser.add_argument('--curvature', type=float, default=1.0,
                       help='Curvature of Poincare ball')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = PoincareVisualizer(curvature=args.curvature)
    
    # Visualize
    viz.visualize_best_subgraphs(
        embeddings_path=args.embeddings_path,
        save_dir=args.save_dir,
        num_subgraphs=args.num_subgraphs,
        max_nodes_per_subgraph=args.max_nodes
    )


if __name__ == '__main__':
    main()