import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class EnhancedTensorBoardLogger:
    """
    Comprehensive TensorBoard logger for hyperbolic time series forecasting.
    
    Logs:
    - Training/validation losses (properly scaled)
    - Component-wise embeddings and reconstructions
    - Hyperbolic geometry visualizations
    - Hierarchy scales
    - Prediction vs ground truth plots
    """
    
    def __init__(self, log_dir, experiment_name='experiment'):
        """
        Args:
            log_dir: str - directory for TensorBoard logs
            experiment_name: str - name of experiment
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Scaling factors for different metrics
        self.loss_scale = 1.0
        self.metric_scale = 1.0
        
        print(f" TensorBoard logger initialized: {self.log_dir}")
        print(f"   Run: tensorboard --logdir={log_dir}")
    
    # ============================================
    # Loss and Metrics Logging
    # ============================================
    
    def log_losses(self, epoch, train_loss, val_loss=None, train_losses_dict=None, 
                   val_losses_dict=None):
        """
        Log training and validation losses with proper scaling.
        
        Args:
            epoch: int
            train_loss: float - total training loss
            val_loss: float - total validation loss
            train_losses_dict: dict - component losses (e.g., {'mse': ..., 'hierarchy': ...})
            val_losses_dict: dict - validation component losses
        """
        # Main losses
        self.writer.add_scalars('Loss/Total', {
            'train': train_loss,
            'val': val_loss if val_loss is not None else 0.0
        }, epoch)
        
        # Log on proper scale (log scale for better visibility)
        if train_loss > 10:
            self.writer.add_scalar('Loss/Train_Log', np.log10(train_loss + 1e-10), epoch)
        if val_loss is not None and val_loss > 10:
            self.writer.add_scalar('Loss/Val_Log', np.log10(val_loss + 1e-10), epoch)
        
        # Component losses
        if train_losses_dict is not None:
            for name, value in train_losses_dict.items():
                self.writer.add_scalar(f'Loss/Train_{name}', value, epoch)
        
        if val_losses_dict is not None:
            for name, value in val_losses_dict.items():
                self.writer.add_scalar(f'Loss/Val_{name}', value, epoch)
    
    def log_metrics(self, epoch, metrics_dict, prefix='Metrics'):
        """
        Log evaluation metrics (MSE, MAE, etc.)
        
        Args:
            epoch: int
            metrics_dict: dict - {'mse': ..., 'mae': ..., 'rmse': ...}
            prefix: str - prefix for metric names
        """
        for name, value in metrics_dict.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, epoch)
            
            # Log on multiple scales for clarity
            if value > 0:
                self.writer.add_scalar(f'{prefix}/{name}_log', np.log10(value + 1e-10), epoch)
        
    # ============================================
    # Hierarchy Scales
    # ============================================
    
    def log_hierarchy_scales(self, epoch, model, manifold_type='lorentz'):
        """
        Log learned hierarchy scales.
        
        Args:
            epoch: int
            model: nn.Module - model with embed_hyperbolic or embed_euclidean
            manifold_type: str - 'lorentz', 'poincare', or 'euclidean'
        """
        # Get encoder
        if manifold_type in ['lorentz', 'poincare']:
            encoder = model.forecater.embed
        else:
            encoder = model.forecaster.embed
        
        if not hasattr(encoder, 'use_hierarchy') or not encoder.use_hierarchy:
            return
        
        if not hasattr(encoder, 'log_scales'):
            return
        
        # Extract scales
        component_names = ['Trend', 'Weekly', 'Daily', 'Residual']
        scales = {}
        
        for i, name in enumerate(component_names):
            scale = torch.exp(encoder.log_scales[i]).item()
            scales[name] = scale
            self.writer.add_scalar(f'Hierarchy/Scale_{name}', scale, epoch)
        
        # Log as bar chart
        self.writer.add_scalars('Hierarchy/All_Scales', scales, epoch)
        
        # Compute and log weights (for Euclidean)
        if manifold_type == 'euclidean':
            weights = {name: 1.0 / scale for name, scale in scales.items()}
            total_weight = sum(weights.values())
            normalized_weights = {name: w / total_weight for name, w in weights.items()}
            
            for name, weight in normalized_weights.items():
                self.writer.add_scalar(f'Hierarchy/Weight_{name}', weight, epoch)
    
    
    # ============================================
    # Predictions vs Ground Truth
    # ============================================
    
    def log_predictions(self, epoch, predictions, targets, num_samples=3, 
                       feature_names=None):
        """
        Log prediction vs ground truth plots.
        
        Args:
            epoch: int
            predictions: tensor [B, pred_len, n_features]
            targets: tensor [B, pred_len, n_features]
            num_samples: int - number of samples to plot
            feature_names: list - names of features
        """
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        
        B, pred_len, n_features = predictions.shape
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # Plot a few samples
        num_samples = min(num_samples, B)
        
        for sample_idx in range(num_samples):
            fig, axes = plt.subplots(n_features, 1, figsize=(12, 3 * n_features))
            if n_features == 1:
                axes = [axes]
            
            for feat_idx, ax in enumerate(axes):
                # Ground truth
                ax.plot(targets[sample_idx, :, feat_idx], 
                       label='Ground Truth', linewidth=2, alpha=0.8, color='blue')
                
                # Prediction
                ax.plot(predictions[sample_idx, :, feat_idx], 
                       label='Prediction', linewidth=2, alpha=0.8, color='red', linestyle='--')
                
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Value')
                ax.set_title(f'{feature_names[feat_idx]} - Sample {sample_idx}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.writer.add_figure(f'Predictions/Sample_{sample_idx}', fig, epoch)
            plt.close(fig)
    
    def log_prediction_errors(self, epoch, predictions, targets):
        """
        Log prediction error distributions and statistics.
        
        Args:
            epoch: int
            predictions: tensor [B, pred_len, n_features]
            targets: tensor [B, pred_len, n_features]
        """
        errors = (predictions - targets).detach().cpu()
        
        # Error statistics
        self.writer.add_scalar('Errors/Mean_Absolute_Error', errors.abs().mean().item(), epoch)
        self.writer.add_scalar('Errors/RMSE', (errors ** 2).mean().sqrt().item(), epoch)
        
        # Error histogram
        self.writer.add_histogram('Errors/Distribution', errors.flatten(), epoch)
        
        # Error over time (averaged over batch and features)
        error_over_time = errors.abs().mean(dim=[0, 2])  # [pred_len]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(error_over_time.numpy(), linewidth=2)
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Prediction Error vs Forecast Horizon')
        ax.grid(True, alpha=0.3)
        plt.close(fig)
    
    # ============================================
    # Hyperbolic Geometry Visualization
    # ============================================
    
    def log_hyperbolic_embeddings_2d(self, epoch, embeddings, labels, 
                                     manifold_type='poincare'):
        """
        Visualize hyperbolic embeddings in 2D (for Poincaré ball).
        
        Args:
            epoch: int
            embeddings: tensor [N, embed_dim] - must be 2D for visualization
            labels: list - labels for each point
            manifold_type: str - 'poincare' or 'lorentz'
        """
        if embeddings.shape[-1] != 2:
            # Use PCA to project to 2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings.detach().cpu().numpy())
        else:
            embeddings_2d = embeddings.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if manifold_type == 'poincare':
            # Draw Poincaré disk boundary
            circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
        
        # Scatter plot
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                           c=range(len(embeddings_2d)), cmap='viridis', 
                           s=50, alpha=0.7)
        
        # Add labels if provided
        if labels is not None and len(labels) == len(embeddings_2d):
            for i, label in enumerate(labels[:min(20, len(labels))]):  # Limit labels
                ax.annotate(str(label), (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                          fontsize=8, alpha=0.7)
        
        ax.set_aspect('equal')
        ax.set_title(f'{manifold_type.capitalize()} Embeddings (2D Projection)')
        ax.grid(True, alpha=0.3)
        
        self.writer.add_figure(f'Geometry/{manifold_type}_embeddings_2d', fig, epoch)
        plt.close(fig)
    
    def log_radial_distances(self, epoch, embeddings, component_names, manifold):
        """
        Log radial distances from origin (hierarchy visualization).
        
        Args:
            epoch: int
            embeddings: dict - {'trend': tensor, 'weekly': tensor, ...}
            component_names: list - component names
            manifold: geoopt manifold
        """
        distances = {}
        
        for name, emb in embeddings.items():
            if emb is None:
                continue
            
            # Compute distance from origin
            if hasattr(manifold, 'dist0'):
                dist = manifold.dist0(emb).mean().item()
            else:
                # Euclidean distance
                dist = torch.norm(emb, dim=-1).mean().item()
            
            distances[name] = dist
            self.writer.add_scalar(f'Geometry/RadialDistance_{name}', dist, epoch)
        
        # Bar chart of distances
        if distances:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(distances.keys(), distances.values())
            ax.set_ylabel('Mean Distance from Origin')
            ax.set_title('Component Hierarchy (Radial Distances)')
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.writer.add_figure('Geometry/Hierarchy_Distances', fig, epoch)
            plt.close(fig)
    
    # ============================================
    # Model Parameters
    # ============================================
    
    def log_model_parameters(self, model):
        """
        Log model parameter statistics (gradients, weights).
        
        Args:
            model: nn.Module
        """
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
    
    def log_learning_rate(self, epoch, optimizer):
        """Log current learning rate"""
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'LearningRate/group_{i}', 
                                  param_group['lr'], epoch)
    
    # ============================================
    # Utility
    # ============================================
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()
        print(f"TensorBoard logger closed")
