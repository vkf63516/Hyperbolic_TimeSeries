import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class WandbLogger:
    """
    Weights & Biases logger for hyperbolic time series forecasting.
    Replaces TensorBoard functionality with wandb.
    """
    
    def __init__(self, project_name="Hyperbolic_TimeSeries", experiment_name=None, config=None):
        """
        Initialize wandb logger.
        
        Args:
            project_name: str - wandb project name
            experiment_name: str - experiment/run name
            config: dict - hyperparameters and configuration to log
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            reinit=True
        )
        
        print(f"Wandb initialized. Project: {project_name}, Run: {experiment_name}")
        print(f"View results at: {wandb.run.get_url()}")
    
    # ============================================
    # Loss Logging
    # ============================================
    
    def log_losses(self, step, train_loss=None, val_loss=None, test_loss=None, 
                   train_losses_dict=None, prefix=''):
        """
        Log training, validation, and test losses.
        
        Args:
            step: int - current step/epoch
            train_loss: float - training loss
            val_loss: float - validation loss
            test_loss: float - test loss
            train_losses_dict: dict - component losses {'trend': ..., 'weekly': ...}
            prefix: str - prefix for metric names
        """
        log_dict = {"step": step}
        
        if train_loss is not None:
            log_dict[f'{prefix}train_loss'] = train_loss
        
        if val_loss is not None:
            log_dict[f'{prefix}val_loss'] = val_loss
        
        if test_loss is not None:
            log_dict[f'{prefix}test_loss'] = test_loss
        
        # Log component losses
        if train_losses_dict is not None:
            for name, value in train_losses_dict.items():
                log_dict[f'{prefix}loss/{name}'] = value
        
        wandb.log(log_dict)
    
    # ============================================
    # Metrics Logging
    # ============================================
    
    def log_metrics(self, step, metrics_dict, prefix='metrics'):
        """
        Log evaluation metrics (MSE, MAE, etc.)
        
        Args:
            step: int - current step/epoch
            metrics_dict: dict - {'mse': ..., 'mae': ..., 'rmse': ...}
            prefix: str - prefix for metric names
        """
        log_dict = {"step": step}
        
        for name, value in metrics_dict.items():
            log_dict[f'{prefix}/{name}'] = value
            
            # Log on log scale for clarity
            if value > 0:
                log_dict[f'{prefix}/{name}_log'] = np.log10(value + 1e-10)
        
        wandb.log(log_dict)
    
    # ============================================
    # Hierarchy Logging
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
            encoder = model.forecater.embed if hasattr(model, 'forecater') else model.forecaster.embed
        else:
            encoder = model.forecaster.embed
        
        if not hasattr(encoder, 'use_hierarchy') or not encoder.use_hierarchy:
            return
        
        if not hasattr(encoder, 'log_scales'):
            return
        
        # Extract scales
        component_names = ['Trend', 'Weekly', 'Daily', 'Residual']
        scales = {}
        log_dict = {"epoch": epoch}
        
        for i, name in enumerate(component_names):
            scale = torch.exp(encoder.log_scales[i]).item()
            scales[name] = scale
            log_dict[f'hierarchy/scale_{name}'] = scale
        
        # Compute and log weights (for Euclidean)
        if manifold_type == 'euclidean':
            weights = {name: 1.0 / scale for name, scale in scales.items()}
            total_weight = sum(weights.values())
            normalized_weights = {name: w / total_weight for name, w in weights.items()}
            
            for name, weight in normalized_weights.items():
                log_dict[f'hierarchy/weight_{name}'] = weight
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(scales.keys(), scales.values())
        ax.set_xlabel('Component')
        ax.set_ylabel('Scale')
        ax.set_title(f'Hierarchy Scales - Epoch {epoch}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        log_dict['hierarchy/scales_chart'] = wandb.Image(fig)
        plt.close(fig)
        
        wandb.log(log_dict)
    
    # ============================================
    # Predictions Logging
    # ============================================
    
    def log_predictions(self, epoch, predictions, targets, num_samples=5):
        """
        Log sample predictions vs ground truth.
        
        Args:
            epoch: int
            predictions: tensor or numpy array [B, T, D]
            targets: tensor or numpy array [B, T, D]
            num_samples: int - number of samples to visualize
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Select random samples
        num_samples = min(num_samples, predictions.shape[0])
        indices = np.random.choice(predictions.shape[0], num_samples, replace=False)
        
        # Create plots
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for idx, ax in zip(indices, axes):
            # Plot first feature/dimension
            if predictions.ndim == 3:
                pred = predictions[idx, :, 0]
                target = targets[idx, :, 0]
            else:
                pred = predictions[idx, :]
                target = targets[idx, :]
            
            ax.plot(target, label='Ground Truth', linewidth=2)
            ax.plot(pred, label='Prediction', linewidth=2, linestyle='--')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.set_title(f'Sample {idx}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        wandb.log({
            "epoch": epoch,
            "predictions/samples": wandb.Image(fig)
        })
        plt.close(fig)
    
    # ============================================
    # Geometry Logging
    # ============================================
    
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
        log_dict = {"epoch": epoch}
        
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
            log_dict[f'geometry/radial_distance_{name}'] = dist
        
        # Bar chart of distances
        if distances:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(distances.keys(), distances.values())
            ax.set_xlabel('Component')
            ax.set_ylabel('Radial Distance from Origin')
            ax.set_title(f'Radial Distances - Epoch {epoch}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            log_dict['geometry/radial_distances_chart'] = wandb.Image(fig)
            plt.close(fig)
        
        wandb.log(log_dict)
    
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
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.config.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        })
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def log_learning_rate(self, step, optimizer):
        """Log current learning rate"""
        log_dict = {"step": step}
        
        for i, param_group in enumerate(optimizer.param_groups):
            log_dict[f'learning_rate/group_{i}'] = param_group['lr']
        
        wandb.log(log_dict)
    
    # ============================================
    # System Metrics
    # ============================================
    
    def log_system_metrics(self, step, gpu_memory_mb=None, epoch_time=None, 
                          speed_per_iter=None, time_left=None):
        """
        Log system metrics like memory usage and speed.
        
        Args:
            step: int
            gpu_memory_mb: float - GPU memory in MB
            epoch_time: float - time per epoch in seconds
            speed_per_iter: float - seconds per iteration
            time_left: float - estimated time left in seconds
        """
        log_dict = {"step": step}
        
        if gpu_memory_mb is not None:
            log_dict['system/gpu_memory_mb'] = gpu_memory_mb
        
        if epoch_time is not None:
            log_dict['system/epoch_time_seconds'] = epoch_time
        
        if speed_per_iter is not None:
            log_dict['system/speed_seconds_per_iter'] = speed_per_iter
        
        if time_left is not None:
            log_dict['system/estimated_time_left_seconds'] = time_left
        
        wandb.log(log_dict)
    
    # ============================================
    # Utility
    # ============================================
    
    def watch_model(self, model, log_freq=100):
        """
        Watch model gradients and parameters.
        
        Args:
            model: nn.Module
            log_freq: int - logging frequency
        """
        wandb.watch(model, log='all', log_freq=log_freq)
    
    def log_artifact(self, file_path, artifact_type='model', name=None):
        """
        Log file as wandb artifact.
        
        Args:
            file_path: str - path to file
            artifact_type: str - type of artifact
            name: str - artifact name
        """
        if name is None:
            name = Path(file_path).stem
        
        artifact = wandb.Artifact(name, type=artifact_type)
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish the wandb run"""
        wandb.finish()
        print("Wandb run finished")