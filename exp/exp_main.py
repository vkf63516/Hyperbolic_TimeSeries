"""
Unified exp_main.py supporting both segment-level and point-level hyperbolic embeddings

Controlled by args.use_segments flag:
- Point-level hyperbolic space (ParallelLorentzBlock)
"""
from wandb_logger import WandbLogger
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import HyperbolicForecasting
from utils.tools import adjust_learning_rate, visual
from utils.metrics import metric
from geoopt import optim as geooptim
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from spec import EarlyStopping, compute_hierarchical_loss_with_manifold_dist
from Decomposition.Orthogonal_Series_Trend_Decomposition import orthogonalMSTL
import pandas as pd

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        self.wandb_logger = None
        super(Exp_Main, self).__init__(args)
        self.use_decomposition = args.use_decomposition
        self.n_basis_components = args.num_basis
        self.orthogonal_lr = args.orthogonal_lr
        self.orthogonal_iters = args.orthogonal_iters
        self.manifold_type = args.manifold_type 
        # Support for both segment-level and point-level
        self.use_segments = args.use_segments  # Default to point-level
        # Initialize decomposition cache
        if self.use_segments:
            self.mstl_period = args.mstl_period
        
        # Initialize wandb logger
        if args.use_wandb:
            # Create experiment name
            experiment_name = f"{args.model}_{args.data}_{args.data_path}_sl{args.seq_len}_pl{args.pred_len}_{args.manifold_type}"
            
            # Prepare config dict
            config = {
                'model': args.model,
                'data': args.data,
                'data_path': args.data_path,
                'seq_len': args.seq_len,
                'pred_len': args.pred_len,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'train_epochs': args.train_epochs,
                'embed_dim': args.embed_dim,
                'hidden_dim': args.hidden_dim,
                'manifold_type': args.manifold_type,
                'use_segments': args.use_segments,
                'use_decomposition': args.use_decomposition,
                'use_attention_pooling': args.use_attention_pooling,
                'loss': args.loss,
                'patience': args.patience,
            }
            
            self.wandb_logger = WandbLogger(
                project_name="Hyperbolic_TimeSeries",
                experiment_name=experiment_name,
                config=config
            )
            
            print(f"Wandb logging enabled. Experiment: {experiment_name}")
        else:
            self.wandb_logger = None        
            
    def _build_model(self):
        model_dict = {
            "HyperbolicForecasting": HyperbolicForecasting
        }
        model = model_dict[self.args.model].Model(self.args).float()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        #print(f"Mode: {'Segment-level' if self.use_segments else 'Point-level'} hyperbolic embeddings")
        if self.wandb_logger is not None:
            self.wandb_logger.log_model_parameters(model)
            self.wandb_logger.watch_model(model, log_freq=100)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    

    def _select_optimizer(self):
        if self.manifold_type == "Euclidean":
            model_geooptim = optim.AdamW(
                    params=self.model.parameters(),
                    lr=self.args.learning_rate,
                    weight_decay=1e-4)
            return model_geooptim
        model_geooptim = geooptim.RiemannianAdam(
            params=self.model.parameters(), 
            lr=self.args.learning_rate,
            weight_decay=1e-4,
            stabilize=5
        )
        return model_geooptim

    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
    
        with torch.no_grad():
            for i, (X_dict, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # ========================================
                # IDENTICAL to train loop up to forward pass
                # ========================================
                trend_x = X_dict['trend'].float().to(self.device, non_blocking=True)
                coarse_x = X_dict['seasonal_coarse'].float().to(self.device, non_blocking=True)
                fine_x = X_dict['seasonal_fine'].float().to(self.device, non_blocking=True)
                resid_x = X_dict['residual'].float().to(self.device, non_blocking=True)
            
                outputs, hyp_outputs = self.model(
                    trend=trend_x,
                    seasonal_coarse=coarse_x,
                    seasonal_fine=fine_x,
                    residual=resid_x,
                )
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                preds = outputs.detach()
                trues = batch_y.detach()

                # Compute validation loss
                loss = criterion(preds, trues)

                # print("Component Loss: ", loss)
                total_loss.append(loss.item())
                
        avg_total_loss = np.average(total_loss)
        self.model.train()
        return avg_total_loss
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
      
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_geooptim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Calculate train steps
        train_steps = len(train_loader)

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_geooptim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )
        train_losses_dict = {}
        max_memory = -1
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (X_dict, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                global_step = epoch * len(train_loader) + i
                iter_count += 1
                model_geooptim.zero_grad()
            
                # Load components
                trend_x = X_dict['trend'].float().to(self.device, non_blocking=True)
                coarse_x = X_dict['seasonal_coarse'].float().to(self.device, non_blocking=True)
                fine_x = X_dict['seasonal_fine'].float().to(self.device, non_blocking=True)
                resid_x = X_dict['residual'].float().to(self.device, non_blocking=True)
                # Ground truth
                batch_y = batch_y.float().to(self.device, non_blocking=True)
            
                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, hyp_outputs = self.model(
                            trend=trend_x,
                            seasonal_coarse=coarse_x,
                            seasonal_fine=fine_x,
                            residual=resid_x,
                        )
                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                        outputs = outputs[:, -self.args.pred_len:, f_dim:]

                        loss = criterion(outputs, batch_y)

                        train_loss.append(loss.item())

                else:
                    outputs, hyp_outputs = self.model(
                        trend=trend_x,
                        seasonal_coarse=coarse_x,
                        seasonal_fine=fine_x,
                        residual=resid_x,
                    )
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    # print(f"Batch size of output {outputs.shape[0]}")
                    # print(f"Batch size of coarse output {coarse_outputs.shape[0]}")

                    loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                # Log iteration metrics
                if i % 100 == 0 and self.wandb_logger is not None:
                    self.wandb_logger.log_losses(
                        global_step,
                        train_loss=loss.item(),
                        train_losses_dict={
                            'loss': loss.item(),
                    })
                    # Log learning rate
                    self.wandb_logger.log_learning_rate(global_step, model_geooptim)
                
                # Console logging
                if (iter_count) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        iter_count, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - iter_count)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    
                    # Log system metrics
                    if self.wandb_logger is not None:
                        self.wandb_logger.log_system_metrics(
                            global_step,
                            speed_per_iter=speed,
                            time_left=left_time
                        )

                # Backward pass
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    scaler.step(model_geooptim)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_geooptim.step()

            # Memory tracking
            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated(device=self.device) / 1024 ** 2
                max_memory = max(max_memory, current_memory)
            
            # Scheduler step (for OneCycleLR)
            if self.args.lradj == 'TST':
                scheduler.step()
        
            # End of epoch
            print(f"Epoch {epoch + 1} Max Memory (MB): {max_memory}")
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            avg_train_loss = np.average(train_loss)
            train_losses_dict = {k: v / len(train_loader) for k, v in train_losses_dict.items()}
            
            # Validation
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            # Log epoch metrics to wandb
            if self.wandb_logger is not None:
                self.wandb_logger.log_losses(
                    epoch,
                    train_loss=avg_train_loss,
                    val_loss=vali_loss,
                    train_losses_dict=train_losses_dict
                )
                
                # Log hierarchy scales
                manifold_type = self.manifold_type                
                # Log system metrics
                epoch_duration = time.time() - epoch_time
                self.wandb_logger.log_system_metrics(
                    epoch,
                    gpu_memory_mb=max_memory if torch.cuda.is_available() else None,
                    epoch_time=epoch_duration
                )

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, avg_train_loss, vali_loss))

            # Early stopping
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # Learning rate adjustment
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_geooptim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Load best model
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # Log model checkpoint as artifact
        if self.wandb_logger is not None:
            self.wandb_logger.log_artifact(best_model_path, artifact_type='model', name='best_model')
        print(f"Final Max Memory (MB): {max_memory}")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(
                os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), 
                map_location=self.device
            ))


        preds = []
        trues = []
        hypers = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (X_dict, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # ========================================
                # Load Input Components (SAME AS TRAIN)
                # ========================================
                trend_x = X_dict['trend'].float().to(self.device, non_blocking=True)
                coarse_x = X_dict['seasonal_coarse'].float().to(self.device, non_blocking=True)
                fine_x = X_dict['seasonal_fine'].float().to(self.device, non_blocking=True)
                resid_x = X_dict['residual'].float().to(self.device, non_blocking=True)
                # ========================================
                # Load Ground Truth (SAME AS TRAIN)
                # ========================================
                batch_y = batch_y.float().to(self.device, non_blocking=True)
            
                # For point-level, extract only prediction part
            
            
                # ========================================
                # Forward Pass (SAME AS TRAIN, but no AMP)
                # ========================================                
                outputs, hyp_outputs = self.model(
                    trend=trend_x,
                    seasonal_coarse=coarse_x,
                    seasonal_fine=fine_x,
                    residual=resid_x,
                )
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                # outputs = trend_outputs + coarse_outputs + fine_outputs + resid_outputs
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # hyp_outputs = hyp_outputs[:, -self.args.pred_len:, f_dim:]

            # ========================================
            # For Segment-Level: Flatten for Metrics
            # ========================================
                if self.args.use_segments:
                    # outputs: [B, num_pred_segs, seg_len] → [B, pred_len]
                    if outputs.dim() == 3:
                        B, num_pred_segs, seg_len = outputs.shape
                        outputs = outputs.reshape(B, num_pred_segs * seg_len)
                
                    if batch_y.dim() == 3:
                        B, num_pred_segs, seg_len = batch_y.shape
                        batch_y = batch_y.reshape(B, num_pred_segs * seg_len)
            
            # ========================================
            # Convert to NumPy
            # ========================================
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                hyper = hyp_outputs.detach().cpu().numpy()
            
                preds.append(outputs)
                trues.append(batch_y)
                hypers.append(hyper)

        # ========================================
        # Concatenate All Batches
        # ========================================
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        hypers = np.concatenate(hypers, axis=0)
    
        print('test shape:', preds.shape, trues.shape, hypers.shape)

    # ========================================
    # Calculate Metrics
    # ========================================
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        f.write('\n\n')
        f.close()
        if self.wandb_logger is not None:
            test_metrics = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape,
                'MSPE': mspe,
                'RSE': rse,
                'CORR': corr
            }
            self.wandb_logger.log_metrics(0, test_metrics, prefix='metrics')
            
            # Log sample predictions
            self.wandb_logger.log_predictions(0, preds, trues, num_samples=5)
            
            # Finish wandb run
            self.wandb_logger.finish()
            print("Wandb logging complete. View results at: https://wandb.ai")
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'hyper.npy', hypers)

        return

    def predict(self, setting, load=False):
        return

