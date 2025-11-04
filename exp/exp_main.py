"""
Unified exp_main.py supporting both segment-level and point-level hyperbolic embeddings

Controlled by args.use_segments flag:
- use_segments=True: Segment-level hyperbolic space (SegmentParallelLorentzBlock)
- use_segments=False: Point-level hyperbolic space (ParallelLorentzBlock)
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TimeBase
from models import HyperbolicMambaForecasting
from utils.tools import adjust_learning_rate, visual
from utils.metrics import metric
from geoopt import optim as geooptim
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from spec import EarlyStopping, compute_hierarchical_loss_with_manifold_dist
from Decomposition.TimeBase_Series_Trend_Decomposition import TimeBaseMSTL
from Decomposition.tensor_utils import build_decomposition_tensors
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.use_decomposition = args.use_decomposition
        self.n_basis_components = args.num_basis
        self.orthogonal_lr = args.orthogonal_lr
        self.orthogonal_iters = args.orthogonal_iters
        
        # Support for both segment-level and point-level
        self.use_segments = args.use_segments  # Default to point-level
        # Initialize decomposition cache
        if self.use_segments:
            self.mstl_period = args.mstl_period
        if args.use_tensorboard:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            log_dir = os.path.join(
                './runs', 
                f"{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_{timestamp}"
            )
            self.writer = SummaryWriter(log_dir)
            
            # Log hyperparameters as text
            hparams_text = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
            self.writer.add_text('Hyperparameters', hparams_text, 0)
            
            print(f"TensorBoard logs will be saved to: {log_dir}")
        else:
            self.writer = None 
        
            
    def _build_model(self):
        model_dict = {
            "HyperbolicMambaForecasting": HyperbolicMambaForecasting
        }
        model = model_dict[self.args.model].Model(self.args).float()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        #print(f"Mode: {'Segment-level' if self.use_segments else 'Point-level'} hyperbolic embeddings")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    

    def _select_optimizer(self):
        model_geooptim = geooptim.RiemannianAdam(
            params=self.model.parameters(), 
            lr=self.args.learning_rate
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
            for i, (X_dict, Y_dict, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # ========================================
                # IDENTICAL to train loop up to forward pass
                # ========================================
                trend_x = X_dict['trend'].float().to(self.device)
                weekly_x = X_dict['seasonal_weekly'].float().to(self.device)
                daily_x = X_dict['seasonal_daily'].float().to(self.device)
                resid_x = X_dict['residual'].float().to(self.device)
            
                trend_y = Y_dict['trend'].float().to(self.device)
                weekly_y = Y_dict['seasonal_weekly'].float().to(self.device)
                daily_y = Y_dict['seasonal_daily'].float().to(self.device)
                resid_y = Y_dict['residual'].float().to(self.device)
            
                target = (trend_y + weekly_y + daily_y + resid_y)
            
                if not self.args.use_segments:
                    target = target[:, -self.args.pred_len:, :]
            
                if target.dim() == 3 and target.shape[-1] == 1:
                    target = target.squeeze(-1)
            
                outputs = self.model(
                    trend=trend_x,
                    seasonal_weekly=weekly_x,
                    seasonal_daily=daily_x,
                    residual=resid_x,
                    teacher_forcing=False                
                )
            
                # Compute validation loss
                loss = criterion(outputs, target)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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
        
        max_memory = -1
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (X_dict, Y_dict, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_geooptim.zero_grad()
            
                # Load components
                trend_x = X_dict['trend'].float().to(self.device)
                weekly_x = X_dict['seasonal_weekly'].float().to(self.device)
                daily_x = X_dict['seasonal_daily'].float().to(self.device)
                resid_x = X_dict['residual'].float().to(self.device)
            
                trend_y = Y_dict['trend'].float().to(self.device)
                weekly_y = Y_dict['seasonal_weekly'].float().to(self.device)
                daily_y = Y_dict['seasonal_daily'].float().to(self.device)
                resid_y = Y_dict['residual'].float().to(self.device)

                # Ground truth
                target = (trend_y + weekly_y + daily_y + resid_y)
            
                # For point-level, extract only prediction part
                if not self.args.use_segments:
                    target = target[:, -self.args.pred_len:, :]
            
                # Squeeze if needed
                if target.dim() == 3 and target.shape[-1] == 1:
                    target = target.squeeze(-1)
            
                # Forward
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(  #Now calls forward() automatically
                            trend=trend_x,
                            seasonal_weekly=weekly_x,
                            seasonal_daily=daily_x,
                            residual=resid_x,
                            teacher_forcing=True,
                            target=target
                        )
                        loss = criterion(outputs, target)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(
                        trend=trend_x,
                        seasonal_weekly=weekly_x,
                        seasonal_daily=daily_x,
                        residual=resid_x,
                        teacher_forcing=True,
                        target=target
                    )
                    loss = criterion(outputs, target)
                    train_loss.append(loss.item())
            
                # Logging
                if (iter_count) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        iter_count, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - iter_count)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    
                    if self.writer is not None:
                        self.writer.add_scalar('Speed/seconds_per_iter', speed, global_step)
                        self.writer.add_scalar('Speed/estimated_time_left_seconds', left_time, global_step)


                # Backward
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_geooptim)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    scaler.step(model_geooptim)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    model_geooptim.step()
                if self.writer is not None:
                    global_step = epoch * train_steps + i
                    self.writer.add_scalar('Loss/train_iter', loss.item(), global_step)

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
            
        
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            # TensorBoard: Log epoch metrics
            if self.writer is not None:
                self.writer.add_scalar('Loss/train_loss', train_loss, epoch)
                self.writer.add_scalar('Loss/vali_loss', vali_loss, epoch)
                self.writer.add_scalar('Learning_Rate/lr', model_geooptim.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Time/epoch_seconds', time.time() - epoch_time, epoch)
                
                # Log GPU memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
                    self.writer.add_scalar('Memory/gpu_allocated_mb', memory_allocated, epoch)
                    self.writer.add_scalar('Memory/gpu_reserved_mb', memory_reserved, epoch)
                
                # Log train vs vali loss comparison
                self.writer.add_scalars('Loss/train_vs_vali', {
                    'train': train_loss,
                    'vali': vali_loss
                }, epoch)

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
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (X_dict, Y_dict, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # ========================================
                # Load Input Components (SAME AS TRAIN)
                # ========================================
                trend_x = X_dict['trend'].float().to(self.device)
                weekly_x = X_dict['seasonal_weekly'].float().to(self.device)
                daily_x = X_dict['seasonal_daily'].float().to(self.device)
                resid_x = X_dict['residual'].float().to(self.device)
            
                # ========================================
                # Load Target Components (SAME AS TRAIN)
                # ========================================
                trend_y = Y_dict['trend'].float().to(self.device)
                weekly_y = Y_dict['seasonal_weekly'].float().to(self.device)
                daily_y = Y_dict['seasonal_daily'].float().to(self.device)
                resid_y = Y_dict['residual'].float().to(self.device)
            
                # ========================================
                # Prepare Ground Truth (SAME AS TRAIN)
                # ========================================
                target = (trend_y + weekly_y + daily_y + resid_y)
            
                # For point-level, extract only prediction part
                if not self.args.use_segments:
                    target = target[:, -self.args.pred_len:, :]
            
                # Squeeze if needed
                if target.dim() == 3 and target.shape[-1] == 1:
                    target = target.squeeze(-1)
            
                # ========================================
                # Forward Pass (SAME AS TRAIN, but no AMP)
                # ========================================
                outputs = self.model(
                    trend=trend_x,
                    seasonal_weekly=weekly_x,
                    seasonal_daily=daily_x,
                    residual=resid_x,
                    teacher_forcing=False
                )
            
            # ========================================
            # For Segment-Level: Flatten for Metrics
            # ========================================
                if self.args.use_segments:
                    # outputs: [B, num_pred_segs, seg_len] → [B, pred_len]
                    if outputs.dim() == 3:
                        B, num_pred_segs, seg_len = outputs.shape
                        outputs = outputs.reshape(B, num_pred_segs * seg_len)
                
                # target: [B, num_pred_segs, seg_len] → [B, pred_len]
                    if target.dim() == 3:
                        B, num_pred_segs, seg_len = target.shape
                        target = target.reshape(B, num_pred_segs * seg_len)
            
            # ========================================
            # Convert to NumPy
            # ========================================
                outputs = outputs.detach().cpu().numpy()
                target = target.detach().cpu().numpy()
            
                preds.append(outputs)
                trues.append(target)

        # ========================================
        # Concatenate All Batches
        # ========================================
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
    
        print('test shape:', preds.shape, trues.shape)

    # ========================================
    # Calculate Metrics
    # ========================================
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n\n')
        f.close()
        if self.writer is not None:
            self.writer.add_scalar('Test/MAE', mae, 0)
            self.writer.add_scalar('Test/MSE', mse, 0)
            self.writer.add_scalar('Test/RMSE', rmse, 0)
            self.writer.add_scalar('Test/MAPE', mape, 0)
            
            # Log all test metrics together
            self.writer.add_scalars('Test/all_metrics', {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'MAPE': mape
            }, 0)
            
            # Close writer
            self.writer.close()
            print(f"TensorBoard logs saved. View with: tensorboard --logdir=./runs")
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        return

