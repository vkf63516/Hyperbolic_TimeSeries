"""
Unified exp_main.py supporting both segment-level and point-level hyperbolic encodedings

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
        self.manifold_type = args.manifold_type 
        self.hyperbolic_weight = args.hyperbolic_weight
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
                'encode_dim': args.encode_dim,
                'hidden_dim': args.hidden_dim,
                'manifold_type': args.manifold_type,
                'use_segments': args.use_segments,
                'use_learnable_decomposition': args.use_learnable_decomposition,
                'use_no_decomposition': args.use_no_decomposition,
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
            "HyperbolicForecasting": HyperbolicForecasting,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        #print(f"Mode: {'Segment-level' if self.use_segments else 'Point-level'} hyperbolic encodedings")
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, hyp_outputs = self.model(batch_x)
                else:
                    outputs, hyp_outputs = self.model(batch_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_geooptim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                    
                        outputs, hyp_loss = self.model(batch_x)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    
                    outputs, hyp_loss = self.model(batch_x)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_geooptim)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                    scaler.update()
                else:
                    back_loss = loss + self.hyperbolic_weight * hyp_loss
                    back_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    model_geooptim.step()
            # Memory tracking
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
            test_loss = self.vali(test_data, test_loader, criterion)
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

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, avg_train_loss, vali_loss, test_loss))

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, hyp_outputs = self.model(batch_x)
                else:
                    
                    outputs, hyp_outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)

        # ========================================
        # Concatenate All Batches
        # ========================================
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    
        print('test shape:', preds.shape, trues.shape)

    # ========================================
    # Calculate Metrics
    # ========================================
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}'.format(mse, mae, rmse))
        
        f = open(self.args.result_file, 'a')
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
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'hyper.npy', hypers)

        return

    def predict(self, setting, load=False):
        return

