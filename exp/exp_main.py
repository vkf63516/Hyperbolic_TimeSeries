"""
Example exp_main.py with TimeBaseMSTL integration for HyperbolicMambaForecasting

This file shows how to integrate TimeBaseMSTL decomposition into the training pipeline.
Key changes from the original exp_main.py:
1. Import TimeBaseMSTL
2. Initialize and fit TimeBaseMSTL on training data
3. Pass decomposed components to the model instead of non-decomposed data
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
from spec import EarlyStopping, prepare_timebase_data_with_mstl
from Decomposition.TimeBase_Series_Trend_Decomposition import TimeBaseMSTL
from Decomposition.tensor_utils import build_decomposition_tensors
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

        
        # Initialize decomposition cache
        if self.use_decomposition:
            self.timebase_mstl = None  # Will be initialized after loading data
            self.decomposition_cache = {}
            self.mstl_period = None
            
    def _build_model(self):
        model_dict = {
            "HyperbolicMambaForecasting": HyperbolicMambaForecasting
        }
        model = model_dict[self.args.model].Model(self.args).float()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _initialize_timebase_mstl(self, train_data, vali_data, test_data):
        """
        Initialize and fit TimeBaseMSTL on training data.
        This should be called once after loading training data.
        """
        print("=" * 70)
        print("Initializing TimeBaseMSTL decomposition...")
        print("=" * 70)
        # train_data["date"] = pd.to_datetime(train_data["date"])
        # vali_data["date"] = pd.to_datetime(vali_data["date"])
        # test_data["date"] = pd.to_datetime(test_data["date"])
        # Convert datasets to DataFrames
        train_df = self._dataset_to_dataframe(train_data)
        val_df = self._dataset_to_dataframe(vali_data)
        test_df = self._dataset_to_dataframe(test_data)

        print("\nDataset Shapes:")
        print(f"  Train: {train_df.shape}")
        print(f"  Val:   {val_df.shape}")
        print(f"  Test:  {test_df.shape}")
        print(f"\n[1/4] Fitting TimeBaseMSTL...")
        
        # Initialize TimeBaseMSTL
        self.timebase_mstl = TimeBaseMSTL(
            n_basis_components=self.n_basis_components,
            orthogonal_lr=self.orthogonal_lr,
            orthogonal_iters=self.orthogonal_iters
        )
        
        # Fit on training data
        self.timebase_mstl.fit(train_df)
        # Auto-detect periods
        self.mstl_period = self.timebase_mstl.steps_per_period[0]
        self.model.seg_len = self.mstl_period
        print(f"Auto-detected MSTL periods: {self.mstl_period}")
        print(f"\n[2/4] Transforming datasets...")

        train_components = self.timebase_mstl.transform(train_df)
        print(train_components)
        print(type(train_components))
        val_components = self.timebase_mstl.transform(val_df)
        test_components = self.timebase_mstl.transform(test_df)

        print(f"\n[3/4] Building decomposition tensors...")
        train_tensors_dict = {}
        val_tensors_dict = {}
        test_tensors_dict = {}
        for feature_idx in train_components.keys():
            train_tensors_dict[feature_idx] = build_decomposition_tensors(train_components[feature_idx])
            val_tensors_dict[feature_idx] = build_decomposition_tensors(val_components[feature_idx])
            test_tensors_dict[feature_idx] = build_decomposition_tensors(test_components[feature_idx])
        print(f"\n[4/4] Creating segments with MSTL period alignment...")
        train_seg, val_seg, test_seg = prepare_timebase_data_with_mstl(
            train_dict=train_tensors_dict,
            val_dict=val_tensors_dict, 
            test_dict=test_tensors_dict,
            mstl_period=self.mstl_period,
            input_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            stride="overlap",
            device=self.device
        )
        
        self.decomposition_cache = {
            'train': train_seg,
            'val': val_seg,
            'test': test_seg
        }
        print(f"✓ Created segments")
        print("=" * 70)
        print("TIMEBASEMSTL INITIALIZATION COMPLETE")
        print("=" * 70)

    def _dataset_to_dataframe(self, dataset):
        """Convert dataset to pandas DataFrame with proper datetime index."""
    
        if hasattr(dataset, 'df') and isinstance(dataset.df, pd.DataFrame):
            return dataset.df
        data_array = dataset.data_x
        df = pd.DataFrame(data_array)
        df.index = pd.to_datetime(dataset.dates)        
        print(df.index)
        #csv_path = os.path.join(self.args.root_path, self.args.data_path)
        #df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df = df.select_dtypes(include=[np.number])
        
        return df

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
        """Validation function supporting decomposed data."""
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            if self.use_decomposition and self.decomposition_cache:
                # Use cached decomposed data
                cache = self.decomposition_cache["val"]
                feature_key = list(cache.keys())[0]
                
                X_dict = cache[feature_key]['X']
                Y_dict = cache[feature_key]['Y']
                
                num_samples = X_dict["trend"].shape[0]
                batch_size = self.args.batch_size
                
                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    
                    # Get batch
                    trend_x = X_dict["trend"][i:end_idx].float().to(self.device)
                    weekly_x = X_dict["seasonal_weekly"][i:end_idx].float().to(self.device)
                    daily_x = X_dict["seasonal_daily"][i:end_idx].float().to(self.device)
                    resid_x = X_dict['residual'][i:end_idx].float().to(self.device)
                    
                    trend_y = Y_dict["trend"][i:end_idx].float().to(self.device)
                    weekly_y = Y_dict["seasonal_weekly"][i:end_idx].float().to(self.device)
                    daily_y = Y_dict["seasonal_daily"][i:end_idx].float().to(self.device)
                    resid_y = Y_dict["residual"][i:end_idx].float().to(self.device)
                    
                    # Embed inputs
                    embed_input = self.model.embedding(trend_x, weekly_x, daily_x, resid_x)
                    z0 = embed_input["combined_h"]
                    
                    # Forecast
                    outputs, _ = self.model.forecaster.forecast(
                        pred_len=self.args.pred_len,
                        z0=z0,
                        teacher_forcing=False,
                        K=getattr(self.args, 'gradient_truncation_K', 6)
                    )
                    
                    # Target
                    target = (trend_y + weekly_y + daily_y + resid_y)
                    B, num_seg, seg_len, C = target.shape
                    target = target.reshape(B, num_seg * seg_len)
                    if C == 1:
                        target = target.squeeze(-1)
                    
                    pred = outputs.detach()
                    true = target.detach()
                    
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())
            
            else:
                # Standard non-decomposed validation
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    
                    outputs = self.model(batch_x)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    pred = outputs.detach()
                    true = batch_y.detach()
                    
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # Initialize TimeBaseMSTL if decomposition is enabled
        if self.use_decomposition:
            self._initialize_timebase_mstl(train_data, vali_data, test_data)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_geooptim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Calculate train steps based on decomposition
        if self.use_decomposition and self.decomposition_cache:
            cache = self.decomposition_cache['train']
            feature_key = list(cache.keys())[0]
            num_samples = cache[feature_key]['X']['trend'].shape[0]
            train_steps = (num_samples + self.args.batch_size - 1) // self.args.batch_size
        else:
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
            
            if self.use_decomposition and self.decomposition_cache:
                # Train with decomposed data
                cache = self.decomposition_cache['train']
                feature_key = list(cache.keys())[0]
                
                X_dict = cache[feature_key]['X']
                Y_dict = cache[feature_key]['Y']
                
                num_samples = X_dict['trend'].shape[0]
                batch_size = self.args.batch_size
                
                # Shuffle indices
                indices = torch.randperm(num_samples)
                
                for i in range(0, num_samples, batch_size):
                    iter_count += 1
                    model_geooptim.zero_grad()
                    
                    # Get batch indices
                    batch_indices = indices[i:min(i + batch_size, num_samples)]
                    
                    # Input components: [B, num_input_seg, seg_len, 1]
                    trend_x = X_dict['trend'][batch_indices].float().to(self.device)
                    weekly_x = X_dict['seasonal_weekly'][batch_indices].float().to(self.device)
                    daily_x = X_dict['seasonal_daily'][batch_indices].float().to(self.device)
                    resid_x = X_dict['residual'][batch_indices].float().to(self.device)
                    
                    # Target components: [B, num_pred_seg, seg_len, 1]
                    trend_y = Y_dict['trend'][batch_indices].float().to(self.device)
                    weekly_y = Y_dict['seasonal_weekly'][batch_indices].float().to(self.device)
                    daily_y = Y_dict['seasonal_daily'][batch_indices].float().to(self.device)
                    resid_y = Y_dict['residual'][batch_indices].float().to(self.device)
                    
                    # Encode inputs
                    encoded_input = self.model.embedding(trend_x, weekly_x, daily_x, resid_x)
                    z0 = encoded_input["combined_h"]
                    
                    # Encode targets for teacher forcing
                    encoded_target = self.model.embedding(trend_y, weekly_y, daily_y, resid_y)
                    z_true_seq = encoded_target["combined_h"]
                    
                    # Ground truth
                    target = (trend_y + weekly_y + daily_y + resid_y)
                    B, num_seg, seg_len, C = target.shape
                    target = target.reshape(B, num_seg * seg_len)
                    if C == 1:
                        target = target.squeeze(-1)
                    
                    # Forward
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, z_pred = self.model.forecaster.forecast(
                                pred_len=self.args.pred_len,
                                z0=z0,
                                teacher_forcing=True,
                                z_true_seq=z_true_seq,
                                K=getattr(self.args, 'gradient_truncation_K', 6)
                            )
                            loss = criterion(outputs, target)
                            train_loss.append(loss.item())
                    else:
                        outputs, z_pred = self.model.forecaster.forecast(
                            pred_len=self.args.pred_len,
                            z0=z0,
                            teacher_forcing=True,
                            z_true_seq=z_true_seq,
                            K=getattr(self.args, 'gradient_truncation_K', 6)
                        )
                        loss = criterion(outputs, target)
                        train_loss.append(loss.item())
                    print(outputs)
                    print(target)
                    # Backward
                    if (iter_count) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            iter_count, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - iter_count)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        # Gradient clipping for stability
                        scaler.unscale_(model_geooptim)
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                        scaler.step(model_geooptim)
                        scaler.update()
                    else:
                        loss.backward()
                        # Gradient clipping for stability
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                        model_geooptim.step()
                    
                    current_memory = torch.cuda.max_memory_allocated(device=self.device) / 1024 ** 2
                    max_memory = max(max_memory, current_memory)
                    
                    if self.args.lradj == 'TST':
                        scheduler.step()
            
            else:
                # Standard training loop
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count += 1
                    model_geooptim.zero_grad()
                    
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    
                    outputs = self.model(batch_x)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    loss = criterion(outputs, batch_y)
                    print(f"Predicted values {outputs}")
                    print(f"True Values: {batch_y}")
                    train_loss.append(loss.item())
                    
                    if (i + 1) % 100 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()))
                    
                    loss.backward()
                    model_geooptim.step()
            
            print(f"Max Memory (MB): {max_memory}")
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_geooptim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            
            # Clear CUDA cache to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            if self.use_decomposition and self.decomposition_cache:
                cache = self.decomposition_cache['test']
                feature_key = list(cache.keys())[0]
                
                X_dict = cache[feature_key]['X']
                Y_dict = cache[feature_key]['Y']
                
                num_samples = X_dict['trend'].shape[0]
                batch_size = self.args.batch_size
                
                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    
                    trend_x = X_dict['trend'][i:end_idx].float().to(self.device)
                    weekly_x = X_dict['seasonal_weekly'][i:end_idx].float().to(self.device)
                    daily_x = X_dict['seasonal_daily'][i:end_idx].float().to(self.device)
                    resid_x = X_dict['residual'][i:end_idx].float().to(self.device)
                    
                    encoded_input = self.model.embedding(trend_x, weekly_x, daily_x, resid_x)
                    z0 = encoded_input["combined_h"]
                    
                    outputs, _ = self.model.forecaster.forecast(
                        pred_len=self.args.pred_len,
                        z0=z0,
                        teacher_forcing=False,
                        K=getattr(self.args, 'gradient_truncation_K', 6)
                    )
                    
                    trend_y = Y_dict['trend'][i:end_idx].float().to(self.device)
                    weekly_y = Y_dict['seasonal_weekly'][i:end_idx].float().to(self.device)
                    daily_y = Y_dict['seasonal_daily'][i:end_idx].float().to(self.device)
                    resid_y = Y_dict['residual'][i:end_idx].float().to(self.device)
                    
                    target = (trend_y + weekly_y + daily_y + resid_y)
                    B, num_seg, seg_len, C = target.shape
                    target = target.reshape(B, num_seg * seg_len)
                    if C == 1:
                        target = target.squeeze(-1)
                    
                    outputs = outputs.detach().cpu().numpy()
                    target = target.detach().cpu().numpy()
                    
                    preds.append(outputs)
                    trues.append(target)
            
            else:
                # Standard testing
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    
                    outputs = self.model(batch_x)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()
                    
                    preds.append(outputs)
                    trues.append(batch_y)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        print('test shape:', preds.shape, trues.shape)

        # result save
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
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return


    def predict(self, setting, load=False):
        # pred_data, pred_loader = self._get_data(flag='pred')

        # if load:
        #     path = os.path.join(self.args.checkpoints, setting)
        #     best_model_path = path + '/' + 'checkpoint.pth'
        #     self.model.load_state_dict(torch.load(best_model_path))

        # preds = []

        # self.model.eval()
        # with torch.no_grad():
        #     for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
        #         batch_x = batch_x.float().to(self.device)
        #         batch_y = batch_y.float()
        #         batch_x_mark = batch_x_mark.float().to(self.device)
        #         batch_y_mark = batch_y_mark.float().to(self.device)

        #         # decoder input
        #         dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
        #             batch_y.device)
        #         dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        #         # encoder - decoder
        #         if self.args.use_amp:
        #             with torch.cuda.amp.autocast():
        #                 if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
        #                     outputs = self.model(batch_x)
        #                 else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        #         else:
        #             if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
        #                 outputs = self.model(batch_x)
        #             else:
    #                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        #         pred = outputs.detach().cpu().numpy()  # .squeeze()
        #         preds.append(pred)

        # preds = np.array(preds)
        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)
        return
