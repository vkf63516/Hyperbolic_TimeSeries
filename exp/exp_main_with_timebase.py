"""
Example exp_main.py with TimeBaseMSTL integration for HyperbolicMambaForecasting

This file shows how to integrate TimeBaseMSTL decomposition into the training pipeline
using the prepare_timebase_data_with_mstl function from spec.py.

Key changes from the original exp_main.py:
1. Import TimeBaseMSTL and prepare_timebase_data_with_mstl
2. Initialize and fit TimeBaseMSTL on training data
3. Use prepare_timebase_data_with_mstl for proper normalization and segmentation
4. Pass decomposed components to the model instead of raw data

Note: This example demonstrates the integration pattern. For production use,
consider implementing HyperbolicTimeSeriesDataset (currently commented in data_loader.py)
which provides a cleaner data pipeline.
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TimeBase
from models import HyperbolicMambaForecasting
from utils.tools import adjust_learning_rate, visual, test_params_flop
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
        self.use_orthogonal = args.use_orthogonal
        self.use_decomposition = getattr(args, 'use_decomposition', False)
        
        # Initialize TimeBaseMSTL if using decomposition
        if self.use_decomposition:
            self.timebase_mstl = None  # Will be initialized after loading data
            self.decomposed_data = {}  # Cache for decomposed data
            self.mstl_period = None
            
    def _build_model(self):
        model_dict = {
            "HyperbolicMambaForecasting": HyperbolicMambaForecasting
        }
        model = model_dict[self.args.model].Model(self.args).float()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params}")

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _initialize_timebase_mstl_with_spec(self, train_data, val_data, test_data):
        """
        Initialize and fit TimeBaseMSTL on training data, then use
        prepare_timebase_data_with_mstl from spec.py for proper preprocessing.
        
        This method:
        1. Fits TimeBaseMSTL on training data
        2. Transforms train/val/test data to get decomposed components
        3. Uses prepare_timebase_data_with_mstl for normalization and segmentation
        4. Caches the preprocessed data for efficient batch access
        """
        print("Initializing TimeBaseMSTL decomposition with spec.py integration...")
        
        # Get raw data as DataFrames
        # Note: Adapt this based on your dataset structure
        train_df = self._dataset_to_dataframe(train_data)
        val_df = self._dataset_to_dataframe(val_data)
        test_df = self._dataset_to_dataframe(test_data)
        
        # Initialize TimeBaseMSTL
        self.timebase_mstl = TimeBaseMSTL(
            n_basis_components=self.args.num_basis,
            orthogonal_lr=self.args.orthogonal_lr,
            orthogonal_iters=self.args.orthogonal_iters
        )
        
        # Fit on training data
        print("Fitting TimeBaseMSTL on training data...")
        self.timebase_mstl.fit(train_df)
        
        # Auto-detect periods
        self.mstl_period = self.timebase_mstl.steps_per_period[0]  # Daily period
        print(f"Auto-detected MSTL period: {self.mstl_period}")
        
        # Transform all datasets to get decomposed components
        print("Transforming datasets with TimeBaseMSTL...")
        train_components = self.timebase_mstl.transform(train_df)
        val_components = self.timebase_mstl.transform(val_df)
        test_components = self.timebase_mstl.transform(test_df)
        
        # Convert decomposed components to tensor dictionaries
        # Format: { feature_name: { component_name: tensor } }
        train_dict = build_decomposition_tensors(train_components, device=self.device)
        val_dict = build_decomposition_tensors(val_components, device=self.device)
        test_dict = build_decomposition_tensors(test_components, device=self.device)
        
        # Use prepare_timebase_data_with_mstl from spec.py for proper preprocessing
        # This handles normalization and segmentation with MSTL period
        print("Preparing segmented data with prepare_timebase_data_with_mstl...")
        train_seg, val_seg, test_seg, scaler, mstl_period = prepare_timebase_data_with_mstl(
            train_dict=train_dict,
            val_dict=val_dict,
            test_dict=test_dict,
            mstl_period=self.mstl_period,
            input_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            stride='overlap',  # or 'period' for non-overlapping samples
            device=self.device
        )
        
        # Cache the preprocessed data
        self.decomposed_data = {
            'train': train_seg,
            'val': val_seg,
            'test': test_seg,
            'scaler': scaler
        }
        
        print(f"✓ TimeBaseMSTL initialization complete")
        print(f"  MSTL period: {mstl_period}")
        print(f"  Scaler type: {type(scaler)}")
    
    def _dataset_to_dataframe(self, dataset):
        """
        Convert dataset to pandas DataFrame for TimeBaseMSTL.
        Adapt this based on your dataset structure.
        """
        if hasattr(dataset, 'data_x'):
            data_array = dataset.data_x
            # Create DataFrame with proper column names
            df = pd.DataFrame(data_array, columns=[f'feat_{i}' for i in range(data_array.shape[1])])
            
            # Add datetime index if available
            if hasattr(dataset, 'data_stamp'):
                # Try to create datetime index from timestamp data
                # This is dataset-specific and may need adjustment
                pass
            
            return df
        else:
            raise ValueError("Cannot convert dataset to DataFrame: unsupported structure")
    
    def _get_decomposed_batch_from_cache(self, batch_indices, flag='train'):
        """
        Retrieve decomposed batch from cached preprocessed data.
        
        Args:
            batch_indices: Indices of samples in the batch
            flag: 'train', 'val', or 'test'
            
        Returns:
            decomposed_components: dict with 'trend', 'daily', 'weekly', 'resid'
        """
        if flag not in self.decomposed_data:
            return None
        
        seg_data = self.decomposed_data[flag]
        
        # Extract components for the batch
        # Note: This depends on the structure returned by prepare_timebase_data_with_mstl
        # Adjust based on actual data structure
        
        # Placeholder - actual implementation depends on seg_data structure
        # The seg_data from prepare_timebase_data_with_mstl has format:
        # { feature: { "X": { comp: tensor }, "Y": { comp: tensor } } }
        
        return None  # Implement based on actual data structure
        """
        Initialize and fit TimeBaseMSTL on training data.
        This should be called once after loading training data.
        """
        print("Initializing TimeBaseMSTL decomposition...")
        
        # Get raw data as DataFrame
        # Note: You'll need to adapt this based on your dataset structure
        # This assumes train_data has a method to get the raw dataframe
        if hasattr(train_data, 'data_x'):
            # Convert to DataFrame with datetime index if available
            data_array = train_data.data_x
            
            # Create a simple DataFrame (you may need to add proper datetime index)
            df = pd.DataFrame(data_array)
            
            # Initialize TimeBaseMSTL
            self.timebase_mstl = TimeBaseMSTL(
                n_basis_components=self.args.num_basis,
                orthogonal_lr=self.args.orthogonal_lr,
                orthogonal_iters=self.args.orthogonal_iters
            )
            
            # Fit on training data
            self.timebase_mstl.fit(df)
            
            # Auto-detect periods
            self.mstl_periods = self.timebase_mstl.steps_per_period
            print(f"Auto-detected MSTL periods: {self.mstl_periods}")
            
        else:
            raise ValueError("Cannot initialize TimeBaseMSTL: dataset structure not recognized")
    
    def _decompose_batch(self, batch_x):
        """
        Decompose a batch of time series using fitted TimeBaseMSTL.
        
        Args:
            batch_x: [B, T, C] raw time series batch
            
        Returns:
            decomposed_components: dict with 'trend', 'daily', 'weekly', 'resid'
                                   each of shape [B, T, 1]
        """
        B, T, C = batch_x.shape
        device = batch_x.device
        
        # Initialize component tensors
        trend_batch = torch.zeros(B, T, 1, device=device)
        daily_batch = torch.zeros(B, T, 1, device=device)
        weekly_batch = torch.zeros(B, T, 1, device=device)
        resid_batch = torch.zeros(B, T, 1, device=device)
        
        # Decompose each sample in the batch
        # Note: This is a simplified version. In production, you'd want to:
        # 1. Use cached decomposition for efficiency
        # 2. Or implement batch decomposition in TimeBaseMSTL
        for b in range(B):
            # Get first channel
            series = batch_x[b, :, 0].detach().cpu().numpy()
            
            # Create DataFrame for this sample
            df_sample = pd.DataFrame({'value': series})
            
            # Transform using fitted TimeBaseMSTL
            decomposition = self.timebase_mstl.transform(df_sample)
            
            # Extract components (assuming single column 'value')
            if 'value' in decomposition:
                comp = decomposition['value']
                trend_batch[b, :, 0] = torch.from_numpy(comp['trend']).float()
                daily_batch[b, :, 0] = torch.from_numpy(comp['seasonal_daily']).float()
                weekly_batch[b, :, 0] = torch.from_numpy(comp['seasonal_weekly']).float()
                resid_batch[b, :, 0] = torch.from_numpy(comp['residual']).float()
        
        return {
            'trend': trend_batch.to(device),
            'daily': daily_batch.to(device),
            'weekly': weekly_batch.to(device),
            'resid': resid_batch.to(device)
        }

    def _select_optimizer(self):
        model_geooptim = geooptim.RiemannianAdam(params=self.model.parameters(), lr=self.args.learning_rate)
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
                        if any(substr in self.args.model for substr in {"Linear", "TST", "SparseTSF", "HyperbolicMambaForecasting"}):
                            # Use decomposition if enabled
                            if self.use_decomposition and self.timebase_mstl is not None:
                                decomposed_components = self._decompose_batch(batch_x)
                                outputs = self.model(batch_x, decomposed_components=decomposed_components)
                            else:
                                outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {"Linear", "TST", "SparseTSF", "HyperbolicMambaForecasting"}):
                        # Use decomposition if enabled
                        if self.use_decomposition and self.timebase_mstl is not None:
                            decomposed_components = self._decompose_batch(batch_x)
                            if self.use_orthogonal:
                                outputs, orthogonal_loss = self.model(batch_x, decomposed_components=decomposed_components)
                            else:
                                outputs = self.model(batch_x, decomposed_components=decomposed_components)
                        else:
                            if self.use_orthogonal:
                                outputs, orthogonal_loss = self.model(batch_x)
                            else:
                                outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # Initialize TimeBaseMSTL if decomposition is enabled
        if self.use_decomposition:
            self._initialize_timebase_mstl(train_data)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        max_memory = -1
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
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
                        if any(substr in self.args.model for substr in {"Linear", "TST", "SparseTSF", "HyperbolicMambaForecasting"}):
                            # Use decomposition if enabled
                            if self.use_decomposition and self.timebase_mstl is not None:
                                decomposed_components = self._decompose_batch(batch_x)
                                outputs = self.model(batch_x, decomposed_components=decomposed_components)
                            else:
                                outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if any(substr in self.args.model for substr in {"Linear", "TST", "SparseTSF", "HyperbolicMambaForecasting"}):
                        # Use decomposition if enabled
                        if self.use_decomposition and self.timebase_mstl is not None:
                            decomposed_components = self._decompose_batch(batch_x)
                            if self.use_orthogonal:
                                outputs, orthogonal_loss = self.model(batch_x, decomposed_components=decomposed_components)
                            else:
                                outputs = self.model(batch_x, decomposed_components=decomposed_components)
                                orthogonal_loss = 0
                        else:
                            if self.use_orthogonal:
                                outputs, orthogonal_loss = self.model(batch_x)
                            else:
                                outputs = self.model(batch_x)
                                orthogonal_loss = 0
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
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
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    back_loss = loss + self.orthogonal_weight*orthogonal_loss 
                    back_loss.backward()
                    model_optim.step()
                current_memory = torch.cuda.max_memory_allocated(device=self.device) / 1024 ** 2
                max_memory = max(max_memory, current_memory)
                
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            print(f"Max Memory (MB): {max_memory}")
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        print(f"Final Max Memory (MB): {max_memory}")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location="cuda:0"))

        preds = []
        trues = []
        inputx = []
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
                        if any(substr in self.args.model for substr in {"Linear", "TST", "SparseTSF", "HyperbolicMambaForecasting"}):
                            # Use decomposition if enabled
                            if self.use_decomposition and self.timebase_mstl is not None:
                                decomposed_components = self._decompose_batch(batch_x)
                                outputs = self.model(batch_x, decomposed_components=decomposed_components)
                            else:
                                outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if any(substr in self.args.model for substr in {"Linear", "TST", "SparseTSF", "HyperbolicMambaForecasting"}):
                        # Use decomposition if enabled
                        if self.use_decomposition and self.timebase_mstl is not None:
                            decomposed_components = self._decompose_batch(batch_x)
                            if self.use_orthogonal:
                                outputs, orthogonal_loss = self.model(batch_x, decomposed_components=decomposed_components)
                            else:
                                outputs = self.model(batch_x, decomposed_components=decomposed_components)
                        else:
                            if self.use_orthogonal:
                                outputs, orthogonal_loss = self.model(batch_x)
                            else:
                                outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))

        return

    def predict(self, setting, load=False):
        return
