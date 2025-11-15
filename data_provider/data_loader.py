import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from Decomposition.TimeBase_Series_Trend_Decomposition import TimeBaseMSTL
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 basis=None, use_segments=False, mstl_period=24):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.basis = basis
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 basis=None, use_segments=False, mstl_period=24):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.basis = basis
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_hour_Decomposition(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 basis=None, use_segments=False, mstl_period=24):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        if basis == None:
            self.n_basis_components = 20
            self.orthogonal_lr = 1e-3
            self.orthogonal_iters = 300
        else:
            self.n_basis_components = basis[0]
            self.orthogonal_lr = basis[1]
            self.orthogonal_iters = basis[2]
        self.use_segments = use_segments
        self.mstl_period = mstl_period
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.timebase_mstl = TimeBaseMSTL(
            n_basis_components=self.n_basis_components,
            orthogonal_lr=self.orthogonal_lr,
            orthogonal_iters=self.orthogonal_iters,
            seq_len=self.seq_len
        )
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ========================================
        # TimeBaseMSTL Decomposition
        # ========================================
        train_data_normalized = data[border1s[0]:border2s[0]]
        train_df = pd.DataFrame(train_data_normalized, columns=df_data.columns)
        train_dates = pd.to_datetime(df_raw['date'][border1s[0]:border2s[0]].values)
        train_df.index = train_dates
        
        print(f"[{self.flag}] Fitting TimeBaseMSTL on training data...")
        self.timebase_mstl.fit(train_df)
        
        # Auto-detect MSTL period if using segments
        if self.use_segments:
            detected_period = self.timebase_mstl.detect_periods(train_df)[0]
            if self.mstl_period != detected_period:
                print(f"[{self.flag}] Overriding mstl_period {self.mstl_period} with detected {detected_period}")
                self.mstl_period = detected_period

        # Transform current split using fitted TimeBaseMSTL
        current_data_normalized = data[border1:border2]
        current_df = pd.DataFrame(current_data_normalized, columns=df_data.columns)
        current_dates = pd.to_datetime(df_raw['date'][border1:border2].values)
        current_df.index = current_dates

        print(f"[{self.flag}] Transforming data with TimeBaseMSTL...")
        components_per_column = self.timebase_mstl.transform(current_df)
        
        # Convert to model-ready format: always store as [T, C] point-level
        self.decomposed_components = self._format_components(components_per_column)
        
        # After: self.decomposed_components = self._format_components(components_per_column)
        
        # Cache temporal length for efficient __len__ computation
        self.temporal_length = self.decomposed_components['trend'].shape[0]


        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
            # ADD THIS:
        print(f"\n{'='*60}")
        print(f"[{self.flag}] DECOMPOSITION DIAGNOSTICS FOR ETTh1")
        print(f"{'='*60}")
        print(f"Data shape: {self.data_x.shape}")
        print(f"Temporal length: {self.temporal_length}")
        print(f"\nOriginal data stats:")
        print(f"  Mean: {self.data_x.mean():.6f}, Std: {self.data_x.std():.6f}")
        print(f"  Range: [{self.data_x.min():.6f}, {self.data_x.max():.6f}]")

        print(f"\nDecomposed components stats:")
        for comp_name in ['trend', 'seasonal_coarse', 'seasonal_fine', 'residual']:
            comp_data = self.decomposed_components[comp_name]
            print(f"  {comp_name:20s}: mean={comp_data.mean():8.6f}, std={comp_data.std():8.6f}, range=[{comp_data.min():8.4f}, {comp_data.max():8.4f}]")

        # Reconstruction check
        recomposed = (self.decomposed_components['trend'] + 
                    self.decomposed_components['seasonal_coarse'] + 
                    self.decomposed_components['seasonal_fine'] + 
                    self.decomposed_components['residual'])
        reconstruction_error = np.abs(recomposed - self.data_x[:len(recomposed)]).mean()
        print(f"\nReconstruction error: {reconstruction_error:.8f}")

        if reconstruction_error > 0.01:
            print(f"⚠️  WARNING: High reconstruction error! Decomposition may be broken.")
        else:
            print(f"✅ Reconstruction looks good.")

        # Check for degenerate components (all zeros or constant)
        for comp_name in ['trend', 'seasonal_coarse', 'seasonal_fine', 'residual']:
            comp_std = self.decomposed_components[comp_name].std()
            if comp_std < 1e-6:
                print(f"⚠️  WARNING: {comp_name} is nearly constant (std={comp_std:.8f})")

        print(f"{'='*60}\n")
        mode_str = "segment-level" if self.use_segments else "point-level"
        print(f"[{self.flag}] Data preparation complete ({mode_str}). Temporal length: {self.temporal_length}")

    def _format_components(self, components_per_column):
        """
        Convert TimeBaseMSTL output to unified [T, num_features] format.
        use_segments is not important for this aspect.
        
        Args:
            components_per_column: dict from TimeBaseMSTL.transform()
                {
                    'feature_name': {
                        'trend': np.array([T,]),
                        'seasonal_coarse': np.array([T,]),
                        'seasonal_fine': np.array([T,]),
                        'residual': np.array([T,])
                    },
                    ...
                }
        
        Returns:
            dict: {
                'trend': np.array([T, num_features]),
                'seasonal_coarse': np.array([T, num_features]),
                'seasonal_fine': np.array([T, num_features]),
                'residual': np.array([T, num_features])
            }
        """
        column_names = list(components_per_column.keys())
        num_features = len(column_names)
        
        if num_features == 1:
            # Single feature case
            col = column_names[0]
            return {
                'trend': components_per_column[col]['trend'].reshape(-1, 1),
                'seasonal_coarse': components_per_column[col]['seasonal_coarse'].reshape(-1, 1),
                'seasonal_fine': components_per_column[col]['seasonal_fine'].reshape(-1, 1),
                'residual': components_per_column[col]['residual'].reshape(-1, 1)
            }
        else:
            # Multiple features case
            return {
                'trend': np.column_stack([
                    components_per_column[col]['trend'] for col in column_names
                ]),
                'seasonal_coarse': np.column_stack([
                    components_per_column[col]['seasonal_coarse'] for col in column_names
                ]),
                'seasonal_fine': np.column_stack([
                    components_per_column[col]['seasonal_fine'] for col in column_names
                ]),
                'residual': np.column_stack([
                    components_per_column[col]['residual'] for col in column_names
                ])
            }

    def _create_segments(self, data, seg_len):
        """
        Create segments that fit
        Only creates full segments that fit within the data.
        
        Args:
            data: [T, C] array
            seg_len: segment length (mstl_period)
        
        Returns:
            segments: [num_segs, seg_len, C] array
        """
        T, C = data.shape
        num_segments = T // seg_len
        effective_len = num_segments * seg_len
        data_trimmed = data[:effective_len]
        segments = data_trimmed.reshape(num_segments, seg_len, C)
        
        return segments  # [num_segs, seg_len, C]

    def __getitem__(self, index):
        """
        Returns decomposed components in dict format.
        
        Point-level mode:
            Returns: X_dict with [seq_len, C] and [label_len + pred_len, C] arrays
        
        Segment-level mode:
            Returns: X_dict with [num_segs, seg_len, C] arrays
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.use_segments:
            # ========================================
            # Segment-Level Mode Overlapping Window
            # ========================================
            X_dict = {}
            
            for comp_name, comp_data in self.decomposed_components.items():
                # Extract window first
                x_seq = comp_data[s_begin:s_end]  # [seq_len, C]
                
                # Creating segments
                X_dict[comp_name] = self._create_segments(x_seq, self.mstl_period)
            
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            seq_y = self.data_y[r_begin:r_end]
            
        else:
            # ========================================
            # Point-Level Mode Representing the segments
            # ========================================
            X_dict = {
                'trend': self.decomposed_components['trend'][s_begin:s_end],
                'seasonal_coarse': self.decomposed_components['seasonal_coarse'][s_begin:s_end],
                'seasonal_fine': self.decomposed_components['seasonal_fine'][s_begin:s_end],
                'residual': self.decomposed_components['residual'][s_begin:s_end]
            }
            
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            seq_y = self.data_y[r_begin:r_end]

        return X_dict, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Calculate number of valid sliding window samples.
        
        Note: decomposed_components are always stored as [T, C] regardless of use_segments.
        Segmentation only happens in __getitem__.
        """
        
        return self.temporal_length - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse scaling for predictions."""
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute_Decomposition(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', 
                 basis=None, use_segments=False, mstl_period=24):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        if basis == None:
            self.n_basis_components = 20
            self.orthogonal_lr = 1e-3
            self.orthogonal_iters = 300
        else:
            self.n_basis_components = basis[0]
            self.orthogonal_lr = basis[1]
            self.orthogonal_iters = basis[2]

        self.timebase_mstl = TimeBaseMSTL(
            n_basis_components=self.n_basis_components,
            orthogonal_lr=self.orthogonal_lr,
            orthogonal_iters=self.orthogonal_iters,
            seq_len=self.seq_len
        )
        self.use_segments = use_segments
        self.mstl_period = mstl_period
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        # ========================================
        # TimeBaseMSTL Decomposition
        # ========================================
        train_data_normalized = data[border1s[0]:border2s[0]]
        train_df = pd.DataFrame(train_data_normalized, columns=df_data.columns)
        train_dates = pd.to_datetime(df_raw['date'][border1s[0]:border2s[0]].values)
        train_df.index = train_dates
        
        print(f"[{self.flag}] Fitting TimeBaseMSTL on training data...")
        self.timebase_mstl.fit(train_df)
        
        # Auto-detect MSTL period if using segments
        if self.use_segments:
            detected_period = self.timebase_mstl.detect_periods(train_df)[0]
            if self.mstl_period != detected_period:
                print(f"[{self.flag}] Overriding mstl_period {self.mstl_period} with detected {detected_period}")
                self.mstl_period = detected_period

        # Transform current split using fitted TimeBaseMSTL
        current_data_normalized = data[border1:border2]
        current_df = pd.DataFrame(current_data_normalized, columns=df_data.columns)
        current_dates = pd.to_datetime(df_raw['date'][border1:border2].values)
        current_df.index = current_dates

        print(f"[{self.flag}] Transforming data with TimeBaseMSTL...")
        components_per_column = self.timebase_mstl.transform(current_df)
        
        # Convert to model-ready format: always store as [T, C] point-level
        self.decomposed_components = self._format_components(components_per_column)
        
        # Cache temporal length for efficient __len__ computation
        self.temporal_length = self.decomposed_components['trend'].shape[0]
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        mode_str = "segment-level" if self.use_segments else "point-level"
        print(f"[{self.flag}] Data preparation complete ({mode_str}). Temporal length: {self.temporal_length}")

    def _format_components(self, components_per_column):
        """
        Convert TimeBaseMSTL output to unified [T, num_features] format.
        use_segments is not important for this aspect.
        
        Args:
            components_per_column: dict from TimeBaseMSTL.transform()
                {
                    'feature_name': {
                        'trend': np.array([T,]),
                        'seasonal_coarse': np.array([T,]),
                        'seasonal_fine': np.array([T,]),
                        'residual': np.array([T,])
                    },
                    ...
                }
        
        Returns:
            dict: {
                'trend': np.array([T, num_features]),
                'seasonal_coarse': np.array([T, num_features]),
                'seasonal_fine': np.array([T, num_features]),
                'residual': np.array([T, num_features])
            }
        """
        column_names = list(components_per_column.keys())
        num_features = len(column_names)
        
        if num_features == 1:
            # Single feature case
            col = column_names[0]
            return {
                'trend': components_per_column[col]['trend'].reshape(-1, 1),
                'seasonal_coarse': components_per_column[col]['seasonal_coarse'].reshape(-1, 1),
                'seasonal_fine': components_per_column[col]['seasonal_fine'].reshape(-1, 1),
                'residual': components_per_column[col]['residual'].reshape(-1, 1)
            }
        else:
            # Multiple features case
            return {
                'trend': np.column_stack([
                    components_per_column[col]['trend'] for col in column_names
                ]),
                'seasonal_coarse': np.column_stack([
                    components_per_column[col]['seasonal_coarse'] for col in column_names
                ]),
                'seasonal_fine': np.column_stack([
                    components_per_column[col]['seasonal_fine'] for col in column_names
                ]),
                'residual': np.column_stack([
                    components_per_column[col]['residual'] for col in column_names
                ])
            }

    def _create_segments(self, data, seg_len):
        """
        Create segments that fit
        Only creates full segments that fit within the data.
        
        Args:
            data: [T, C] array
            seg_len: segment length (mstl_period)
        
        Returns:
            segments: [num_segs, seg_len, C] array
        """
        T, C = data.shape
        num_segments = T // seg_len
        effective_len = num_segments * seg_len
        data_trimmed = data[:effective_len]
        segments = data_trimmed.reshape(num_segments, seg_len, C)
        
        return segments  # [num_segs, seg_len, C]

    def __getitem__(self, index):
        """
        Returns decomposed components in dict format.
        
        Point-level mode:
            Returns: X_dict with [seq_len, C] and [label_len + pred_len, C] arrays
        
        Segment-level mode:
            Returns: X_dict with [num_segs, seg_len, C] arrays
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.use_segments:
            # ========================================
            # Segment-Level Mode Overlapping Window
            # ========================================
            X_dict = {}
            
            for comp_name, comp_data in self.decomposed_components.items():
                # Extract window first
                x_seq = comp_data[s_begin:s_end]  # [seq_len, C]
                
                # Creating segments
                X_dict[comp_name] = self._create_segments(x_seq, self.mstl_period)
            
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            seq_y = self.data_y[r_begin:r_end]
            
        else:
            # ========================================
            # Point-Level Mode Representing the segments
            # ========================================
            X_dict = {
                'trend': self.decomposed_components['trend'][s_begin:s_end],
                'seasonal_coarse': self.decomposed_components['seasonal_coarse'][s_begin:s_end],
                'seasonal_fine': self.decomposed_components['seasonal_fine'][s_begin:s_end],
                'residual': self.decomposed_components['residual'][s_begin:s_end]
            }
            
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            seq_y = self.data_y[r_begin:r_end]

        return X_dict, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Calculate number of valid sliding window samples.
        
        Note: decomposed_components are always stored as [T, C] regardless of use_segments.
        Segmentation only happens in __getitem__.
        """
        
        return self.temporal_length - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse scaling for predictions."""
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 basis=None, use_segments=False, mstl_period=24):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.basis = basis
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.dates = df_stamp['date'].values
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_Decomposition(Dataset):
    """
    Custom dataset with TimeBaseMSTL decomposition.
    Unified implementation supporting both point-level and segment-level modes.
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 basis=None, use_segments=False, mstl_period=24):
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Segment/Point configuration
        self.use_segments = use_segments
        self.mstl_period = mstl_period
        
        # TimeBaseMSTL basis parameters
        if basis == None:
            self.n_basis_components = 20
            self.orthogonal_lr = 1e-3
            self.orthogonal_iters = 300
        else:
            self.n_basis_components = basis[0]
            self.orthogonal_lr = basis[1]
            self.orthogonal_iters = basis[2]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.timebase_mstl = TimeBaseMSTL(
            n_basis_components=self.n_basis_components,
            orthogonal_lr=self.orthogonal_lr,
            orthogonal_iters=self.orthogonal_iters,
            seq_len=self.seq_len
        )
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, 
        self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ========================================
        # TimeBaseMSTL Decomposition
        # ========================================
        train_data_normalized = data[border1s[0]:border2s[0]]
        train_df = pd.DataFrame(train_data_normalized, columns=df_data.columns)
        train_dates = pd.to_datetime(df_raw['date'][border1s[0]:border2s[0]].values)
        train_df.index = train_dates
        
        print(f"[{self.flag}] Fitting TimeBaseMSTL on training data...")
        self.timebase_mstl.fit(train_df)
        
        # Auto-detect MSTL period if using segments
        if self.use_segments:
            detected_period = self.timebase_mstl.detect_periods(train_df)[0]
            if self.mstl_period != detected_period:
                print(f"[{self.flag}] Overriding mstl_period {self.mstl_period} with detected {detected_period}")
                self.mstl_period = detected_period

        # Transform current split using fitted TimeBaseMSTL
        current_data_normalized = data[border1:border2]
        current_df = pd.DataFrame(current_data_normalized, columns=df_data.columns)
        current_dates = pd.to_datetime(df_raw['date'][border1:border2].values)
        current_df.index = current_dates

        print(f"[{self.flag}] Transforming data with TimeBaseMSTL...")
        components_per_column = self.timebase_mstl.transform(current_df)
        
        # Convert to model-ready format: always store as [T, C] point-level
        self.decomposed_components = self._format_components(components_per_column)
        
        # Cache temporal length for efficient __len__ computation
        self.temporal_length = self.decomposed_components['trend'].shape[0]
        
        # ========================================
        # Time Features
        # ========================================
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        self.dates = df_stamp['date'].values    
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        self.data_stamp = data_stamp
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
            # ADD THIS:
        print(f"\n{'='*60}")
        print(f"[{self.flag}] DECOMPOSITION DIAGNOSTICS FOR Weather")
        print(f"{'='*60}")
        print(f"Data shape: {self.data_x.shape}")
        print(f"Temporal length: {self.temporal_length}")
        print(f"\nOriginal data stats:")
        print(f"  Mean: {self.data_x.mean():.6f}, Std: {self.data_x.std():.6f}")
        print(f"  Range: [{self.data_x.min():.6f}, {self.data_x.max():.6f}]")

        print(f"\nDecomposed components stats:")
        for comp_name in ['trend', 'seasonal_coarse', 'seasonal_fine', 'residual']:
            comp_data = self.decomposed_components[comp_name]
            print(f"  {comp_name:20s}: mean={comp_data.mean():8.6f}, std={comp_data.std():8.6f}, range=[{comp_data.min():8.4f}, {comp_data.max():8.4f}]")

        # Reconstruction check
        recomposed = (self.decomposed_components['trend'] + 
                    self.decomposed_components['seasonal_coarse'] + 
                    self.decomposed_components['seasonal_fine'] + 
                    self.decomposed_components['residual'])
        reconstruction_error = np.abs(recomposed - self.data_x[:len(recomposed)]).mean()
        print(f"\nReconstruction error: {reconstruction_error:.8f}")

        if reconstruction_error > 0.01:
            print(f"⚠️  WARNING: High reconstruction error! Decomposition may be broken.")
        else:
            print(f"✅ Reconstruction looks good.")

        # Check for degenerate components (all zeros or constant)
        for comp_name in ['trend', 'seasonal_coarse', 'seasonal_fine', 'residual']:
            comp_std = self.decomposed_components[comp_name].std()
            if comp_std < 1e-6:
                print(f"⚠️  WARNING: {comp_name} is nearly constant (std={comp_std:.8f})")

        print(f"{'='*60}\n")
       
        mode_str = "segment-level" if self.use_segments else "point-level"
        print(f"[{self.flag}] Data preparation complete ({mode_str}). Temporal length: {self.temporal_length}")

    def _format_components(self, components_per_column):
        """
        Convert TimeBaseMSTL output to unified [T, num_features] format.
        use_segments is not important for this aspect.
        
        Args:
            components_per_column: dict from TimeBaseMSTL.transform()
                {
                    'feature_name': {
                        'trend': np.array([T,]),
                        'seasonal_coarse': np.array([T,]),
                        'seasonal_fine': np.array([T,]),
                        'residual': np.array([T,])
                    },
                    ...
                }
        
        Returns:
            dict: {
                'trend': np.array([T, num_features]),
                'seasonal_coarse': np.array([T, num_features]),
                'seasonal_fine': np.array([T, num_features]),
                'residual': np.array([T, num_features])
            }
        """
        column_names = list(components_per_column.keys())
        num_features = len(column_names)
        
        if num_features == 1:
            # Single feature case
            col = column_names[0]
            return {
                'trend': components_per_column[col]['trend'].reshape(-1, 1),
                'seasonal_coarse': components_per_column[col]['seasonal_coarse'].reshape(-1, 1),
                'seasonal_fine': components_per_column[col]['seasonal_fine'].reshape(-1, 1),
                'residual': components_per_column[col]['residual'].reshape(-1, 1)
            }
        else:
            # Multiple features case
            return {
                'trend': np.column_stack([
                    components_per_column[col]['trend'] for col in column_names
                ]),
                'seasonal_coarse': np.column_stack([
                    components_per_column[col]['seasonal_coarse'] for col in column_names
                ]),
                'seasonal_fine': np.column_stack([
                    components_per_column[col]['seasonal_fine'] for col in column_names
                ]),
                'residual': np.column_stack([
                    components_per_column[col]['residual'] for col in column_names
                ])
            }

    def _create_segments(self, data, seg_len):
        """
        Create segments that fit
        Only creates full segments that fit within the data.
        
        Args:
            data: [T, C] array
            seg_len: segment length (mstl_period)
        
        Returns:
            segments: [num_segs, seg_len, C] array
        """
        T, C = data.shape
        num_segments = T // seg_len
        effective_len = num_segments * seg_len
        data_trimmed = data[:effective_len]
        segments = data_trimmed.reshape(num_segments, seg_len, C)
        
        return segments  # [num_segs, seg_len, C]

    def __getitem__(self, index):
        """
        Returns decomposed components in dict format.
        
        Point-level mode:
            Returns: X_dict with [seq_len, C] and [label_len + pred_len, C] arrays
        
        Segment-level mode:
            Returns: X_dict with [num_segs, seg_len, C] arrays
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        if self.use_segments:
            # ========================================
            # Segment-Level Mode Overlapping Window
            # ========================================
            X_dict = {}
            
            for comp_name, comp_data in self.decomposed_components.items():
                # Extract window first
                x_seq = comp_data[s_begin:s_end]  # [seq_len, C]
                
                # Creating segments
                X_dict[comp_name] = self._create_segments(x_seq, self.mstl_period)
            
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            seq_y = self.data_y[r_begin:r_end]
            
        else:
            # ========================================
            # Point-Level Mode Representing the segments
            # ========================================
            X_dict = {
                'trend': self.decomposed_components['trend'][s_begin:s_end],
                'seasonal_coarse': self.decomposed_components['seasonal_coarse'][s_begin:s_end],
                'seasonal_fine': self.decomposed_components['seasonal_fine'][s_begin:s_end],
                'residual': self.decomposed_components['residual'][s_begin:s_end]
            }
            
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
            seq_y = self.data_y[r_begin:r_end]

        return X_dict, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        Calculate number of valid sliding window samples.
        
        Note: decomposed_components are always stored as [T, C] regardless of use_segments.
        Segmentation only happens in __getitem__.
        """
        
        return self.temporal_length - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """Inverse scaling for predictions."""
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)        
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
