# TimeBaseMSTL Integration Guide for HyperbolicMambaForecasting

## Overview

This guide explains how to properly integrate TimeBaseMSTL decomposition with HyperbolicMambaForecasting using the existing infrastructure in the codebase.

## Architecture Components

### 1. TimeBaseMSTL (`Decomposition/TimeBase_Series_Trend_Decomposition.py`)
- Learns orthogonal basis functions from training data
- Decomposes time series into: trend, seasonal_daily, seasonal_weekly, residual
- Auto-detects periods using `timesteps_from_index()`

### 2. HyperbolicMambaForecasting Model (`models/HyperbolicMambaForecasting.py`)
- Accepts pre-decomposed components via `decomposed_components` parameter
- Encodes components to Lorentz manifold using ParallelLorentzBlock
- Performs autoregressive forecasting on the manifold
- Reconstructs predictions back to Euclidean space

### 3. Data Preprocessing (`spec.py`)
- `prepare_timebase_data_with_mstl()`: Normalizes and segments decomposed data
- `Create_Segments_With_MSTL_Period()`: Creates sliding windows with MSTL period
- Handles proper normalization and period-aligned segmentation

### 4. HyperbolicTimeSeriesDataset (`data_provider/data_loader.py` - currently commented)
- **Should be integrated with `prepare_timebase_data_with_mstl()`**
- Provides efficient batch access to decomposed, segmented data
- Caches decomposition for training efficiency

## Recommended Integration Path

### Option 1: Using HyperbolicTimeSeriesDataset (Best Practice)

The key challenge is that `data_provider()` is called separately for each split (train/val/test), but TimeBaseMSTL needs to:
- Be fit once on training data
- Transform all three splits consistently
- Share the fitted instance across datasets

**Solution: Use module-level caching in data_factory.py**

1. **Update `data_provider/data_factory.py`**:

```python
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
from Decomposition.TimeBase_Series_Trend_Decomposition import TimeBaseMSTL
from Decomposition.tensor_utils import build_decomposition_tensors
from spec import prepare_timebase_data_with_mstl
import pandas as pd

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}

# Module-level cache for TimeBaseMSTL preprocessing
_timebase_cache = {}

def _initialize_timebase_preprocessing(args):
    """
    Initialize TimeBaseMSTL and preprocess all splits.
    Called once on first data_provider call.
    """
    print("Initializing TimeBaseMSTL preprocessing...")
    
    # Load data using the same pattern as Dataset_Custom
    # Read entire CSV file once
    df_raw = pd.read_csv(os.path.join(args.root_path, args.data_path))
    
    # Split data using same border logic as Dataset_Custom (70/20/10 split)
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    
    # Extract splits as DataFrames
    train_df = df_raw.iloc[0:num_train]
    val_df = df_raw.iloc[num_train:num_train + num_vali]
    test_df = df_raw.iloc[num_train + num_vali:]
    
    # Initialize and fit TimeBaseMSTL on training data
    timebase_mstl = TimeBaseMSTL(
        n_basis_components=args.num_basis,
        orthogonal_lr=args.orthogonal_lr,
        orthogonal_iters=args.orthogonal_iters
    )
    timebase_mstl.fit(train_df)
    
    # Auto-detect period
    mstl_period = timebase_mstl.steps_per_period[0]
    print(f"Auto-detected MSTL period: {mstl_period}")
    
    # Transform all datasets
    train_components = timebase_mstl.transform(train_df)
    val_components = timebase_mstl.transform(val_df)
    test_components = timebase_mstl.transform(test_df)
    
    # Convert to tensor dictionaries
    train_dict = build_decomposition_tensors(train_components)
    val_dict = build_decomposition_tensors(val_components)
    test_dict = build_decomposition_tensors(test_components)
    
    # Use prepare_timebase_data_with_mstl for normalization and segmentation
    train_seg, val_seg, test_seg, scaler, _ = prepare_timebase_data_with_mstl(
        train_dict=train_dict,
        val_dict=val_dict,
        test_dict=test_dict,
        mstl_period=mstl_period,
        input_len=args.seq_len,
        pred_len=args.pred_len,
        stride='overlap',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Cache for reuse across train/val/test calls
    _timebase_cache[args.data] = {
        'train': train_seg,
        'val': val_seg,
        'test': test_seg,
        'scaler': scaler,
        'mstl_period': mstl_period
    }
    print("TimeBaseMSTL preprocessing complete")

def data_provider(args, flag):
    # Check if we should use TimeBaseMSTL preprocessing
    use_timebase = (
        args.model == 'HyperbolicMambaForecasting' and 
        getattr(args, 'use_decomposition', False)
    )
    
    if use_timebase:
        # Initialize preprocessing on first call
        if args.data not in _timebase_cache:
            _initialize_timebase_preprocessing(args)
        
        # Create HyperbolicTimeSeriesDataset with preprocessed data
        from data_provider.data_loader import HyperbolicTimeSeriesDataset
        data_set = HyperbolicTimeSeriesDataset(
            preprocessed_data=_timebase_cache[args.data][flag],
            scaler=_timebase_cache[args.data]['scaler'],
            args=args,
            flag=flag
        )
    else:
        # Standard data loading
        Data = data_dict[args.data]
        timeenc = 0 if args.embed != 'timeF' else 1
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq
        )
    
    # Create DataLoader
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True if flag == 'train' else False
        drop_last = False
        batch_size = args.batch_size
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader
```

2. **Uncomment and update HyperbolicTimeSeriesDataset** in `data_provider/data_loader.py`:

```python
class HyperbolicTimeSeriesDataset(Dataset):
    """
    Dataset for hyperbolic forecasting with TimeBaseMSTL decomposition.
    Receives pre-processed data from data_factory to avoid redundant preprocessing.
    """
    
    def __init__(self, preprocessed_data, scaler, args, flag='train'):
        """
        Args:
            preprocessed_data: Output from prepare_timebase_data_with_mstl for this split
            scaler: Fitted scaler from preprocessing
            args: Configuration arguments
            flag: 'train', 'val', or 'test'
        """
        self.data = preprocessed_data
        self.scaler = scaler
        self.args = args
        self.flag = flag
    
    def __len__(self):
        # The structure from prepare_timebase_data_with_mstl is:
        # {feature: {"X": {comp: tensor}, "Y": {comp: tensor}}}
        # where tensor shape is [N_samples, N_segments, P, C]
        
        # Get first feature to determine sample count
        first_feature = next(iter(self.data.keys()))
        first_comp = next(iter(self.data[first_feature]["X"].keys()))
        return self.data[first_feature]["X"][first_comp].shape[0]
    
    def __getitem__(self, idx):
        """
        Return decomposed components for a single sample.
        
        Returns:
            Dictionary with input and target decomposed components:
            {
                'input': {'trend': [T, 1], 'daily': [T, 1], 'weekly': [T, 1], 'resid': [T, 1]},
                'target': {'trend': [T', 1], 'daily': [T', 1], 'weekly': [T', 1], 'resid': [T', 1]}
            }
        """
        # Extract components for this sample from prepare_timebase_data_with_mstl structure
        sample = {}
        for feature_name, feature_data in self.data.items():
            sample['input'] = {}
            sample['target'] = {}
            
            for comp_name in ['trend', 'seasonal_daily', 'seasonal_weekly', 'residual']:
                # Get input and target components
                if comp_name in feature_data["X"]:
                    sample['input'][comp_name] = feature_data["X"][comp_name][idx]
                if comp_name in feature_data["Y"]:
                    sample['target'][comp_name] = feature_data["Y"][comp_name][idx]
        
        return sample
```

3. **Key Differences from Dataset_Custom**:

| Aspect | Dataset_Custom | HyperbolicTimeSeriesDataset |
|--------|----------------|----------------------------|
| Data Loading | Reads from files internally | Receives preprocessed data |
| Parameters | `root_path`, `data_path`, `flag` | `preprocessed_data`, `scaler`, `args`, `flag` |
| Initialization | Independent per split | Shares preprocessing via cache |
| TimeBaseMSTL | Not used | Preprocessed in data_factory |

This approach:
- ✅ Maintains the existing data_factory pattern
- ✅ Fits TimeBaseMSTL once and shares across splits
- ✅ Uses module-level cache to avoid redundant preprocessing
- ✅ Compatible with existing training loops

### Option 2: Using exp_main_with_timebase.py (Reference Implementation)

See `exp/exp_main_with_timebase.py` for a complete example that:
- Calls `prepare_timebase_data_with_mstl` directly in exp_main
- Decomposes batches on-the-fly (less efficient but demonstrates the flow)
- Shows the complete integration pattern

## Key Points

1. **Auto-detection**: `mstl_period` is auto-detected via `timesteps_from_index()` - no user input needed
2. **Normalization**: `prepare_timebase_data_with_mstl` handles proper normalization
3. **Segmentation**: Data is segmented with MSTL period for period-aligned processing
4. **Efficiency**: Using HyperbolicTimeSeriesDataset with cached decomposition is most efficient
5. **Backward Compatibility**: Model works with both decomposed and raw data

## Usage Example

```bash
python run.py \
  --is_training 1 \
  --model HyperbolicMambaForecasting \
  --data ETTh1 \
  --seq_len 96 \
  --pred_len 96 \
  --embed_dim 32 \
  --hidden_dim 64 \
  --use_decomposition \
  --num_basis 10 \
  --orthogonal_lr 1e-3 \
  --orthogonal_iters 500
```

## References

- TimeBaseMSTL: `Decomposition/TimeBase_Series_Trend_Decomposition.py`
- Preprocessing: `spec.py` - `prepare_timebase_data_with_mstl()`
- Model: `models/HyperbolicMambaForecasting.py`
- Example: `exp/exp_main_with_timebase.py`
- Dataset (commented): `data_provider/data_loader.py` - `HyperbolicTimeSeriesDataset`
