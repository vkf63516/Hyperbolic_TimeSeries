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

1. **Uncomment and update HyperbolicTimeSeriesDataset** in `data_provider/data_loader.py`:

```python
class HyperbolicTimeSeriesDataset(Dataset):
    """
    Dataset for hyperbolic forecasting with TimeBaseMSTL decomposition.
    Integrates with prepare_timebase_data_with_mstl from spec.py.
    """
    
    def __init__(self, train_df, val_df, test_df, args, flag='train'):
        self.args = args
        self.flag = flag
        
        # Initialize TimeBaseMSTL
        self.timebase_mstl = TimeBaseMSTL(
            n_basis_components=args.num_basis,
            orthogonal_lr=args.orthogonal_lr,
            orthogonal_iters=args.orthogonal_iters
        )
        
        # Fit on training data
        print("Fitting TimeBaseMSTL...")
        self.timebase_mstl.fit(train_df)
        
        # Get auto-detected period
        self.mstl_period = self.timebase_mstl.steps_per_period[0]
        print(f"Auto-detected MSTL period: {self.mstl_period}")
        
        # Transform all datasets
        train_components = self.timebase_mstl.transform(train_df)
        val_components = self.timebase_mstl.transform(val_df)
        test_components = self.timebase_mstl.transform(test_df)
        
        # Convert to tensor dictionaries
        from Decomposition.tensor_utils import build_decomposition_tensors
        train_dict = build_decomposition_tensors(train_components)
        val_dict = build_decomposition_tensors(val_components)
        test_dict = build_decomposition_tensors(test_components)
        
        # Use prepare_timebase_data_with_mstl for proper preprocessing
        from spec import prepare_timebase_data_with_mstl
        train_seg, val_seg, test_seg, self.scaler, _ = prepare_timebase_data_with_mstl(
            train_dict=train_dict,
            val_dict=val_dict,
            test_dict=test_dict,
            mstl_period=self.mstl_period,
            input_len=args.seq_len,
            pred_len=args.pred_len,
            stride='overlap',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Store appropriate split
        if flag == 'train':
            self.data = train_seg
        elif flag == 'val':
            self.data = val_seg
        else:
            self.data = test_seg
    
    def __len__(self):
        # Return number of samples based on segmented data structure
        # Adjust based on actual structure from prepare_timebase_data_with_mstl
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return decomposed components for the sample
        # Format: trend, daily, weekly, resid each of shape [T, 1]
        # Adjust based on actual structure from prepare_timebase_data_with_mstl
        sample = self.data[idx]
        
        # Extract and format components
        # This depends on the output structure of prepare_timebase_data_with_mstl
        return sample
```

2. **Update data_factory.py** to use HyperbolicTimeSeriesDataset:

```python
from data_provider.data_loader import HyperbolicTimeSeriesDataset

def data_provider(args, flag):
    if args.model == 'HyperbolicMambaForecasting' and args.use_decomposition:
        # Load raw DataFrames
        train_df = load_dataframe(args, 'train')
        val_df = load_dataframe(args, 'val')
        test_df = load_dataframe(args, 'test')
        
        # Create dataset with TimeBaseMSTL integration
        data_set = HyperbolicTimeSeriesDataset(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            args=args,
            flag=flag
        )
    else:
        # Use standard datasets
        data_set = Dataset_Custom(...)
    
    data_loader = DataLoader(data_set, ...)
    return data_set, data_loader
```

3. **Update exp_main.py** to use decomposed components:

```python
# In training loop
for batch in train_loader:
    if isinstance(batch, dict):  # Decomposed batch from HyperbolicTimeSeriesDataset
        batch_x = batch['input']
        batch_y = batch['target']
        decomposed_components = {
            'trend': batch['trend'],
            'daily': batch['daily'],
            'weekly': batch['weekly'],
            'resid': batch['resid']
        }
        outputs = self.model(batch_x, decomposed_components=decomposed_components)
    else:
        # Standard batch
        outputs = self.model(batch_x)
```

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
