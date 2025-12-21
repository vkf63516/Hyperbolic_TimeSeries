# Hyperbolic_TimeSeries
This repository implements Hyperbolic Time Series Forecasting with orthogonalMSTL decomposition. Below is a description of each file in the project:
Main Files
run.py
Download the datasets from Google Drive
The main entry point for training and evaluation. This script:

    Defines command-line arguments for configuring the model and training process
    Supports hyperbolic forecasting with orthogonalMSTL decomposition
    Configures hyperparameters including:
        encodeding dimensions, hidden dimensions, and curvature for hyperbolic space
        Data loading parameters (dataset type, paths, features)
        Training settings (batch size, learning rate, epochs)
        MSTL decomposition settings (period, number of basis components)
        Manifold type selection (Lorentzian, Poincare, or Euclidean)
    Imports and initializes the main experiment class Exp_Main

spec.py

Contains utility functions and implementations for hyperbolic geometry operations:

    compute_hierarchical_loss_with_manifold_dist(): Enforces hierarchical relationships between time series components (trend < coarse < fine < residual) using hyperbolic distances from origin
    segment_safe_expmap0(): Safe exponential mapping from tangent space to manifold for segmented data with norm clamping
    safe_expmap(): Safe exponential mapping for non-origin base points
    Implements margin-based hierarchy loss and entailment loss for parent-child relationships in hyperbolic space

Shell Scripts
weather_euclidean.sh

Training script for weather dataset using Euclidean manifold:

    Configures a 336→96 forecasting task on weather data
    Uses MS (multivariate predict univariate) features
    21 input features, 32 encodeding dimensions, 256 hidden dimensions
    Enables decomposition, hierarchy scaling, and automatic mixed precision (AMP)

Experiment Framework
exp/exp_basic.py

Base experiment class providing:

    Device management (GPU/MPS/CPU)
    Model initialization and device placement
    Abstract methods for data loading, training, validation, and testing
    Foundation for concrete experiment implementations

exp/exp_main.py

Main experiment orchestrator that extends Exp_Basic:

    Initialization:
        Configures decomposition settings (orthogonalMSTL parameters)
        Sets up TensorBoard logging for experiment tracking
        Supports both segment-level and point-level hyperbolic encodedings via use_segments flag
        Initializes manifold type (Lorentzian, Poincare, or Euclidean)
    Model Building: Constructs the HyperbolicForecasting model with parameter counting
    Optimizer Selection: Chooses between standard AdamW (Euclidean) or Riemannian optimizers (hyperbolic)
    Data Provider: Interfaces with data factory to load train/val/test datasets
    Training Loop: Implements the full training procedure with:
        Teacher forcing support
        Hierarchical loss computation for hyperbolic encodedings
        TensorBoard metric logging
        Early stopping and checkpointing
    Validation & Testing: Evaluates model performance with MSE and MAE metrics

Data Loading
data_provider/data_loader.py

Implements PyTorch Dataset classes for time series data:

    Dataset_ETT_hour: Loads ETT (Electricity Transformer Temperature) hourly data
        Handles train/val/test splits with predefined borders
        Applies StandardScaler normalization using training statistics
        orthogonalMSTL Integration:
            Fits decomposition on training data
            Auto-detects MSTL period from data frequency
            Transforms data into trend, seasonal_coarse, seasonal_fine, and residual components
            Stores decomposed components in [T, C] format for both point-level and segment-level modes
        Generates time features (month, day, weekday, hour) or uses time encodedings
        Returns sliding windows of decomposed components for forecasting
        Supports multivariate (M), univariate (S), and multivariate-to-univariate (MS) forecasting tasks

Decomposition Module
Decomposition/Orthogonal_Series_Trend_Decomposition.py

Implements orthogonalMSTL - a orthogonal-inspired decomposition using learned orthogonal basis functions:

    orthogonalMSTL Class:
        Period Detection: detect_periods() automatically infers fine and coarse periods from data frequency
        Segment Extraction: extract_periodic_segments() extracts repeating patterns from time series
        Orthogonal Basis Learning: learn_orthogonal_basis() learns basis functions via gradient-based orthogonalization (alternative to PCA)
        Decomposition: Breaks time series into:
            Trend: Long-term patterns
            seasonal_coarse: coarse cyclical patterns
            Seasonal_fine: fine cyclical patterns
            Residual: Remaining noise/irregularities
        Workflow:
            fit(df): Learns basis functions from training data
            transform(df): Decomposes new data using learned basis
            Returns per-feature decomposition dictionary

Decomposition/visualization_utils.py

Visualization tools for analyzing decomposition quality:

    plot_component_grid(): Plots all components (trend, coarse, fine, residual) for a single feature
    plot_variance_contribution(): Bar chart showing variance explained by each component across features
    plot_component_correlation_maps(): Heatmaps showing cross-feature correlations within each component

Decomposition/Series_Trend_Decomposition.py

Alternative MSTL decomposition using statsmodels:

    mstl_decomposition_for_window(): Standard MSTL decomposition with hourly, fine, and coarse periods
    timesteps_based_on_frequency(): Computes period lengths based on data frequency
    Provides a baseline comparison to orthogonalMSTL

encodeding Modules
encode/segment_mlp_encode_lorentz.py

Implements segment-level encodedings for Lorentzian (hyperbolic) manifold:

    SegmentMLPencode: MLP encoder for segmented time series data
        Encodes each segment independently
        Uses attention pooling over segments
        Supports both segmented and non-segmented inputs
    SegmentedParallelLorentz: Parallel encodedings for different time series components with hierarchical scaling
        Configurable segment lengths for trend, coarse, fine, and residual components
        Learnable hierarchy scales as parameters

encode/mlp_encode_euclidean.py

Implements encodedings for Euclidean space:

    MLPencode: Basic MLP encoder with layer normalization and GELU activation
        Optional attention pooling or mean pooling across time
        Produces Euclidean latent vectors
    ParallelEuclideanencode: Parallel encoders for different decomposed components (trend, fine, coarse, residual)
        Optional learnable hierarchy scales
        Separate encodeding networks for each component

Helper Modules
spec.py

Helper functions for training:

    adjust_learning_rate(): Learning rate scheduling with multiple strategies (type1-6, TST, constant)
    EarlyStopping: Early stopping callback with patience counter and checkpoint saving
    dotdict: Dictionary wrapper for dot-notation attribute access
