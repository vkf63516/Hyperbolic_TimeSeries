### HyperForecast: Velocity-Driven Hyperbolic Dynamics with Learnable Decomposition for Time Series Forecasting ###

This repository contains the official implementation of HyperForecast, a hyperbolic geometry–based framework for long-term time series forecasting. The model combines learnable multi-scale decomposition with geodesic dynamics in hyperbolic space to capture hierarchical temporal structure across frequencies.

HyperForecast models trend, seasonal, and residual components as trajectories on a hyperbolic manifold and performs forecasting via velocity-driven geodesic evolution, providing both competitive accuracy and interpretable hierarchical representations.

Key Features

Learnable Multi-Scale Decomposition
Automatically separates time series into trend, coarse seasonal, fine seasonal, and residual components.

Hyperbolic Latent Space Modeling
Embeds decomposed temporal segments in hyperbolic space (Poincaré or Lorentz models) to encode hierarchical frequency structure.

Velocity-Driven Geodesic Forecasting
Forecasts future values by evolving latent representations along geodesics using learned velocity fields.

Manifold-Aware Training
Includes hierarchy and temporal consistency regularization to stabilize long-horizon forecasting.

Acknowledgements

We would like to thank the authors of the following open-source projects for their valuable contributions, which provides significant help for our work:

[Time-Series-Library (THUML)](https://github.com/thuml/Time-Series-Library)
[TimeBase (ICML 2025)](https://github.com/hqh0728/TimeBase)
[DLinear (AAAI 2023)](https://github.com/cure-lab/LTSF-Linear/tree/main)