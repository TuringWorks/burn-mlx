# burn-mlx

[![Crates.io](https://img.shields.io/crates/v/burn-mlx.svg)](https://crates.io/crates/burn-mlx)
[![Documentation](https://docs.rs/burn-mlx/badge.svg)](https://docs.rs/burn-mlx)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**MLX backend for Burn** — native Apple Silicon GPU acceleration for deep learning.

This crate provides a [Burn](https://github.com/tracel-ai/burn) backend using Apple's [MLX](https://github.com/ml-explore/mlx) framework, enabling high-performance machine learning on M1/M2/M3/M4 Macs.

## Features

- **Native Apple Silicon**: Direct GPU acceleration via Metal
- **Unified Memory**: Zero-copy data sharing between CPU and GPU
- **Lazy Evaluation**: Automatic operation fusion and optimization
- **Full Burn Backend**: FloatTensorOps, IntTensorOps, BoolTensorOps, ModuleOps, ActivationOps
- **Training Support**: Pooling operations with backward passes for autodiff

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust 1.75+

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
burn-mlx = "0.1"
burn = "0.16"
```

## Quick Start

```rust
use burn::tensor::Tensor;
use burn_mlx::{Mlx, MlxDevice};

// Create tensors on Apple Silicon GPU
let device = MlxDevice::Gpu;
let a: Tensor<Mlx, 2> = Tensor::ones([2, 3], &device);
let b: Tensor<Mlx, 2> = Tensor::ones([2, 3], &device);
let c = a + b;

println!("Result shape: {:?}", c.shape());
```

## Using with Autodiff

```rust
use burn::backend::Autodiff;
use burn_mlx::Mlx;

type TrainBackend = Autodiff<Mlx>;

// Now use TrainBackend for training with automatic differentiation
```

## Pooling Operations

burn-mlx provides full support for pooling operations with both forward and backward passes, enabling their use in training workflows.

### Average Pooling

```rust
use burn::tensor::Tensor;
use burn::nn::pool::{AvgPool2d, AvgPool2dConfig};
use burn_mlx::{Mlx, MlxDevice};

let device = MlxDevice::Gpu;

// Create a 4D tensor: [batch, channels, height, width]
let input: Tensor<Mlx, 4> = Tensor::ones([1, 3, 32, 32], &device);

// Create avg pool layer with 2x2 kernel and stride 2
let config = AvgPool2dConfig::new([2, 2]).with_strides([2, 2]);
let pool = AvgPool2d::new(config);

let output = pool.forward(input);
// Output shape: [1, 3, 16, 16]
```

### Max Pooling

```rust
use burn::tensor::Tensor;
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn_mlx::{Mlx, MlxDevice};

let device = MlxDevice::Gpu;

let input: Tensor<Mlx, 4> = Tensor::ones([1, 3, 32, 32], &device);

// Create max pool layer with 2x2 kernel and stride 2
let config = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]);
let pool = MaxPool2d::new(config);

let output = pool.forward(input);
// Output shape: [1, 3, 16, 16]
```

### 1D Pooling

```rust
use burn::tensor::Tensor;
use burn::nn::pool::{AvgPool1d, AvgPool1dConfig, MaxPool1d, MaxPool1dConfig};
use burn_mlx::{Mlx, MlxDevice};

let device = MlxDevice::Gpu;

// Create a 3D tensor: [batch, channels, length]
let input: Tensor<Mlx, 3> = Tensor::ones([1, 64, 128], &device);

// Average pooling
let avg_config = AvgPool1dConfig::new(4).with_stride(4);
let avg_pool = AvgPool1d::new(avg_config);
let avg_output = avg_pool.forward(input.clone());
// Output shape: [1, 64, 32]

// Max pooling
let max_config = MaxPool1dConfig::new(4).with_stride(4);
let max_pool = MaxPool1d::new(max_config);
let max_output = max_pool.forward(input);
// Output shape: [1, 64, 32]
```

### Adaptive Pooling

```rust
use burn::tensor::Tensor;
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig};
use burn_mlx::{Mlx, MlxDevice};

let device = MlxDevice::Gpu;

let input: Tensor<Mlx, 4> = Tensor::ones([1, 512, 14, 14], &device);

// Adaptive pool to fixed output size (common before FC layers)
let config = AdaptiveAvgPool2dConfig::new([1, 1]);
let pool = AdaptiveAvgPool2d::new(config);

let output = pool.forward(input);
// Output shape: [1, 512, 1, 1]
```

## Low-Level Tensor API

```rust
use burn_mlx::{MlxTensor, MlxDevice};

let device = MlxDevice::Gpu;

// Create tensors
let a = MlxTensor::<f32>::ones(&[1024, 1024], device);
let b = MlxTensor::<f32>::ones(&[1024, 1024], device);

// Operations
let c = a.matmul(&b);
let d = c.relu();
let e = d.softmax();

// Evaluate lazy computation
e.eval().expect("evaluation failed");
```

## Supported Operations

### Tensor Operations
- Arithmetic: add, sub, mul, div, matmul
- Math: exp, log, sqrt, abs, neg, pow
- Reductions: sum, mean, max, min, argmax, argmin
- Shape: reshape, transpose, permute, expand, slice, flip, scatter

### Activation Functions
- ReLU, Sigmoid, Tanh, GELU, LeakyReLU
- Softmax, LogSoftmax, HardSigmoid

### Neural Network Layers
- Conv1d, Conv2d (with proper NCHW layout handling)
- Embedding lookup
- **Pooling** (full forward and backward support):
  - AvgPool1d, AvgPool2d
  - MaxPool1d, MaxPool2d
  - MaxPool2d with indices
  - AdaptiveAvgPool1d, AdaptiveAvgPool2d

## Implementation Details

### Pooling Operations

The pooling operations are implemented using MLX's `as_strided` function combined with reduction operations:

1. **Forward Pass**: Uses `as_strided` to create sliding window views over the input, then applies `mean_axes` (avg pool) or `max_axes` (max pool) for reduction.

2. **Backward Pass**:
   - **AvgPool**: Distributes gradients evenly across each pooling window using `scatter_add`
   - **MaxPool**: Uses saved indices from forward pass to scatter gradients to max positions

3. **Layout Handling**: Automatically converts between Burn's NCHW format and MLX's native NHWC format.

## Performance

On Apple M-series chips, burn-mlx leverages:
- Metal Performance Shaders for optimized GPU kernels
- Unified memory architecture for efficient data transfer
- Lazy evaluation for automatic operation fusion

Typical matmul performance (1024x1024):
- ~12ms per operation on M1/M2
- Scales well with larger matrices

## Limitations

- macOS only (Apple Silicon required)
- Conv3d and ConvTranspose operations are placeholders
- Quantization support is minimal
- Dilation in pooling operations is not yet supported

## License

Apache-2.0

## Acknowledgments

- [Burn](https://github.com/tracel-ai/burn) - Rust deep learning framework
- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [mlx-rs](https://github.com/oxideai/mlx-rs) - Rust bindings for MLX
