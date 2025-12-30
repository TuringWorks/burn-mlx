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
- Shape: reshape, transpose, permute, expand, slice

### Activation Functions
- ReLU, Sigmoid, Tanh, GELU, LeakyReLU
- Softmax, LogSoftmax, HardSigmoid

### Neural Network Layers
- Conv1d, Conv2d (with proper NCHW layout handling)
- Embedding lookup
- Pooling (placeholder implementations)

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
- Some operations use placeholder implementations (pooling backward passes)
- Quantization support is minimal

## License

Apache-2.0

## Acknowledgments

- [Burn](https://github.com/tracel-ai/burn) - Rust deep learning framework
- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- [mlx-rs](https://github.com/oxideai/mlx-rs) - Rust bindings for MLX
