//! Burn backend for Apple MLX framework.
//!
//! This crate provides a full Burn backend implementation that leverages
//! Apple's MLX framework for high-performance machine learning on Apple Silicon.
//!
//! ## Features
//!
//! - **Full Burn Backend**: Implements `FloatTensorOps`, `IntTensorOps`,
//!   `BoolTensorOps`, `ModuleOps`, and `ActivationOps`
//! - **Unified Memory**: Zero-copy data sharing between CPU and GPU
//! - **Lazy Evaluation**: Automatic compute graph optimization
//! - **Apple Silicon Optimized**: Native support for M-series Neural Engine
//!
//! ## Requirements
//!
//! - macOS 13.3+ (Ventura)
//! - Apple Silicon (M1/M2/M3/M4/M5)
//! - Rust 1.82+
//!
//! ## Usage
//!
//! ```ignore
//! use burn::prelude::*;
//! use burn_mlx::Mlx;
//!
//! // Use MLX as the backend
//! type Backend = Mlx;
//!
//! // Create tensors
//! let device = <Backend as burn::tensor::backend::Backend>::Device::default();
//! let x: Tensor<Backend, 2> = Tensor::ones([2, 3], &device);
//! let y: Tensor<Backend, 2> = Tensor::ones([3, 4], &device);
//! let z = x.matmul(y);
//! ```
//!
//! ## With Autodiff
//!
//! ```ignore
//! use burn::prelude::*;
//! use burn_autodiff::Autodiff;
//! use burn_mlx::Mlx;
//!
//! type TrainBackend = Autodiff<Mlx>;
//!
//! // Now you can use automatic differentiation with MLX
//! ```

mod device;
mod element;
mod tensor;
mod backend;
mod ops;

// Public exports
pub use backend::{Mlx, MlxTensorPrimitive, MlxQuantizedTensorPrimitive};
pub use device::MlxDevice;
pub use element::MlxElement;
pub use tensor::MlxTensor;

/// Re-export mlx-rs types for advanced usage.
pub mod mlx {
    pub use mlx_rs::*;
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::{backend::Backend, Tensor, TensorData, Shape};

    #[test]
    fn test_device_creation() {
        let _device = MlxDevice::Gpu;
        let _cpu = MlxDevice::Cpu;
    }

    #[test]
    fn test_tensor_creation_raw() {
        let tensor = MlxTensor::<f32>::ones(&[2, 3], MlxDevice::Gpu);
        assert_eq!(tensor.shape(), vec![2, 3]);
    }

    #[test]
    fn test_tensor_operations_raw() {
        let a = MlxTensor::<f32>::ones(&[2, 3], MlxDevice::Gpu);
        let b = MlxTensor::<f32>::ones(&[2, 3], MlxDevice::Gpu);
        let c = a.add(&b);
        assert_eq!(c.shape(), vec![2, 3]);
    }

    #[test]
    fn test_matmul_raw() {
        let a = MlxTensor::<f32>::ones(&[2, 3], MlxDevice::Gpu);
        let b = MlxTensor::<f32>::ones(&[3, 4], MlxDevice::Gpu);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), vec![2, 4]);
    }

    #[test]
    fn test_burn_backend_tensor_creation() {
        let device = MlxDevice::default();

        // Test from_data
        let data = TensorData::from([1.0f32, 2.0, 3.0, 4.0]);
        let tensor: Tensor<Mlx, 1> = Tensor::from_data(data, &device);
        assert_eq!(tensor.shape().dims, [4]);
    }

    #[test]
    fn test_burn_backend_arithmetic() {
        let device = MlxDevice::default();

        let a: Tensor<Mlx, 2> = Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0]], &device);
        let b: Tensor<Mlx, 2> = Tensor::from_data([[5.0f32, 6.0], [7.0, 8.0]], &device);

        let sum = a.clone() + b.clone();
        let diff = a.clone() - b.clone();
        let prod = a.clone() * b.clone();
        let quot = a / b;

        assert_eq!(sum.shape().dims, [2, 2]);
        assert_eq!(diff.shape().dims, [2, 2]);
        assert_eq!(prod.shape().dims, [2, 2]);
        assert_eq!(quot.shape().dims, [2, 2]);
    }

    #[test]
    fn test_burn_backend_matmul() {
        let device = MlxDevice::default();

        let a: Tensor<Mlx, 2> = Tensor::from_data([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let b: Tensor<Mlx, 2> = Tensor::from_data([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &device);

        let c = a.matmul(b);
        assert_eq!(c.shape().dims, [2, 2]);
    }

    #[test]
    fn test_burn_backend_activations() {
        let device = MlxDevice::default();

        let x: Tensor<Mlx, 1> = Tensor::from_data([-1.0f32, 0.0, 1.0, 2.0], &device);

        let relu = burn_tensor::activation::relu(x.clone());
        let sigmoid = burn_tensor::activation::sigmoid(x.clone());
        let softmax = burn_tensor::activation::softmax(x.clone(), 0);

        assert_eq!(relu.shape().dims, [4]);
        assert_eq!(sigmoid.shape().dims, [4]);
        assert_eq!(softmax.shape().dims, [4]);
    }
}
