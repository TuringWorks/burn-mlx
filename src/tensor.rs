//! MLX tensor wrapper for Burn.

use crate::device::MlxDevice;
use crate::element::MlxElement;
use mlx_rs::Array;
use std::marker::PhantomData;

/// MLX tensor primitive.
#[derive(Debug, Clone)]
pub struct MlxTensor<E: MlxElement> {
    /// The underlying MLX array.
    pub(crate) array: Array,
    /// The device this tensor lives on.
    pub(crate) device: MlxDevice,
    /// Phantom data for element type.
    pub(crate) _element: PhantomData<E>,
}

impl<E: MlxElement> MlxTensor<E> {
    /// Create a new MLX tensor from an array.
    pub fn new(array: Array, device: MlxDevice) -> Self {
        Self {
            array,
            device,
            _element: PhantomData,
        }
    }

    /// Get the underlying MLX array.
    pub fn array(&self) -> &Array {
        &self.array
    }

    /// Get the device.
    pub fn device(&self) -> &MlxDevice {
        &self.device
    }

    /// Get the shape.
    pub fn shape(&self) -> Vec<usize> {
        self.array.shape().iter().map(|&s| s as usize).collect()
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.array.ndim()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.array.size()
    }

    /// Evaluate the lazy computation graph.
    pub fn eval(&self) -> Result<(), mlx_rs::error::Exception> {
        self.array.eval()
    }
}

impl MlxTensor<f32> {
    /// Create a zeros tensor.
    pub fn zeros(shape: &[i32], device: MlxDevice) -> Self {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);
        let array = Array::zeros::<f32>(shape).expect("Failed to create zeros array");
        Self::new(array, device)
    }

    /// Create a ones tensor.
    pub fn ones(shape: &[i32], device: MlxDevice) -> Self {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);
        let array = Array::ones::<f32>(shape).expect("Failed to create ones array");
        Self::new(array, device)
    }

    /// Create a tensor from a slice.
    pub fn from_slice(data: &[f32], shape: &[i32], device: MlxDevice) -> Self {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);
        let array = Array::from_slice(data, shape);
        Self::new(array, device)
    }

    /// Element-wise addition.
    pub fn add(&self, other: &Self) -> Self {
        let array = mlx_rs::ops::add(&self.array, &other.array).expect("Failed to add");
        Self::new(array, self.device.clone())
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        let array = mlx_rs::ops::subtract(&self.array, &other.array).expect("Failed to subtract");
        Self::new(array, self.device.clone())
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        let array = mlx_rs::ops::multiply(&self.array, &other.array).expect("Failed to multiply");
        Self::new(array, self.device.clone())
    }

    /// Element-wise division.
    pub fn div(&self, other: &Self) -> Self {
        let array = mlx_rs::ops::divide(&self.array, &other.array).expect("Failed to divide");
        Self::new(array, self.device.clone())
    }

    /// Matrix multiplication.
    pub fn matmul(&self, other: &Self) -> Self {
        let array = mlx_rs::ops::matmul(&self.array, &other.array).expect("Failed to matmul");
        Self::new(array, self.device.clone())
    }

    /// ReLU activation.
    pub fn relu(&self) -> Self {
        let zero = Array::from_f32(0.0);
        let array = mlx_rs::ops::maximum(&self.array, &zero).expect("Failed to relu");
        Self::new(array, self.device.clone())
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self) -> Self {
        let array = mlx_rs::ops::sigmoid(&self.array).expect("Failed to sigmoid");
        Self::new(array, self.device.clone())
    }

    /// Tanh activation.
    pub fn tanh_act(&self) -> Self {
        let array = mlx_rs::ops::tanh(&self.array).expect("Failed to tanh");
        Self::new(array, self.device.clone())
    }

    /// Softmax along the last dimension.
    pub fn softmax(&self) -> Self {
        let array = mlx_rs::ops::softmax(&self.array, None).expect("Failed to softmax");
        Self::new(array, self.device.clone())
    }

    /// Sum along a dimension.
    pub fn sum_dim(&self, dim: i32, keepdim: bool) -> Self {
        let array = mlx_rs::ops::sum_axis(&self.array, dim, keepdim).expect("Failed to sum");
        Self::new(array, self.device.clone())
    }

    /// Mean along a dimension.
    pub fn mean_dim(&self, dim: i32) -> Self {
        let array = mlx_rs::ops::mean_axis(&self.array, dim, true).expect("Failed to mean");
        Self::new(array, self.device.clone())
    }

    /// Exponential.
    pub fn exp(&self) -> Self {
        let array = mlx_rs::ops::exp(&self.array).expect("Failed to exp");
        Self::new(array, self.device.clone())
    }

    /// Natural log.
    pub fn log(&self) -> Self {
        let array = mlx_rs::ops::log(&self.array).expect("Failed to log");
        Self::new(array, self.device.clone())
    }

    /// Square root.
    pub fn sqrt(&self) -> Self {
        let array = mlx_rs::ops::sqrt(&self.array).expect("Failed to sqrt");
        Self::new(array, self.device.clone())
    }

    /// Absolute value.
    pub fn abs(&self) -> Self {
        let array = mlx_rs::ops::abs(&self.array).expect("Failed to abs");
        Self::new(array, self.device.clone())
    }

    /// Negation.
    pub fn neg(&self) -> Self {
        let array = mlx_rs::ops::negative(&self.array).expect("Failed to neg");
        Self::new(array, self.device.clone())
    }
}

// Additional methods (reshape_to, transpose_all, broadcast_to) are defined in ops/base.rs

// SAFETY: MLX Array implements Send. We need Sync for Burn's trait bounds.
unsafe impl<E: MlxElement> Send for MlxTensor<E> {}
unsafe impl<E: MlxElement> Sync for MlxTensor<E> {}
