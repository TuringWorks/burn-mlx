//! Numeric tensor operations for MLX backend.

use crate::tensor::MlxTensor;

/// Numeric operations on f32 MLX tensors.
impl MlxTensor<f32> {
    /// Element-wise addition.
    pub fn add(&self, rhs: &MlxTensor<f32>) -> MlxTensor<f32> {
        let array = mlx_rs::ops::add(&self.array, &rhs.array)
            .expect("Failed to add arrays");
        MlxTensor::new(array, self.device)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, rhs: &MlxTensor<f32>) -> MlxTensor<f32> {
        let array = mlx_rs::ops::subtract(&self.array, &rhs.array)
            .expect("Failed to subtract arrays");
        MlxTensor::new(array, self.device)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, rhs: &MlxTensor<f32>) -> MlxTensor<f32> {
        let array = mlx_rs::ops::multiply(&self.array, &rhs.array)
            .expect("Failed to multiply arrays");
        MlxTensor::new(array, self.device)
    }

    /// Element-wise division.
    pub fn div(&self, rhs: &MlxTensor<f32>) -> MlxTensor<f32> {
        let array = mlx_rs::ops::divide(&self.array, &rhs.array)
            .expect("Failed to divide arrays");
        MlxTensor::new(array, self.device)
    }

    /// Matrix multiplication.
    pub fn matmul(&self, rhs: &MlxTensor<f32>) -> MlxTensor<f32> {
        let array = self.array.matmul(&rhs.array)
            .expect("Failed to matmul arrays");
        MlxTensor::new(array, self.device)
    }

    /// Sum along dimension.
    pub fn sum_dim(&self, dim: usize, keepdims: bool) -> MlxTensor<f32> {
        let array = mlx_rs::ops::sum_axis(&self.array, dim as i32, keepdims)
            .expect("Failed to sum array");
        MlxTensor::new(array, self.device)
    }

    /// Mean along dimension.
    pub fn mean_dim(&self, dim: usize) -> MlxTensor<f32> {
        let array = mlx_rs::ops::mean_axis(&self.array, dim as i32, true)
            .expect("Failed to mean array");
        MlxTensor::new(array, self.device)
    }

    /// Exponential.
    pub fn exp(&self) -> MlxTensor<f32> {
        let array = mlx_rs::ops::exp(&self.array)
            .expect("Failed to exp array");
        MlxTensor::new(array, self.device)
    }

    /// Natural logarithm.
    pub fn log(&self) -> MlxTensor<f32> {
        let array = mlx_rs::ops::log(&self.array)
            .expect("Failed to log array");
        MlxTensor::new(array, self.device)
    }

    /// Square root.
    pub fn sqrt(&self) -> MlxTensor<f32> {
        let array = mlx_rs::ops::sqrt(&self.array)
            .expect("Failed to sqrt array");
        MlxTensor::new(array, self.device)
    }

    /// Absolute value.
    pub fn abs(&self) -> MlxTensor<f32> {
        let array = mlx_rs::ops::abs(&self.array)
            .expect("Failed to abs array");
        MlxTensor::new(array, self.device)
    }

    /// Negation.
    pub fn neg(&self) -> MlxTensor<f32> {
        let array = mlx_rs::ops::negative(&self.array)
            .expect("Failed to neg array");
        MlxTensor::new(array, self.device)
    }
}
