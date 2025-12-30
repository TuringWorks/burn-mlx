//! Activation function operations for MLX backend.

use crate::tensor::MlxTensor;

/// Activation operations on f32 MLX tensors.
impl MlxTensor<f32> {
    /// ReLU activation.
    pub fn relu(&self) -> MlxTensor<f32> {
        let zero = mlx_rs::Array::from_f32(0.0);
        let array = mlx_rs::ops::maximum(&self.array, &zero)
            .expect("Failed to relu array");
        MlxTensor::new(array, self.device)
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self) -> MlxTensor<f32> {
        let array = mlx_rs::ops::sigmoid(&self.array)
            .expect("Failed to sigmoid array");
        MlxTensor::new(array, self.device)
    }

    /// Tanh activation.
    pub fn tanh_act(&self) -> MlxTensor<f32> {
        let array = mlx_rs::ops::tanh(&self.array)
            .expect("Failed to tanh array");
        MlxTensor::new(array, self.device)
    }

    /// Softmax activation (along last dimension).
    pub fn softmax(&self) -> MlxTensor<f32> {
        let array = mlx_rs::ops::softmax(&self.array, None)
            .expect("Failed to softmax array");
        MlxTensor::new(array, self.device)
    }
}
