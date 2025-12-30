//! Base tensor operations for MLX backend.

use crate::tensor::MlxTensor;

/// Base operations on f32 MLX tensors.
impl MlxTensor<f32> {
    /// Reshape this tensor.
    pub fn reshape_to(&self, shape: &[i32]) -> Self {
        let array = self.array.reshape(shape)
            .expect("Failed to reshape array");
        MlxTensor::new(array, self.device)
    }

    /// Transpose this tensor (reverses all axes).
    pub fn transpose_all(&self) -> Self {
        let array = mlx_rs::ops::transpose(&self.array)
            .expect("Failed to transpose array");
        MlxTensor::new(array, self.device)
    }

    /// Expand tensor to a new shape.
    pub fn broadcast_to(&self, shape: &[i32]) -> Self {
        let array = mlx_rs::ops::broadcast_to(&self.array, shape)
            .expect("Failed to broadcast array");
        MlxTensor::new(array, self.device)
    }
}

/// Concatenate tensors along a dimension.
pub fn concat(tensors: &[&MlxTensor<f32>], dim: usize) -> MlxTensor<f32> {
    if tensors.is_empty() {
        panic!("Cannot concatenate empty list of tensors");
    }

    let device = tensors[0].device;
    let arrays: Vec<&mlx_rs::Array> = tensors.iter().map(|t| &t.array).collect();

    let array = mlx_rs::ops::concatenate_axis(&arrays, dim as i32)
        .expect("Failed to concatenate arrays");
    MlxTensor::new(array, device)
}
