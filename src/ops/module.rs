//! Module operations (linear, etc.) for MLX backend.

use crate::element::MlxElement;
use crate::tensor::MlxTensor;

/// Module operations on MLX tensors.
impl<E: MlxElement> MlxTensor<E> {
    /// Linear layer (matmul + optional bias).
    /// input: [batch, in_features]
    /// weight: [out_features, in_features]
    /// output: [batch, out_features]
    pub fn linear(&self, weight: &MlxTensor<E>, bias: Option<&MlxTensor<E>>) -> MlxTensor<E> {
        let weight_t = mlx_rs::ops::transpose(&weight.array, None)
            .expect("Failed to transpose weight");
        let out = self.array.matmul(&weight_t)
            .expect("Failed to matmul");

        let result = if let Some(b) = bias {
            mlx_rs::ops::add(&out, &b.array)
                .expect("Failed to add bias")
        } else {
            out
        };

        MlxTensor::new(result, self.device)
    }

    /// Layer normalization.
    pub fn layer_norm(&self, weight: Option<&MlxTensor<E>>, bias: Option<&MlxTensor<E>>, epsilon: f32) -> MlxTensor<E> {
        // Normalize over the last dimension
        let mean = mlx_rs::ops::mean(&self.array, &[-1], true)
            .expect("Failed to compute mean");
        let var = mlx_rs::ops::var(&self.array, &[-1], true, 0)
            .expect("Failed to compute var");

        let eps = mlx_rs::Array::from_f32(epsilon);
        let var_eps = mlx_rs::ops::add(&var, &eps).expect("Failed to add eps");
        let std = mlx_rs::ops::sqrt(&var_eps).expect("Failed to sqrt");

        let centered = mlx_rs::ops::subtract(&self.array, &mean).expect("Failed to subtract");
        let normalized = mlx_rs::ops::divide(&centered, &std).expect("Failed to divide");

        let scaled = if let Some(w) = weight {
            mlx_rs::ops::multiply(&normalized, &w.array).expect("Failed to scale")
        } else {
            normalized
        };

        let result = if let Some(b) = bias {
            mlx_rs::ops::add(&scaled, &b.array).expect("Failed to add bias")
        } else {
            scaled
        };

        MlxTensor::new(result, self.device)
    }

    /// Embedding lookup.
    pub fn embedding(&self, indices: &MlxTensor<i32>) -> MlxTensor<E> {
        let array = mlx_rs::ops::indexing::take(&self.array, &indices.array, 0)
            .expect("Failed to embedding lookup");
        MlxTensor::new(array, self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MlxDevice;

    #[test]
    fn test_linear() {
        let input = MlxTensor::<f32>::ones(&[2, 4], MlxDevice::Gpu);
        let weight = MlxTensor::<f32>::ones(&[3, 4], MlxDevice::Gpu);
        let output = input.linear(&weight, None);
        assert_eq!(output.shape(), vec![2, 3]);
    }
}
