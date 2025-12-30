//! Additional ops implementations for MLX backend.

use burn_tensor::{
    ops::{ActivationOps, QTensorOps, TransactionOps},
    quantization::QuantizationScheme,
    Shape, TensorData,
};
use mlx_rs::Array;

use crate::backend::{Mlx, MlxQuantizedTensorPrimitive, MlxTensorPrimitive};
use crate::device::MlxDevice;

// ActivationOps - most methods have default implementations
impl ActivationOps<Self> for Mlx {
    fn relu(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let zero = Array::from_f32(0.0);
        let array = mlx_rs::ops::maximum(&tensor.array, &zero).expect("relu");
        MlxTensorPrimitive::new(array)
    }

    fn sigmoid(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::sigmoid(&tensor.array).expect("sigmoid");
        MlxTensorPrimitive::new(array)
    }

    fn gelu(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Simplified: x * sigmoid(1.702 * x)
        let coef = Array::from_f32(1.702);
        let scaled = mlx_rs::ops::multiply(&tensor.array, &coef).expect("multiply");
        let sigmoid = mlx_rs::ops::sigmoid(&scaled).expect("sigmoid");
        let array = mlx_rs::ops::multiply(&tensor.array, &sigmoid).expect("multiply");
        MlxTensorPrimitive::new(array)
    }

    fn leaky_relu(tensor: MlxTensorPrimitive, negative_slope: f32) -> MlxTensorPrimitive {
        let array = mlx_rs::nn::leaky_relu(&tensor.array, negative_slope).expect("leaky_relu");
        MlxTensorPrimitive::new(array)
    }

    fn hard_sigmoid(tensor: MlxTensorPrimitive, alpha: f32, beta: f32) -> MlxTensorPrimitive {
        let alpha_arr = Array::from_f32(alpha);
        let beta_arr = Array::from_f32(beta);
        let scaled = mlx_rs::ops::multiply(&tensor.array, &alpha_arr).expect("multiply");
        let shifted = mlx_rs::ops::add(&scaled, &beta_arr).expect("add");
        let zero = Array::from_f32(0.0);
        let one = Array::from_f32(1.0);
        let array = mlx_rs::ops::clip(&shifted, (&zero, &one)).expect("clip");
        MlxTensorPrimitive::new(array)
    }

    fn log_sigmoid(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let sig = mlx_rs::ops::sigmoid(&tensor.array).expect("sigmoid");
        let array = mlx_rs::ops::log(&sig).expect("log");
        MlxTensorPrimitive::new(array)
    }

    fn prelu(tensor: MlxTensorPrimitive, alpha: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let zero = Array::from_f32(0.0);
        let pos = mlx_rs::ops::maximum(&tensor.array, &zero).expect("max");
        let neg = mlx_rs::ops::minimum(&tensor.array, &zero).expect("min");
        let scaled_neg = mlx_rs::ops::multiply(&alpha.array, &neg).expect("multiply");
        let array = mlx_rs::ops::add(&pos, &scaled_neg).expect("add");
        MlxTensorPrimitive::new(array)
    }

    fn gelu_backward(x: MlxTensorPrimitive, grad: MlxTensorPrimitive) -> MlxTensorPrimitive {
        // Backward pass for GELU - placeholder
        grad
    }

    fn relu_backward(x: MlxTensorPrimitive, grad: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let zero = Array::from_f32(0.0);
        let mask = mlx_rs::ops::gt(&x.array, &zero).expect("greater");
        let mask_float = mask.as_type::<f32>().expect("cast");
        let array = mlx_rs::ops::multiply(&grad.array, &mask_float).expect("multiply");
        MlxTensorPrimitive::new(array)
    }
}

// QTensorOps - Quantization operations (placeholder)
impl QTensorOps<Self> for Mlx {
    fn q_from_data(data: TensorData, device: &MlxDevice) -> MlxQuantizedTensorPrimitive {
        let tensor = <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_from_data(
            data.convert::<f32>(),
            device,
        );
        MlxQuantizedTensorPrimitive {
            tensor,
            scheme: crate::backend::QuantizationScheme::None,
        }
    }

    fn quantize(
        tensor: MlxTensorPrimitive,
        scheme: &QuantizationScheme,
        qparams: burn_tensor::quantization::QuantizationParametersPrimitive<Self>,
    ) -> MlxQuantizedTensorPrimitive {
        MlxQuantizedTensorPrimitive {
            tensor,
            scheme: crate::backend::QuantizationScheme::None,
        }
    }

    fn dequantize(tensor: MlxQuantizedTensorPrimitive) -> MlxTensorPrimitive {
        tensor.tensor
    }

    fn q_device(tensor: &MlxQuantizedTensorPrimitive) -> MlxDevice {
        MlxDevice::Gpu
    }

    fn q_to_device(
        tensor: MlxQuantizedTensorPrimitive,
        device: &MlxDevice,
    ) -> MlxQuantizedTensorPrimitive {
        tensor
    }

    fn q_reshape(tensor: MlxQuantizedTensorPrimitive, shape: Shape) -> MlxQuantizedTensorPrimitive {
        let reshaped = <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_reshape(
            tensor.tensor,
            shape,
        );
        MlxQuantizedTensorPrimitive {
            tensor: reshaped,
            scheme: tensor.scheme,
        }
    }

    async fn q_into_data(tensor: MlxQuantizedTensorPrimitive) -> TensorData {
        <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_into_data(tensor.tensor).await
    }

    fn q_swap_dims(
        tensor: MlxQuantizedTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> MlxQuantizedTensorPrimitive {
        let swapped = <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_swap_dims(
            tensor.tensor,
            dim1,
            dim2,
        );
        MlxQuantizedTensorPrimitive {
            tensor: swapped,
            scheme: tensor.scheme,
        }
    }

    fn q_permute(
        tensor: MlxQuantizedTensorPrimitive,
        axes: &[usize],
    ) -> MlxQuantizedTensorPrimitive {
        let permuted = <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_permute(
            tensor.tensor,
            axes,
        );
        MlxQuantizedTensorPrimitive {
            tensor: permuted,
            scheme: tensor.scheme,
        }
    }

    fn q_flip(
        tensor: MlxQuantizedTensorPrimitive,
        axes: &[usize],
    ) -> MlxQuantizedTensorPrimitive {
        let flipped = <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_flip(
            tensor.tensor,
            axes,
        );
        MlxQuantizedTensorPrimitive {
            tensor: flipped,
            scheme: tensor.scheme,
        }
    }

    fn q_select(
        tensor: MlxQuantizedTensorPrimitive,
        dim: usize,
        indices: MlxTensorPrimitive,
    ) -> MlxQuantizedTensorPrimitive {
        let selected = <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_select(
            tensor.tensor,
            dim,
            indices,
        );
        MlxQuantizedTensorPrimitive {
            tensor: selected,
            scheme: tensor.scheme,
        }
    }

    fn q_slice(
        tensor: MlxQuantizedTensorPrimitive,
        ranges: &[std::ops::Range<usize>],
    ) -> MlxQuantizedTensorPrimitive {
        let sliced = <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_slice(
            tensor.tensor,
            ranges,
        );
        MlxQuantizedTensorPrimitive {
            tensor: sliced,
            scheme: tensor.scheme,
        }
    }

    fn q_expand(
        tensor: MlxQuantizedTensorPrimitive,
        shape: Shape,
    ) -> MlxQuantizedTensorPrimitive {
        let expanded = <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_expand(
            tensor.tensor,
            shape,
        );
        MlxQuantizedTensorPrimitive {
            tensor: expanded,
            scheme: tensor.scheme,
        }
    }
}

// TransactionOps - transaction batching (default impl)
impl TransactionOps<Self> for Mlx {}
