//! Additional ops implementations for MLX backend.

use burn_tensor::{
    backend::ExecutionError,
    ops::{ActivationOps, QTensorOps, TransactionOps},
    quantization::QuantScheme,
    Shape, Slice, TensorData,
};

use crate::backend::{Mlx, MlxQuantizedTensorPrimitive, MlxTensorPrimitive};
use crate::device::MlxDevice;
use crate::element::FloatMlxElement;

// ActivationOps - most methods have default implementations
impl<F: FloatMlxElement> ActivationOps<Self> for Mlx<F> {
    fn relu(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let zero = F::f64_scalar_array(0.0);
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
        let coef = F::f64_scalar_array(1.702);
        let scaled = mlx_rs::ops::multiply(&tensor.array, &coef).expect("multiply");
        let sigmoid = mlx_rs::ops::sigmoid(&scaled).expect("sigmoid");
        let array = mlx_rs::ops::multiply(&tensor.array, &sigmoid).expect("multiply");
        MlxTensorPrimitive::new(array)
    }

    fn leaky_relu(tensor: MlxTensorPrimitive, negative_slope: F) -> MlxTensorPrimitive {
        let slope_f32 = num_traits::ToPrimitive::to_f32(&negative_slope).unwrap();
        let array = mlx_rs::nn::leaky_relu(&tensor.array, slope_f32).expect("leaky_relu");
        MlxTensorPrimitive::new(array)
    }

    fn hard_sigmoid(tensor: MlxTensorPrimitive, alpha: F, beta: F) -> MlxTensorPrimitive {
        let alpha_arr = F::scalar_array(alpha);
        let beta_arr = F::scalar_array(beta);
        let scaled = mlx_rs::ops::multiply(&tensor.array, &alpha_arr).expect("multiply");
        let shifted = mlx_rs::ops::add(&scaled, &beta_arr).expect("add");
        let zero = F::f64_scalar_array(0.0);
        let one = F::f64_scalar_array(1.0);
        let array = mlx_rs::ops::clip(&shifted, (&zero, &one)).expect("clip");
        MlxTensorPrimitive::new(array)
    }

    fn log_sigmoid(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let sig = mlx_rs::ops::sigmoid(&tensor.array).expect("sigmoid");
        let array = mlx_rs::ops::log(&sig).expect("log");
        MlxTensorPrimitive::new(array)
    }

    fn prelu(tensor: MlxTensorPrimitive, alpha: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let zero = F::f64_scalar_array(0.0);
        let pos = mlx_rs::ops::maximum(&tensor.array, &zero).expect("max");
        let neg = mlx_rs::ops::minimum(&tensor.array, &zero).expect("min");
        let scaled_neg = mlx_rs::ops::multiply(&alpha.array, &neg).expect("multiply");
        let array = mlx_rs::ops::add(&pos, &scaled_neg).expect("add");
        MlxTensorPrimitive::new(array)
    }

    fn gelu_backward(_x: MlxTensorPrimitive, grad: MlxTensorPrimitive) -> MlxTensorPrimitive {
        grad
    }

    fn relu_backward(x: MlxTensorPrimitive, grad: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let zero = F::f64_scalar_array(0.0);
        let mask = mlx_rs::ops::gt(&x.array, &zero).expect("greater");
        let mask_float = F::cast_array(&mask);
        let array = mlx_rs::ops::multiply(&grad.array, &mask_float).expect("multiply");
        MlxTensorPrimitive::new(array)
    }
}

// QTensorOps - Quantization operations (placeholder)
impl<F: FloatMlxElement> QTensorOps<Self> for Mlx<F> {
    fn q_from_data(data: TensorData, device: &MlxDevice) -> MlxQuantizedTensorPrimitive {
        let tensor = <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_from_data(
            data.convert::<F>(),
            device,
        );
        MlxQuantizedTensorPrimitive {
            tensor,
            scheme: QuantScheme::default(),
        }
    }

    fn quantize(
        tensor: MlxTensorPrimitive,
        _scheme: &QuantScheme,
        _qparams: burn_tensor::quantization::QuantizationParametersPrimitive<Self>,
    ) -> MlxQuantizedTensorPrimitive {
        MlxQuantizedTensorPrimitive {
            tensor,
            scheme: QuantScheme::default(),
        }
    }

    fn dequantize(tensor: MlxQuantizedTensorPrimitive) -> MlxTensorPrimitive {
        tensor.tensor
    }

    fn q_device(_tensor: &MlxQuantizedTensorPrimitive) -> MlxDevice {
        MlxDevice::Gpu
    }

    fn q_to_device(
        tensor: MlxQuantizedTensorPrimitive,
        _device: &MlxDevice,
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

    async fn q_into_data(tensor: MlxQuantizedTensorPrimitive) -> Result<TensorData, ExecutionError> {
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
        slices: &[Slice],
    ) -> MlxQuantizedTensorPrimitive {
        let sliced = <Self as burn_tensor::ops::FloatTensorOps<Self>>::float_slice(
            tensor.tensor,
            slices,
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
impl<F: FloatMlxElement> TransactionOps<Self> for Mlx<F> {}
