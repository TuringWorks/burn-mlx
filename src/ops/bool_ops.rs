//! Boolean tensor operations for MLX backend.

use burn_tensor::{ops::BoolTensorOps, Shape, TensorData};
use mlx_rs::Array;
use std::ops::Range;

use crate::backend::{Mlx, MlxTensorPrimitive};
use crate::device::MlxDevice;

impl BoolTensorOps<Self> for Mlx {
    fn bool_from_data(data: TensorData, device: &MlxDevice) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);

        let shape: Vec<i32> = data.shape.iter().map(|&s| s as i32).collect();
        let values: Vec<bool> = data.to_vec().expect("Failed to convert data to bool vec");
        let array = Array::from_slice(&values, &shape);

        MlxTensorPrimitive::new(array)
    }

    async fn bool_into_data(tensor: MlxTensorPrimitive) -> TensorData {
        tensor.array.eval().expect("Failed to evaluate tensor");
        let shape = tensor.shape().to_vec();
        let data: Vec<bool> = tensor.array.as_slice().to_vec();
        TensorData::new(data, shape)
    }

    fn bool_device(tensor: &MlxTensorPrimitive) -> MlxDevice {
        MlxDevice::Gpu
    }

    fn bool_to_device(tensor: MlxTensorPrimitive, device: &MlxDevice) -> MlxTensorPrimitive {
        let _ = device;
        tensor
    }

    fn bool_empty(shape: Shape, device: &MlxDevice) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);

        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = Array::zeros::<bool>(&shape_i32).expect("Failed to create empty bool array");

        MlxTensorPrimitive::new(array)
    }

    fn bool_reshape(tensor: MlxTensorPrimitive, shape: Shape) -> MlxTensorPrimitive {
        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = tensor.array.reshape(&shape_i32).expect("Failed to reshape");
        MlxTensorPrimitive::new(array)
    }

    fn bool_slice(tensor: MlxTensorPrimitive, ranges: &[Range<usize>]) -> MlxTensorPrimitive {
        // Placeholder - need proper slice implementation
        tensor
    }

    fn bool_slice_assign(
        tensor: MlxTensorPrimitive,
        ranges: &[Range<usize>],
        value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        tensor
    }

    fn bool_into_int(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = tensor.array.as_type::<i32>().expect("Failed to cast to int");
        MlxTensorPrimitive::new(array)
    }

    fn bool_into_float(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = tensor.array.as_type::<f32>().expect("Failed to cast to float");
        MlxTensorPrimitive::new(array)
    }

    fn bool_not(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::logical_not(&tensor.array).expect("Failed to logical_not");
        MlxTensorPrimitive::new(array)
    }

    fn bool_swap_dims(tensor: MlxTensorPrimitive, dim1: usize, dim2: usize) -> MlxTensorPrimitive {
        let ndim = tensor.shape().len();
        let mut axes: Vec<i32> = (0..ndim as i32).collect();
        axes.swap(dim1, dim2);
        let array = mlx_rs::ops::transpose_axes(&tensor.array, &axes).expect("Failed to swap dims");
        MlxTensorPrimitive::new(array)
    }

    fn bool_permute(tensor: MlxTensorPrimitive, axes: &[usize]) -> MlxTensorPrimitive {
        let axes_i32: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
        let array = mlx_rs::ops::transpose_axes(&tensor.array, &axes_i32).expect("Failed to permute");
        MlxTensorPrimitive::new(array)
    }

    fn bool_flip(tensor: MlxTensorPrimitive, axes: &[usize]) -> MlxTensorPrimitive {
        // Placeholder - MLX doesn't have direct flip
        tensor
    }

    fn bool_expand(tensor: MlxTensorPrimitive, shape: Shape) -> MlxTensorPrimitive {
        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = mlx_rs::ops::broadcast_to(&tensor.array, &shape_i32).expect("Failed to expand");
        MlxTensorPrimitive::new(array)
    }

    fn bool_equal(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::eq(&lhs.array, &rhs.array).expect("Failed to equal");
        MlxTensorPrimitive::new(array)
    }

    fn bool_any(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::any(&tensor.array, false).expect("Failed to any");
        MlxTensorPrimitive::new(array)
    }

    fn bool_any_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::any_axis(&tensor.array, dim as i32, true).expect("Failed to any_dim");
        MlxTensorPrimitive::new(array)
    }

    fn bool_all(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::all(&tensor.array, false).expect("Failed to all");
        MlxTensorPrimitive::new(array)
    }

    fn bool_all_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::all_axis(&tensor.array, dim as i32, true).expect("Failed to all_dim");
        MlxTensorPrimitive::new(array)
    }

    async fn bool_argwhere(_tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        // MLX argwhere may not be available in mlx-rs bindings
        // Placeholder: return empty tensor
        let empty = mlx_rs::Array::zeros::<i32>(&[0, 1]).expect("Failed to create empty array");
        MlxTensorPrimitive::new(empty)
    }

    fn bool_repeat_dim(tensor: MlxTensorPrimitive, dim: usize, times: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::repeat_axis::<bool>(tensor.array, dim as i32, times as i32).expect("Failed to repeat");
        MlxTensorPrimitive::new(array)
    }
}
