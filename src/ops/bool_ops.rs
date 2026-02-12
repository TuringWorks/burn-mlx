//! Boolean tensor operations for MLX backend.

use burn_tensor::{backend::ExecutionError, ops::BoolTensorOps, Shape, Slice, TensorData};
use mlx_rs::Array;
use mlx_rs::ops::indexing::{take_axis, take_along_axis};

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

    async fn bool_into_data(tensor: MlxTensorPrimitive) -> Result<TensorData, ExecutionError> {
        tensor.array.eval().expect("Failed to evaluate tensor");
        let shape = tensor.shape().to_vec();
        let data: Vec<bool> = tensor.array.as_slice().to_vec();
        Ok(TensorData::new(data, shape))
    }

    fn bool_device(_tensor: &MlxTensorPrimitive) -> MlxDevice {
        MlxDevice::Gpu
    }

    fn bool_to_device(tensor: MlxTensorPrimitive, _device: &MlxDevice) -> MlxTensorPrimitive {
        tensor
    }

    fn bool_empty(shape: Shape, device: &MlxDevice) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);

        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = Array::zeros::<bool>(&shape_i32).expect("Failed to create empty bool array");

        MlxTensorPrimitive::new(array)
    }

    fn bool_zeros(shape: Shape, device: &MlxDevice) -> MlxTensorPrimitive {
        Self::bool_empty(shape, device)
    }

    fn bool_ones(shape: Shape, device: &MlxDevice) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);

        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = Array::ones::<bool>(&shape_i32).expect("Failed to create ones bool array");

        MlxTensorPrimitive::new(array)
    }

    fn bool_reshape(tensor: MlxTensorPrimitive, shape: Shape) -> MlxTensorPrimitive {
        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = tensor.array.reshape(&shape_i32).expect("Failed to reshape");
        MlxTensorPrimitive::new(array)
    }

    fn bool_slice(tensor: MlxTensorPrimitive, slices: &[Slice]) -> MlxTensorPrimitive {
        let shape = tensor.shape().to_vec();
        let starts: Vec<i32> = slices.iter().enumerate().map(|(i, s)| {
            let range = s.to_range(*shape.get(i).unwrap_or(&0));
            range.start as i32
        }).collect();
        let stops: Vec<i32> = slices.iter().enumerate().map(|(i, s)| {
            let range = s.to_range(*shape.get(i).unwrap_or(&0));
            range.end as i32
        }).collect();
        let array = mlx_rs::ops::slice(&tensor.array, &starts, &stops, None)
            .expect("Failed to slice");
        MlxTensorPrimitive::new(array)
    }

    fn bool_slice_assign(
        tensor: MlxTensorPrimitive,
        slices: &[Slice],
        value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        let shape = tensor.shape().to_vec();
        let starts: Vec<i32> = slices.iter().enumerate().map(|(i, s)| {
            let range = s.to_range(*shape.get(i).unwrap_or(&0));
            range.start as i32
        }).collect();
        let stops: Vec<i32> = slices.iter().enumerate().map(|(i, s)| {
            let range = s.to_range(*shape.get(i).unwrap_or(&0));
            range.end as i32
        }).collect();
        let array = mlx_rs::ops::slice_update(&tensor.array, &value.array, &starts, &stops, None)
            .expect("Failed to slice_assign");
        MlxTensorPrimitive::new(array)
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

    fn bool_and(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::logical_and(&lhs.array, &rhs.array).expect("Failed to logical_and");
        MlxTensorPrimitive::new(array)
    }

    fn bool_or(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::logical_or(&lhs.array, &rhs.array).expect("Failed to logical_or");
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
        let axes_i32: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
        let array = mlx_rs::ops::flip(&tensor.array, &axes_i32[..])
            .expect("Failed to flip");
        MlxTensorPrimitive::new(array)
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

    fn bool_equal_elem(lhs: MlxTensorPrimitive, rhs: bool) -> MlxTensorPrimitive {
        let scalar = Array::from_slice(&[rhs], &[1]);
        let array = mlx_rs::ops::eq(&lhs.array, &scalar).expect("Failed to equal_elem");
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

    fn bool_mask_where(
        tensor: MlxTensorPrimitive,
        mask: MlxTensorPrimitive,
        value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::r#where(&mask.array, &value.array, &tensor.array)
            .expect("Failed to mask_where");
        MlxTensorPrimitive::new(array)
    }

    fn bool_mask_fill(
        tensor: MlxTensorPrimitive,
        mask: MlxTensorPrimitive,
        value: bool,
    ) -> MlxTensorPrimitive {
        let fill_val = Array::from_slice(&[value], &[1]);
        let fill_broadcast = mlx_rs::ops::broadcast_to(&fill_val, tensor.array.shape())
            .expect("Failed to broadcast");
        let array = mlx_rs::ops::r#where(&mask.array, &fill_broadcast, &tensor.array)
            .expect("Failed to mask_fill");
        MlxTensorPrimitive::new(array)
    }

    fn bool_gather(
        dim: usize,
        tensor: MlxTensorPrimitive,
        indices: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        let array = take_along_axis(&tensor.array, &indices.array, dim as i32)
            .expect("Failed to gather");
        MlxTensorPrimitive::new(array)
    }

    fn bool_scatter_or(
        dim: usize,
        tensor: MlxTensorPrimitive,
        indices: MlxTensorPrimitive,
        value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        let array = tensor.array.put_along_axis(&indices.array, &value.array, dim as i32)
            .expect("Failed to scatter_or");
        MlxTensorPrimitive::new(array)
    }

    async fn bool_argwhere(_tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        // MLX argwhere may not be available in mlx-rs bindings
        let empty = mlx_rs::Array::zeros::<i32>(&[0, 1]).expect("Failed to create empty array");
        MlxTensorPrimitive::new(empty)
    }

    fn bool_repeat_dim(tensor: MlxTensorPrimitive, dim: usize, times: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::repeat_axis::<bool>(tensor.array, dim as i32, times as i32).expect("Failed to repeat");
        MlxTensorPrimitive::new(array)
    }

    fn bool_unfold(
        tensor: MlxTensorPrimitive,
        dim: usize,
        size: usize,
        step: usize,
    ) -> MlxTensorPrimitive {
        let shape = tensor.shape().to_vec();
        let dim_size = shape[dim];
        let num_windows = (dim_size - size) / step + 1;

        let mut window_indices = Vec::new();
        for w in 0..num_windows {
            let start = w * step;
            for i in 0..size {
                window_indices.push((start + i) as i32);
            }
        }

        let indices = Array::from_slice(&window_indices, &[(num_windows * size) as i32]);
        let gathered = take_axis(&tensor.array, &indices, dim as i32).expect("take");

        let mut new_shape: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        new_shape[dim] = num_windows as i32;
        new_shape.push(size as i32);
        let array = gathered.reshape(&new_shape).expect("reshape");

        MlxTensorPrimitive::new(array)
    }
}
