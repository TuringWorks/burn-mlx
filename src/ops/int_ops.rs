//! Integer tensor operations for MLX backend.

use burn_tensor::{ops::IntTensorOps, Distribution, Shape, TensorData};
use mlx_rs::Array;
use mlx_rs::ops::indexing::{argmax_axis, argmin_axis, take_axis, take_along_axis};
use std::ops::Range;

use crate::backend::{Mlx, MlxTensorPrimitive};
use crate::device::MlxDevice;

impl IntTensorOps<Self> for Mlx {
    fn int_from_data(data: TensorData, device: &MlxDevice) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);

        let shape: Vec<i32> = data.shape.iter().map(|&s| s as i32).collect();
        let values: Vec<i32> = data.to_vec().expect("Failed to convert data to i32 vec");
        let array = Array::from_slice(&values, &shape);

        MlxTensorPrimitive::new(array)
    }

    async fn int_into_data(tensor: MlxTensorPrimitive) -> TensorData {
        tensor.array.eval().expect("Failed to evaluate tensor");
        let shape = tensor.shape().to_vec();
        let data: Vec<i32> = tensor.array.as_slice().to_vec();
        TensorData::new(data, shape)
    }

    fn int_device(tensor: &MlxTensorPrimitive) -> MlxDevice {
        MlxDevice::Gpu
    }

    fn int_to_device(tensor: MlxTensorPrimitive, device: &MlxDevice) -> MlxTensorPrimitive {
        let _ = device;
        tensor
    }

    fn int_empty(shape: Shape, device: &MlxDevice) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);
        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = Array::zeros::<i32>(&shape_i32).expect("Failed to create empty int array");
        MlxTensorPrimitive::new(array)
    }

    fn int_zeros(shape: Shape, device: &MlxDevice) -> MlxTensorPrimitive {
        Self::int_empty(shape, device)
    }

    fn int_ones(shape: Shape, device: &MlxDevice) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);
        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = Array::ones::<i32>(&shape_i32).expect("Failed to create ones int array");
        MlxTensorPrimitive::new(array)
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &MlxDevice,
    ) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);
        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();

        let array = match distribution {
            Distribution::Uniform(low, high) => {
                mlx_rs::random::randint::<i32, i32>(low as i32, high as i32, &shape_i32, None)
                    .expect("Failed to create uniform random int array")
            }
            _ => {
                mlx_rs::random::randint::<i32, i32>(0, 100, &shape_i32, None)
                    .expect("Failed to create random int array")
            }
        };
        MlxTensorPrimitive::new(array)
    }

    fn int_add(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::add(&lhs.array, &rhs.array).expect("Failed to add");
        MlxTensorPrimitive::new(array)
    }

    fn int_add_scalar(lhs: MlxTensorPrimitive, rhs: i32) -> MlxTensorPrimitive {
        let scalar = Array::from_int(rhs);
        let array = mlx_rs::ops::add(&lhs.array, &scalar).expect("Failed to add scalar");
        MlxTensorPrimitive::new(array)
    }

    fn int_sub(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::subtract(&lhs.array, &rhs.array).expect("Failed to subtract");
        MlxTensorPrimitive::new(array)
    }

    fn int_sub_scalar(lhs: MlxTensorPrimitive, rhs: i32) -> MlxTensorPrimitive {
        let scalar = Array::from_int(rhs);
        let array = mlx_rs::ops::subtract(&lhs.array, &scalar).expect("Failed to subtract scalar");
        MlxTensorPrimitive::new(array)
    }

    fn int_mul(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::multiply(&lhs.array, &rhs.array).expect("Failed to multiply");
        MlxTensorPrimitive::new(array)
    }

    fn int_mul_scalar(lhs: MlxTensorPrimitive, rhs: i32) -> MlxTensorPrimitive {
        let scalar = Array::from_int(rhs);
        let array = mlx_rs::ops::multiply(&lhs.array, &scalar).expect("Failed to multiply scalar");
        MlxTensorPrimitive::new(array)
    }

    fn int_div(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::divide(&lhs.array, &rhs.array).expect("Failed to divide");
        MlxTensorPrimitive::new(array)
    }

    fn int_div_scalar(lhs: MlxTensorPrimitive, rhs: i32) -> MlxTensorPrimitive {
        let scalar = Array::from_int(rhs);
        let array = mlx_rs::ops::divide(&lhs.array, &scalar).expect("Failed to divide scalar");
        MlxTensorPrimitive::new(array)
    }

    fn int_remainder(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::remainder(&lhs.array, &rhs.array).expect("Failed to remainder");
        MlxTensorPrimitive::new(array)
    }

    fn int_remainder_scalar(lhs: MlxTensorPrimitive, rhs: i32) -> MlxTensorPrimitive {
        let scalar = Array::from_int(rhs);
        let array = mlx_rs::ops::remainder(&lhs.array, &scalar).expect("Failed to remainder scalar");
        MlxTensorPrimitive::new(array)
    }

    fn int_neg(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::negative(&tensor.array).expect("Failed to negate");
        MlxTensorPrimitive::new(array)
    }

    fn int_abs(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::abs(&tensor.array).expect("Failed to abs");
        MlxTensorPrimitive::new(array)
    }

    fn int_swap_dims(tensor: MlxTensorPrimitive, dim1: usize, dim2: usize) -> MlxTensorPrimitive {
        let ndim = tensor.shape().len();
        let mut axes: Vec<i32> = (0..ndim as i32).collect();
        axes.swap(dim1, dim2);
        let array = mlx_rs::ops::transpose_axes(&tensor.array, &axes).expect("Failed to swap dims");
        MlxTensorPrimitive::new(array)
    }

    fn int_permute(tensor: MlxTensorPrimitive, axes: &[usize]) -> MlxTensorPrimitive {
        let axes_i32: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
        let array = mlx_rs::ops::transpose_axes(&tensor.array, &axes_i32).expect("Failed to permute");
        MlxTensorPrimitive::new(array)
    }

    fn int_flip(tensor: MlxTensorPrimitive, axes: &[usize]) -> MlxTensorPrimitive {
        // Placeholder - MLX doesn't have direct flip
        tensor
    }

    fn int_reshape(tensor: MlxTensorPrimitive, shape: Shape) -> MlxTensorPrimitive {
        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = tensor.array.reshape(&shape_i32).expect("Failed to reshape");
        MlxTensorPrimitive::new(array)
    }

    fn int_slice(tensor: MlxTensorPrimitive, ranges: &[Range<usize>]) -> MlxTensorPrimitive {
        // Use Array slicing method
        let starts: Vec<i32> = ranges.iter().map(|r| r.start as i32).collect();
        let ends: Vec<i32> = ranges.iter().map(|r| r.end as i32).collect();
        // Placeholder - need proper slice implementation
        tensor
    }

    fn int_slice_assign(
        tensor: MlxTensorPrimitive,
        ranges: &[Range<usize>],
        value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        tensor
    }

    fn int_mask_where(
        tensor: MlxTensorPrimitive,
        mask: MlxTensorPrimitive,
        value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::r#where(&mask.array, &value.array, &tensor.array)
            .expect("Failed to mask_where");
        MlxTensorPrimitive::new(array)
    }

    fn int_mask_fill(tensor: MlxTensorPrimitive, mask: MlxTensorPrimitive, value: i32) -> MlxTensorPrimitive {
        let fill_val = Array::from_int(value);
        let fill_broadcast = mlx_rs::ops::broadcast_to(&fill_val, tensor.array.shape())
            .expect("Failed to broadcast");
        let array = mlx_rs::ops::r#where(&mask.array, &fill_broadcast, &tensor.array)
            .expect("Failed to mask_fill");
        MlxTensorPrimitive::new(array)
    }

    fn int_gather(dim: usize, tensor: MlxTensorPrimitive, indices: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = take_along_axis(&tensor.array, &indices.array, dim as i32)
            .expect("Failed to gather");
        MlxTensorPrimitive::new(array)
    }

    fn int_scatter(dim: usize, tensor: MlxTensorPrimitive, indices: MlxTensorPrimitive, value: MlxTensorPrimitive) -> MlxTensorPrimitive {
        tensor
    }

    fn int_select(tensor: MlxTensorPrimitive, dim: usize, indices: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = take_axis(&tensor.array, &indices.array, dim as i32).expect("Failed to select");
        MlxTensorPrimitive::new(array)
    }

    fn int_select_assign(tensor: MlxTensorPrimitive, dim: usize, indices: MlxTensorPrimitive, value: MlxTensorPrimitive) -> MlxTensorPrimitive {
        tensor
    }

    fn int_equal(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::eq(&lhs.array, &rhs.array).expect("Failed to equal");
        MlxTensorPrimitive::new(array)
    }

    fn int_equal_elem(lhs: MlxTensorPrimitive, rhs: i32) -> MlxTensorPrimitive {
        let scalar = Array::from_int(rhs);
        let array = mlx_rs::ops::eq(&lhs.array, &scalar).expect("Failed to equal_elem");
        MlxTensorPrimitive::new(array)
    }

    fn int_greater(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::gt(&lhs.array, &rhs.array).expect("Failed to greater");
        MlxTensorPrimitive::new(array)
    }

    fn int_greater_elem(lhs: MlxTensorPrimitive, rhs: i32) -> MlxTensorPrimitive {
        let scalar = Array::from_int(rhs);
        let array = mlx_rs::ops::gt(&lhs.array, &scalar).expect("Failed to greater_elem");
        MlxTensorPrimitive::new(array)
    }

    fn int_greater_equal(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::ge(&lhs.array, &rhs.array).expect("Failed to greater_equal");
        MlxTensorPrimitive::new(array)
    }

    fn int_greater_equal_elem(lhs: MlxTensorPrimitive, rhs: i32) -> MlxTensorPrimitive {
        let scalar = Array::from_int(rhs);
        let array = mlx_rs::ops::ge(&lhs.array, &scalar).expect("Failed to greater_equal_elem");
        MlxTensorPrimitive::new(array)
    }

    fn int_lower(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::lt(&lhs.array, &rhs.array).expect("Failed to lower");
        MlxTensorPrimitive::new(array)
    }

    fn int_lower_elem(lhs: MlxTensorPrimitive, rhs: i32) -> MlxTensorPrimitive {
        let scalar = Array::from_int(rhs);
        let array = mlx_rs::ops::lt(&lhs.array, &scalar).expect("Failed to lower_elem");
        MlxTensorPrimitive::new(array)
    }

    fn int_lower_equal(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::le(&lhs.array, &rhs.array).expect("Failed to lower_equal");
        MlxTensorPrimitive::new(array)
    }

    fn int_lower_equal_elem(lhs: MlxTensorPrimitive, rhs: i32) -> MlxTensorPrimitive {
        let scalar = Array::from_int(rhs);
        let array = mlx_rs::ops::le(&lhs.array, &scalar).expect("Failed to lower_equal_elem");
        MlxTensorPrimitive::new(array)
    }

    fn int_sum(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::sum(&tensor.array, false).expect("Failed to sum");
        MlxTensorPrimitive::new(array)
    }

    fn int_sum_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::sum_axis(&tensor.array, dim as i32, true).expect("Failed to sum_dim");
        MlxTensorPrimitive::new(array)
    }

    fn int_prod(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::prod(&tensor.array, false).expect("Failed to prod");
        MlxTensorPrimitive::new(array)
    }

    fn int_prod_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::prod_axis(&tensor.array, dim as i32, true).expect("Failed to prod_dim");
        MlxTensorPrimitive::new(array)
    }

    fn int_mean_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::mean_axis(&tensor.array, dim as i32, true).expect("Failed to mean_dim");
        MlxTensorPrimitive::new(array)
    }

    fn int_argmax(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = argmax_axis(&tensor.array, dim as i32, true).expect("Failed to argmax");
        MlxTensorPrimitive::new(array)
    }

    fn int_argmin(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = argmin_axis(&tensor.array, dim as i32, true).expect("Failed to argmin");
        MlxTensorPrimitive::new(array)
    }

    fn int_max(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::max(&tensor.array, false).expect("Failed to max");
        MlxTensorPrimitive::new(array)
    }

    fn int_max_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::max_axis(&tensor.array, dim as i32, true).expect("Failed to max_dim");
        MlxTensorPrimitive::new(array)
    }

    fn int_max_dim_with_indices(tensor: MlxTensorPrimitive, dim: usize) -> (MlxTensorPrimitive, MlxTensorPrimitive) {
        let values = mlx_rs::ops::max_axis(&tensor.array, dim as i32, true).expect("Failed to max_dim");
        let indices = argmax_axis(&tensor.array, dim as i32, true).expect("Failed to argmax");
        (MlxTensorPrimitive::new(values), MlxTensorPrimitive::new(indices))
    }

    fn int_min(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::min(&tensor.array, false).expect("Failed to min");
        MlxTensorPrimitive::new(array)
    }

    fn int_min_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::min_axis(&tensor.array, dim as i32, true).expect("Failed to min_dim");
        MlxTensorPrimitive::new(array)
    }

    fn int_min_dim_with_indices(tensor: MlxTensorPrimitive, dim: usize) -> (MlxTensorPrimitive, MlxTensorPrimitive) {
        let values = mlx_rs::ops::min_axis(&tensor.array, dim as i32, true).expect("Failed to min_dim");
        let indices = argmin_axis(&tensor.array, dim as i32, true).expect("Failed to argmin");
        (MlxTensorPrimitive::new(values), MlxTensorPrimitive::new(indices))
    }

    fn int_into_float(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = tensor.array.as_type::<f32>().expect("Failed to cast to float");
        MlxTensorPrimitive::new(array)
    }

    fn int_expand(tensor: MlxTensorPrimitive, shape: Shape) -> MlxTensorPrimitive {
        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = mlx_rs::ops::broadcast_to(&tensor.array, &shape_i32).expect("Failed to expand");
        MlxTensorPrimitive::new(array)
    }

    fn int_sign(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::sign(&tensor.array).expect("Failed to sign");
        MlxTensorPrimitive::new(array)
    }

    fn int_sort(tensor: MlxTensorPrimitive, dim: usize, _descending: bool) -> MlxTensorPrimitive {
        let sorted = mlx_rs::ops::sort_axis(&tensor.array, dim as i32).expect("Failed to sort");
        MlxTensorPrimitive::new(sorted)
    }

    fn int_sort_with_indices(tensor: MlxTensorPrimitive, dim: usize, _descending: bool) -> (MlxTensorPrimitive, MlxTensorPrimitive) {
        let sorted = mlx_rs::ops::sort_axis(&tensor.array, dim as i32).expect("Failed to sort");
        let indices = mlx_rs::ops::argsort_axis(&tensor.array, dim as i32).expect("Failed to argsort");
        (MlxTensorPrimitive::new(sorted), MlxTensorPrimitive::new(indices))
    }

    fn int_argsort(tensor: MlxTensorPrimitive, dim: usize, _descending: bool) -> MlxTensorPrimitive {
        let indices = mlx_rs::ops::argsort_axis(&tensor.array, dim as i32).expect("Failed to argsort");
        MlxTensorPrimitive::new(indices)
    }

    fn int_any(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::any(&tensor.array, false).expect("Failed to any");
        MlxTensorPrimitive::new(array)
    }

    fn int_any_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::any_axis(&tensor.array, dim as i32, true).expect("Failed to any_dim");
        MlxTensorPrimitive::new(array)
    }

    fn int_all(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::all(&tensor.array, false).expect("Failed to all");
        MlxTensorPrimitive::new(array)
    }

    fn int_all_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::all_axis(&tensor.array, dim as i32, true).expect("Failed to all_dim");
        MlxTensorPrimitive::new(array)
    }
}
