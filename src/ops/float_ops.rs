//! Float tensor operations for MLX backend.

use burn_tensor::{ops::FloatTensorOps, Distribution, FloatDType, Shape, TensorData};
use mlx_rs::Array;
use mlx_rs::ops::indexing::{take_axis, take_along_axis};
use std::ops::Range;

use crate::backend::{Mlx, MlxTensorPrimitive};
use crate::device::MlxDevice;

impl FloatTensorOps<Self> for Mlx {
    fn float_from_data(data: TensorData, device: &MlxDevice) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);

        let shape: Vec<i32> = data.shape.iter().map(|&s| s as i32).collect();
        let values: Vec<f32> = data.to_vec().expect("Failed to convert data to f32 vec");
        let array = Array::from_slice(&values, &shape);

        MlxTensorPrimitive::new(array)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &MlxDevice,
    ) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);

        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();

        let array = match distribution {
            Distribution::Default => {
                mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &shape_i32, None)
                    .expect("Failed to create uniform random array")
            }
            Distribution::Uniform(low, high) => {
                mlx_rs::random::uniform::<f32, f32>(low as f32, high as f32, &shape_i32, None)
                    .expect("Failed to create uniform random array")
            }
            Distribution::Normal(mean, std) => {
                mlx_rs::random::normal::<f32>(&shape_i32, None, None, None)
                    .map(|arr| {
                        let std_arr = Array::from_f32(std as f32);
                        let mean_arr = Array::from_f32(mean as f32);
                        let scaled = mlx_rs::ops::multiply(&arr, &std_arr).expect("multiply");
                        mlx_rs::ops::add(&scaled, &mean_arr).expect("add")
                    })
                    .expect("Failed to create normal random array")
            }
            Distribution::Bernoulli(prob) => {
                let uniform = mlx_rs::random::uniform::<f32, f32>(0.0, 1.0, &shape_i32, None)
                    .expect("Failed to create uniform");
                let threshold = Array::from_f32(prob as f32);
                let bool_arr = mlx_rs::ops::lt(&uniform, &threshold).expect("lt");
                bool_arr.as_type::<f32>().expect("cast to f32")
            }
        };

        MlxTensorPrimitive::new(array)
    }

    async fn float_into_data(tensor: MlxTensorPrimitive) -> TensorData {
        tensor.array.eval().expect("Failed to evaluate tensor");
        let shape = tensor.shape().to_vec();
        let data: Vec<f32> = tensor.array.as_slice().to_vec();
        TensorData::new(data, shape)
    }

    fn float_device(tensor: &MlxTensorPrimitive) -> MlxDevice {
        let _ = tensor;
        MlxDevice::Gpu
    }

    fn float_to_device(tensor: MlxTensorPrimitive, device: &MlxDevice) -> MlxTensorPrimitive {
        let _ = device;
        tensor
    }

    fn float_empty(shape: Shape, device: &MlxDevice) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);

        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = Array::zeros::<f32>(&shape_i32).expect("Failed to create empty array");

        MlxTensorPrimitive::new(array)
    }

    fn float_add(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::add(&lhs.array, &rhs.array).expect("Failed to add");
        MlxTensorPrimitive::new(array)
    }

    fn float_add_scalar(lhs: MlxTensorPrimitive, rhs: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(rhs);
        let array = mlx_rs::ops::add(&lhs.array, &scalar).expect("Failed to add scalar");
        MlxTensorPrimitive::new(array)
    }

    fn float_sub(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::subtract(&lhs.array, &rhs.array).expect("Failed to subtract");
        MlxTensorPrimitive::new(array)
    }

    fn float_sub_scalar(lhs: MlxTensorPrimitive, rhs: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(rhs);
        let array = mlx_rs::ops::subtract(&lhs.array, &scalar).expect("Failed to subtract scalar");
        MlxTensorPrimitive::new(array)
    }

    fn float_mul(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::multiply(&lhs.array, &rhs.array).expect("Failed to multiply");
        MlxTensorPrimitive::new(array)
    }

    fn float_mul_scalar(lhs: MlxTensorPrimitive, rhs: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(rhs);
        let array = mlx_rs::ops::multiply(&lhs.array, &scalar).expect("Failed to multiply scalar");
        MlxTensorPrimitive::new(array)
    }

    fn float_div(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::divide(&lhs.array, &rhs.array).expect("Failed to divide");
        MlxTensorPrimitive::new(array)
    }

    fn float_div_scalar(lhs: MlxTensorPrimitive, rhs: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(rhs);
        let array = mlx_rs::ops::divide(&lhs.array, &scalar).expect("Failed to divide scalar");
        MlxTensorPrimitive::new(array)
    }

    fn float_remainder(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::remainder(&lhs.array, &rhs.array).expect("Failed to remainder");
        MlxTensorPrimitive::new(array)
    }

    fn float_remainder_scalar(lhs: MlxTensorPrimitive, rhs: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(rhs);
        let array =
            mlx_rs::ops::remainder(&lhs.array, &scalar).expect("Failed to remainder scalar");
        MlxTensorPrimitive::new(array)
    }

    fn float_matmul(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = lhs.array.matmul(&rhs.array).expect("Failed to matmul");
        MlxTensorPrimitive::new(array)
    }

    fn float_neg(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::negative(&tensor.array).expect("Failed to negate");
        MlxTensorPrimitive::new(array)
    }

    fn float_recip(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let one = Array::from_f32(1.0);
        let array = mlx_rs::ops::divide(&one, &tensor.array).expect("Failed to recip");
        MlxTensorPrimitive::new(array)
    }

    fn float_swap_dims(
        tensor: MlxTensorPrimitive,
        dim1: usize,
        dim2: usize,
    ) -> MlxTensorPrimitive {
        let ndim = tensor.shape().len();
        let mut axes: Vec<i32> = (0..ndim as i32).collect();
        axes.swap(dim1, dim2);
        let array =
            mlx_rs::ops::transpose_axes(&tensor.array, &axes).expect("Failed to swap dims");
        MlxTensorPrimitive::new(array)
    }

    fn float_permute(tensor: MlxTensorPrimitive, axes: &[usize]) -> MlxTensorPrimitive {
        let axes_i32: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
        let array =
            mlx_rs::ops::transpose_axes(&tensor.array, &axes_i32).expect("Failed to permute");
        MlxTensorPrimitive::new(array)
    }

    fn float_flip(tensor: MlxTensorPrimitive, _axes: &[usize]) -> MlxTensorPrimitive {
        // MLX doesn't have direct flip - use indexing
        // Placeholder: return tensor as-is for now
        tensor
    }

    fn float_reshape(tensor: MlxTensorPrimitive, shape: Shape) -> MlxTensorPrimitive {
        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = tensor.array.reshape(&shape_i32).expect("Failed to reshape");
        MlxTensorPrimitive::new(array)
    }

    fn float_gather(
        dim: usize,
        tensor: MlxTensorPrimitive,
        indices: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        let array =
            take_along_axis(&tensor.array, &indices.array, dim as i32).expect("Failed to gather");
        MlxTensorPrimitive::new(array)
    }

    fn float_scatter(
        _dim: usize,
        tensor: MlxTensorPrimitive,
        _indices: MlxTensorPrimitive,
        _value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        // Placeholder - needs proper scatter implementation
        tensor
    }

    fn float_select(
        tensor: MlxTensorPrimitive,
        dim: usize,
        indices: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        let array =
            take_axis(&tensor.array, &indices.array, dim as i32).expect("Failed to select");
        MlxTensorPrimitive::new(array)
    }

    fn float_select_assign(
        tensor: MlxTensorPrimitive,
        _dim: usize,
        _indices: MlxTensorPrimitive,
        _value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        // Placeholder
        tensor
    }

    fn float_slice(tensor: MlxTensorPrimitive, _ranges: &[Range<usize>]) -> MlxTensorPrimitive {
        // MLX doesn't have direct slice - would need to implement via indexing
        // Placeholder: return tensor as-is for now
        tensor
    }

    fn float_slice_assign(
        tensor: MlxTensorPrimitive,
        _ranges: &[Range<usize>],
        _value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        // Placeholder
        tensor
    }

    fn float_mask_where(
        tensor: MlxTensorPrimitive,
        mask: MlxTensorPrimitive,
        value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::r#where(&mask.array, &value.array, &tensor.array)
            .expect("Failed to mask_where");
        MlxTensorPrimitive::new(array)
    }

    fn float_mask_fill(
        tensor: MlxTensorPrimitive,
        mask: MlxTensorPrimitive,
        value: f32,
    ) -> MlxTensorPrimitive {
        let fill_val = Array::from_f32(value);
        let fill_broadcast =
            mlx_rs::ops::broadcast_to(&fill_val, tensor.array.shape()).expect("Failed to broadcast");
        let array = mlx_rs::ops::r#where(&mask.array, &fill_broadcast, &tensor.array)
            .expect("Failed to mask_fill");
        MlxTensorPrimitive::new(array)
    }

    fn float_equal(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::eq(&lhs.array, &rhs.array).expect("Failed to equal");
        MlxTensorPrimitive::new(array)
    }

    fn float_equal_elem(lhs: MlxTensorPrimitive, rhs: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(rhs);
        let array = mlx_rs::ops::eq(&lhs.array, &scalar).expect("Failed to equal_elem");
        MlxTensorPrimitive::new(array)
    }

    fn float_greater(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::gt(&lhs.array, &rhs.array).expect("Failed to greater");
        MlxTensorPrimitive::new(array)
    }

    fn float_greater_elem(lhs: MlxTensorPrimitive, rhs: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(rhs);
        let array = mlx_rs::ops::gt(&lhs.array, &scalar).expect("Failed to greater_elem");
        MlxTensorPrimitive::new(array)
    }

    fn float_greater_equal(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::ge(&lhs.array, &rhs.array).expect("Failed to greater_equal");
        MlxTensorPrimitive::new(array)
    }

    fn float_greater_equal_elem(lhs: MlxTensorPrimitive, rhs: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(rhs);
        let array = mlx_rs::ops::ge(&lhs.array, &scalar).expect("Failed to greater_equal_elem");
        MlxTensorPrimitive::new(array)
    }

    fn float_lower(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::lt(&lhs.array, &rhs.array).expect("Failed to lower");
        MlxTensorPrimitive::new(array)
    }

    fn float_lower_elem(lhs: MlxTensorPrimitive, rhs: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(rhs);
        let array = mlx_rs::ops::lt(&lhs.array, &scalar).expect("Failed to lower_elem");
        MlxTensorPrimitive::new(array)
    }

    fn float_lower_equal(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::le(&lhs.array, &rhs.array).expect("Failed to lower_equal");
        MlxTensorPrimitive::new(array)
    }

    fn float_lower_equal_elem(lhs: MlxTensorPrimitive, rhs: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(rhs);
        let array = mlx_rs::ops::le(&lhs.array, &scalar).expect("Failed to lower_equal_elem");
        MlxTensorPrimitive::new(array)
    }

    fn float_sum(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::sum(&tensor.array, false).expect("Failed to sum");
        MlxTensorPrimitive::new(array)
    }

    fn float_sum_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array =
            mlx_rs::ops::sum_axis(&tensor.array, dim as i32, true).expect("Failed to sum_dim");
        MlxTensorPrimitive::new(array)
    }

    fn float_prod(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::prod(&tensor.array, false).expect("Failed to prod");
        MlxTensorPrimitive::new(array)
    }

    fn float_prod_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array =
            mlx_rs::ops::prod_axis(&tensor.array, dim as i32, true).expect("Failed to prod_dim");
        MlxTensorPrimitive::new(array)
    }

    fn float_mean(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::mean(&tensor.array, false).expect("Failed to mean");
        MlxTensorPrimitive::new(array)
    }

    fn float_mean_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array =
            mlx_rs::ops::mean_axis(&tensor.array, dim as i32, true).expect("Failed to mean_dim");
        MlxTensorPrimitive::new(array)
    }

    fn float_exp(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::exp(&tensor.array).expect("Failed to exp");
        MlxTensorPrimitive::new(array)
    }

    fn float_log(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::log(&tensor.array).expect("Failed to log");
        MlxTensorPrimitive::new(array)
    }

    fn float_log1p(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::log1p(&tensor.array).expect("Failed to log1p");
        MlxTensorPrimitive::new(array)
    }

    fn float_powf(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::power(&lhs.array, &rhs.array).expect("Failed to powf");
        MlxTensorPrimitive::new(array)
    }

    fn float_powf_scalar(tensor: MlxTensorPrimitive, value: f32) -> MlxTensorPrimitive {
        let scalar = Array::from_f32(value);
        let array = mlx_rs::ops::power(&tensor.array, &scalar).expect("Failed to powf_scalar");
        MlxTensorPrimitive::new(array)
    }

    fn float_sqrt(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::sqrt(&tensor.array).expect("Failed to sqrt");
        MlxTensorPrimitive::new(array)
    }

    fn float_abs(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::abs(&tensor.array).expect("Failed to abs");
        MlxTensorPrimitive::new(array)
    }

    fn float_cos(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::cos(&tensor.array).expect("Failed to cos");
        MlxTensorPrimitive::new(array)
    }

    fn float_sin(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::sin(&tensor.array).expect("Failed to sin");
        MlxTensorPrimitive::new(array)
    }

    fn float_tanh(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::tanh(&tensor.array).expect("Failed to tanh");
        MlxTensorPrimitive::new(array)
    }

    fn float_erf(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::erf(&tensor.array).expect("Failed to erf");
        MlxTensorPrimitive::new(array)
    }

    fn float_argmax(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::indexing::argmax_axis(&tensor.array, dim as i32, true)
            .expect("Failed to argmax");
        MlxTensorPrimitive::new(array)
    }

    fn float_argmin(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::indexing::argmin_axis(&tensor.array, dim as i32, true)
            .expect("Failed to argmin");
        MlxTensorPrimitive::new(array)
    }

    fn float_max(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::max(&tensor.array, false).expect("Failed to max");
        MlxTensorPrimitive::new(array)
    }

    fn float_max_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array =
            mlx_rs::ops::max_axis(&tensor.array, dim as i32, true).expect("Failed to max_dim");
        MlxTensorPrimitive::new(array)
    }

    fn float_max_dim_with_indices(
        tensor: MlxTensorPrimitive,
        dim: usize,
    ) -> (MlxTensorPrimitive, MlxTensorPrimitive) {
        let values =
            mlx_rs::ops::max_axis(&tensor.array, dim as i32, true).expect("Failed to max_dim");
        let indices = mlx_rs::ops::indexing::argmax_axis(&tensor.array, dim as i32, true)
            .expect("Failed to argmax");
        (
            MlxTensorPrimitive::new(values),
            MlxTensorPrimitive::new(indices),
        )
    }

    fn float_min(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::min(&tensor.array, false).expect("Failed to min");
        MlxTensorPrimitive::new(array)
    }

    fn float_min_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array =
            mlx_rs::ops::min_axis(&tensor.array, dim as i32, true).expect("Failed to min_dim");
        MlxTensorPrimitive::new(array)
    }

    fn float_min_dim_with_indices(
        tensor: MlxTensorPrimitive,
        dim: usize,
    ) -> (MlxTensorPrimitive, MlxTensorPrimitive) {
        let values =
            mlx_rs::ops::min_axis(&tensor.array, dim as i32, true).expect("Failed to min_dim");
        let indices = mlx_rs::ops::indexing::argmin_axis(&tensor.array, dim as i32, true)
            .expect("Failed to argmin");
        (
            MlxTensorPrimitive::new(values),
            MlxTensorPrimitive::new(indices),
        )
    }

    fn float_into_int(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = tensor.array.as_type::<i32>().expect("Failed to cast to int");
        MlxTensorPrimitive::new(array)
    }

    fn float_clamp(tensor: MlxTensorPrimitive, min: f32, max: f32) -> MlxTensorPrimitive {
        let min_arr = Array::from_f32(min);
        let max_arr = Array::from_f32(max);
        let array = mlx_rs::ops::clip(&tensor.array, (&min_arr, &max_arr)).expect("Failed to clamp");
        MlxTensorPrimitive::new(array)
    }

    fn float_clamp_min(tensor: MlxTensorPrimitive, min: f32) -> MlxTensorPrimitive {
        let min_arr = Array::from_f32(min);
        let array = mlx_rs::ops::maximum(&tensor.array, &min_arr).expect("Failed to clamp_min");
        MlxTensorPrimitive::new(array)
    }

    fn float_clamp_max(tensor: MlxTensorPrimitive, max: f32) -> MlxTensorPrimitive {
        let max_arr = Array::from_f32(max);
        let array = mlx_rs::ops::minimum(&tensor.array, &max_arr).expect("Failed to clamp_max");
        MlxTensorPrimitive::new(array)
    }

    fn float_expand(tensor: MlxTensorPrimitive, shape: Shape) -> MlxTensorPrimitive {
        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array =
            mlx_rs::ops::broadcast_to(&tensor.array, &shape_i32).expect("Failed to expand");
        MlxTensorPrimitive::new(array)
    }

    fn float_sign(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::sign(&tensor.array).expect("Failed to sign");
        MlxTensorPrimitive::new(array)
    }

    fn float_any(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::any(&tensor.array, false).expect("Failed to any");
        MlxTensorPrimitive::new(array)
    }

    fn float_any_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array =
            mlx_rs::ops::any_axis(&tensor.array, dim as i32, true).expect("Failed to any_dim");
        MlxTensorPrimitive::new(array)
    }

    fn float_all(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::all(&tensor.array, false).expect("Failed to all");
        MlxTensorPrimitive::new(array)
    }

    fn float_all_dim(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array =
            mlx_rs::ops::all_axis(&tensor.array, dim as i32, true).expect("Failed to all_dim");
        MlxTensorPrimitive::new(array)
    }

    fn float_sort(tensor: MlxTensorPrimitive, dim: usize, _descending: bool) -> MlxTensorPrimitive {
        let sorted = mlx_rs::ops::sort_axis(&tensor.array, dim as i32).expect("Failed to sort");
        // Note: MLX sort is ascending only; descending would need flip
        MlxTensorPrimitive::new(sorted)
    }

    fn float_sort_with_indices(
        tensor: MlxTensorPrimitive,
        dim: usize,
        _descending: bool,
    ) -> (MlxTensorPrimitive, MlxTensorPrimitive) {
        let sorted = mlx_rs::ops::sort_axis(&tensor.array, dim as i32).expect("Failed to sort");
        let indices = mlx_rs::ops::argsort_axis(&tensor.array, dim as i32).expect("Failed to argsort");
        (
            MlxTensorPrimitive::new(sorted),
            MlxTensorPrimitive::new(indices),
        )
    }

    fn float_argsort(tensor: MlxTensorPrimitive, dim: usize, _descending: bool) -> MlxTensorPrimitive {
        let indices = mlx_rs::ops::argsort_axis(&tensor.array, dim as i32).expect("Failed to argsort");
        MlxTensorPrimitive::new(indices)
    }

    fn float_cast(tensor: MlxTensorPrimitive, dtype: FloatDType) -> MlxTensorPrimitive {
        let array = match dtype {
            FloatDType::F32 => tensor.array.as_type::<f32>().expect("cast to f32"),
            FloatDType::F64 => tensor.array.as_type::<f64>().expect("cast to f64"),
            _ => tensor.array, // Keep as-is for unsupported types
        };
        MlxTensorPrimitive::new(array)
    }

    fn float_round(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::round(&tensor.array, 0).expect("Failed to round");
        MlxTensorPrimitive::new(array)
    }

    fn float_floor(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::floor(&tensor.array).expect("Failed to floor");
        MlxTensorPrimitive::new(array)
    }

    fn float_ceil(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::ceil(&tensor.array).expect("Failed to ceil");
        MlxTensorPrimitive::new(array)
    }
}
