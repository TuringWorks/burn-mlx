//! Float tensor operations for MLX backend.

use burn_tensor::{backend::ExecutionError, ops::FloatTensorOps, Distribution, FloatDType, Shape, Slice, TensorData};
use half::{bf16, f16};
use mlx_rs::Array;
use mlx_rs::ops::indexing::{take_axis, take_along_axis};

use crate::backend::{Mlx, MlxTensorPrimitive};
use crate::device::MlxDevice;
use crate::element::FloatMlxElement;

impl<F: FloatMlxElement> FloatTensorOps<Self> for Mlx<F> {
    fn float_from_data(data: TensorData, device: &MlxDevice) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);

        let shape: Vec<i32> = data.shape.iter().map(|&s| s as i32).collect();
        let values: Vec<F> = data.to_vec().expect("Failed to convert data to vec");
        let array = F::array_from_slice(&values, &shape);

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

        // Generate random values in f32 (widest support in mlx-rs random API),
        // then cast to the target float type F.
        let array_f32 = match distribution {
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

        let array = F::cast_array(&array_f32);
        MlxTensorPrimitive::new(array)
    }

    async fn float_into_data(tensor: MlxTensorPrimitive) -> Result<TensorData, ExecutionError> {
        tensor.array.eval().expect("Failed to evaluate tensor");
        let shape = tensor.shape().to_vec();
        let data: Vec<F> = F::array_to_vec(&tensor.array);
        Ok(TensorData::new(data, shape))
    }

    fn float_device(tensor: &MlxTensorPrimitive) -> MlxDevice {
        let _ = tensor;
        MlxDevice::Gpu
    }

    fn float_to_device(tensor: MlxTensorPrimitive, device: &MlxDevice) -> MlxTensorPrimitive {
        let _ = device;
        tensor
    }

    fn float_empty(shape: Shape, device: &MlxDevice, _dtype: FloatDType) -> MlxTensorPrimitive {
        let mlx_device = device.to_mlx_device();
        mlx_rs::Device::set_default(&mlx_device);

        let shape_i32: Vec<i32> = shape.dims.iter().map(|&s| s as i32).collect();
        let array = F::zeros_array(&shape_i32);

        MlxTensorPrimitive::new(array)
    }

    fn float_add(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::add(&lhs.array, &rhs.array).expect("Failed to add");
        MlxTensorPrimitive::new(array)
    }

    fn float_add_scalar(lhs: MlxTensorPrimitive, rhs: F) -> MlxTensorPrimitive {
        let scalar = F::scalar_array(rhs);
        let array = mlx_rs::ops::add(&lhs.array, &scalar).expect("Failed to add scalar");
        MlxTensorPrimitive::new(array)
    }

    fn float_sub(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::subtract(&lhs.array, &rhs.array).expect("Failed to subtract");
        MlxTensorPrimitive::new(array)
    }

    fn float_sub_scalar(lhs: MlxTensorPrimitive, rhs: F) -> MlxTensorPrimitive {
        let scalar = F::scalar_array(rhs);
        let array = mlx_rs::ops::subtract(&lhs.array, &scalar).expect("Failed to subtract scalar");
        MlxTensorPrimitive::new(array)
    }

    fn float_mul(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::multiply(&lhs.array, &rhs.array).expect("Failed to multiply");
        MlxTensorPrimitive::new(array)
    }

    fn float_mul_scalar(lhs: MlxTensorPrimitive, rhs: F) -> MlxTensorPrimitive {
        let scalar = F::scalar_array(rhs);
        let array = mlx_rs::ops::multiply(&lhs.array, &scalar).expect("Failed to multiply scalar");
        MlxTensorPrimitive::new(array)
    }

    fn float_div(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::divide(&lhs.array, &rhs.array).expect("Failed to divide");
        MlxTensorPrimitive::new(array)
    }

    fn float_div_scalar(lhs: MlxTensorPrimitive, rhs: F) -> MlxTensorPrimitive {
        let scalar = F::scalar_array(rhs);
        let array = mlx_rs::ops::divide(&lhs.array, &scalar).expect("Failed to divide scalar");
        MlxTensorPrimitive::new(array)
    }

    fn float_remainder(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::remainder(&lhs.array, &rhs.array).expect("Failed to remainder");
        MlxTensorPrimitive::new(array)
    }

    fn float_remainder_scalar(lhs: MlxTensorPrimitive, rhs: F) -> MlxTensorPrimitive {
        let scalar = F::scalar_array(rhs);
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
        let one = F::f64_scalar_array(1.0);
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

    fn float_flip(tensor: MlxTensorPrimitive, axes: &[usize]) -> MlxTensorPrimitive {
        let axes_i32: Vec<i32> = axes.iter().map(|&a| a as i32).collect();
        let array = mlx_rs::ops::flip(&tensor.array, &axes_i32[..])
            .expect("Failed to flip");
        MlxTensorPrimitive::new(array)
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

    fn float_scatter_add(
        dim: usize,
        tensor: MlxTensorPrimitive,
        indices: MlxTensorPrimitive,
        value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        let array = tensor.array.put_along_axis(&indices.array, &value.array, dim as i32)
            .expect("Failed to scatter_add");
        MlxTensorPrimitive::new(array)
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

    fn float_select_add(
        tensor: MlxTensorPrimitive,
        dim: usize,
        indices: MlxTensorPrimitive,
        value: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        let array = tensor.array.put_along_axis(&indices.array, &value.array, dim as i32)
            .expect("Failed to select_add");
        MlxTensorPrimitive::new(array)
    }

    fn float_slice(tensor: MlxTensorPrimitive, slices: &[Slice]) -> MlxTensorPrimitive {
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

    fn float_slice_assign(
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
        value: F,
    ) -> MlxTensorPrimitive {
        let fill_val = F::scalar_array(value);
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

    fn float_equal_elem(lhs: MlxTensorPrimitive, rhs: F) -> MlxTensorPrimitive {
        let scalar = F::scalar_array(rhs);
        let array = mlx_rs::ops::eq(&lhs.array, &scalar).expect("Failed to equal_elem");
        MlxTensorPrimitive::new(array)
    }

    fn float_greater(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::gt(&lhs.array, &rhs.array).expect("Failed to greater");
        MlxTensorPrimitive::new(array)
    }

    fn float_greater_elem(lhs: MlxTensorPrimitive, rhs: F) -> MlxTensorPrimitive {
        let scalar = F::scalar_array(rhs);
        let array = mlx_rs::ops::gt(&lhs.array, &scalar).expect("Failed to greater_elem");
        MlxTensorPrimitive::new(array)
    }

    fn float_greater_equal(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::ge(&lhs.array, &rhs.array).expect("Failed to greater_equal");
        MlxTensorPrimitive::new(array)
    }

    fn float_greater_equal_elem(lhs: MlxTensorPrimitive, rhs: F) -> MlxTensorPrimitive {
        let scalar = F::scalar_array(rhs);
        let array = mlx_rs::ops::ge(&lhs.array, &scalar).expect("Failed to greater_equal_elem");
        MlxTensorPrimitive::new(array)
    }

    fn float_lower(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::lt(&lhs.array, &rhs.array).expect("Failed to lower");
        MlxTensorPrimitive::new(array)
    }

    fn float_lower_elem(lhs: MlxTensorPrimitive, rhs: F) -> MlxTensorPrimitive {
        let scalar = F::scalar_array(rhs);
        let array = mlx_rs::ops::lt(&lhs.array, &scalar).expect("Failed to lower_elem");
        MlxTensorPrimitive::new(array)
    }

    fn float_lower_equal(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::le(&lhs.array, &rhs.array).expect("Failed to lower_equal");
        MlxTensorPrimitive::new(array)
    }

    fn float_lower_equal_elem(lhs: MlxTensorPrimitive, rhs: F) -> MlxTensorPrimitive {
        let scalar = F::scalar_array(rhs);
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
        Self::float_powf_scalar_impl(tensor, value)
    }

    fn float_powf_scalar_impl(tensor: MlxTensorPrimitive, value: f32) -> MlxTensorPrimitive {
        let scalar = F::f64_scalar_array(value as f64);
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

    fn float_clamp(tensor: MlxTensorPrimitive, min: F, max: F) -> MlxTensorPrimitive {
        let min_arr = F::scalar_array(min);
        let max_arr = F::scalar_array(max);
        let array = mlx_rs::ops::clip(&tensor.array, (&min_arr, &max_arr)).expect("Failed to clamp");
        MlxTensorPrimitive::new(array)
    }

    fn float_clamp_min(tensor: MlxTensorPrimitive, min: F) -> MlxTensorPrimitive {
        let min_arr = F::scalar_array(min);
        let array = mlx_rs::ops::maximum(&tensor.array, &min_arr).expect("Failed to clamp_min");
        MlxTensorPrimitive::new(array)
    }

    fn float_clamp_max(tensor: MlxTensorPrimitive, max: F) -> MlxTensorPrimitive {
        let max_arr = F::scalar_array(max);
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
            FloatDType::F16 => tensor.array.as_type::<f16>().expect("cast to f16"),
            FloatDType::BF16 => tensor.array.as_type::<bf16>().expect("cast to bf16"),
            FloatDType::F32 => tensor.array.as_type::<f32>().expect("cast to f32"),
            FloatDType::F64 => tensor.array.as_type::<f64>().expect("cast to f64"),
            _ => tensor.array,
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

    fn float_trunc(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        // trunc(x) = sign(x) * floor(abs(x))
        let abs_val = mlx_rs::ops::abs(&tensor.array).expect("abs");
        let floored = mlx_rs::ops::floor(&abs_val).expect("floor");
        let sign_val = mlx_rs::ops::sign(&tensor.array).expect("sign");
        let array = mlx_rs::ops::multiply(&sign_val, &floored).expect("multiply");
        MlxTensorPrimitive::new(array)
    }

    fn float_tan(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::tan(&tensor.array).expect("Failed to tan");
        MlxTensorPrimitive::new(array)
    }

    fn float_cosh(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::cosh(&tensor.array).expect("Failed to cosh");
        MlxTensorPrimitive::new(array)
    }

    fn float_sinh(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::sinh(&tensor.array).expect("Failed to sinh");
        MlxTensorPrimitive::new(array)
    }

    fn float_acos(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::acos(&tensor.array).expect("Failed to acos");
        MlxTensorPrimitive::new(array)
    }

    fn float_acosh(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::acosh(&tensor.array).expect("Failed to acosh");
        MlxTensorPrimitive::new(array)
    }

    fn float_asin(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::asin(&tensor.array).expect("Failed to asin");
        MlxTensorPrimitive::new(array)
    }

    fn float_asinh(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::asinh(&tensor.array).expect("Failed to asinh");
        MlxTensorPrimitive::new(array)
    }

    fn float_atan(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::atan(&tensor.array).expect("Failed to atan");
        MlxTensorPrimitive::new(array)
    }

    fn float_atanh(tensor: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::atanh(&tensor.array).expect("Failed to atanh");
        MlxTensorPrimitive::new(array)
    }

    fn float_atan2(lhs: MlxTensorPrimitive, rhs: MlxTensorPrimitive) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::atan2(&lhs.array, &rhs.array).expect("Failed to atan2");
        MlxTensorPrimitive::new(array)
    }

    fn float_cross(
        lhs: MlxTensorPrimitive,
        rhs: MlxTensorPrimitive,
        dim: usize,
    ) -> MlxTensorPrimitive {
        let dim_i32 = dim as i32;

        let a0 = take_axis(&lhs.array, &Array::from_int(0), dim_i32).expect("take");
        let a1 = take_axis(&lhs.array, &Array::from_int(1), dim_i32).expect("take");
        let a2 = take_axis(&lhs.array, &Array::from_int(2), dim_i32).expect("take");

        let b0 = take_axis(&rhs.array, &Array::from_int(0), dim_i32).expect("take");
        let b1 = take_axis(&rhs.array, &Array::from_int(1), dim_i32).expect("take");
        let b2 = take_axis(&rhs.array, &Array::from_int(2), dim_i32).expect("take");

        let r0 = mlx_rs::ops::subtract(
            &mlx_rs::ops::multiply(&a1, &b2).expect("mul"),
            &mlx_rs::ops::multiply(&a2, &b1).expect("mul"),
        ).expect("sub");
        let r1 = mlx_rs::ops::subtract(
            &mlx_rs::ops::multiply(&a2, &b0).expect("mul"),
            &mlx_rs::ops::multiply(&a0, &b2).expect("mul"),
        ).expect("sub");
        let r2 = mlx_rs::ops::subtract(
            &mlx_rs::ops::multiply(&a0, &b1).expect("mul"),
            &mlx_rs::ops::multiply(&a1, &b0).expect("mul"),
        ).expect("sub");

        let array = mlx_rs::ops::stack_axis(&[&r0, &r1, &r2], dim_i32).expect("stack");
        MlxTensorPrimitive::new(array)
    }

    fn float_cumsum(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::cumsum(&tensor.array, dim as i32, None, None)
            .expect("Failed to cumsum");
        MlxTensorPrimitive::new(array)
    }

    fn float_cumprod(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::cumprod(&tensor.array, dim as i32, None, None)
            .expect("Failed to cumprod");
        MlxTensorPrimitive::new(array)
    }

    fn float_cummin(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::cummin(&tensor.array, dim as i32, None, None)
            .expect("Failed to cummin");
        MlxTensorPrimitive::new(array)
    }

    fn float_cummax(tensor: MlxTensorPrimitive, dim: usize) -> MlxTensorPrimitive {
        let array = mlx_rs::ops::cummax(&tensor.array, dim as i32, None, None)
            .expect("Failed to cummax");
        MlxTensorPrimitive::new(array)
    }

    fn float_unfold(
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
