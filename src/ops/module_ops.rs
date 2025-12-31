//! Module operations for MLX backend (neural network primitives).

use burn_tensor::ops::{
    ConvOptions, ConvTransposeOptions, DeformConv2dBackward, DeformConvOptions,
    InterpolateOptions, MaxPool1dWithIndices, MaxPool2dBackward, MaxPool2dWithIndices, ModuleOps,
};
use mlx_rs::Array;
use mlx_rs::ops::indexing::take_axis;

use crate::backend::{Mlx, MlxTensorPrimitive};

/// Helper function to compute pooling using as_strided approach.
/// This follows the pattern from mlx-rs nn/pooling.rs.
///
/// Input shape for 2D: [N, H, W, C] (NHWC format - MLX native)
/// Returns: pooled output with shape [N, out_H, out_W, C]
fn pool2d_strided<F>(
    x: &Array,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    pooling_op: F,
) -> Array
where
    F: Fn(&Array, &[i32]) -> Result<Array, mlx_rs::error::Exception>,
{
    let shape = x.shape();
    let n = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];

    let kh = kernel_size[0] as i32;
    let kw = kernel_size[1] as i32;
    let sh = stride[0] as i64;
    let sw = stride[1] as i64;

    // Calculate output dimensions
    let out_h = (h as i32 - kh) / stride[0] as i32 + 1;
    let out_w = (w as i32 - kw) / stride[1] as i32 + 1;

    // Build final shape: [N, out_H, out_W, kH, kW, C]
    let final_shape = vec![n, out_h, out_w, kh, kw, c];

    // Compute strides for the original array
    // Original layout is [N, H, W, C] with strides computed from shape
    let orig_strides: Vec<i64> = {
        let mut strides = vec![1i64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as i64;
        }
        strides
    };

    // Final strides: [N_stride, H_stride*sh, W_stride*sw, H_stride, W_stride, C_stride]
    let final_strides = vec![
        orig_strides[0],           // N stride
        orig_strides[1] * sh,      // out_H stride (moves by stride[0] in H dimension)
        orig_strides[2] * sw,      // out_W stride (moves by stride[1] in W dimension)
        orig_strides[1],           // kH stride (moves by 1 in H dimension)
        orig_strides[2],           // kW stride (moves by 1 in W dimension)
        orig_strides[3],           // C stride
    ];

    // Create strided view
    let strided = mlx_rs::ops::as_strided(x, &final_shape[..], &final_strides[..], None)
        .expect("as_strided");

    // Apply pooling operation on kernel dimensions (axes -3 and -2, i.e., 3 and 4)
    // This reduces [N, out_H, out_W, kH, kW, C] -> [N, out_H, out_W, C]
    let axes = [-3, -2];
    pooling_op(&strided, &axes).expect("pooling reduction")
}

/// Helper function for 1D pooling using as_strided approach.
/// Input shape: [N, L, C] (NLC format - MLX native)
/// Returns: pooled output with shape [N, out_L, C]
fn pool1d_strided<F>(
    x: &Array,
    kernel_size: usize,
    stride: usize,
    pooling_op: F,
) -> Array
where
    F: Fn(&Array, &[i32]) -> Result<Array, mlx_rs::error::Exception>,
{
    let shape = x.shape();
    let n = shape[0];
    let l = shape[1];
    let c = shape[2];

    let k = kernel_size as i32;
    let s = stride as i64;

    // Calculate output dimension
    let out_l = (l as i32 - k) / stride as i32 + 1;

    // Build final shape: [N, out_L, K, C]
    let final_shape = vec![n, out_l, k, c];

    // Compute strides for the original array [N, L, C]
    let orig_strides: Vec<i64> = {
        let mut strides = vec![1i64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as i64;
        }
        strides
    };

    // Final strides: [N_stride, L_stride*s, L_stride, C_stride]
    let final_strides = vec![
        orig_strides[0],           // N stride
        orig_strides[1] * s,       // out_L stride
        orig_strides[1],           // K stride
        orig_strides[2],           // C stride
    ];

    // Create strided view
    let strided = mlx_rs::ops::as_strided(x, &final_shape[..], &final_strides[..], None)
        .expect("as_strided");

    // Apply pooling operation on kernel dimension (axis -2, i.e., 2)
    let axes = [-2];
    pooling_op(&strided, &axes).expect("pooling reduction")
}

/// Helper for max_pool2d_with_indices.
/// Returns both max values and flat indices into the padded input.
/// Input shape: [N, H, W, C] (NHWC format)
/// Returns: (output [N, out_H, out_W, C], indices [N, out_H, out_W, C])
fn max_pool2d_with_indices_impl(
    x: &Array,
    kernel_size: [usize; 2],
    stride: [usize; 2],
) -> (Array, Array) {
    let shape = x.shape();
    let n = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];

    let kh = kernel_size[0] as i32;
    let kw = kernel_size[1] as i32;
    let sh = stride[0] as i64;
    let sw = stride[1] as i64;

    // Calculate output dimensions
    let out_h = (h as i32 - kh) / stride[0] as i32 + 1;
    let out_w = (w as i32 - kw) / stride[1] as i32 + 1;

    // Build final shape: [N, out_H, out_W, kH, kW, C]
    let final_shape = vec![n, out_h, out_w, kh, kw, c];

    // Compute strides for the original array [N, H, W, C]
    let orig_strides: Vec<i64> = {
        let mut strides = vec![1i64; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1] as i64;
        }
        strides
    };

    // Final strides: [N_stride, H_stride*sh, W_stride*sw, H_stride, W_stride, C_stride]
    let final_strides = vec![
        orig_strides[0],
        orig_strides[1] * sh,
        orig_strides[2] * sw,
        orig_strides[1],
        orig_strides[2],
        orig_strides[3],
    ];

    // Create strided view: [N, out_H, out_W, kH, kW, C]
    let strided = mlx_rs::ops::as_strided(x, &final_shape[..], &final_strides[..], None)
        .expect("as_strided");

    // Flatten kernel dimensions: [N, out_H, out_W, kH*kW, C]
    let flat_kernel = kh * kw;
    let reshaped = strided.reshape(&[n, out_h, out_w, flat_kernel, c]).expect("reshape");

    // Get max values: reduce on axis 3 (the flattened kernel axis)
    let output = reshaped.max_axis(3, None).expect("max_axis");

    // Get argmax indices within each kernel window (axis 3)
    let local_indices = mlx_rs::ops::indexing::argmax_axis(&reshaped, 3, None).expect("argmax");

    // Convert local indices (within kernel) to flat indices into padded NHWC input
    // For each output position (n, oh, ow, c), the local_idx tells us which element
    // in the kH*kW kernel was the max.
    //
    // The actual position in the padded input (NHWC layout) is:
    //   n * (H * W * C) + (oh * stride[0] + local_h) * (W * C) + (ow * stride[1] + local_w) * C + c
    // where local_h = local_idx / kW, local_w = local_idx % kW
    //
    // We need to compute this index for the backward pass.

    // Create coordinate arrays for output positions
    // Shape of output/indices: [N, out_H, out_W, C]
    let out_h_size = out_h as usize;
    let out_w_size = out_w as usize;
    let n_size = n as usize;
    let c_size = c as usize;
    let h_size = h as usize;
    let w_size = w as usize;

    // Create index arrays for n, oh, ow, c dimensions
    // n_idx: [N, 1, 1, 1] broadcast to [N, out_H, out_W, C]
    let n_range: Vec<i32> = (0..n_size as i32).collect();
    let n_idx = Array::from_slice(&n_range, &[n_size as i32])
        .reshape(&[n, 1, 1, 1]).expect("reshape");

    // oh_idx: [1, out_H, 1, 1]
    let oh_range: Vec<i32> = (0..out_h_size as i32).collect();
    let oh_idx = Array::from_slice(&oh_range, &[out_h_size as i32])
        .reshape(&[1, out_h, 1, 1]).expect("reshape");

    // ow_idx: [1, 1, out_W, 1]
    let ow_range: Vec<i32> = (0..out_w_size as i32).collect();
    let ow_idx = Array::from_slice(&ow_range, &[out_w_size as i32])
        .reshape(&[1, 1, out_w, 1]).expect("reshape");

    // c_idx: [1, 1, 1, C]
    let c_range: Vec<i32> = (0..c_size as i32).collect();
    let c_idx = Array::from_slice(&c_range, &[c_size as i32])
        .reshape(&[1, 1, 1, c]).expect("reshape");

    // Compute local_h and local_w from local_indices
    let kw_arr = Array::from_int(kw);
    let local_h = mlx_rs::ops::floor_divide(&local_indices, &kw_arr).expect("div");
    let local_w = mlx_rs::ops::remainder(&local_indices, &kw_arr).expect("rem");

    // Compute actual h and w positions in padded input
    let sh_arr = Array::from_int(stride[0] as i32);
    let sw_arr = Array::from_int(stride[1] as i32);

    // actual_h = oh * stride[0] + local_h
    let actual_h = mlx_rs::ops::add(
        &mlx_rs::ops::multiply(&oh_idx, &sh_arr).expect("mul"),
        &local_h
    ).expect("add");

    // actual_w = ow * stride[1] + local_w
    let actual_w = mlx_rs::ops::add(
        &mlx_rs::ops::multiply(&ow_idx, &sw_arr).expect("mul"),
        &local_w
    ).expect("add");

    // Compute flat index: n * (H * W * C) + h * (W * C) + w * C + c
    let hwc = Array::from_int((h_size * w_size * c_size) as i32);
    let wc = Array::from_int((w_size * c_size) as i32);
    let c_stride = Array::from_int(c_size as i32);

    let flat_indices = mlx_rs::ops::add(
        &mlx_rs::ops::add(
            &mlx_rs::ops::add(
                &mlx_rs::ops::multiply(&n_idx, &hwc).expect("mul"),
                &mlx_rs::ops::multiply(&actual_h, &wc).expect("mul")
            ).expect("add"),
            &mlx_rs::ops::multiply(&actual_w, &c_stride).expect("mul")
        ).expect("add"),
        &c_idx
    ).expect("add");

    (output, flat_indices)
}

impl ModuleOps<Self> for Mlx {
    fn conv1d(
        x: MlxTensorPrimitive,
        weight: MlxTensorPrimitive,
        bias: Option<MlxTensorPrimitive>,
        options: ConvOptions<1>,
    ) -> MlxTensorPrimitive {
        // MLX conv1d: expects [N, L, C_in], weight [C_out, K, C_in]
        // Burn uses [N, C_in, L], weight [C_out, C_in, K]

        // Transpose input from [N, C_in, L] to [N, L, C_in]
        let x_t = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 1]).expect("transpose");

        // Transpose weight from [C_out, C_in, K] to [C_out, K, C_in]
        let w_t = mlx_rs::ops::transpose_axes(&weight.array, &[0, 2, 1]).expect("transpose");

        let result = mlx_rs::ops::conv1d(
            &x_t,
            &w_t,
            options.stride[0] as i32,
            options.padding[0] as i32,
            options.dilation[0] as i32,
            options.groups as i32,
        ).expect("conv1d");

        // Transpose output back from [N, L_out, C_out] to [N, C_out, L_out]
        let mut output = mlx_rs::ops::transpose_axes(&result, &[0, 2, 1]).expect("transpose");

        // Add bias if provided
        if let Some(b) = bias {
            // Reshape bias from [C_out] to [1, C_out, 1]
            let b_shape = b.shape();
            let b_reshaped = b.array.reshape(&[1, b_shape[0] as i32, 1]).expect("reshape bias");
            output = mlx_rs::ops::add(&output, &b_reshaped).expect("add bias");
        }

        MlxTensorPrimitive::new(output)
    }

    fn conv2d(
        x: MlxTensorPrimitive,
        weight: MlxTensorPrimitive,
        bias: Option<MlxTensorPrimitive>,
        options: ConvOptions<2>,
    ) -> MlxTensorPrimitive {
        // MLX conv2d: expects [N, H, W, C_in], weight [C_out, Kh, Kw, C_in]
        // Burn uses [N, C_in, H, W], weight [C_out, C_in, Kh, Kw]

        // Transpose input from [N, C_in, H, W] to [N, H, W, C_in]
        let x_t = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 3, 1]).expect("transpose");

        // Transpose weight from [C_out, C_in, Kh, Kw] to [C_out, Kh, Kw, C_in]
        let w_t = mlx_rs::ops::transpose_axes(&weight.array, &[0, 2, 3, 1]).expect("transpose");

        let stride = (options.stride[0] as i32, options.stride[1] as i32);
        let padding = (options.padding[0] as i32, options.padding[1] as i32);
        let dilation = (options.dilation[0] as i32, options.dilation[1] as i32);

        let result = mlx_rs::ops::conv2d(
            &x_t,
            &w_t,
            stride,
            padding,
            dilation,
            options.groups as i32,
        ).expect("conv2d");

        // Transpose output back from [N, H_out, W_out, C_out] to [N, C_out, H_out, W_out]
        let mut output = mlx_rs::ops::transpose_axes(&result, &[0, 3, 1, 2]).expect("transpose");

        // Add bias if provided
        if let Some(b) = bias {
            let b_shape = b.shape();
            let b_reshaped = b.array.reshape(&[1, b_shape[0] as i32, 1, 1]).expect("reshape bias");
            output = mlx_rs::ops::add(&output, &b_reshaped).expect("add bias");
        }

        MlxTensorPrimitive::new(output)
    }

    fn conv3d(
        x: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _bias: Option<MlxTensorPrimitive>,
        _options: ConvOptions<3>,
    ) -> MlxTensorPrimitive {
        // MLX doesn't have native conv3d - placeholder
        x
    }

    fn conv_transpose1d(
        x: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _bias: Option<MlxTensorPrimitive>,
        _options: ConvTransposeOptions<1>,
    ) -> MlxTensorPrimitive {
        // conv_transpose1d is complex in MLX - placeholder
        x
    }

    fn conv_transpose2d(
        x: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _bias: Option<MlxTensorPrimitive>,
        _options: ConvTransposeOptions<2>,
    ) -> MlxTensorPrimitive {
        // conv_transpose2d is complex in MLX - placeholder
        x
    }

    fn conv_transpose3d(
        x: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _bias: Option<MlxTensorPrimitive>,
        _options: ConvTransposeOptions<3>,
    ) -> MlxTensorPrimitive {
        // Placeholder
        x
    }

    fn deform_conv2d(
        _x: MlxTensorPrimitive,
        _offset: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _mask: Option<MlxTensorPrimitive>,
        _bias: Option<MlxTensorPrimitive>,
        _options: DeformConvOptions<2>,
    ) -> MlxTensorPrimitive {
        // Deformable convolution is not supported in MLX - placeholder
        let shape = [1i32, 1, 1, 1];
        let array = Array::zeros::<f32>(&shape).expect("zeros");
        MlxTensorPrimitive::new(array)
    }

    fn deform_conv2d_backward(
        _x: MlxTensorPrimitive,
        _offset: MlxTensorPrimitive,
        _weight: MlxTensorPrimitive,
        _mask: Option<MlxTensorPrimitive>,
        _bias: Option<MlxTensorPrimitive>,
        _out_grad: MlxTensorPrimitive,
        _options: DeformConvOptions<2>,
    ) -> DeformConv2dBackward<Mlx> {
        // Placeholder
        let shape = [1i32, 1, 1, 1];
        let zeros = MlxTensorPrimitive::new(Array::zeros::<f32>(&shape).expect("zeros"));
        DeformConv2dBackward::new(
            zeros.clone(),
            zeros.clone(),
            zeros.clone(),
            Some(zeros.clone()),
            Some(zeros),
        )
    }

    fn avg_pool1d(
        x: MlxTensorPrimitive,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        _count_include_pad: bool,
    ) -> MlxTensorPrimitive {
        // Burn uses NCL format, MLX uses NLC format
        // Transpose from [N, C, L] to [N, L, C]
        let x_nhwc = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 1]).expect("transpose");

        // Apply padding if needed
        let x_padded = if padding > 0 {
            let pad = padding as i32;
            // Pad only the L dimension (axis 1 in NLC format)
            // PadWidth for [N, L, C]: [(0,0), (pad,pad), (0,0)]
            mlx_rs::ops::pad(
                &x_nhwc,
                &[(0, 0), (pad, pad), (0, 0)],
                None,
                None,
            ).expect("pad")
        } else {
            x_nhwc
        };

        // Apply pooling using as_strided + mean_axes
        let pooled = pool1d_strided(&x_padded, kernel_size, stride, |arr, axes| {
            arr.mean_axes(axes, None)
        });

        // Transpose back from [N, L, C] to [N, C, L]
        let output = mlx_rs::ops::transpose_axes(&pooled, &[0, 2, 1]).expect("transpose");

        MlxTensorPrimitive::new(output)
    }

    fn avg_pool2d(
        x: MlxTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        _count_include_pad: bool,
    ) -> MlxTensorPrimitive {
        // Burn uses NCHW format, MLX uses NHWC format
        // Transpose from [N, C, H, W] to [N, H, W, C]
        let x_nhwc = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 3, 1]).expect("transpose");

        // Apply padding if needed
        let x_padded = if padding[0] > 0 || padding[1] > 0 {
            let pad_h = padding[0] as i32;
            let pad_w = padding[1] as i32;
            // Pad H and W dimensions (axes 1 and 2 in NHWC format)
            // PadWidth for [N, H, W, C]: [(0,0), (pad_h,pad_h), (pad_w,pad_w), (0,0)]
            mlx_rs::ops::pad(
                &x_nhwc,
                &[(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)],
                None,
                None,
            ).expect("pad")
        } else {
            x_nhwc
        };

        // Apply pooling using as_strided + mean_axes
        let pooled = pool2d_strided(&x_padded, kernel_size, stride, |arr, axes| {
            arr.mean_axes(axes, None)
        });

        // Transpose back from [N, H, W, C] to [N, C, H, W]
        let output = mlx_rs::ops::transpose_axes(&pooled, &[0, 3, 1, 2]).expect("transpose");

        MlxTensorPrimitive::new(output)
    }

    fn avg_pool2d_backward(
        x: MlxTensorPrimitive,
        grad: MlxTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        _count_include_pad: bool,
    ) -> MlxTensorPrimitive {
        // Burn uses NCHW format
        let input_shape = x.shape();
        let n = input_shape[0];
        let c = input_shape[1];
        let h = input_shape[2];
        let w = input_shape[3];

        let kh = kernel_size[0];
        let kw = kernel_size[1];
        let sh = stride[0];
        let sw = stride[1];
        let pad_h = padding[0];
        let pad_w = padding[1];

        // Padded input dimensions
        let h_padded = h + 2 * pad_h;
        let w_padded = w + 2 * pad_w;

        // Output dimensions
        let out_h = (h_padded - kh) / sh + 1;
        let out_w = (w_padded - kw) / sw + 1;

        let pool_size = (kh * kw) as f32;

        // Transpose grad from NCHW to NHWC for processing
        let grad_nhwc = mlx_rs::ops::transpose_axes(&grad.array, &[0, 2, 3, 1]).expect("transpose");

        // Scale gradient by 1/pool_size
        let scale = Array::from_f32(1.0 / pool_size);
        let grad_scaled = mlx_rs::ops::multiply(&grad_nhwc, &scale).expect("multiply");

        // Create zeros for padded input gradient (NHWC format)
        let grad_input_padded = Array::zeros::<f32>(&[
            n as i32,
            h_padded as i32,
            w_padded as i32,
            c as i32,
        ]).expect("zeros");

        // For avg pooling backward, each output gradient contributes equally to all
        // input positions in its window. We use scatter_add to accumulate gradients.
        //
        // For each output position (oh, ow), the window covers:
        //   h_start = oh * stride[0]
        //   w_start = ow * stride[1]
        //   positions: (h_start..h_start+kH, w_start..w_start+kW)

        // Create flat indices for all input positions that receive gradients
        // We need to iterate over all output positions and all kernel positions

        // Build index arrays
        // For each (oh, ow, kh_off, kw_off), compute flat index into padded input

        let mut all_indices: Vec<i32> = Vec::with_capacity(n * out_h * out_w * kh * kw * c);
        let mut all_n_indices: Vec<i32> = Vec::with_capacity(n * out_h * out_w * kh * kw * c);
        let mut update_indices: Vec<usize> = Vec::with_capacity(n * out_h * out_w * kh * kw * c);

        for ni in 0..n {
            for ohi in 0..out_h {
                for owi in 0..out_w {
                    let h_start = ohi * sh;
                    let w_start = owi * sw;
                    for khi in 0..kh {
                        for kwi in 0..kw {
                            let hi = h_start + khi;
                            let wi = w_start + kwi;
                            for ci in 0..c {
                                // Flat index in NHWC layout
                                let flat_idx = (ni * h_padded * w_padded * c
                                    + hi * w_padded * c
                                    + wi * c
                                    + ci) as i32;
                                all_indices.push(flat_idx);
                                all_n_indices.push(ni as i32);
                                // Index into the flat grad_scaled array
                                let grad_idx = ni * out_h * out_w * c
                                    + ohi * out_w * c
                                    + owi * c
                                    + ci;
                                update_indices.push(grad_idx);
                            }
                        }
                    }
                }
            }
        }

        // Flatten the scaled gradient and gather the values we need
        let grad_flat = grad_scaled.flatten(None, None).expect("flatten");
        let update_idx_arr = Array::from_slice(
            &update_indices.iter().map(|&x| x as i32).collect::<Vec<_>>(),
            &[update_indices.len() as i32],
        );
        let updates = take_axis(&grad_flat, &update_idx_arr, 0).expect("take");

        // Flatten the input gradient and use scatter_add
        let grad_input_flat = grad_input_padded.flatten(None, None).expect("flatten");
        let indices_arr = Array::from_slice(&all_indices, &[all_indices.len() as i32]);

        // Use scatter_add: add updates to grad_input_flat at indices
        let result_flat = mlx_rs::ops::scatter_add(
            &grad_input_flat,
            &[&indices_arr],
            &updates,
            &[0],
        ).expect("scatter_add");

        // Reshape back to NHWC
        let result_nhwc = result_flat.reshape(&[
            n as i32,
            h_padded as i32,
            w_padded as i32,
            c as i32,
        ]).expect("reshape");

        // Remove padding if present
        let result_unpadded = if pad_h > 0 || pad_w > 0 {
            mlx_rs::ops::slice(
                &result_nhwc,
                &[0, pad_h as i32, pad_w as i32, 0],
                &[n as i32, (pad_h + h) as i32, (pad_w + w) as i32, c as i32],
                None,
            ).expect("slice")
        } else {
            result_nhwc
        };

        // Transpose back from NHWC to NCHW
        let output = mlx_rs::ops::transpose_axes(&result_unpadded, &[0, 3, 1, 2]).expect("transpose");

        MlxTensorPrimitive::new(output)
    }

    fn max_pool1d(
        x: MlxTensorPrimitive,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        _dilation: usize,
    ) -> MlxTensorPrimitive {
        // Burn uses NCL format, MLX uses NLC format
        // Transpose from [N, C, L] to [N, L, C]
        let x_nlc = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 1]).expect("transpose");

        // Apply padding if needed (use -inf for max pooling)
        let x_padded = if padding > 0 {
            let pad = padding as i32;
            let neg_inf = Array::from_f32(f32::NEG_INFINITY);
            mlx_rs::ops::pad(
                &x_nlc,
                &[(0, 0), (pad, pad), (0, 0)],
                neg_inf,
                None,
            ).expect("pad")
        } else {
            x_nlc
        };

        // Apply pooling using as_strided + max_axes
        let pooled = pool1d_strided(&x_padded, kernel_size, stride, |arr, axes| {
            arr.max_axes(axes, None)
        });

        // Transpose back from [N, L, C] to [N, C, L]
        let output = mlx_rs::ops::transpose_axes(&pooled, &[0, 2, 1]).expect("transpose");

        MlxTensorPrimitive::new(output)
    }

    fn max_pool2d(
        x: MlxTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        _dilation: [usize; 2],
    ) -> MlxTensorPrimitive {
        // Burn uses NCHW format, MLX uses NHWC format
        // Transpose from [N, C, H, W] to [N, H, W, C]
        let x_nhwc = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 3, 1]).expect("transpose");

        // Apply padding if needed (use -inf for max pooling)
        let x_padded = if padding[0] > 0 || padding[1] > 0 {
            let pad_h = padding[0] as i32;
            let pad_w = padding[1] as i32;
            let neg_inf = Array::from_f32(f32::NEG_INFINITY);
            mlx_rs::ops::pad(
                &x_nhwc,
                &[(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)],
                neg_inf,
                None,
            ).expect("pad")
        } else {
            x_nhwc
        };

        // Apply pooling using as_strided + max_axes
        let pooled = pool2d_strided(&x_padded, kernel_size, stride, |arr, axes| {
            arr.max_axes(axes, None)
        });

        // Transpose back from [N, H, W, C] to [N, C, H, W]
        let output = mlx_rs::ops::transpose_axes(&pooled, &[0, 3, 1, 2]).expect("transpose");

        MlxTensorPrimitive::new(output)
    }

    fn max_pool1d_with_indices(
        x: MlxTensorPrimitive,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> MaxPool1dWithIndices<Mlx> {
        let output = Self::max_pool1d(x, kernel_size, stride, padding, dilation);
        // Create dummy indices (placeholder)
        let indices = MlxTensorPrimitive::new(
            Array::zeros::<i32>(&output.array.shape().iter().map(|&s| s as i32).collect::<Vec<_>>())
                .expect("zeros")
        );
        MaxPool1dWithIndices::new(output, indices)
    }

    fn max_pool2d_with_indices(
        x: MlxTensorPrimitive,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        _dilation: [usize; 2],
    ) -> MaxPool2dWithIndices<Mlx> {
        // Burn uses NCHW format, MLX uses NHWC format
        // Transpose from [N, C, H, W] to [N, H, W, C]
        let x_nhwc = mlx_rs::ops::transpose_axes(&x.array, &[0, 2, 3, 1]).expect("transpose");

        // Apply padding if needed (use -inf for max pooling)
        let x_padded = if padding[0] > 0 || padding[1] > 0 {
            let pad_h = padding[0] as i32;
            let pad_w = padding[1] as i32;
            let neg_inf = Array::from_f32(f32::NEG_INFINITY);
            mlx_rs::ops::pad(
                &x_nhwc,
                &[(0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)],
                neg_inf,
                None,
            ).expect("pad")
        } else {
            x_nhwc
        };

        // Get max values and indices
        let (output_nhwc, indices_nhwc) = max_pool2d_with_indices_impl(&x_padded, kernel_size, stride);

        // Transpose back from [N, H, W, C] to [N, C, H, W]
        let output = mlx_rs::ops::transpose_axes(&output_nhwc, &[0, 3, 1, 2]).expect("transpose");
        let indices = mlx_rs::ops::transpose_axes(&indices_nhwc, &[0, 3, 1, 2]).expect("transpose");

        MaxPool2dWithIndices::new(
            MlxTensorPrimitive::new(output),
            MlxTensorPrimitive::new(indices),
        )
    }

    fn max_pool2d_with_indices_backward(
        x: MlxTensorPrimitive,
        _kernel_size: [usize; 2],
        _stride: [usize; 2],
        padding: [usize; 2],
        _dilation: [usize; 2],
        output_grad: MlxTensorPrimitive,
        indices: MlxTensorPrimitive,
    ) -> MaxPool2dBackward<Mlx> {
        // The indices contain flat indices into the padded NHWC input tensor.
        // We need to scatter the gradients to those positions.

        let input_shape = x.shape();
        let n = input_shape[0];
        let c = input_shape[1];
        let h = input_shape[2];
        let w = input_shape[3];

        let pad_h = padding[0];
        let pad_w = padding[1];

        // Padded dimensions
        let h_padded = h + 2 * pad_h;
        let w_padded = w + 2 * pad_w;

        // Create zeros for padded input gradient (NHWC flattened)
        let total_size = n * h_padded * w_padded * c;
        let grad_input_flat = Array::zeros::<f32>(&[total_size as i32]).expect("zeros");

        // Transpose grad and indices from NCHW to NHWC to match index computation
        let grad_nhwc = mlx_rs::ops::transpose_axes(&output_grad.array, &[0, 2, 3, 1]).expect("transpose");
        let indices_nhwc = mlx_rs::ops::transpose_axes(&indices.array, &[0, 2, 3, 1]).expect("transpose");

        // Flatten both
        let grad_flat = grad_nhwc.flatten(None, None).expect("flatten");
        let indices_flat = indices_nhwc.flatten(None, None).expect("flatten");

        // Scatter the gradients to the positions indicated by indices
        let result_flat = mlx_rs::ops::scatter_add(
            &grad_input_flat,
            &[&indices_flat],
            &grad_flat,
            &[0],
        ).expect("scatter_add");

        // Reshape to NHWC
        let result_nhwc = result_flat.reshape(&[
            n as i32,
            h_padded as i32,
            w_padded as i32,
            c as i32,
        ]).expect("reshape");

        // Remove padding if present
        let result_unpadded = if pad_h > 0 || pad_w > 0 {
            mlx_rs::ops::slice(
                &result_nhwc,
                &[0, pad_h as i32, pad_w as i32, 0],
                &[n as i32, (pad_h + h) as i32, (pad_w + w) as i32, c as i32],
                None,
            ).expect("slice")
        } else {
            result_nhwc
        };

        // Transpose back from NHWC to NCHW
        let output = mlx_rs::ops::transpose_axes(&result_unpadded, &[0, 3, 1, 2]).expect("transpose");

        MaxPool2dBackward::new(MlxTensorPrimitive::new(output))
    }

    fn adaptive_avg_pool1d(x: MlxTensorPrimitive, output_size: usize) -> MlxTensorPrimitive {
        // Calculate kernel_size and stride to achieve output_size
        let input_size = x.shape()[2];
        let stride = input_size / output_size;
        let kernel_size = input_size - (output_size - 1) * stride;
        Self::avg_pool1d(x, kernel_size, stride, 0, true)
    }

    fn adaptive_avg_pool2d(x: MlxTensorPrimitive, output_size: [usize; 2]) -> MlxTensorPrimitive {
        let input_h = x.shape()[2];
        let input_w = x.shape()[3];

        let stride_h = input_h / output_size[0];
        let stride_w = input_w / output_size[1];

        let kernel_h = input_h - (output_size[0] - 1) * stride_h;
        let kernel_w = input_w - (output_size[1] - 1) * stride_w;

        Self::avg_pool2d(x, [kernel_h, kernel_w], [stride_h, stride_w], [0, 0], true)
    }

    fn adaptive_avg_pool2d_backward(
        x: MlxTensorPrimitive,
        _grad: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        // Placeholder: return zeros with input shape
        let shape: Vec<i32> = x.shape().iter().map(|&s| s as i32).collect();
        let output = Array::zeros::<f32>(&shape).expect("zeros");
        MlxTensorPrimitive::new(output)
    }

    fn interpolate(
        x: MlxTensorPrimitive,
        _output_size: [usize; 2],
        _options: InterpolateOptions,
    ) -> MlxTensorPrimitive {
        // MLX doesn't have direct interpolate - placeholder
        x
    }

    fn interpolate_backward(
        x: MlxTensorPrimitive,
        _grad: MlxTensorPrimitive,
        _output_size: [usize; 2],
        _options: InterpolateOptions,
    ) -> MlxTensorPrimitive {
        // Placeholder: return zeros with input shape
        let shape: Vec<i32> = x.shape().iter().map(|&s| s as i32).collect();
        let output = Array::zeros::<f32>(&shape).expect("zeros");
        MlxTensorPrimitive::new(output)
    }

    fn embedding(
        weights: MlxTensorPrimitive,
        indices: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        // Embedding lookup - gather rows from weights based on indices
        let array = take_axis(&weights.array, &indices.array, 0)
            .expect("embedding");
        MlxTensorPrimitive::new(array)
    }

    fn embedding_backward(
        weights: MlxTensorPrimitive,
        _output_grad: MlxTensorPrimitive,
        _indices: MlxTensorPrimitive,
    ) -> MlxTensorPrimitive {
        // Scatter gradients back to weights
        // Placeholder - proper implementation needed
        weights
    }
}
