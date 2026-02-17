//! Element type mappings between Burn and MLX.

use burn_tensor::{DType, Element};
use half::{bf16, f16};
use mlx_rs::{Array, Dtype};
use num_traits::{Float, FromPrimitive};

/// Trait for elements that can be used with MLX.
pub trait MlxElement: Element + Clone + Send + Sync + 'static {
    /// Get the MLX data type for this element.
    fn mlx_dtype() -> Dtype;

    /// Get the Burn DType for this element.
    fn dtype() -> DType;
}

impl MlxElement for f32 {
    fn mlx_dtype() -> Dtype {
        Dtype::Float32
    }
    fn dtype() -> DType {
        DType::F32
    }
}

impl MlxElement for f64 {
    fn mlx_dtype() -> Dtype {
        Dtype::Float64
    }
    fn dtype() -> DType {
        DType::F64
    }
}

impl MlxElement for f16 {
    fn mlx_dtype() -> Dtype {
        Dtype::Float16
    }
    fn dtype() -> DType {
        DType::F16
    }
}

impl MlxElement for bf16 {
    fn mlx_dtype() -> Dtype {
        Dtype::Bfloat16
    }
    fn dtype() -> DType {
        DType::BF16
    }
}

impl MlxElement for i32 {
    fn mlx_dtype() -> Dtype {
        Dtype::Int32
    }
    fn dtype() -> DType {
        DType::I32
    }
}

impl MlxElement for i64 {
    fn mlx_dtype() -> Dtype {
        Dtype::Int64
    }
    fn dtype() -> DType {
        DType::I64
    }
}

impl MlxElement for i16 {
    fn mlx_dtype() -> Dtype {
        Dtype::Int16
    }
    fn dtype() -> DType {
        DType::I16
    }
}

impl MlxElement for i8 {
    fn mlx_dtype() -> Dtype {
        Dtype::Int8
    }
    fn dtype() -> DType {
        DType::I8
    }
}

impl MlxElement for u8 {
    fn mlx_dtype() -> Dtype {
        Dtype::Uint8
    }
    fn dtype() -> DType {
        DType::U8
    }
}

impl MlxElement for u16 {
    fn mlx_dtype() -> Dtype {
        Dtype::Uint16
    }
    fn dtype() -> DType {
        DType::U16
    }
}

impl MlxElement for u32 {
    fn mlx_dtype() -> Dtype {
        Dtype::Uint32
    }
    fn dtype() -> DType {
        DType::U32
    }
}

impl MlxElement for u64 {
    fn mlx_dtype() -> Dtype {
        Dtype::Uint64
    }
    fn dtype() -> DType {
        DType::U64
    }
}

impl MlxElement for bool {
    fn mlx_dtype() -> Dtype {
        Dtype::Bool
    }
    fn dtype() -> DType {
        DType::Bool
    }
}

/// Trait for float elements that can be used as the primary float type in the MLX backend.
///
/// This enables `Mlx<F>` to be generic over the float precision (f32, f16, bf16, f64).
pub trait FloatMlxElement: MlxElement + Float + FromPrimitive {
    /// Create a scalar MLX array from this value.
    fn scalar_array(value: Self) -> Array;

    /// Create a scalar MLX array from an f64 constant.
    fn f64_scalar_array(value: f64) -> Array {
        Self::scalar_array(Self::from_f64(value).unwrap())
    }

    /// Create an MLX array from a slice of elements.
    fn array_from_slice(data: &[Self], shape: &[i32]) -> Array;

    /// Create a zeros array in this element's dtype.
    fn zeros_array(shape: &[i32]) -> Array;

    /// Create a ones array in this element's dtype.
    fn ones_array(shape: &[i32]) -> Array;

    /// Read an MLX array's data as a vector of this element type.
    fn array_to_vec(array: &Array) -> Vec<Self>;

    /// Cast an MLX array to this element's dtype.
    fn cast_array(array: &Array) -> Array;
}

impl FloatMlxElement for f32 {
    fn scalar_array(value: Self) -> Array { Array::from_f32(value) }
    fn array_from_slice(data: &[Self], shape: &[i32]) -> Array { Array::from_slice(data, shape) }
    fn zeros_array(shape: &[i32]) -> Array { Array::zeros::<f32>(shape).expect("zeros") }
    fn ones_array(shape: &[i32]) -> Array { Array::ones::<f32>(shape).expect("ones") }
    fn array_to_vec(array: &Array) -> Vec<Self> { array.as_slice::<f32>().to_vec() }
    fn cast_array(array: &Array) -> Array { array.as_type::<f32>().expect("cast") }
}

impl FloatMlxElement for f16 {
    fn scalar_array(value: Self) -> Array { Array::from_slice(&[value], &[1]) }
    fn array_from_slice(data: &[Self], shape: &[i32]) -> Array { Array::from_slice(data, shape) }
    fn zeros_array(shape: &[i32]) -> Array { Array::zeros::<f16>(shape).expect("zeros") }
    fn ones_array(shape: &[i32]) -> Array { Array::ones::<f16>(shape).expect("ones") }
    fn array_to_vec(array: &Array) -> Vec<Self> { array.as_slice::<f16>().to_vec() }
    fn cast_array(array: &Array) -> Array { array.as_type::<f16>().expect("cast") }
}

impl FloatMlxElement for bf16 {
    fn scalar_array(value: Self) -> Array { Array::from_slice(&[value], &[1]) }
    fn array_from_slice(data: &[Self], shape: &[i32]) -> Array { Array::from_slice(data, shape) }
    fn zeros_array(shape: &[i32]) -> Array { Array::zeros::<bf16>(shape).expect("zeros") }
    fn ones_array(shape: &[i32]) -> Array { Array::ones::<bf16>(shape).expect("ones") }
    fn array_to_vec(array: &Array) -> Vec<Self> { array.as_slice::<bf16>().to_vec() }
    fn cast_array(array: &Array) -> Array { array.as_type::<bf16>().expect("cast") }
}

impl FloatMlxElement for f64 {
    fn scalar_array(value: Self) -> Array {
        // f64 doesn't implement FromSliceElement in mlx-rs, route through f32
        let arr = Array::from_f32(value as f32);
        arr.as_type::<f64>().expect("cast to f64")
    }
    fn array_from_slice(data: &[Self], shape: &[i32]) -> Array {
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let arr = Array::from_slice(&f32_data, shape);
        arr.as_type::<f64>().expect("cast to f64")
    }
    fn zeros_array(shape: &[i32]) -> Array {
        let arr = Array::zeros::<f32>(shape).expect("zeros");
        arr.as_type::<f64>().expect("cast to f64")
    }
    fn ones_array(shape: &[i32]) -> Array {
        let arr = Array::ones::<f32>(shape).expect("ones");
        arr.as_type::<f64>().expect("cast to f64")
    }
    fn array_to_vec(array: &Array) -> Vec<Self> {
        let arr = array.as_type::<f32>().expect("cast to f32");
        arr.as_slice::<f32>().iter().map(|&v| v as f64).collect()
    }
    fn cast_array(array: &Array) -> Array { array.as_type::<f64>().expect("cast") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_mapping() {
        assert_eq!(f32::mlx_dtype(), Dtype::Float32);
        assert_eq!(f64::mlx_dtype(), Dtype::Float64);
        assert_eq!(i32::mlx_dtype(), Dtype::Int32);
        assert_eq!(bool::mlx_dtype(), Dtype::Bool);
    }
}
