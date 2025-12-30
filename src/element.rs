//! Element type mappings between Burn and MLX.

use burn_tensor::{DType, Element};
use half::{bf16, f16};
use mlx_rs::Dtype;

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
