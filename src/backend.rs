//! MLX Backend implementation for Burn.

use burn_tensor::backend::Backend;
use burn_tensor::{DType, TensorMetadata};
use burn_tensor::quantization::QuantScheme;
use mlx_rs::Array;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::MlxDevice;

// Global seed for random number generation
static SEED: AtomicU64 = AtomicU64::new(0);

/// MLX tensor primitive with shape tracking.
#[derive(Debug, Clone)]
pub struct MlxTensorPrimitive {
    /// The underlying MLX array.
    pub array: Array,
    /// Cached shape for fast access.
    pub shape: Vec<usize>,
}

impl MlxTensorPrimitive {
    /// Create a new tensor primitive.
    pub fn new(array: Array) -> Self {
        let shape = array.shape().iter().map(|&s| s as usize).collect();
        Self { array, shape }
    }

    /// Get the array reference.
    pub fn array(&self) -> &Array {
        &self.array
    }

    /// Get the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

// SAFETY: MLX arrays can be sent between threads.
// MLX uses internal synchronization for its compute graph.
unsafe impl Send for MlxTensorPrimitive {}
unsafe impl Sync for MlxTensorPrimitive {}

impl TensorMetadata for MlxTensorPrimitive {
    fn dtype(&self) -> DType {
        // Map MLX dtype to Burn dtype
        match self.array.dtype() {
            mlx_rs::Dtype::Float32 => DType::F32,
            mlx_rs::Dtype::Float16 => DType::F16,
            mlx_rs::Dtype::Bfloat16 => DType::BF16,
            mlx_rs::Dtype::Float64 => DType::F64,
            mlx_rs::Dtype::Int32 => DType::I32,
            mlx_rs::Dtype::Int64 => DType::I64,
            mlx_rs::Dtype::Bool => DType::Bool,
            _ => DType::F32, // Default fallback
        }
    }

    fn shape(&self) -> burn_tensor::Shape {
        burn_tensor::Shape::from(self.shape.clone())
    }
}

/// Quantized tensor primitive (placeholder for future implementation).
#[derive(Debug, Clone)]
pub struct MlxQuantizedTensorPrimitive {
    /// The underlying tensor (stored as float for now).
    pub tensor: MlxTensorPrimitive,
    /// Quantization scheme.
    pub scheme: QuantScheme,
}

// SAFETY: Same as MlxTensorPrimitive
unsafe impl Send for MlxQuantizedTensorPrimitive {}
unsafe impl Sync for MlxQuantizedTensorPrimitive {}

impl TensorMetadata for MlxQuantizedTensorPrimitive {
    fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    fn shape(&self) -> burn_tensor::Shape {
        burn_tensor::Shape::from(self.tensor.shape.clone())
    }
}

impl burn_tensor::quantization::QTensorPrimitive for MlxQuantizedTensorPrimitive {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }
}

/// MLX Backend for Burn.
#[derive(Debug, Default, Clone, Copy)]
pub struct Mlx;

impl Backend for Mlx {
    type Device = MlxDevice;

    type FloatTensorPrimitive = MlxTensorPrimitive;
    type FloatElem = f32;

    type IntTensorPrimitive = MlxTensorPrimitive;
    type IntElem = i32;

    type BoolTensorPrimitive = MlxTensorPrimitive;
    type BoolElem = bool;

    type QuantizedTensorPrimitive = MlxQuantizedTensorPrimitive;

    fn name(_device: &Self::Device) -> String {
        "mlx".to_string()
    }

    fn seed(_device: &Self::Device, seed: u64) {
        SEED.store(seed, Ordering::SeqCst);
        // MLX uses its own seeding mechanism
        let _ = mlx_rs::random::seed(seed);
    }

    fn supports_dtype(_device: &Self::Device, dtype: DType) -> bool {
        matches!(
            dtype,
            DType::F32 | DType::F64 | DType::F16 | DType::BF16 | DType::I32 | DType::I64 | DType::Bool
        )
    }
}

/// Get the current seed value.
pub fn get_seed() -> u64 {
    SEED.load(Ordering::SeqCst)
}
