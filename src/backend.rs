//! MLX Backend implementation for Burn.

use burn_tensor::backend::Backend;
use burn_tensor::TensorMetadata;
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
    fn dtype(&self) -> burn_tensor::DType {
        // Map MLX dtype to Burn dtype
        match self.array.dtype() {
            mlx_rs::Dtype::Float32 => burn_tensor::DType::F32,
            mlx_rs::Dtype::Float16 => burn_tensor::DType::F16,
            mlx_rs::Dtype::Bfloat16 => burn_tensor::DType::BF16,
            mlx_rs::Dtype::Float64 => burn_tensor::DType::F64,
            mlx_rs::Dtype::Int32 => burn_tensor::DType::I32,
            mlx_rs::Dtype::Int64 => burn_tensor::DType::I64,
            mlx_rs::Dtype::Bool => burn_tensor::DType::Bool,
            _ => burn_tensor::DType::F32, // Default fallback
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
    pub scheme: QuantizationScheme,
}

/// Quantization scheme.
#[derive(Debug, Clone, Copy, Default)]
pub enum QuantizationScheme {
    #[default]
    None,
}

// SAFETY: Same as MlxTensorPrimitive
unsafe impl Send for MlxQuantizedTensorPrimitive {}
unsafe impl Sync for MlxQuantizedTensorPrimitive {}

impl TensorMetadata for MlxQuantizedTensorPrimitive {
    fn dtype(&self) -> burn_tensor::DType {
        self.tensor.dtype()
    }

    fn shape(&self) -> burn_tensor::Shape {
        burn_tensor::Shape::from(self.tensor.shape.clone())
    }
}

impl burn_tensor::quantization::QTensorPrimitive for MlxQuantizedTensorPrimitive {
    fn scheme(&self) -> &burn_tensor::quantization::QuantizationScheme {
        // Return a reference to a static scheme
        static SYMMETRIC: burn_tensor::quantization::QuantizationScheme =
            burn_tensor::quantization::QuantizationScheme::PerTensorSymmetric(
                burn_tensor::quantization::QuantizationType::QInt8,
            );
        &SYMMETRIC
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
    type QuantizedEncoding = i8;

    fn name() -> String {
        "mlx".to_string()
    }

    fn seed(seed: u64) {
        SEED.store(seed, Ordering::SeqCst);
        // MLX uses its own seeding mechanism
        mlx_rs::random::seed(seed);
    }

    fn sync(device: &Self::Device) {
        // MLX is lazy-evaluated; sync forces evaluation
        // This is a no-op in MLX as synchronization happens implicitly
        // when reading tensor values
        let _ = device;
    }
}

/// Get the current seed value.
pub fn get_seed() -> u64 {
    SEED.load(Ordering::SeqCst)
}
