//! MLX device types for Burn.

use burn_tensor::backend::DeviceOps;
use std::fmt;

/// MLX device types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MlxDevice {
    /// CPU execution via MLX.
    Cpu,
    /// GPU execution on Apple Silicon.
    #[default]
    Gpu,
}

impl MlxDevice {
    /// Convert to mlx-rs Device.
    pub fn to_mlx_device(&self) -> mlx_rs::Device {
        match self {
            MlxDevice::Cpu => mlx_rs::Device::cpu(),
            MlxDevice::Gpu => mlx_rs::Device::gpu(),
        }
    }
}

impl fmt::Display for MlxDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MlxDevice::Cpu => write!(f, "MLX CPU"),
            MlxDevice::Gpu => write!(f, "MLX GPU"),
        }
    }
}

impl burn_tensor::backend::Device for MlxDevice {
    fn from_id(device_id: burn_tensor::backend::DeviceId) -> Self {
        match device_id.type_id {
            0 => MlxDevice::Cpu,
            _ => MlxDevice::Gpu,
        }
    }

    fn to_id(&self) -> burn_tensor::backend::DeviceId {
        match self {
            MlxDevice::Cpu => burn_tensor::backend::DeviceId::new(0, 0),
            MlxDevice::Gpu => burn_tensor::backend::DeviceId::new(1, 0),
        }
    }

    fn device_count(_type_id: u16) -> usize {
        1
    }
}

impl DeviceOps for MlxDevice {}
