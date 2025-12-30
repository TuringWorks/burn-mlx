//! MLX tensor operations for Burn backend.
//!
//! This module provides implementations of Burn's tensor operation traits
//! for the MLX backend.

mod base;
mod float_ops;
mod int_ops;
mod bool_ops;
mod module_ops;
mod other_ops;

// Re-export operations for internal use
#[allow(unused_imports)]
pub use base::concat;
