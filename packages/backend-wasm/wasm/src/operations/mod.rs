/*!
 * Tensor operations for WebAssembly backend
 * 
 * This module contains all the tensor operations supported by the WASM backend.
 * Operations are organized by type and use SIMD optimizations where available.
 * 
 * # Memory Safety
 * 
 * Operations use a pattern of casting `*const u8` to `*mut u8` for output tensors.
 * This is safe by design - see MEMORY_SAFETY.md for detailed explanation of why
 * this pattern is memory-safe within our arena-based architecture.
 */

pub mod unary;
pub mod binary;
pub mod matmul;
pub mod view;
pub mod reduction;
pub mod softmax;

use wasm_bindgen::prelude::*;
use crate::types::{WasmOperation, WasmTensorMeta, WasmResult, WasmError};

// Placeholder for new arena-based operation dispatcher
// Will be implemented in Phase 1 of the rewrite