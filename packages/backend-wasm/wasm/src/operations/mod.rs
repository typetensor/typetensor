/*!
 * Tensor operations implementation for WebAssembly backend
 * 
 * This module provides high-performance implementations of tensor operations
 * optimized for WebAssembly, with SIMD support where available.
 * 
 * Uses direct ownership model for immutable tensor buffers - no RefCell needed.
 */

pub mod unary;
pub mod binary;
pub mod matmul;
pub mod view;
pub mod reduction;

use wasm_bindgen::prelude::*;
use crate::types::{WasmOperation, WasmTensorMeta, WasmResult, WasmError};
use crate::memory::{WasmMemoryManager, WasmBufferHandle};

/// Main operation dispatcher for WASM backend - Direct ownership model
#[wasm_bindgen]
pub struct WasmOperationDispatcher {
    memory_manager: WasmMemoryManager,  // Direct ownership - no RefCell!
}

#[wasm_bindgen]
impl WasmOperationDispatcher {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmOperationDispatcher {
        crate::utils::log_with_timing("Initializing WASM operation dispatcher");
        
        WasmOperationDispatcher {
            memory_manager: WasmMemoryManager::new(),
        }
    }

    /// Execute a tensor operation with direct ownership - no RefCell conflicts
    #[wasm_bindgen]
    pub fn execute_unary(
        &mut self,
        operation: WasmOperation,
        input: &WasmBufferHandle,
        input_meta: &WasmTensorMeta,
        output_meta: &WasmTensorMeta,
        output_handle: Option<WasmBufferHandle>,
    ) -> Result<WasmBufferHandle, JsValue> {
        let result = self.execute_internal(
            operation, 
            vec![input.clone()], 
            vec![input_meta.clone()], 
            output_meta.clone(), 
            output_handle
        );
        result.map_err(|e| e.into())
    }
    
    #[wasm_bindgen]
    pub fn execute_binary(
        &mut self,
        operation: WasmOperation,
        input_a: &WasmBufferHandle,
        input_b: &WasmBufferHandle,
        input_meta_a: &WasmTensorMeta,
        input_meta_b: &WasmTensorMeta,
        output_meta: &WasmTensorMeta,
        output_handle: Option<WasmBufferHandle>,
    ) -> Result<WasmBufferHandle, JsValue> {
        let result = self.execute_internal(
            operation,
            vec![input_a.clone(), input_b.clone()],
            vec![input_meta_a.clone(), input_meta_b.clone()],
            output_meta.clone(),
            output_handle
        );
        result.map_err(|e| e.into())
    }

    fn execute_internal(
        &mut self,
        operation: WasmOperation,
        inputs: Vec<WasmBufferHandle>,
        input_metas: Vec<WasmTensorMeta>,
        output_meta: WasmTensorMeta,
        output_handle: Option<WasmBufferHandle>,
    ) -> WasmResult<WasmBufferHandle> {
        // Create output buffer if not provided
        let (mut output, needs_initialization) = match output_handle {
            Some(handle) => (handle, false), // Already initialized
            None => {
                // Create empty buffer for the operation result
                let (handle, _write_ptr) = self.memory_manager.create_empty_buffer(output_meta.byte_size());
                (handle, true) // Needs initialization
            }
        };

        let result = match operation {
            // Creation operation
            WasmOperation::Create => {
                // For create operations, just return success
                Ok(())
            }

            // Unary operations
            WasmOperation::Neg | WasmOperation::Abs | WasmOperation::Sin | WasmOperation::Cos |
            WasmOperation::Exp | WasmOperation::Log | WasmOperation::Sqrt | WasmOperation::Square => {
                if inputs.len() != 1 {
                    return Err(WasmError::InvalidInput);
                }
                unary::execute_unary_op(
                    &mut self.memory_manager,
                    operation,
                    &inputs[0],
                    &input_metas[0],
                    &output,
                    &output_meta,
                )?;
                Ok(())
            }

            // Binary operations
            WasmOperation::Add | WasmOperation::Sub | WasmOperation::Mul | WasmOperation::Div => {
                if inputs.len() != 2 {
                    return Err(WasmError::InvalidInput);
                }
                binary::execute_binary_op(
                    &mut self.memory_manager,
                    operation,
                    &inputs[0],
                    &inputs[1],
                    &input_metas[0],
                    &input_metas[1],
                    &output,
                    &output_meta,
                )?;
                Ok(())
            }

            // Matrix operations
            WasmOperation::Matmul => {
                if inputs.len() != 2 {
                    return Err(WasmError::InvalidInput);
                }
                matmul::execute_matmul_op(
                    &mut self.memory_manager,
                    operation,
                    &inputs[0],
                    &inputs[1],
                    &input_metas[0],
                    &input_metas[1],
                    &output,
                    &output_meta,
                )?;
                Ok(())
            }

            // View operations
            WasmOperation::Reshape | WasmOperation::View | WasmOperation::Slice | WasmOperation::Flatten |
            WasmOperation::Permute | WasmOperation::Transpose | WasmOperation::Squeeze |
            WasmOperation::Unsqueeze | WasmOperation::Expand | WasmOperation::Tile => {
                if inputs.len() != 1 {
                    return Err(WasmError::InvalidInput);
                }
                view::execute_view_op(
                    &mut self.memory_manager,
                    operation,
                    &inputs[0],
                    &input_metas[0],
                    &output,
                    &output_meta,
                )?;
                Ok(())
            }

            // Reduction operations
            WasmOperation::Sum | WasmOperation::Mean | WasmOperation::Max |
            WasmOperation::Min | WasmOperation::Prod => {
                if inputs.len() != 1 {
                    return Err(WasmError::InvalidInput);
                }
                reduction::execute_reduction_op(
                    &mut self.memory_manager,
                    operation,
                    &inputs[0],
                    &input_metas[0],
                    &output,
                    &output_meta,
                )?;
                Ok(())
            }

            // Not yet implemented operations
            _ => Err(WasmError::NotImplemented),
        }?;
        
        // Mark buffer as initialized if it was newly created
        if needs_initialization {
            self.memory_manager.mark_buffer_initialized(&mut output);
        }
        
        Ok(output)
    }

    /// Get memory usage statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> crate::memory::WasmMemoryStats {
        self.memory_manager.get_memory_stats()
    }

    /// Create buffer with data (atomic operation - replaces separate allocate+write)
    #[wasm_bindgen]
    pub fn create_buffer_with_data(&mut self, data: &[u8]) -> WasmBufferHandle {
        self.memory_manager.create_buffer_with_data(data)
    }
    
    /// Create buffer with JS Uint8Array data (atomic operation for TypeScript bridge)
    #[wasm_bindgen]
    pub fn create_buffer_with_js_data(&mut self, js_array: &js_sys::Uint8Array) -> WasmBufferHandle {
        let data = js_array.to_vec();
        self.memory_manager.create_buffer_with_data(&data)
    }

    /// Get read pointer for immutable buffer access
    /// Note: This is an internal method, not exposed to JS
    pub(crate) fn get_read_ptr(&self, handle: &WasmBufferHandle) -> *const u8 {
        self.memory_manager.get_read_ptr(handle)
    }
    
    /// Release buffer back to pool for reuse
    #[wasm_bindgen]
    pub fn release_buffer(&mut self, handle: WasmBufferHandle) -> bool {
        self.memory_manager.release_buffer(handle)
    }
    
    /// Copy buffer data to JavaScript Uint8Array
    #[wasm_bindgen]
    pub fn copy_buffer_to_js(&self, handle: &WasmBufferHandle) -> js_sys::Uint8Array {
        let ptr = self.memory_manager.get_read_ptr(handle);
        let data = unsafe { std::slice::from_raw_parts(ptr, handle.size()) };
        js_sys::Uint8Array::from(data)
    }
    
    /// Compact memory pools to reduce memory usage
    #[wasm_bindgen]
    pub fn compact_pools(&mut self) {
        self.memory_manager.compact_pools();
    }
    
    /// Clone a buffer handle (increments reference count)
    #[wasm_bindgen]
    pub fn clone_buffer_handle(&self, handle: &WasmBufferHandle) -> WasmBufferHandle {
        // Increment reference count for the buffer
        self.memory_manager.increment_ref_count(handle.id());
        
        // Return a cloned handle
        handle.clone()
    }
}