
pub mod unary;
pub mod binary;
pub mod matmul;
pub mod view;
pub mod reduction;
pub mod softmax;

use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use crate::types::{WasmOperation, WasmTensorMeta, WasmResult, WasmError};
use crate::memory::{WasmMemoryManager, WasmBufferHandle};

/// Main operation dispatcher for WASM backend
#[wasm_bindgen]
pub struct WasmOperationDispatcher {
    memory_manager: RefCell<WasmMemoryManager>,
}

#[wasm_bindgen]
impl WasmOperationDispatcher {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmOperationDispatcher {
        WasmOperationDispatcher {
            memory_manager: RefCell::new(WasmMemoryManager::new()),
        }
    }

    /// Execute a tensor operation
    #[wasm_bindgen]
    pub fn execute_unary(
        &self,
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
        &self,
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
        &self,
        operation: WasmOperation,
        inputs: Vec<WasmBufferHandle>,
        input_metas: Vec<WasmTensorMeta>,
        output_meta: WasmTensorMeta,
        output_handle: Option<WasmBufferHandle>,
    ) -> WasmResult<WasmBufferHandle> {
        let (mut output, needs_initialization) = match output_handle {
            Some(handle) => (handle, false),
            None => {
                let (handle, _write_ptr) = self.memory_manager.borrow_mut()
                    .create_empty_buffer(output_meta.byte_size())
                    .map_err(|e| WasmError::MemoryAllocationFailed)?;
                (handle, true)
            }
        };

        let result = match operation {
            WasmOperation::Create => {
                Ok(())
            }

            WasmOperation::Neg | WasmOperation::Abs | WasmOperation::Sin | WasmOperation::Cos |
            WasmOperation::Exp | WasmOperation::Log | WasmOperation::Sqrt | WasmOperation::Square => {
                if inputs.len() != 1 {
                    return Err(WasmError::InvalidInput);
                }
                unary::execute_unary_op(
                    &mut *self.memory_manager.borrow_mut(),
                    operation,
                    &inputs[0],
                    &input_metas[0],
                    &output,
                    &output_meta,
                )?;
                Ok(())
            }

            WasmOperation::Add | WasmOperation::Sub | WasmOperation::Mul | WasmOperation::Div => {
                if inputs.len() != 2 {
                    return Err(WasmError::InvalidInput);
                }
                binary::execute_binary_op(
                    &mut *self.memory_manager.borrow_mut(),
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

            WasmOperation::Matmul => {
                if inputs.len() != 2 {
                    return Err(WasmError::InvalidInput);
                }
                matmul::execute_matmul_op(
                    &mut *self.memory_manager.borrow_mut(),
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

            WasmOperation::Reshape | WasmOperation::View | WasmOperation::Slice | WasmOperation::Flatten |
            WasmOperation::Permute | WasmOperation::Transpose | WasmOperation::Squeeze |
            WasmOperation::Unsqueeze | WasmOperation::Expand | WasmOperation::Tile => {
                if inputs.len() != 1 {
                    return Err(WasmError::InvalidInput);
                }
                view::execute_view_op(
                    &mut *self.memory_manager.borrow_mut(),
                    operation,
                    &inputs[0],
                    &input_metas[0],
                    &output,
                    &output_meta,
                )?;
                Ok(())
            }

            WasmOperation::Sum | WasmOperation::Mean | WasmOperation::Max |
            WasmOperation::Min | WasmOperation::Prod => {
                if inputs.len() != 1 {
                    return Err(WasmError::InvalidInput);
                }
                reduction::execute_reduction_op(
                    &mut *self.memory_manager.borrow_mut(),
                    operation,
                    &inputs[0],
                    &input_metas[0],
                    &output,
                    &output_meta,
                )?;
                Ok(())
            }
            WasmOperation::Softmax | WasmOperation::LogSoftmax => {
                if inputs.len() != 1 {
                    return Err(WasmError::InvalidInput);
                }
                softmax::execute_softmax_op(
                    &mut *self.memory_manager.borrow_mut(),
                    operation,
                    &inputs[0],
                    &input_metas[0],
                    &output,
                    &output_meta,
                    None, // Default axis (last dimension)
                )?;
                Ok(())
            }

            _ => Err(WasmError::NotImplemented),
        }?;
        
        if needs_initialization {
            self.memory_manager.borrow().mark_buffer_initialized(&mut output);
        }
        Ok(output)
    }

    /// Get memory usage statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> crate::memory::WasmMemoryStats {
        self.memory_manager.borrow().get_memory_stats()
    }

    /// Create buffer with data
    #[wasm_bindgen]
    pub fn create_buffer_with_data(&self, data: &[u8]) -> WasmBufferHandle {
        match self.memory_manager.borrow_mut().create_buffer_with_data(data) {
            Ok(handle) => handle,
            Err(e) => {
                wasm_bindgen::throw_str(&format!("Memory allocation failed: {}", e));
            }
        }
    }
    
    /// Create buffer with JS Uint8Array data
    #[wasm_bindgen]
    pub fn create_buffer_with_js_data(&self, js_array: &js_sys::Uint8Array) -> WasmBufferHandle {
        let data = js_array.to_vec();
        match self.memory_manager.borrow_mut().create_buffer_with_data(&data) {
            Ok(handle) => handle,
            Err(e) => {
                wasm_bindgen::throw_str(&format!("Memory allocation failed: {}", e));
            }
        }
    }

    /// Get read pointer for immutable buffer access
    pub(crate) fn get_read_ptr(&self, handle: &WasmBufferHandle) -> *const u8 {
        // Return the pointer directly from the handle to avoid borrow conflicts
        handle.get_read_ptr()
    }
    
    /// Release buffer back to pool for reuse
    #[wasm_bindgen]
    pub fn release_buffer(&self, handle: WasmBufferHandle) -> bool {
        self.memory_manager.borrow_mut().release_buffer(handle)
    }
    
    /// Copy buffer data to JavaScript Uint8Array
    #[wasm_bindgen]
    pub fn copy_buffer_to_js(&self, handle: &WasmBufferHandle) -> js_sys::Uint8Array {
        // Use handle directly to avoid RefCell borrow conflicts
        let ptr = handle.get_read_ptr();
        let data = unsafe { std::slice::from_raw_parts(ptr, handle.size()) };
        let copied_data = data.to_vec();
        js_sys::Uint8Array::from(&copied_data[..])
    }
    
    /// Compact memory pools to reduce memory usage
    #[wasm_bindgen]
    pub fn compact_pools(&self) {
        self.memory_manager.borrow_mut().compact_pools();
    }
    
    /// Perform intensive cleanup - for use during benchmarks or stress tests
    #[wasm_bindgen]
    pub fn intensive_cleanup(&self) {
        let mut manager = self.memory_manager.borrow_mut();
        manager.compact_pools();
        // Could add more aggressive cleanup here if needed
    }
    
    /// Clone a buffer handle
    #[wasm_bindgen]
    pub fn clone_buffer_handle(&self, handle: &WasmBufferHandle) -> WasmBufferHandle {
        self.memory_manager.borrow().increment_ref_count(handle.id());
        handle.clone()
    }
    
    /// Execute a reduction operation with axis information
    #[wasm_bindgen]
    pub fn execute_reduction(
        &self,
        operation: WasmOperation,
        input: &WasmBufferHandle,
        input_meta: &WasmTensorMeta,
        output_meta: &WasmTensorMeta,
        axes: Option<Vec<i32>>,
        keep_dims: bool,
        output_handle: Option<WasmBufferHandle>,
    ) -> Result<WasmBufferHandle, JsValue> {
        let axes_usize = axes.map(|v| v.into_iter().map(|x| x as usize).collect::<Vec<_>>());
        let result = self.execute_reduction_internal(
            operation,
            input.clone(),
            input_meta.clone(),
            output_meta.clone(),
            axes_usize,
            keep_dims,
            output_handle
        );
        result.map_err(|e| e.into())
    }
    
    /// Execute a softmax operation with axis information
    #[wasm_bindgen]
    pub fn execute_softmax(
        &self,
        operation: WasmOperation,
        input: &WasmBufferHandle,
        input_meta: &WasmTensorMeta,
        output_meta: &WasmTensorMeta,
        axis: i32,
        output_handle: Option<WasmBufferHandle>,
    ) -> Result<WasmBufferHandle, JsValue> {
        let result = self.execute_softmax_internal(
            operation,
            input.clone(),
            input_meta.clone(),
            output_meta.clone(),
            Some(axis),
            output_handle,
        );
        result.map_err(|e| e.into())
    }
    
    fn execute_reduction_internal(
        &self,
        operation: WasmOperation,
        input: WasmBufferHandle,
        input_meta: WasmTensorMeta,
        output_meta: WasmTensorMeta,
        axes: Option<Vec<usize>>,
        keep_dims: bool,
        output_handle: Option<WasmBufferHandle>,
    ) -> WasmResult<WasmBufferHandle> {
        let (mut output, needs_initialization) = match output_handle {
            Some(handle) => (handle, false),
            None => {
                let (handle, _write_ptr) = self.memory_manager.borrow_mut()
                    .create_empty_buffer(output_meta.byte_size())
                    .map_err(|e| WasmError::MemoryAllocationFailed)?;
                (handle, true)
            }
        };
        
        reduction::execute_reduction_op_with_axes(
            &mut *self.memory_manager.borrow_mut(),
            operation,
            &input,
            &input_meta,
            &output,
            &output_meta,
            axes.as_deref(),
            keep_dims,
        )?;
        
        if needs_initialization {
            self.memory_manager.borrow().mark_buffer_initialized(&mut output);
        }
        Ok(output)
    }
    
    fn execute_softmax_internal(
        &self,
        operation: WasmOperation,
        input: WasmBufferHandle,
        input_meta: WasmTensorMeta,
        output_meta: WasmTensorMeta,
        axis: Option<i32>,
        output_handle: Option<WasmBufferHandle>,
    ) -> WasmResult<WasmBufferHandle> {
        let (mut output, needs_initialization) = match output_handle {
            Some(handle) => (handle, false),
            None => {
                let (handle, _write_ptr) = self.memory_manager.borrow_mut()
                    .create_empty_buffer(output_meta.byte_size())
                    .map_err(|e| WasmError::MemoryAllocationFailed)?;
                (handle, true)
            }
        };
        
        softmax::execute_softmax_op(
            &mut *self.memory_manager.borrow_mut(),
            operation,
            &input,
            &input_meta,
            &output,
            &output_meta,
            axis,
        )?;
        
        if needs_initialization {
            self.memory_manager.borrow().mark_buffer_initialized(&mut output);
        }
        Ok(output)
    }
}