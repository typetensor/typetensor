
pub mod unary;
pub mod binary;
pub mod matmul;
pub mod view;
pub mod reduction;
pub mod softmax;

use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use crate::types::{WasmOperation, WasmTensorMeta, WasmResult, WasmError};
use crate::memory::{WasmMemoryManager, WasmBufferHandle, allocate_direct, allocate_direct_with_data};

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
        output_handle: WasmBufferHandle,  // Required - always pre-allocated
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
        output_handle: WasmBufferHandle,  // Required - always pre-allocated
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
        output_handle: WasmBufferHandle,  // Required - always pre-allocated
    ) -> WasmResult<WasmBufferHandle> {
        // Output is always pre-allocated - no RefCell borrow needed
        let mut output = output_handle;

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
        
        // Output is always pre-allocated and initialized
        output.mark_initialized();
        Ok(output)
    }

    /// Get memory usage statistics with graceful degradation
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> crate::memory::WasmMemoryStats {
        // Use try_borrow for read access, return fallback stats if busy
        self.memory_manager.try_borrow()
            .map(|manager| manager.get_memory_stats())
            .unwrap_or_else(|_| crate::memory::WasmMemoryStats::busy_fallback())
    }

    /// Create buffer with data using fast-path/slow-path pattern
    #[wasm_bindgen]
    pub fn create_buffer_with_data(&self, data: &[u8]) -> WasmBufferHandle {
        let size = data.len();
        
        // FAST PATH: Try pool (non-blocking)
        if let Ok(mut manager) = self.memory_manager.try_borrow_mut() {
            if let Some(mut handle) = manager.try_get_buffer(size) {
                // Copy data to pooled buffer
                unsafe { 
                    std::ptr::copy_nonoverlapping(data.as_ptr(), handle.ptr_mut(), size);
                }
                handle.mark_initialized();
                return handle;
            }
        }
        
        // SLOW PATH: Direct allocation (no RefCell)
        match allocate_direct_with_data(data) {
            Ok(handle) => handle,
            Err(e) => {
                wasm_bindgen::throw_str(&format!("Memory allocation failed: {}", e));
            }
        }
    }
    
    /// Create buffer with JS Uint8Array data using fast-path/slow-path pattern
    #[wasm_bindgen]
    pub fn create_buffer_with_js_data(&self, js_array: &js_sys::Uint8Array) -> WasmBufferHandle {
        let data = js_array.to_vec();
        let size = data.len();
        
        // FAST PATH: Try pool (non-blocking)
        if let Ok(mut manager) = self.memory_manager.try_borrow_mut() {
            if let Some(mut handle) = manager.try_get_buffer(size) {
                // Copy data to pooled buffer
                unsafe { 
                    std::ptr::copy_nonoverlapping(data.as_ptr(), handle.ptr_mut(), size);
                }
                handle.mark_initialized();
                return handle;
            }
        }
        
        // SLOW PATH: Direct allocation (no RefCell)
        match allocate_direct_with_data(&data) {
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
    
    /// Release buffer back to pool for reuse using fast-path/slow-path pattern
    #[wasm_bindgen]
    pub fn release_buffer(&self, handle: WasmBufferHandle) -> bool {
        // Check if it's a pooled buffer first
        if handle.is_pooled() {
            // Try pool return (non-blocking)
            if let Ok(mut manager) = self.memory_manager.try_borrow_mut() {
                // try_release_buffer consumes handle - only call if pool is available
                return manager.try_release_buffer(handle);
            }
            // Pool is busy - fallback to direct deallocation
            handle.deallocate_direct();
            true
        } else {
            // Direct buffer - deallocate immediately
            handle.deallocate_direct();
            true
        }
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
    
    /// Get buffer info for creating zero-copy views
    /// Returns: [ptr, size, initialized]
    #[wasm_bindgen]
    pub fn get_buffer_view_info(&self, handle: &WasmBufferHandle) -> js_sys::Uint32Array {
        let info = vec![
            handle.ptr() as u32,
            handle.size() as u32,
            if handle.is_initialized() { 1 } else { 0 }
        ];
        js_sys::Uint32Array::from(&info[..])
    }
    
    /// Compact memory pools to reduce memory usage (non-blocking)
    #[wasm_bindgen]
    pub fn compact_pools(&self) {
        // Only compact if not busy - skip if reentrancy detected
        if let Ok(mut manager) = self.memory_manager.try_borrow_mut() {
            manager.compact_pools();
        }
        // If busy, skip compaction - it's not critical
    }
    
    /// Perform intensive cleanup - for use during benchmarks or stress tests (non-blocking)
    #[wasm_bindgen]
    pub fn intensive_cleanup(&self) {
        // Only cleanup if not busy - skip if reentrancy detected
        if let Ok(mut manager) = self.memory_manager.try_borrow_mut() {
            manager.compact_pools();
            // Could add more aggressive cleanup here if needed
        }
        // If busy, skip cleanup - it's not critical
    }
    
    /// Clone a buffer handle with proper reference counting
    #[wasm_bindgen]
    pub fn clone_buffer_handle(&self, handle: &WasmBufferHandle) -> WasmBufferHandle {
        // FAST PATH: Try to increment reference count (non-blocking)
        if let Ok(mut manager) = self.memory_manager.try_borrow_mut() {
            if let Some(cloned_handle) = manager.clone_buffer_handle(handle) {
                return cloned_handle;
            }
        }
        
        // SLOW PATH: If memory manager is busy or buffer not found in pool,
        // create a new direct allocation clone (defensive fallback)
        // This preserves safety even under reentrancy
        match self.create_defensive_clone(handle) {
            Ok(cloned_handle) => cloned_handle,
            Err(e) => {
                wasm_bindgen::throw_str(&format!("Failed to clone buffer handle: {}", e));
            }
        }
    }
    
    /// Create a defensive clone when memory manager is busy
    /// This creates a new allocation with copied data to preserve safety
    fn create_defensive_clone(&self, handle: &WasmBufferHandle) -> Result<WasmBufferHandle, String> {
        // Read data from original handle
        let src_ptr = handle.get_read_ptr();
        let size = handle.size();
        
        // Create new direct allocation
        let mut new_handle = crate::memory::allocate_direct(size)?;
        
        // Copy data to new allocation
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, new_handle.ptr_mut(), size);
        }
        
        new_handle.mark_initialized();
        Ok(new_handle)
    }
    
    /// Create an empty buffer without data copy using fast-path/slow-path pattern
    #[wasm_bindgen]
    pub fn create_empty_buffer(&self, size: usize) -> WasmBufferHandle {
        // FAST PATH: Try pool (non-blocking)
        if let Ok(mut manager) = self.memory_manager.try_borrow_mut() {
            if let Ok((mut handle, ptr)) = manager.create_empty_buffer(size) {
                // Zero initialize the buffer
                unsafe {
                    std::ptr::write_bytes(ptr, 0, size);
                }
                handle.mark_initialized();
                return handle;
            }
        }
        
        // SLOW PATH: Direct allocation (no RefCell)
        match allocate_direct(size) {
            Ok(mut handle) => {
                // Zero initialize the buffer
                unsafe {
                    std::ptr::write_bytes(handle.ptr_mut(), 0, size);
                }
                handle.mark_initialized();
                handle
            }
            Err(e) => {
                wasm_bindgen::throw_str(&format!("Memory allocation failed: {}", e));
            }
        }
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
        output_handle: WasmBufferHandle,  // Required - always pre-allocated
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
        output_handle: WasmBufferHandle,  // Required - always pre-allocated
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
        output_handle: WasmBufferHandle,  // Required - always pre-allocated
    ) -> WasmResult<WasmBufferHandle> {
        // Output is always pre-allocated - no RefCell borrow needed
        let mut output = output_handle;
        
        reduction::execute_reduction_op_with_axes(
            operation,
            &input,
            &input_meta,
            &output,
            &output_meta,
            axes.as_deref(),
            keep_dims,
        )?;
        
        // Output is always pre-allocated and needs initialization
        output.mark_initialized();
        Ok(output)
    }
    
    fn execute_softmax_internal(
        &self,
        operation: WasmOperation,
        input: WasmBufferHandle,
        input_meta: WasmTensorMeta,
        output_meta: WasmTensorMeta,
        axis: Option<i32>,
        output_handle: WasmBufferHandle,  // Required - always pre-allocated
    ) -> WasmResult<WasmBufferHandle> {
        // Output is always pre-allocated - no RefCell borrow needed
        let mut output = output_handle;
        
        softmax::execute_softmax_op(
            operation,
            &input,
            &input_meta,
            &output,
            &output_meta,
            axis,
        )?;
        
        // Output is always pre-allocated and needs initialization
        output.mark_initialized();
        Ok(output)
    }
}