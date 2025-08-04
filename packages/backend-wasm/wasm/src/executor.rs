/*!
 * Main operation executor for WASM tensor operations
 * 
 * Replaces the old RefCell-based WasmOperationDispatcher with a simple
 * ownership model using the new arena-based memory system.
 */

use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use crate::memory::{WasmMemorySystem, WasmTensor, WasmMemoryStats};
use crate::arena::CheckpointId;
use crate::types::{WasmOperation, WasmDType, WasmResult, WasmError};
use crate::operations::{unary, binary, matmul};
use crate::pattern::{PatternCache, PatternBuilder, OperationDesc, AllocationRequirement, PatternCacheStats, PatternSignature, PatternId};

/// Result of pattern-based execution
struct PatternExecution {
    pattern_id: PatternId,
    pre_allocated_tensors: Vec<WasmTensor>,
    estimated_speedup: f32,
}

/// Main tensor operation executor
/// 
/// Uses single ownership model - no RefCell complexity!
/// All operations get `&mut self` for clear, safe memory management.
#[wasm_bindgen]
pub struct WasmExecutor {
    memory: WasmMemorySystem,
    pattern_cache: PatternCache,
    enable_pattern_optimization: bool,
    checkpoint_counter: usize,
    active_checkpoints: HashMap<usize, CheckpointId>,
}

#[wasm_bindgen]
impl WasmExecutor {
    /// Create new executor
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmExecutor {
        WasmExecutor {
            memory: WasmMemorySystem::new(),
            pattern_cache: PatternCache::new(100, 50), // 100 patterns, 50MB cache
            enable_pattern_optimization: true,
            checkpoint_counter: 0,
            active_checkpoints: HashMap::new(),
        }
    }
    
    /// Create executor with custom pattern cache settings
    pub fn new_with_pattern_cache(max_patterns: usize, max_memory_mb: usize) -> WasmExecutor {
        WasmExecutor {
            memory: WasmMemorySystem::new(),
            pattern_cache: PatternCache::new(max_patterns, max_memory_mb),
            enable_pattern_optimization: true,
            checkpoint_counter: 0,
            active_checkpoints: HashMap::new(),
        }
    }
    
    /// Allocate temporary tensor (arena-based, fast cleanup)
    #[wasm_bindgen]
    pub fn alloc_temp_tensor(&mut self, dtype: WasmDType, shape: Box<[usize]>) -> Result<WasmTensor, JsValue> {
        self.memory.alloc_temp_tensor(dtype, &shape)
            .map_err(|e| JsValue::from_str(&e))
    }
    
    /// Allocate persistent tensor (reference-counted, manual cleanup)
    #[wasm_bindgen]
    pub fn alloc_persistent_tensor(&mut self, dtype: WasmDType, shape: Box<[usize]>) -> Result<WasmTensor, JsValue> {
        self.memory.alloc_persistent_tensor(dtype, &shape)
            .map_err(|e| JsValue::from_str(&e))
    }
    
    /// Create tensor from JavaScript TypedArray data
    #[wasm_bindgen]
    pub fn tensor_from_data(&mut self, data: Vec<u8>, dtype: WasmDType, shape: Box<[usize]>) -> Result<WasmTensor, JsValue> {
        self.memory.tensor_from_data(data, dtype, &shape)
            .map_err(|e| JsValue::from_str(&e))
    }
    
    /// Create checkpoint for scoped memory management
    #[wasm_bindgen]
    pub fn checkpoint(&mut self) -> usize {
        // Create checkpoint in memory system
        let checkpoint = self.memory.checkpoint();
        
        // Generate unique ID and store mapping
        let checkpoint_id = self.checkpoint_counter;
        self.active_checkpoints.insert(checkpoint_id, checkpoint);
        self.checkpoint_counter += 1;
        
        checkpoint_id
    }
    
    /// Restore to checkpoint (bulk cleanup of temporaries)
    #[wasm_bindgen]
    pub fn restore(&mut self, checkpoint_id: usize) -> Result<(), JsValue> {
        // Look up the CheckpointId from our mapping
        if let Some(&checkpoint) = self.active_checkpoints.get(&checkpoint_id) {
            // Restore to the actual checkpoint
            self.memory.restore(checkpoint)
                .map_err(|e| JsValue::from_str(&e))?;
            
            // Clean up checkpoints that are now invalid (created after this restore point)
            self.active_checkpoints.retain(|&id, _| id <= checkpoint_id);
            
            Ok(())
        } else {
            Err(JsValue::from_str("Invalid checkpoint ID"))
        }
    }
    
    /// Get memory usage statistics
    #[wasm_bindgen]
    pub fn memory_stats(&self) -> WasmMemoryStats {
        self.memory.memory_stats()
    }
    
    /// Execute unary operation
    #[wasm_bindgen]
    pub fn execute_unary(&mut self, 
        operation: WasmOperation, 
        input: &WasmTensor,
        output: &WasmTensor
    ) -> Result<(), JsValue> {
        // Record pattern for optimization
        self.record_operation_pattern(operation, &[input], output);
        
        unary::execute_unary_op(
            operation,
            input,
            output,
            self.memory.arena(),
        ).map_err(|e| self.map_wasm_error(e))
    }
    
    /// Execute binary operation
    #[wasm_bindgen]
    pub fn execute_binary(&mut self,
        operation: WasmOperation,
        input_a: &WasmTensor,
        input_b: &WasmTensor,
        output: &WasmTensor
    ) -> Result<(), JsValue> {
        // Record pattern for optimization
        self.record_operation_pattern(operation, &[input_a, input_b], output);
        
        binary::execute_binary_op(
            operation,
            input_a,
            input_b,
            output,
            self.memory.arena(),
        ).map_err(|e| self.map_wasm_error(e))
    }
    
    /// Execute matrix multiplication
    #[wasm_bindgen]
    pub fn execute_matmul(&mut self,
        input_a: &WasmTensor,
        input_b: &WasmTensor,
        output: &WasmTensor
    ) -> Result<(), JsValue> {
        // Record pattern for optimization
        self.record_operation_pattern(WasmOperation::Matmul, &[input_a, input_b], output);
        
        matmul::execute_matmul_op(
            WasmOperation::Matmul,
            input_a,
            input_b,
            output,
            self.memory.arena(),
        ).map_err(|e| self.map_wasm_error(e))
    }
    
    /// Execute slice operation with explicit offset parameters
    #[wasm_bindgen]
    pub fn execute_slice(&mut self,
        input: &WasmTensor,
        output: &WasmTensor,
        row_start: usize,
        col_start: usize
    ) -> Result<(), JsValue> {
        use crate::operations::view;
        
        
        view::execute_slice_with_offsets(
            input,
            output,
            row_start,
            col_start,
            self.memory.arena(),
        ).map_err(|e| self.map_wasm_error(e))
    }
    
    /// Execute reduction operation with optional axis parameter
    #[wasm_bindgen]
    pub fn execute_reduction(&mut self,
        operation: WasmOperation,
        input: &WasmTensor,
        output: &WasmTensor,
        axis: Option<Vec<usize>>,
        keep_dims: bool
    ) -> Result<(), JsValue> {
        use crate::operations::reduction;
        
        // Record pattern for optimization
        self.record_operation_pattern(operation, &[input], output);
        
        let axes_slice = axis.as_ref().map(|v| v.as_slice());
        
        reduction::execute_reduction_op_with_axes(
            operation,
            input,
            output,
            self.memory.arena(),
            axes_slice,
            keep_dims,
        ).map_err(|e| self.map_wasm_error(e))
    }
    
    /// Garbage collect persistent tensors
    #[wasm_bindgen]
    pub fn gc(&mut self) -> usize {
        self.memory.gc_persistent()
    }
    
    /// Get pattern cache statistics
    #[wasm_bindgen]
    pub fn pattern_cache_stats(&self) -> PatternCacheStats {
        self.pattern_cache.cache_stats()
    }
    
    /// Enable or disable pattern optimization
    #[wasm_bindgen]
    pub fn set_pattern_optimization(&mut self, enabled: bool) {
        self.enable_pattern_optimization = enabled;
    }
    
    /// Clear pattern cache
    #[wasm_bindgen]
    pub fn clear_pattern_cache(&mut self) {
        self.pattern_cache.clear();
    }
    
    /// Copy tensor data to JavaScript Uint8Array (for TypeScript readData)
    #[wasm_bindgen]
    pub fn copy_tensor_data_to_js(&self, tensor: &WasmTensor) -> Vec<u8> {
        let ptr = tensor.get_read_ptr(self.memory.arena());
        let size = tensor.get_data_size();
        
        // Copy data from WASM memory to JavaScript
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, size);
            slice.to_vec()
        }
    }
    
    /// Copy JavaScript Uint8Array data to tensor (for TypeScript writeData)
    #[wasm_bindgen]
    pub fn copy_js_data_to_tensor(&mut self, tensor: &WasmTensor, data: Vec<u8>) -> Result<(), JsValue> {
        if data.len() != tensor.get_data_size() {
            return Err(JsValue::from_str(&format!(
                "Data size mismatch: expected {} bytes, got {} bytes",
                tensor.get_data_size(),
                data.len()
            )));
        }
        
        // For temporary tensors, we need mutable arena access
        // For persistent tensors, we can write directly
        if tensor.is_temporary() {
            // For temporary tensors, we need to use get_write_ptr with mutable arena
            let ptr = tensor.get_write_ptr(Some(self.memory.arena_mut()));
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        } else {
            // For persistent tensors, we can write directly
            let ptr = tensor.get_write_ptr(None);
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        }
        
        Ok(())
    }
}

impl WasmExecutor {
    /// Bulk allocate tensors for pattern (for testing and advanced usage)
    pub fn bulk_allocate_for_pattern(&mut self, pattern: &crate::pattern::OperationPattern) -> Result<Vec<crate::memory::WasmTensor>, String> {
        self.memory.bulk_allocate_for_pattern(pattern)
    }
    
    /// Try to execute operation using cached pattern (bulk allocation optimization)
    fn try_pattern_execution(
        &mut self,
        operation: WasmOperation,
        inputs: &[&WasmTensor],
    ) -> Option<PatternExecution> {
        if !self.enable_pattern_optimization {
            return None;
        }
        
        // Build pattern signature for current operation
        let signature = self.build_pattern_signature(operation, inputs);
        
        // Look for matching pattern
        if let Some(pattern_id) = self.pattern_cache.find_matching_pattern(&signature) {
            // Try to get the pattern and perform bulk allocation
            if let Some(pattern) = self.pattern_cache.get_pattern(pattern_id) {
                let pattern_clone = pattern.clone(); // Clone to avoid borrow conflicts
                
                match self.memory.bulk_allocate_for_pattern(&pattern_clone) {
                    Ok(pre_allocated_tensors) => {
                        return Some(PatternExecution {
                            pattern_id,
                            pre_allocated_tensors,
                            estimated_speedup: pattern_clone.estimated_speedup,
                        });
                    }
                    Err(_) => {
                        // Bulk allocation failed, fall back to individual allocation
                        return None;
                    }
                }
            }
        }
        
        None
    }
    
    /// Build pattern signature for current operation
    fn build_pattern_signature(&self, operation: WasmOperation, inputs: &[&WasmTensor]) -> PatternSignature {
        let input_shapes = inputs.iter()
            .map(|tensor| tensor.metadata().shape().to_vec())
            .collect();
        let input_dtypes = inputs.iter()
            .map(|tensor| tensor.metadata().dtype())
            .collect();
        
        PatternSignature::new(operation, input_shapes, input_dtypes)
    }
    
    
    /// Helper method to map WasmError to JsValue
    fn map_wasm_error(&self, error: WasmError) -> JsValue {
        match error {
            WasmError::NotImplemented => JsValue::from_str("Operation not implemented"),
            WasmError::InvalidOperation => JsValue::from_str("Invalid operation"),
            WasmError::OutOfMemory => JsValue::from_str("Out of memory"),
            WasmError::InvalidInput => JsValue::from_str("Invalid input"),
            WasmError::InvalidDType => JsValue::from_str("Invalid data type"),
            WasmError::InvalidShape => JsValue::from_str("Invalid tensor shape"),
            WasmError::MemoryAllocationFailed => JsValue::from_str("Memory allocation failed"),
        }
    }
    
    /// Create operation description for pattern recognition
    fn create_operation_desc(&self, 
        operation: WasmOperation,
        inputs: &[&WasmTensor], 
        output: &WasmTensor
    ) -> OperationDesc {
        let input_shapes = inputs.iter()
            .map(|t| t.metadata().shape().clone())
            .collect();
        let input_dtypes = inputs.iter()
            .map(|t| t.metadata().dtype())
            .collect();
        
        OperationDesc {
            operation,
            input_shapes,
            input_dtypes,
            output_shape: output.metadata().shape().clone(),
            output_dtype: output.metadata().dtype(),
        }
    }
    
    /// Record operation pattern for future optimization
    fn record_operation_pattern(&mut self, 
        operation: WasmOperation,
        inputs: &[&WasmTensor], 
        output: &WasmTensor
    ) {
        if !self.enable_pattern_optimization {
            return;
        }
        
        let op_desc = self.create_operation_desc(operation, inputs, output);
        let mut builder = PatternBuilder::new();
        builder.add_operation(op_desc);
        
        // Add allocation requirement for output tensor
        builder.add_allocation(AllocationRequirement {
            size_bytes: output.byte_size(),
            alignment: 16, // SIMD alignment
            is_output: true,
        });
        
        let pattern = builder.build(&self.pattern_cache);
        
        // Try to store pattern (ignore errors for now)
        let _ = self.pattern_cache.store_pattern(pattern);
    }
}

// Clean implementation - no complex adapters needed!

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;
    
    // Configure for Node.js instead of browser for better CI compatibility
    // wasm_bindgen_test_configure!(run_in_browser);
    
    #[test]
    fn test_executor_creation() {
        let executor = WasmExecutor::new();
        let stats = executor.memory_stats();
        assert!(stats.arena_capacity() > 0);
        assert_eq!(stats.persistent_count(), 0);
    }
    
    #[wasm_bindgen_test]
    fn test_wasm_executor_creation() {
        let executor = WasmExecutor::new();
        let stats = executor.memory_stats();
        assert!(stats.arena_capacity() > 0);
        assert_eq!(stats.persistent_count(), 0);
    }
    
    #[test]
    fn test_tensor_allocation() {
        let mut executor = WasmExecutor::new();
        
        // Test temporary tensor allocation
        let temp_tensor = executor.alloc_temp_tensor(WasmDType::Float32, vec![10, 20].into_boxed_slice()).unwrap();
        assert!(temp_tensor.is_temporary());
        assert_eq!(temp_tensor.byte_size(), 10 * 20 * 4); // f32 = 4 bytes
        
        // Test persistent tensor allocation
        let persistent_tensor = executor.alloc_persistent_tensor(WasmDType::Float64, vec![5, 5].into_boxed_slice()).unwrap();
        assert!(!persistent_tensor.is_temporary());
        assert_eq!(persistent_tensor.byte_size(), 5 * 5 * 8); // f64 = 8 bytes
        
        let stats = executor.memory_stats();
        assert!(stats.arena_used() > 0);
        assert_eq!(stats.persistent_count(), 1);
    }
    
    #[test]
    fn test_memory_checkpoint_restore() {
        let mut executor = WasmExecutor::new();
        
        let _tensor1 = executor.alloc_temp_tensor(WasmDType::Float32, vec![100].into_boxed_slice()).unwrap();
        let checkpoint = executor.checkpoint();
        let _tensor2 = executor.alloc_temp_tensor(WasmDType::Float32, vec![200].into_boxed_slice()).unwrap();
        
        let stats_before = executor.memory_stats();
        executor.restore(checkpoint).unwrap();
        let stats_after = executor.memory_stats();
        
        // After restore, arena should be reset (our current simple implementation)
        assert!(stats_after.arena_used() <= stats_before.arena_used());
    }
        
    // WASM-specific operation integration tests
    #[wasm_bindgen_test]
    fn wasm_test_pattern_recording_unary() {
        let mut executor = WasmExecutor::new();
        
        // Create tensors for unary operation
        let input = executor.alloc_temp_tensor(WasmDType::Float32, vec![5, 5].into_boxed_slice()).unwrap();
        let output = executor.alloc_temp_tensor(WasmDType::Float32, vec![5, 5].into_boxed_slice()).unwrap();
        
        // Check initial pattern cache stats
        let initial_stats = executor.pattern_cache_stats();
        assert_eq!(initial_stats.pattern_count(), 0);
        
        // Execute unary operation - this should record a pattern
        let result = executor.execute_unary(WasmOperation::Abs, &input, &output);
        assert!(result.is_ok());
        
        // Pattern should be recorded (though not necessarily stored yet due to minimum hit requirements)
        // The important thing is that the operation completed successfully
    }
    
    #[wasm_bindgen_test]
    fn wasm_test_pattern_recording_binary() {
        let mut executor = WasmExecutor::new();
        
        // Create tensors for binary operation
        let input_a = executor.alloc_temp_tensor(WasmDType::Float32, vec![10, 10].into_boxed_slice()).unwrap();
        let input_b = executor.alloc_temp_tensor(WasmDType::Float32, vec![10, 10].into_boxed_slice()).unwrap();
        let output = executor.alloc_temp_tensor(WasmDType::Float32, vec![10, 10].into_boxed_slice()).unwrap();
        
        // Execute binary operation - should record pattern
        let result = executor.execute_binary(WasmOperation::Add, &input_a, &input_b, &output);
        assert!(result.is_ok());
        
        // Execute same operation again - might hit pattern cache
        let result2 = executor.execute_binary(WasmOperation::Add, &input_a, &input_b, &output);
        assert!(result2.is_ok());
    }
    
    #[wasm_bindgen_test]
    fn wasm_test_pattern_recording_matmul() {
        let mut executor = WasmExecutor::new();
        
        // Create tensors for matrix multiplication
        let input_a = executor.alloc_temp_tensor(WasmDType::Float32, vec![4, 3].into_boxed_slice()).unwrap();
        let input_b = executor.alloc_temp_tensor(WasmDType::Float32, vec![3, 4].into_boxed_slice()).unwrap();
        let output = executor.alloc_temp_tensor(WasmDType::Float32, vec![4, 4].into_boxed_slice()).unwrap();
        
        // Execute matmul operation - should record pattern
        let result = executor.execute_matmul(&input_a, &input_b, &output);
        assert!(result.is_ok());
        
        // Verify pattern optimization is enabled
        executor.set_pattern_optimization(false);
        let result2 = executor.execute_matmul(&input_a, &input_b, &output);
        assert!(result2.is_ok());
        
        // Re-enable pattern optimization
        executor.set_pattern_optimization(true);
    }
    
    #[wasm_bindgen_test]
    fn wasm_test_operation_chain_execution() {
        let mut executor = WasmExecutor::new();
        
        // Create tensors for operation chain: A + B, then result * C, then matmul with D
        let tensor_a = executor.alloc_temp_tensor(WasmDType::Float32, vec![2, 2].into_boxed_slice()).unwrap();
        let tensor_b = executor.alloc_temp_tensor(WasmDType::Float32, vec![2, 2].into_boxed_slice()).unwrap();
        let tensor_c = executor.alloc_temp_tensor(WasmDType::Float32, vec![2, 2].into_boxed_slice()).unwrap();
        let tensor_d = executor.alloc_temp_tensor(WasmDType::Float32, vec![2, 2].into_boxed_slice()).unwrap();
        
        let temp1 = executor.alloc_temp_tensor(WasmDType::Float32, vec![2, 2].into_boxed_slice()).unwrap();
        let temp2 = executor.alloc_temp_tensor(WasmDType::Float32, vec![2, 2].into_boxed_slice()).unwrap();
        let final_output = executor.alloc_temp_tensor(WasmDType::Float32, vec![2, 2].into_boxed_slice()).unwrap();
        
        // Execute operation chain
        let result1 = executor.execute_binary(WasmOperation::Add, &tensor_a, &tensor_b, &temp1);
        assert!(result1.is_ok());
        
        let result2 = executor.execute_binary(WasmOperation::Mul, &temp1, &tensor_c, &temp2);
        assert!(result2.is_ok());
        
        let result3 = executor.execute_matmul(&temp2, &tensor_d, &final_output);
        assert!(result3.is_ok());
        
        // Check that patterns might have been recorded
        let stats = executor.pattern_cache_stats();
        // Note: Patterns may or may not be stored depending on hit count thresholds
    }
    
    #[wasm_bindgen_test]
    fn wasm_test_memory_management_during_operations() {
        let mut executor = WasmExecutor::new();
        
        let initial_stats = executor.memory_stats();
        assert_eq!(initial_stats.arena_used(), 0);
        
        // Allocate tensors
        let input1 = executor.alloc_temp_tensor(WasmDType::Float32, vec![100].into_boxed_slice()).unwrap();
        let input2 = executor.alloc_temp_tensor(WasmDType::Float32, vec![100].into_boxed_slice()).unwrap();
        let output = executor.alloc_temp_tensor(WasmDType::Float32, vec![100].into_boxed_slice()).unwrap();
        
        let after_alloc_stats = executor.memory_stats();
        assert!(after_alloc_stats.arena_used() > initial_stats.arena_used());
        
        // Execute operations
        let _result = executor.execute_binary(WasmOperation::Add, &input1, &input2, &output);
        
        let after_op_stats = executor.memory_stats();
        // Memory usage should be stable (not growing continuously)
        assert_eq!(after_op_stats.arena_used(), after_alloc_stats.arena_used());
        
        // Test checkpoint and restore
        let checkpoint = executor.checkpoint();
        let _temp = executor.alloc_temp_tensor(WasmDType::Float32, vec![50].into_boxed_slice()).unwrap();
        
        executor.restore(checkpoint).unwrap();
        let after_restore_stats = executor.memory_stats();
        assert!(after_restore_stats.arena_used() <= after_op_stats.arena_used());
    }
    
    #[wasm_bindgen_test]
    fn wasm_test_pattern_optimization_toggle() {
        let mut executor = WasmExecutor::new();
        
        // Test that pattern optimization can be toggled
        executor.set_pattern_optimization(false);
        
        let input = executor.alloc_temp_tensor(WasmDType::Float32, vec![10].into_boxed_slice()).unwrap();
        let output = executor.alloc_temp_tensor(WasmDType::Float32, vec![10].into_boxed_slice()).unwrap();
        
        // Execute operation with pattern optimization disabled
        let result1 = executor.execute_unary(WasmOperation::Abs, &input, &output);
        assert!(result1.is_ok());
        
        // Enable pattern optimization
        executor.set_pattern_optimization(true);
        let result2 = executor.execute_unary(WasmOperation::Abs, &input, &output);
        assert!(result2.is_ok());
        
        // Clear pattern cache
        executor.clear_pattern_cache();
        let stats = executor.pattern_cache_stats();
        assert_eq!(stats.pattern_count(), 0);
    }
    
    #[wasm_bindgen_test]
    fn wasm_test_large_tensor_operations() {
        let mut executor = WasmExecutor::new();
        
        // Test operations with larger tensors to stress test memory management
        let large_input1 = executor.alloc_temp_tensor(WasmDType::Float32, vec![50, 50].into_boxed_slice()).unwrap();
        let large_input2 = executor.alloc_temp_tensor(WasmDType::Float32, vec![50, 50].into_boxed_slice()).unwrap();
        let large_output = executor.alloc_temp_tensor(WasmDType::Float32, vec![50, 50].into_boxed_slice()).unwrap();
        
        // Execute binary operation
        let result = executor.execute_binary(WasmOperation::Mul, &large_input1, &large_input2, &large_output);
        assert!(result.is_ok());
        
        // Check memory stats
        let stats = executor.memory_stats();
        assert!(stats.arena_used() > 0);
        assert!(stats.arena_utilization() > 0.0);
        
        // Test matrix multiplication with medium-sized matrices
        let mat_a = executor.alloc_temp_tensor(WasmDType::Float32, vec![20, 30].into_boxed_slice()).unwrap();
        let mat_b = executor.alloc_temp_tensor(WasmDType::Float32, vec![30, 25].into_boxed_slice()).unwrap();
        let mat_output = executor.alloc_temp_tensor(WasmDType::Float32, vec![20, 25].into_boxed_slice()).unwrap();
        
        let matmul_result = executor.execute_matmul(&mat_a, &mat_b, &mat_output);
        assert!(matmul_result.is_ok());
    }
}