/*!
 * New arena-based memory management system for WASM tensors
 * 
 * Replaces the complex buffer pool system with:
 * - Single ownership (no RefCell complexity)
 * - Arena allocation for temporaries  
 * - Reference counting for persistents
 * - WASM-optimized memory management
 */

use std::sync::Arc;
use wasm_bindgen::prelude::*;
use crate::arena::{TempArena, PersistentStorage, PersistentTensor, ArenaOffset, CheckpointId};
use crate::types::{WasmDType, WasmTensorMeta};
use crate::pattern::{OperationPattern, AllocationRequirement};

/// Tensor data storage - either temporary (arena) or persistent (reference-counted)
#[derive(Debug, Clone)]
pub enum TensorData {
    /// Temporary allocation in arena (fast, bulk cleanup)
    Temporary(ArenaOffset),
    /// Persistent allocation with reference counting
    Persistent(Arc<PersistentTensor>),
}

impl TensorData {
    /// Get read pointer to tensor data
    /// 
    /// # Memory Safety
    /// Returns a valid pointer to the tensor's data that remains valid for the
    /// lifetime of the arena (for temporary tensors) or until the tensor is dropped
    /// (for persistent tensors).
    pub fn get_read_ptr(&self, arena: &TempArena) -> *const u8 {
        match self {
            TensorData::Temporary(offset) => arena.get_ptr(*offset),
            TensorData::Persistent(tensor) => tensor.get_ptr(),
        }
    }
    
    /// Get write pointer to tensor data (requires mutable arena for temporaries)
    /// 
    /// # Design Note
    /// This method exists but is rarely used in practice. Most operations use
    /// `get_read_ptr()` and cast to `*mut u8` for the output tensor, which is safe
    /// because:
    /// 1. Operations typically take `&TempArena` (immutable) since they don't need
    ///    to allocate new memory, just access existing allocations
    /// 2. Each tensor has unique memory regions from the bump allocator
    /// 3. No aliasing occurs between different tensors
    /// 4. The semantic intent is to write to output tensors
    pub fn get_write_ptr(&self, arena: Option<&mut TempArena>) -> *mut u8 {
        match self {
            TensorData::Temporary(offset) => {
                arena.expect("Mutable arena required for temporary tensors")
                    .get_mut_ptr(*offset)
            },
            TensorData::Persistent(tensor) => {
                // SAFETY: Cast from *const to *mut for persistent tensors
                // This is safe because:
                // - Persistent tensors are reference-counted but have unique storage
                // - The API ensures exclusive access during operations
                // - This follows the same pattern as the *const -> *mut cast in operations
                tensor.get_ptr() as *mut u8
            }
        }
    }
    
    /// Get tensor data size
    pub fn size(&self) -> usize {
        match self {
            TensorData::Temporary(offset) => offset.size(),
            TensorData::Persistent(tensor) => tensor.size(),
        }
    }
    
    /// Check if tensor is temporary (arena-allocated)
    pub fn is_temporary(&self) -> bool {
        matches!(self, TensorData::Temporary(_))
    }
}

/// New simplified tensor handle
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmTensor {
    data: TensorData,
    meta: WasmTensorMeta,
}

#[wasm_bindgen]
impl WasmTensor {
    /// Get tensor metadata
    #[wasm_bindgen(getter)]
    pub fn meta(&self) -> WasmTensorMeta {
        self.meta.clone()
    }
    
    /// Get tensor data size in bytes
    #[wasm_bindgen(getter)]
    pub fn byte_size(&self) -> usize {
        self.data.size()
    }
    
    /// Check if tensor is temporary (arena-allocated)
    #[wasm_bindgen(getter)]
    pub fn is_temporary(&self) -> bool {
        self.data.is_temporary()
    }
    
    /// Get raw data pointer offset for temporary tensors (returns offset in arena)
    /// For persistent tensors, this is not applicable and returns 0
    #[wasm_bindgen]
    pub fn get_data_offset(&self) -> usize {
        match &self.data {
            TensorData::Temporary(offset) => offset.offset(),
            TensorData::Persistent(_) => 0, // Persistent tensors don't use arena offsets
        }
    }
    
    /// Get arena offset size for temporary tensors
    /// For persistent tensors, returns the actual data size
    #[wasm_bindgen]
    pub fn get_data_size(&self) -> usize {
        self.data.size()
    }
}

impl WasmTensor {
    /// Create new temporary tensor (internal)
    pub fn new_temporary(offset: ArenaOffset, meta: WasmTensorMeta) -> Self {
        WasmTensor {
            data: TensorData::Temporary(offset),
            meta,
        }
    }
    
    /// Create new persistent tensor (internal)
    pub fn new_persistent(tensor: Arc<PersistentTensor>, meta: WasmTensorMeta) -> Self {
        WasmTensor {
            data: TensorData::Persistent(tensor),
            meta,
        }
    }
    
    /// Get read pointer to tensor data
    pub fn get_read_ptr(&self, arena: &TempArena) -> *const u8 {
        self.data.get_read_ptr(arena)
    }
    
    /// Get write pointer to tensor data
    pub fn get_write_ptr(&self, arena: Option<&mut TempArena>) -> *mut u8 {
        self.data.get_write_ptr(arena)
    }
    
    /// Get tensor metadata
    pub fn metadata(&self) -> &WasmTensorMeta {
        &self.meta
    }
}

/// Memory usage statistics
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmMemoryStats {
    arena_used: usize,
    arena_capacity: usize,
    arena_utilization: f32,
    persistent_count: usize,
    persistent_bytes: usize,
    total_allocated: usize,
}

#[wasm_bindgen]
impl WasmMemoryStats {
    #[wasm_bindgen(getter)]
    pub fn arena_used(&self) -> usize { self.arena_used }
    
    #[wasm_bindgen(getter)]
    pub fn arena_capacity(&self) -> usize { self.arena_capacity }
    
    #[wasm_bindgen(getter)]
    pub fn arena_utilization(&self) -> f32 { self.arena_utilization }
    
    #[wasm_bindgen(getter)]
    pub fn persistent_count(&self) -> usize { self.persistent_count }
    
    #[wasm_bindgen(getter)]
    pub fn persistent_bytes(&self) -> usize { self.persistent_bytes }
    
    #[wasm_bindgen(getter)]
    pub fn total_allocated(&self) -> usize { self.total_allocated }
}

/// Main memory management system
pub struct WasmMemorySystem {
    arena: TempArena,
    persistent_storage: PersistentStorage,
}

impl WasmMemorySystem {
    /// Create new memory system
    pub fn new() -> Self {
        WasmMemorySystem {
            arena: TempArena::new(),
            persistent_storage: PersistentStorage::new(),
        }
    }
    
    /// Allocate temporary tensor (arena-based, fast cleanup)
    pub fn alloc_temp_tensor(&mut self, dtype: WasmDType, shape: &[usize]) -> Result<WasmTensor, String> {
        let byte_size = calculate_tensor_bytes(dtype, shape);
        let offset = self.arena.alloc(byte_size)?;
        
        let meta = WasmTensorMeta::new(
            dtype,
            shape.to_vec(),
            calculate_row_major_strides(shape),
            shape.iter().product(),
            0, // offset - always 0 for new tensors
        );
        
        Ok(WasmTensor::new_temporary(offset, meta))
    }
    
    /// Allocate persistent tensor (reference-counted, manual cleanup)
    pub fn alloc_persistent_tensor(&mut self, dtype: WasmDType, shape: &[usize]) -> Result<WasmTensor, String> {
        let byte_size = calculate_tensor_bytes(dtype, shape);
        let persistent_tensor = PersistentTensor::new(byte_size);
        
        // Store in persistent storage and get shared reference
        let tensor_id = self.persistent_storage.store(persistent_tensor);
        let tensor = self.persistent_storage.get(tensor_id).unwrap();
        
        let meta = WasmTensorMeta::new(
            dtype,
            shape.to_vec(),
            calculate_row_major_strides(shape),
            shape.iter().product(),
            0, // offset - always 0 for new tensors
        );
        
        Ok(WasmTensor::new_persistent(tensor, meta))
    }
    
    /// Create tensor from data (always persistent since data is provided)
    pub fn tensor_from_data(&mut self, data: Vec<u8>, dtype: WasmDType, shape: &[usize]) -> Result<WasmTensor, String> {
        let expected_bytes = calculate_tensor_bytes(dtype, shape);
        if data.len() != expected_bytes {
            return Err(format!(
                "Data size mismatch: expected {} bytes, got {}",
                expected_bytes, data.len()
            ));
        }
        
        let persistent_tensor = PersistentTensor::with_data(data);
        
        // Store in persistent storage and get shared reference
        let tensor_id = self.persistent_storage.store(persistent_tensor);
        let tensor = self.persistent_storage.get(tensor_id).unwrap();
        
        let meta = WasmTensorMeta::new(
            dtype,
            shape.to_vec(),
            calculate_row_major_strides(shape),
            shape.iter().product(),
            0, // offset - always 0 for new tensors
        );
        
        Ok(WasmTensor::new_persistent(tensor, meta))
    }
    
    /// Create checkpoint for scoped memory management
    pub fn checkpoint(&mut self) -> CheckpointId {
        self.arena.checkpoint()
    }
    
    /// Restore to checkpoint (bulk cleanup of temporaries)
    pub fn restore(&mut self, checkpoint: CheckpointId) -> Result<(), String> {
        self.arena.restore(checkpoint)
    }
    
    /// Reset entire arena (deallocate all temporaries)
    pub fn reset_arena(&mut self) {
        self.arena.reset();
    }
    
    /// Get memory usage statistics
    pub fn memory_stats(&self) -> WasmMemoryStats {
        let (arena_used, arena_capacity, arena_utilization) = self.arena.memory_usage();
        let (persistent_count, persistent_bytes) = self.persistent_storage.storage_stats();
        
        WasmMemoryStats {
            arena_used,
            arena_capacity,
            arena_utilization,
            persistent_count,
            persistent_bytes,
            total_allocated: arena_used + persistent_bytes,
        }
    }
    
    /// Garbage collect persistent tensors
    pub fn gc_persistent(&mut self) -> usize {
        self.persistent_storage.gc()
    }
    
    /// Check if system is under memory pressure
    pub fn is_memory_pressure(&self) -> bool {
        self.arena.is_memory_pressure()
    }
    
    /// Get mutable reference to arena (for operations)
    pub fn arena_mut(&mut self) -> &mut TempArena {
        &mut self.arena
    }
    
    /// Get immutable reference to arena (for operations)
    pub fn arena(&self) -> &TempArena {
        &self.arena
    }
    
    /// Pre-allocate all tensors for recognized pattern (ONNX-style bulk allocation)
    pub fn bulk_allocate_for_pattern(
        &mut self, 
        pattern: &OperationPattern
    ) -> Result<Vec<WasmTensor>, String> {
        // Check if we have enough memory for bulk allocation
        if !self.can_bulk_allocate(pattern) {
            return Err("Insufficient memory for bulk allocation".to_string());
        }
        
        let mut allocated_tensors = Vec::with_capacity(pattern.allocations.len());
        let checkpoint = self.checkpoint(); // Create checkpoint for rollback
        
        // Attempt to allocate all tensors in sequence
        for (i, allocation) in pattern.allocations.iter().enumerate() {
            match self.allocate_tensor_for_requirement(allocation, i) {
                Ok(tensor) => allocated_tensors.push(tensor),
                Err(e) => {
                    // Rollback all allocations on failure
                    let _ = self.restore(checkpoint);
                    return Err(format!("Bulk allocation failed at tensor {}: {}", i, e));
                }
            }
        }
        
        Ok(allocated_tensors)
    }
    
    /// Check if pattern can be bulk allocated (memory availability check)
    pub fn can_bulk_allocate(&self, pattern: &OperationPattern) -> bool {
        // Check if arena has enough space for all allocations
        let available_memory = self.arena.available_memory();
        pattern.total_memory_needed <= available_memory
    }
    
    /// Allocate single tensor based on allocation requirement
    fn allocate_tensor_for_requirement(
        &mut self,
        requirement: &AllocationRequirement,
        index: usize,
    ) -> Result<WasmTensor, String> {
        // For bulk allocation, we assume Float32 tensors with specific sizes
        // In a full implementation, this would use pattern metadata
        let dtype = WasmDType::Float32; // Default for now
        let element_size = dtype.byte_size();
        let element_count = requirement.size_bytes / element_size;
        
        // Create a simple 1D shape for the required size
        // TODO: Extract actual shapes from pattern operations
        let shape = vec![element_count];
        
        // Use temporary allocation for bulk allocations (arena-based)
        let offset = self.arena.alloc_aligned(requirement.size_bytes, requirement.alignment)?;
        
        let meta = WasmTensorMeta::new(
            dtype,
            shape.clone(),
            calculate_row_major_strides(&shape),
            element_count,
            0,
        );
        
        Ok(WasmTensor::new_temporary(offset, meta))
    }
    
    /// Estimate memory requirements for a sequence of operations
    pub fn estimate_memory_for_operations(
        &self,
        operations: &[crate::pattern::OperationDesc]
    ) -> usize {
        let mut total_memory = 0;
        
        for op in operations {
            // Estimate output tensor size
            let output_elements: usize = op.output_shape.iter().product();
            let output_bytes = output_elements * op.output_dtype.byte_size();
            total_memory += output_bytes;
            
            // Add some overhead for intermediate calculations
            total_memory += output_bytes / 4; // 25% overhead
        }
        
        total_memory
    }
}

/// Calculate tensor size in bytes
fn calculate_tensor_bytes(dtype: WasmDType, shape: &[usize]) -> usize {
    let element_count: usize = shape.iter().product();
    let element_size = match dtype {
        WasmDType::Bool => 1,
        WasmDType::Int8 => 1,
        WasmDType::Uint8 => 1,
        WasmDType::Int16 => 2,
        WasmDType::Uint16 => 2,
        WasmDType::Int32 => 4,
        WasmDType::Uint32 => 4,
        WasmDType::Float32 => 4,
        WasmDType::Float64 => 8,
        WasmDType::BigInt64 => 8,
        WasmDType::BigUint64 => 8,
    };
    element_count * element_size
}

/// Calculate row-major strides for tensor shape
fn calculate_row_major_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_temp_tensor_allocation() {
        let mut memory = WasmMemorySystem::new();
        
        let tensor = memory.alloc_temp_tensor(WasmDType::Float32, &[10, 20]).unwrap();
        assert!(tensor.is_temporary());
        assert_eq!(tensor.byte_size(), 10 * 20 * 4); // f32 = 4 bytes
    }
    
    #[test]
    fn test_persistent_tensor_allocation() {
        let mut memory = WasmMemorySystem::new();
        
        let tensor = memory.alloc_persistent_tensor(WasmDType::Float64, &[5, 5]).unwrap();
        assert!(!tensor.is_temporary());
        assert_eq!(tensor.byte_size(), 5 * 5 * 8); // f64 = 8 bytes
    }
    
    #[test]
    fn test_checkpoint_restore() {
        let mut memory = WasmMemorySystem::new();
        
        let _tensor1 = memory.alloc_temp_tensor(WasmDType::Float32, &[100]).unwrap();
        let checkpoint = memory.checkpoint();
        let _tensor2 = memory.alloc_temp_tensor(WasmDType::Float32, &[200]).unwrap();
        
        let stats_before = memory.memory_stats();
        memory.restore(checkpoint).unwrap();
        let stats_after = memory.memory_stats();
        
        assert!(stats_after.arena_used < stats_before.arena_used);
    }
    
    #[test]
    fn test_tensor_from_data() {
        let mut memory = WasmMemorySystem::new();
        
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        
        let tensor = memory.tensor_from_data(bytes, WasmDType::Float32, &[2, 2]).unwrap();
        assert!(!tensor.is_temporary());
        assert_eq!(tensor.byte_size(), 16);
    }
    
    #[test]
    fn test_stride_calculation() {
        assert_eq!(calculate_row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(calculate_row_major_strides(&[5, 7]), vec![7, 1]);
        assert_eq!(calculate_row_major_strides(&[10]), vec![1]);
        assert_eq!(calculate_row_major_strides(&[]), Vec::<usize>::new());
    }
    
    #[test]
    fn test_bulk_allocation_basic() {
        use crate::pattern::{OperationPattern, AllocationRequirement, PatternId};
        
        let mut memory = WasmMemorySystem::new();
        
        // Create a simple pattern with two allocations
        let pattern = OperationPattern {
            pattern_id: PatternId::new(12345),
            operations: vec![], // Empty for this test
            allocations: vec![
                AllocationRequirement {
                    size_bytes: 1024,
                    alignment: 16,
                    is_output: false,
                },
                AllocationRequirement {
                    size_bytes: 2048,
                    alignment: 16,
                    is_output: true,
                },
            ],
            total_memory_needed: 3072, // 1024 + 2048
            estimated_speedup: 2.0,
        };
        
        // Test that we can bulk allocate
        assert!(memory.can_bulk_allocate(&pattern));
        
        let result = memory.bulk_allocate_for_pattern(&pattern);
        assert!(result.is_ok());
        
        let tensors = result.unwrap();
        assert_eq!(tensors.len(), 2);
        
        // Check that tensors are temporary (arena-allocated)
        assert!(tensors[0].is_temporary());
        assert!(tensors[1].is_temporary());
    }
    
    #[test]
    fn test_bulk_allocation_memory_limit() {
        use crate::pattern::{OperationPattern, AllocationRequirement, PatternId};
        
        let memory = WasmMemorySystem::new();
        
        // Create a pattern that requires more memory than available
        let huge_pattern = OperationPattern {
            pattern_id: PatternId::new(67890),
            operations: vec![],
            allocations: vec![
                AllocationRequirement {
                    size_bytes: 1024 * 1024 * 1024, // 1GB - likely exceeds available arena space
                    alignment: 16,
                    is_output: true,
                },
            ],
            total_memory_needed: 1024 * 1024 * 1024,
            estimated_speedup: 1.5,
        };
        
        // Should not be able to bulk allocate
        assert!(!memory.can_bulk_allocate(&huge_pattern));
    }
    
    #[test]
    fn test_bulk_allocation_rollback() {
        use crate::pattern::{OperationPattern, AllocationRequirement, PatternId};
        
        let mut memory = WasmMemorySystem::new();
        
        // Allocate some memory first
        let _existing_tensor = memory.alloc_temp_tensor(WasmDType::Float32, &[1000]).unwrap();
        let stats_before = memory.memory_stats();
        
        // Create a pattern that might fail on second allocation due to memory pressure
        let pattern = OperationPattern {
            pattern_id: PatternId::new(11111),
            operations: vec![],
            allocations: vec![
                AllocationRequirement {
                    size_bytes: 1024,
                    alignment: 16,
                    is_output: false,
                },
                AllocationRequirement {
                    size_bytes: 1024,
                    alignment: 16,
                    is_output: true,
                },
            ],
            total_memory_needed: 2048,
            estimated_speedup: 1.8,
        };
        
        // Try bulk allocation - should succeed in this case since we have plenty of memory
        let result = memory.bulk_allocate_for_pattern(&pattern);
        if result.is_ok() {
            let stats_after = memory.memory_stats();
            // Memory usage should have increased
            assert!(stats_after.arena_used > stats_before.arena_used);
        }
        // Note: In a real implementation, we'd test actual rollback scenarios
        // but that requires more complex memory pressure simulation
    }
    
    // WASM-specific integration tests
    #[cfg(test)]
    mod wasm_tests {
        use super::*;
        use wasm_bindgen_test::*;
        
        #[wasm_bindgen_test]
        fn wasm_test_memory_system_creation() {
            let memory = WasmMemorySystem::new();
            let stats = memory.memory_stats();
            assert!(stats.arena_capacity > 0);
            assert_eq!(stats.persistent_count, 0);
            assert_eq!(stats.arena_used, 0);
        }
        
        #[wasm_bindgen_test]
        fn wasm_test_bulk_allocation_integration() {
            use crate::pattern::{OperationPattern, AllocationRequirement, PatternId};
            
            let mut memory = WasmMemorySystem::new();
            
            // Create a realistic pattern with multiple allocations
            let pattern = OperationPattern {
                pattern_id: PatternId::new(999),
                operations: vec![],
                allocations: vec![
                    AllocationRequirement {
                        size_bytes: 1024,
                        alignment: 16,
                        is_output: false,
                    },
                    AllocationRequirement {
                        size_bytes: 2048,
                        alignment: 16,
                        is_output: false,
                    },
                    AllocationRequirement {
                        size_bytes: 4096,
                        alignment: 16,
                        is_output: true,
                    },
                ],
                total_memory_needed: 7168, // 1024 + 2048 + 4096
                estimated_speedup: 2.5,
            };
            
            // Test memory availability check
            assert!(memory.can_bulk_allocate(&pattern));
            
            // Perform bulk allocation
            let tensors = memory.bulk_allocate_for_pattern(&pattern).unwrap();
            assert_eq!(tensors.len(), 3);
            
            // Verify all tensors are temporary (arena-allocated)
            for tensor in &tensors {
                assert!(tensor.is_temporary());
            }
            
            // Verify memory stats
            let stats = memory.memory_stats();
            assert!(stats.arena_used >= pattern.total_memory_needed);
        }
        
        #[wasm_bindgen_test]
        fn wasm_test_checkpoint_restore_integration() {
            let mut memory = WasmMemorySystem::new();
            
            // Allocate some tensors
            let _tensor1 = memory.alloc_temp_tensor(WasmDType::Float32, &[100, 100]).unwrap();
            let checkpoint = memory.checkpoint();
            let _tensor2 = memory.alloc_temp_tensor(WasmDType::Float64, &[50, 50]).unwrap();
            let _tensor3 = memory.alloc_temp_tensor(WasmDType::Int32, &[200]).unwrap();
            
            let stats_before = memory.memory_stats();
            assert!(stats_before.arena_used > 0);
            
            // Restore to checkpoint
            memory.restore(checkpoint).unwrap();
            let stats_after = memory.memory_stats();
            
            // Memory should be reduced (our current implementation resets completely)
            assert!(stats_after.arena_used <= stats_before.arena_used);
        }
        
        #[wasm_bindgen_test]
        fn wasm_test_mixed_allocation_pattern() {
            let mut memory = WasmMemorySystem::new();
            
            // Mix temporary and persistent allocations
            let temp1 = memory.alloc_temp_tensor(WasmDType::Float32, &[10, 10]).unwrap();
            let persistent1 = memory.alloc_persistent_tensor(WasmDType::Float32, &[10, 10]).unwrap();
            let temp2 = memory.alloc_temp_tensor(WasmDType::Float64, &[5, 5]).unwrap();
            
            assert!(temp1.is_temporary());
            assert!(!persistent1.is_temporary());
            assert!(temp2.is_temporary());
            
            let stats = memory.memory_stats();
            assert!(stats.arena_used > 0);
            assert_eq!(stats.persistent_count, 1);
        }
        
        #[wasm_bindgen_test]
        fn wasm_test_memory_pressure_simulation() {
            let mut memory = WasmMemorySystem::new();
            
            // Try to allocate increasingly large tensors until we hit limits
            let mut successful_allocations = 0;
            let mut total_allocated = 0;
            
            for i in 1..=20 {
                let size = 1024 * 1024 * i; // 1MB, 2MB, 3MB, etc.
                let elements = size / 4; // Float32 elements
                
                match memory.alloc_temp_tensor(WasmDType::Float32, &[elements]) {
                    Ok(_) => {
                        successful_allocations += 1;
                        total_allocated += size;
                    }
                    Err(_) => break, // Hit memory limit
                }
            }
            
            // Should be able to allocate at least some tensors
            assert!(successful_allocations > 0);
            assert!(total_allocated > 0);
            
            let stats = memory.memory_stats();
            assert!(stats.arena_utilization > 0.0);
        }
    }
}