/*!
 * Arena-based memory allocator for WASM tensor operations
 * 
 * Implements a hybrid system:
 * - TempArena: Bump allocator for temporary tensors (bulk cleanup)
 * - PersistentStorage: Reference-counted long-lived tensors
 * - WASM-optimized: 4GB limit aware, 16-byte SIMD alignment
 */

use std::sync::Arc;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// Maximum WASM linear memory (4GB - some headroom)
const MAX_WASM_MEMORY: usize = 3 * 1024 * 1024 * 1024; // 3GB
/// SIMD alignment requirement
const SIMD_ALIGNMENT: usize = 16;
/// Initial arena size (conservative start)
const INITIAL_ARENA_SIZE: usize = 64 * 1024 * 1024; // 64MB
/// Maximum single allocation size
const MAX_ALLOCATION_SIZE: usize = 512 * 1024 * 1024; // 512MB

/// Checkpoint ID for scoped memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointId(usize);

/// Arena offset for temporary allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaOffset {
    offset: usize,
    size: usize,
}

impl ArenaOffset {
    pub fn new(offset: usize, size: usize) -> Self {
        ArenaOffset { offset, size }
    }
    
    pub fn offset(&self) -> usize {
        self.offset
    }
    
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Temporary arena for short-lived tensors
pub struct TempArena {
    memory: Vec<u8>,
    current: usize,
    limit: usize,
    checkpoints: Vec<usize>,
    total_allocated: usize,
}

impl TempArena {
    /// Create new temporary arena
    pub fn new() -> Self {
        let mut memory = Vec::with_capacity(INITIAL_ARENA_SIZE);
        // Ensure the memory is actually allocated
        memory.resize(INITIAL_ARENA_SIZE, 0);
        
        TempArena {
            memory,
            current: 0,
            limit: INITIAL_ARENA_SIZE,
            checkpoints: Vec::new(),
            total_allocated: 0,
        }
    }
    
    /// Allocate SIMD-aligned memory in arena
    pub fn alloc(&mut self, size: usize) -> Result<ArenaOffset, String> {
        if size == 0 {
            return Ok(ArenaOffset::new(0, 0));
        }
        
        if size > MAX_ALLOCATION_SIZE {
            return Err(format!("Allocation too large: {} bytes", size));
        }
        
        // Align to SIMD boundary
        let aligned_current = align_up(self.current, SIMD_ALIGNMENT);
        let aligned_size = align_up(size, SIMD_ALIGNMENT);
        
        // Check if we need to grow the arena
        if aligned_current + aligned_size > self.limit {
            self.grow_arena(aligned_size)?;
        }
        
        let offset = aligned_current;
        self.current = aligned_current + aligned_size;
        self.total_allocated += aligned_size;
        
        Ok(ArenaOffset::new(offset, size))
    }
    
    /// Get pointer to arena memory at offset
    pub fn get_ptr(&self, offset: ArenaOffset) -> *const u8 {
        if offset.offset() + offset.size() > self.limit {
            panic!("Arena offset out of bounds");
        }
        unsafe { self.memory.as_ptr().add(offset.offset()) }
    }
    
    /// Get mutable pointer to arena memory at offset
    pub fn get_mut_ptr(&mut self, offset: ArenaOffset) -> *mut u8 {
        if offset.offset() + offset.size() > self.limit {
            panic!("Arena offset out of bounds");
        }
        unsafe { self.memory.as_mut_ptr().add(offset.offset()) }
    }
    
    /// Create checkpoint for scoped memory management
    pub fn checkpoint(&mut self) -> CheckpointId {
        let id = self.checkpoints.len();
        self.checkpoints.push(self.current);
        CheckpointId(id)
    }
    
    /// Restore to checkpoint (bulk deallocation)
    pub fn restore(&mut self, checkpoint: CheckpointId) -> Result<(), String> {
        if checkpoint.0 >= self.checkpoints.len() {
            return Err("Invalid checkpoint ID".to_string());
        }
        
        let restore_point = self.checkpoints[checkpoint.0];
        if restore_point > self.current {
            return Err("Cannot restore to future checkpoint".to_string());
        }
        
        // Bulk deallocate by resetting pointer
        self.current = restore_point;
        
        // Remove later checkpoints
        self.checkpoints.truncate(checkpoint.0 + 1);
        
        Ok(())
    }
    
    /// Reset entire arena (deallocate everything)
    pub fn reset(&mut self) {
        self.current = 0;
        self.checkpoints.clear();
        self.total_allocated = 0;
    }
    
    /// Get memory usage statistics
    pub fn memory_usage(&self) -> (usize, usize, f32) {
        let used = self.current;
        let capacity = self.limit;
        let utilization = if capacity > 0 { used as f32 / capacity as f32 } else { 0.0 };
        (used, capacity, utilization)
    }
    
    /// Grow arena to accommodate larger allocations
    fn grow_arena(&mut self, needed_size: usize) -> Result<(), String> {
        // Calculate new size (double current or fit needed size, whichever is larger)
        let new_size = std::cmp::max(self.limit * 2, self.limit + needed_size);
        
        // Check WASM memory limits
        if new_size > MAX_WASM_MEMORY {
            return Err(format!("Arena would exceed WASM memory limit: {} bytes", new_size));
        }
        
        // Resize the memory vector
        self.memory.resize(new_size, 0);
        self.limit = new_size;
        
        Ok(())
    }
    
    /// Check if arena is approaching memory pressure
    pub fn is_memory_pressure(&self) -> bool {
        self.limit > MAX_WASM_MEMORY / 2  // Over 1.5GB
    }
    
    /// Allocate aligned memory in arena (for bulk allocation optimization)
    pub fn alloc_aligned(&mut self, size: usize, alignment: usize) -> Result<ArenaOffset, String> {
        if size == 0 {
            return Ok(ArenaOffset::new(0, 0));
        }
        
        if size > MAX_ALLOCATION_SIZE {
            return Err(format!("Allocation too large: {} bytes", size));
        }
        
        // Use the larger of requested alignment or SIMD alignment
        let required_alignment = std::cmp::max(alignment, SIMD_ALIGNMENT);
        
        // Align current position
        let aligned_current = align_up(self.current, required_alignment);
        let aligned_size = align_up(size, required_alignment);
        
        // Check if we need to grow the arena
        if aligned_current + aligned_size > self.limit {
            self.grow_arena(aligned_size)?;
        }
        
        let offset = aligned_current;
        self.current = aligned_current + aligned_size;
        self.total_allocated += aligned_size;
        
        Ok(ArenaOffset::new(offset, size))
    }
    
    /// Get available memory in arena
    pub fn available_memory(&self) -> usize {
        self.limit.saturating_sub(self.current)
    }
}

/// Persistent tensor data with reference counting
#[derive(Debug)]
pub struct PersistentTensor {
    data: Vec<u8>,
    size: usize,
}

impl PersistentTensor {
    pub fn new(size: usize) -> Self {
        let mut data = Vec::with_capacity(align_up(size, SIMD_ALIGNMENT));
        data.resize(align_up(size, SIMD_ALIGNMENT), 0);
        
        PersistentTensor { data, size }
    }
    
    pub fn with_data(data: Vec<u8>) -> Self {
        let size = data.len();
        PersistentTensor { data, size }
    }
    
    pub fn get_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
    
    pub fn get_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }
    
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Tensor ID for persistent storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(u64);

/// Persistent storage for long-lived tensors
pub struct PersistentStorage {
    tensors: HashMap<TensorId, Arc<PersistentTensor>>,
    next_id: u64,
}

impl PersistentStorage {
    pub fn new() -> Self {
        PersistentStorage {
            tensors: HashMap::new(),
            next_id: 1,
        }
    }
    
    /// Store a persistent tensor
    pub fn store(&mut self, tensor: PersistentTensor) -> TensorId {
        let id = TensorId(self.next_id);
        self.next_id += 1;
        self.tensors.insert(id, Arc::new(tensor));
        id
    }
    
    /// Get reference to persistent tensor
    pub fn get(&self, id: TensorId) -> Option<Arc<PersistentTensor>> {
        self.tensors.get(&id).cloned()
    }
    
    /// Remove persistent tensor (if no other references exist)
    pub fn remove(&mut self, id: TensorId) -> bool {
        self.tensors.remove(&id).is_some()
    }
    
    /// Get storage statistics
    pub fn storage_stats(&self) -> (usize, usize) {
        let count = self.tensors.len();
        let total_bytes: usize = self.tensors.values()
            .map(|t| t.size())
            .sum();
        (count, total_bytes)
    }
    
    /// Garbage collect unreferenced tensors
    pub fn gc(&mut self) -> usize {
        let initial_count = self.tensors.len();
        self.tensors.retain(|_, tensor| Arc::strong_count(tensor) > 1);
        initial_count - self.tensors.len()
    }
}

/// Align value up to boundary
#[inline]
fn align_up(value: usize, boundary: usize) -> usize {
    (value + boundary - 1) & !(boundary - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arena_basic_allocation() {
        let mut arena = TempArena::new();
        
        let offset1 = arena.alloc(64).unwrap();
        let offset2 = arena.alloc(128).unwrap();
        
        assert_eq!(offset1.size(), 64);
        assert_eq!(offset2.size(), 128);
        assert!(offset2.offset() >= offset1.offset() + align_up(64, SIMD_ALIGNMENT));
    }
    
    #[test]
    fn test_arena_checkpoint_restore() {
        let mut arena = TempArena::new();
        
        let _offset1 = arena.alloc(64).unwrap();
        let checkpoint = arena.checkpoint();
        let _offset2 = arena.alloc(128).unwrap();
        
        let (used_before, _, _) = arena.memory_usage();
        arena.restore(checkpoint).unwrap();
        let (used_after, _, _) = arena.memory_usage();
        
        assert!(used_after < used_before);
    }
    
    #[test]
    fn test_simd_alignment() {
        let mut arena = TempArena::new();
        
        let offset = arena.alloc(17).unwrap(); // Odd size
        let ptr = arena.get_ptr(offset);
        
        // Pointer should be SIMD aligned
        assert_eq!(ptr as usize % SIMD_ALIGNMENT, 0);
    }
    
    #[test]
    fn test_persistent_storage() {
        let mut storage = PersistentStorage::new();
        
        let tensor = PersistentTensor::new(1024);
        let id = storage.store(tensor);
        
        let retrieved = storage.get(id).unwrap();
        assert_eq!(retrieved.size(), 1024);
        
        let (count, bytes) = storage.storage_stats();
        assert_eq!(count, 1);
        assert_eq!(bytes, 1024);
    }
}