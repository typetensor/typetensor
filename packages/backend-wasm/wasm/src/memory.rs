use wasm_bindgen::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const BUFFER_SIZE_CLASSES: &[usize] = &[
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
];

const MEMORY_ALIGNMENT: usize = 16; // SIMD requires 16-byte alignment
const CACHE_LINE_SIZE: usize = 64;  // Keep cache line awareness

static TOTAL_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static NEXT_BUFFER_ID: AtomicUsize = AtomicUsize::new(1);

/// Buffer identifier
pub type BufferId = usize;

/// Buffer source - tracks whether buffer is managed by pool or allocated directly
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferSource {
    Pooled,    // Managed by pool, return to pool on drop
    Direct,    // Direct allocation, deallocate immediately on drop
}

/// Size class for buffer pools
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferSizeClass {
    Size16B = 0,
    Size64B = 1,
    Size256B = 2,
    Size1KB = 3,
    Size4KB = 4,
    Size16KB = 5,
    Size64KB = 6,
    Size256KB = 7,
    Size1MB = 8,
    Size4MB = 9,
    Size16MB = 10,
}

impl BufferSizeClass {
    fn from_size(size: usize) -> Self {
        match size {
            0..=16 => BufferSizeClass::Size16B,
            17..=64 => BufferSizeClass::Size64B,
            65..=256 => BufferSizeClass::Size256B,
            257..=1024 => BufferSizeClass::Size1KB,
            1025..=4096 => BufferSizeClass::Size4KB,
            4097..=16384 => BufferSizeClass::Size16KB,
            16385..=65536 => BufferSizeClass::Size64KB,
            65537..=262144 => BufferSizeClass::Size256KB,
            262145..=1048576 => BufferSizeClass::Size1MB,
            1048577..=4194304 => BufferSizeClass::Size4MB,
            _ => BufferSizeClass::Size16MB,
        }
    }
    
    fn actual_size(&self) -> usize {
        BUFFER_SIZE_CLASSES[*self as usize]
    }
}

/// Information about an active buffer with reference counting
#[derive(Debug)]
struct BufferInfo {
    ptr: *mut u8,
    size: usize,
    size_class: BufferSizeClass,
    ref_count: AtomicUsize,
}

/// Simple buffer pool for a specific size class
#[derive(Debug)]
struct BufferPool {
    size_class: BufferSizeClass,
    available_buffers: Vec<*mut u8>,
    allocated_count: usize,
    max_pool_size: usize,
}

impl BufferPool {
    fn new(size_class: BufferSizeClass) -> Self {
        // Set appropriate pool sizes based on buffer size
        let max_pool_size = match size_class {
            BufferSizeClass::Size16B => 1000,   // Tiny buffers - lots allowed
            BufferSizeClass::Size64B => 1000,   // Very common size for small tensors
            BufferSizeClass::Size256B => 800,   // Common size
            BufferSizeClass::Size1KB => 500,    // Medium size
            BufferSizeClass::Size4KB => 400,    // Larger tensors
            BufferSizeClass::Size16KB => 300,   // Big tensors
            BufferSizeClass::Size64KB => 200,   // Large tensors
            BufferSizeClass::Size256KB => 150,  // Very large
            BufferSizeClass::Size1MB => 100,    // Huge tensors
            BufferSizeClass::Size4MB => 50,     // Massive tensors
            BufferSizeClass::Size16MB => 25,    // Enormous tensors
        };
        
        BufferPool {
            size_class,
            available_buffers: Vec::new(),
            allocated_count: 0,
            max_pool_size,
        }
    }
    
    fn get_buffer(&mut self) -> *mut u8 {
        if let Some(ptr) = self.available_buffers.pop() {
            ptr
        } else {
            let size = self.size_class.actual_size();
            let aligned_size = align_size(size, MEMORY_ALIGNMENT);
            
            let layout = std::alloc::Layout::from_size_align(aligned_size, MEMORY_ALIGNMENT)
                .expect("Invalid layout for buffer allocation");
            
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                // Return null instead of panicking to allow graceful error handling
                return std::ptr::null_mut();
            }
            
            unsafe { ptr.write_bytes(0, aligned_size) };
            
            TOTAL_ALLOCATED.fetch_add(aligned_size, Ordering::Relaxed);
            self.allocated_count += 1;
            
            ptr
        }
    }
    
    fn return_buffer(&mut self, ptr: *mut u8) {
        let size = self.size_class.actual_size();
        unsafe { ptr.write_bytes(0, size) };
        
        self.available_buffers.push(ptr);
    }
    
    fn cleanup(&mut self) {
        let size = self.size_class.actual_size();
        let aligned_size = align_size(size, MEMORY_ALIGNMENT);
        
        let layout = std::alloc::Layout::from_size_align(aligned_size, MEMORY_ALIGNMENT)
            .expect("Invalid layout for buffer deallocation");
        
        for ptr in self.available_buffers.drain(..) {
            unsafe { std::alloc::dealloc(ptr, layout) };
            TOTAL_ALLOCATED.fetch_sub(aligned_size, Ordering::Relaxed);
        }
        
        self.allocated_count = 0;
    }
}

impl Drop for BufferPool {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Handle to a buffer in WASM memory
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmBufferHandle {
    id: BufferId,
    ptr: *mut u8,
    size: usize,
    size_class: BufferSizeClass,
    source: BufferSource,
    initialized: bool,
}

#[wasm_bindgen]
impl WasmBufferHandle {
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> usize {
        self.id
    }

    #[wasm_bindgen(getter)]
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get pointer for reading
    pub fn get_read_ptr(&self) -> *const u8 {
        if !self.initialized {
            panic!("Attempt to read from uninitialized buffer");
        }
        self.ptr as *const u8
    }
    
    /// Create a shallow copy of this handle
    pub fn clone_handle(&self) -> WasmBufferHandle {
        self.clone()
    }
    
    /// Mark buffer as initialized
    pub(crate) fn mark_initialized(&mut self) {
        self.initialized = true;
    }
    
    /// Get mutable pointer for writing data
    pub(crate) fn ptr_mut(&mut self) -> *mut u8 {
        self.ptr
    }
    
    /// Check if buffer is from pool or direct allocation
    pub(crate) fn is_pooled(&self) -> bool {
        self.source == BufferSource::Pooled
    }
    
    /// Get the raw pointer as a number for JavaScript view creation
    /// This is safe because JavaScript views are bounds-checked
    #[wasm_bindgen(getter)]
    pub fn ptr(&self) -> usize {
        self.ptr as usize
    }
    
    /// Check if the buffer is initialized and safe to read
    #[wasm_bindgen(getter)]
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl WasmBufferHandle {
    /// Create a new pooled buffer handle
    pub(crate) fn new_pooled(id: BufferId, ptr: *mut u8, size: usize, size_class: BufferSizeClass, initialized: bool) -> Self {
        WasmBufferHandle {
            id,
            ptr,
            size,
            size_class,
            source: BufferSource::Pooled,
            initialized,
        }
    }
    
    /// Create a new direct buffer handle
    pub(crate) fn new_direct(ptr: *mut u8, size: usize, size_class: BufferSizeClass) -> Self {
        let id = NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
        WasmBufferHandle {
            id,
            ptr,
            size,
            size_class,
            source: BufferSource::Direct,
            initialized: false,  // Direct buffers start uninitialized
        }
    }
    
    /// Deallocate direct buffer immediately (for reentrancy scenarios)
    pub(crate) fn deallocate_direct(self) {
        if self.source == BufferSource::Direct {
            let size = self.size_class.actual_size();
            let aligned_size = align_size(size, MEMORY_ALIGNMENT);
            
            let layout = std::alloc::Layout::from_size_align(aligned_size, MEMORY_ALIGNMENT)
                .expect("Invalid layout for direct deallocation");
            
            unsafe { 
                std::alloc::dealloc(self.ptr, layout);
            }
            
            TOTAL_ALLOCATED.fetch_sub(aligned_size, Ordering::Relaxed);
        }
        // If pooled buffer, this is a no-op - should be returned to pool instead
    }
}

/// Buffer lifecycle manager - handles creation, reference counting, and disposal
#[derive(Debug)]
pub struct BufferLifecycleManager {
    pools: Vec<BufferPool>,
    active_buffers: HashMap<BufferId, Arc<BufferInfo>>,
}

/// Compute memory manager - handles read/write operations during computations
#[derive(Debug)]
pub struct ComputeMemoryManager {
    // Placeholder for compute-specific state
    // Currently all compute operations use buffer handles directly
}

/// Direct allocation functions (stateless, no RefCell involved)
/// These are used as fallback when pool is busy due to reentrancy

/// Allocate buffer directly from system allocator
pub(crate) fn allocate_direct(size: usize) -> Result<WasmBufferHandle, String> {
    let size_class = BufferSizeClass::from_size(size);
    let actual_size = size_class.actual_size();
    let aligned_size = align_size(actual_size, MEMORY_ALIGNMENT);
    
    let layout = std::alloc::Layout::from_size_align(aligned_size, MEMORY_ALIGNMENT)
        .map_err(|_| "Invalid layout for direct allocation".to_string())?;
    
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        return Err("Direct allocation failed - out of memory".to_string());
    }
    
    // Zero the memory (this might trigger reentrancy, but no RefCell involved)
    unsafe { ptr.write_bytes(0, aligned_size) };
    
    // Update global counter atomically
    TOTAL_ALLOCATED.fetch_add(aligned_size, Ordering::Relaxed);
    
    Ok(WasmBufferHandle::new_direct(ptr, size, size_class))
}

/// Allocate buffer directly with provided data
pub(crate) fn allocate_direct_with_data(data: &[u8]) -> Result<WasmBufferHandle, String> {
    let mut handle = allocate_direct(data.len())?;
    
    // Copy data to the allocated buffer
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), handle.ptr_mut(), data.len());
    }
    
    handle.mark_initialized();
    Ok(handle)
}

impl BufferLifecycleManager {
    pub fn new() -> BufferLifecycleManager {
        let mut pools = Vec::with_capacity(BUFFER_SIZE_CLASSES.len());
        
        pools.push(BufferPool::new(BufferSizeClass::Size16B));
        pools.push(BufferPool::new(BufferSizeClass::Size64B));
        pools.push(BufferPool::new(BufferSizeClass::Size256B));
        pools.push(BufferPool::new(BufferSizeClass::Size1KB));
        pools.push(BufferPool::new(BufferSizeClass::Size4KB));
        pools.push(BufferPool::new(BufferSizeClass::Size16KB));
        pools.push(BufferPool::new(BufferSizeClass::Size64KB));
        pools.push(BufferPool::new(BufferSizeClass::Size256KB));
        pools.push(BufferPool::new(BufferSizeClass::Size1MB));
        pools.push(BufferPool::new(BufferSizeClass::Size4MB));
        pools.push(BufferPool::new(BufferSizeClass::Size16MB));
        
        BufferLifecycleManager {
            pools,
            active_buffers: HashMap::new(),
        }
    }

    /// Create buffer with data
    pub fn create_buffer_with_data(&mut self, data: &[u8]) -> Result<WasmBufferHandle, String> {
        let size = data.len();
        let size_class = BufferSizeClass::from_size(size);
        let id = NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
        
        let pool = &mut self.pools[size_class as usize];
        let ptr = pool.get_buffer();
        if ptr.is_null() {
            return Err(format!("Failed to allocate buffer of size {}", size));
        }
        
        // Copy data to buffer
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, size);
        }
        
        let buffer_info = Arc::new(BufferInfo {
            ptr,
            size,
            size_class,
            ref_count: AtomicUsize::new(1),
        });
        
        self.active_buffers.insert(id, buffer_info);
        TOTAL_ALLOCATED.fetch_add(size, Ordering::Relaxed);
        
        Ok(WasmBufferHandle::new_pooled(id, ptr, size, size_class, true))
    }

    /// Create empty buffer
    pub fn create_empty_buffer(&mut self, size: usize) -> Result<(WasmBufferHandle, *mut u8), String> {
        let size_class = BufferSizeClass::from_size(size);
        let id = NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
        
        let pool = &mut self.pools[size_class as usize];
        let ptr = pool.get_buffer();
        if ptr.is_null() {
            return Err(format!("Failed to allocate buffer of size {}", size));
        }
        
        let buffer_info = Arc::new(BufferInfo {
            ptr,
            size,
            size_class,
            ref_count: AtomicUsize::new(1),
        });
        
        self.active_buffers.insert(id, buffer_info);
        TOTAL_ALLOCATED.fetch_add(size, Ordering::Relaxed);
        
        let handle = WasmBufferHandle::new_pooled(id, ptr, size, size_class, false);
        
        Ok((handle, ptr))
    }

    /// Release buffer back to pool for reuse
    pub fn release_buffer(&mut self, handle: WasmBufferHandle) -> bool {
        if let Some(buffer_info) = self.active_buffers.get(&handle.id) {
            let prev_count = buffer_info.ref_count.fetch_sub(1, Ordering::AcqRel);
            
            if prev_count == 1 {
                if let Some(buffer_info) = self.active_buffers.remove(&handle.id) {
                    let pool = &mut self.pools[buffer_info.size_class as usize];
                    pool.return_buffer(buffer_info.ptr);
                }
            }
            true
        } else {
            false
        }
    }

    /// Try to get buffer from pool (non-blocking)
    /// Returns None if no buffer available or if pool is empty
    pub fn try_get_buffer(&mut self, size: usize) -> Option<WasmBufferHandle> {
        let size_class = BufferSizeClass::from_size(size);
        let id = NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
        
        let pool = &mut self.pools[size_class as usize];
        let ptr = pool.get_buffer();
        if ptr.is_null() {
            return None; // Pool allocation failed
        }
        
        let buffer_info = Arc::new(BufferInfo {
            ptr,
            size,
            size_class,
            ref_count: AtomicUsize::new(1),
        });
        
        self.active_buffers.insert(id, buffer_info);
        TOTAL_ALLOCATED.fetch_add(size, Ordering::Relaxed);
        
        Some(WasmBufferHandle::new_pooled(id, ptr, size, size_class, false))
    }
    
    /// Try to release buffer back to pool (non-blocking)
    /// Returns true if successful, false if buffer not found or not pooled
    pub fn try_release_buffer(&mut self, handle: WasmBufferHandle) -> bool {
        if handle.source != BufferSource::Pooled {
            return false; // Not a pooled buffer
        }
        
        if let Some(buffer_info) = self.active_buffers.get(&handle.id) {
            let prev_count = buffer_info.ref_count.fetch_sub(1, Ordering::AcqRel);
            
            if prev_count == 1 {
                if let Some(buffer_info) = self.active_buffers.remove(&handle.id) {
                    let pool = &mut self.pools[buffer_info.size_class as usize];
                    pool.return_buffer(buffer_info.ptr);
                }
            }
            true
        } else {
            false
        }
    }

    /// Clone a buffer handle with proper reference counting
    /// Returns Some(cloned_handle) if successful, None if buffer not found
    pub fn clone_buffer_handle(&mut self, handle: &WasmBufferHandle) -> Option<WasmBufferHandle> {
        if handle.source != BufferSource::Pooled {
            // Direct buffers can't be cloned with reference counting - 
            // caller should use defensive clone
            return None;
        }
        
        if let Some(buffer_info) = self.active_buffers.get(&handle.id) {
            // Increment reference count atomically
            buffer_info.ref_count.fetch_add(1, Ordering::AcqRel);
            
            // Create new handle with same ID - shares the buffer
            let cloned_handle = WasmBufferHandle::new_pooled(
                handle.id, 
                handle.ptr, 
                handle.size, 
                handle.size_class, 
                handle.initialized
            );
            
            Some(cloned_handle)
        } else {
            // Buffer not found in active buffers
            None
        }
    }

    /// Compact memory pools
    pub fn compact_pools(&mut self) {
        for pool in &mut self.pools {
            pool.cleanup();
        }
    }

    /// Get current memory usage statistics
    pub fn get_memory_stats(&self) -> WasmMemoryStats {
        let mut stats_by_class = Vec::new();
        
        for (i, pool) in self.pools.iter().enumerate() {
            let size_class_bytes = BUFFER_SIZE_CLASSES[i];
            stats_by_class.push(PoolStats {
                size_class_bytes,
                available_buffers: pool.available_buffers.len(),
                allocated_buffers: pool.allocated_count,
            });
        }
        
        WasmMemoryStats {
            total_allocated_bytes: TOTAL_ALLOCATED.load(Ordering::Relaxed),
            active_buffers: self.active_buffers.len(),
            pool_stats: stats_by_class,
        }
    }
}

impl ComputeMemoryManager {
    pub fn new() -> ComputeMemoryManager {
        ComputeMemoryManager {}
    }

    /// Get read pointer
    pub fn get_read_ptr(&self, handle: &WasmBufferHandle) -> *const u8 {
        handle.get_read_ptr()
    }
    
    /// Get write pointer for buffer initialization
    pub fn get_write_ptr(&self, handle: &WasmBufferHandle) -> *mut u8 {
        if handle.initialized {
            panic!("Attempt to write to already initialized buffer");
        }
        handle.ptr
    }
}

/// WebAssembly memory manager with buffer pools
#[wasm_bindgen]
pub struct WasmMemoryManager {
    buffer_lifecycle: BufferLifecycleManager,
    compute_manager: ComputeMemoryManager,
}

#[wasm_bindgen]
impl WasmMemoryManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmMemoryManager {
        WasmMemoryManager {
            buffer_lifecycle: BufferLifecycleManager::new(),
            compute_manager: ComputeMemoryManager::new(),
        }
    }

    /// Create buffer with data
    pub fn create_buffer_with_data(&mut self, data: &[u8]) -> Result<WasmBufferHandle, String> {
        self.buffer_lifecycle.create_buffer_with_data(data)
    }
    
    /// Create empty buffer for writing
    pub(crate) fn create_empty_buffer(&mut self, size: usize) -> Result<(WasmBufferHandle, *mut u8), String> {
        self.buffer_lifecycle.create_empty_buffer(size)
    }

    /// Get read pointer
    pub(crate) fn get_read_ptr(&self, handle: &WasmBufferHandle) -> *const u8 {
        self.compute_manager.get_read_ptr(handle)
    }
    
    /// Get write pointer for buffer initialization
    pub(crate) fn get_write_ptr(&self, handle: &WasmBufferHandle) -> *mut u8 {
        self.compute_manager.get_write_ptr(handle)
    }
    
    /// Release buffer back to pool for reuse
    pub fn release_buffer(&mut self, handle: WasmBufferHandle) -> bool {
        self.buffer_lifecycle.release_buffer(handle)
    }

    /// Get current memory usage statistics
    #[wasm_bindgen]
    pub fn get_memory_stats(&self) -> WasmMemoryStats {
        self.buffer_lifecycle.get_memory_stats()
    }
    
    /// Mark a buffer as initialized after writing to it
    pub(crate) fn mark_buffer_initialized(&self, handle: &mut WasmBufferHandle) {
        handle.mark_initialized();
    }
    
    /// Increment reference count for a buffer
    pub(crate) fn increment_ref_count(&self, buffer_id: BufferId) {
        // TODO: Move this to buffer lifecycle manager
        // For now, we need to access the buffer_lifecycle's active_buffers
        // This will be refactored properly
    }
    
    /// Try to get buffer from pool (non-blocking)
    pub(crate) fn try_get_buffer(&mut self, size: usize) -> Option<WasmBufferHandle> {
        self.buffer_lifecycle.try_get_buffer(size)
    }
    
    /// Try to release buffer back to pool (non-blocking)
    pub(crate) fn try_release_buffer(&mut self, handle: WasmBufferHandle) -> bool {
        self.buffer_lifecycle.try_release_buffer(handle)
    }

    /// Clone a buffer handle with proper reference counting
    pub(crate) fn clone_buffer_handle(&mut self, handle: &WasmBufferHandle) -> Option<WasmBufferHandle> {
        self.buffer_lifecycle.clone_buffer_handle(handle)
    }

    /// Compact pools by deallocating excess buffers
    pub fn compact_pools(&mut self) {
        self.buffer_lifecycle.compact_pools();
    }
}

/// Statistics for a single buffer pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub size_class_bytes: usize,
    pub available_buffers: usize,
    pub allocated_buffers: usize,
}

/// Memory usage statistics
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WasmMemoryStats {
    total_allocated_bytes: usize,
    active_buffers: usize,
    pool_stats: Vec<PoolStats>,
}

#[wasm_bindgen]
impl WasmMemoryStats {
    #[wasm_bindgen(getter)]
    pub fn total_allocated_bytes(&self) -> usize {
        self.total_allocated_bytes
    }

    #[wasm_bindgen(getter)]
    pub fn active_buffers(&self) -> usize {
        self.active_buffers
    }
    
    #[wasm_bindgen]
    pub fn get_pool_summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Total allocated: {} bytes\n", self.total_allocated_bytes));
        summary.push_str(&format!("Active buffers: {}\n", self.active_buffers));
        summary.push_str("Pool details:\n");
        
        for stats in &self.pool_stats {
            summary.push_str(&format!(
                "  {} bytes: {} available, {} allocated\n",
                stats.size_class_bytes,
                stats.available_buffers,
                stats.allocated_buffers
            ));
        }
        
        summary
    }
    
    /// Create fallback stats for when memory manager is busy (reentrancy scenario)
    pub(crate) fn busy_fallback() -> Self {
        WasmMemoryStats {
            total_allocated_bytes: TOTAL_ALLOCATED.load(Ordering::Relaxed),
            active_buffers: 0,  // Can't count active buffers when manager is busy
            pool_stats: vec![], // Can't get pool stats when manager is busy
        }
    }
}

fn align_size(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_size_class_mapping() {
        assert_eq!(BufferSizeClass::from_size(10), BufferSizeClass::Size16B);
        assert_eq!(BufferSizeClass::from_size(50), BufferSizeClass::Size64B);
        assert_eq!(BufferSizeClass::from_size(1000), BufferSizeClass::Size1KB);
        assert_eq!(BufferSizeClass::from_size(100000), BufferSizeClass::Size256KB);
    }

    #[test]
    fn test_buffer_creation_and_release() {
        let mut manager = WasmMemoryManager::new();
        let data = vec![1u8, 2, 3, 4, 5];
        
        // Create buffer with data
        let handle = manager.create_buffer_with_data(&data).unwrap();
        assert_eq!(handle.size(), 5);
        
        // Read data back
        let ptr = manager.get_read_ptr(&handle);
        let read_data = unsafe { std::slice::from_raw_parts(ptr, 5) };
        assert_eq!(read_data, &[1, 2, 3, 4, 5]);
        
        // Release buffer
        assert!(manager.release_buffer(handle));
    }

    #[test]
    fn test_buffer_pool_reuse() {
        let mut manager = WasmMemoryManager::new();
        
        // Create and release a buffer
        let data1 = vec![1u8, 2, 3, 4];
        let handle1 = manager.create_buffer_with_data(&data1).unwrap();
        let ptr1 = handle1.ptr;
        manager.release_buffer(handle1);
        
        // Create another buffer of the same size - should reuse the same memory
        let data2 = vec![5u8, 6, 7, 8];
        let handle2 = manager.create_buffer_with_data(&data2).unwrap();
        let ptr2 = handle2.ptr;
        
        // Should reuse the same pointer (after zero-ing)
        assert_eq!(ptr1, ptr2);
        
        // Data should be the new data
        let read_data = unsafe { std::slice::from_raw_parts(ptr2, 4) };
        assert_eq!(read_data, &[5, 6, 7, 8]);
        
        manager.release_buffer(handle2);
    }

    #[test]
    fn test_empty_buffer_creation() {
        let mut manager = WasmMemoryManager::new();
        
        // Create empty buffer
        let (handle, write_ptr) = manager.create_empty_buffer(100).unwrap();
        assert_eq!(handle.size(), 100);
        
        // Write some data
        unsafe {
            for i in 0..100 {
                *write_ptr.add(i) = (i % 256) as u8;
            }
        }
        
        // Read data back
        let read_ptr = manager.get_read_ptr(&handle);
        let read_data = unsafe { std::slice::from_raw_parts(read_ptr, 100) };
        
        for i in 0..100 {
            assert_eq!(read_data[i], (i % 256) as u8);
        }
        
        manager.release_buffer(handle);
    }

    #[test]
    fn test_memory_stats() {
        let mut manager = WasmMemoryManager::new();
        
        let initial_stats = manager.get_memory_stats();
        assert_eq!(initial_stats.active_buffers(), 0);
        
        // Create some buffers
        let data = vec![1u8; 1000];
        let handle1 = manager.create_buffer_with_data(&data).unwrap();
        let handle2 = manager.create_buffer_with_data(&data).unwrap();
        
        let stats = manager.get_memory_stats();
        assert_eq!(stats.active_buffers(), 2);
        assert!(stats.total_allocated_bytes() > 0);
        
        // Release buffers
        manager.release_buffer(handle1);
        manager.release_buffer(handle2);
        
        let final_stats = manager.get_memory_stats();
        assert_eq!(final_stats.active_buffers(), 0);
    }
}