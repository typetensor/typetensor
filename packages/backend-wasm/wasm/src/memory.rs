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
}

/// WebAssembly memory manager with buffer pools
#[wasm_bindgen]
pub struct WasmMemoryManager {
    pools: Vec<BufferPool>,
    active_buffers: HashMap<BufferId, Arc<BufferInfo>>,
}

#[wasm_bindgen]
impl WasmMemoryManager {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmMemoryManager {
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
        
        WasmMemoryManager {
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
        
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, size);
        }
        
        self.active_buffers.insert(id, Arc::new(BufferInfo {
            ptr,
            size,
            size_class,
            ref_count: AtomicUsize::new(1),
        }));
        
        Ok(WasmBufferHandle {
            id,
            ptr,
            size,
            size_class,
            initialized: true,
        })
    }
    
    /// Create empty buffer for writing
    pub(crate) fn create_empty_buffer(&mut self, size: usize) -> Result<(WasmBufferHandle, *mut u8), String> {
        let size_class = BufferSizeClass::from_size(size);
        let id = NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed);
        
        let pool = &mut self.pools[size_class as usize];
        let ptr = pool.get_buffer();
        
        if ptr.is_null() {
            return Err(format!("Failed to allocate buffer of size {}", size));
        }
        
        self.active_buffers.insert(id, Arc::new(BufferInfo {
            ptr,
            size,
            size_class,
            ref_count: AtomicUsize::new(1),
        }));
        
        let handle = WasmBufferHandle {
            id,
            ptr,
            size,
            size_class,
            initialized: false,
        };
        
        Ok((handle, ptr))
    }

    /// Get read pointer
    pub(crate) fn get_read_ptr(&self, handle: &WasmBufferHandle) -> *const u8 {
        handle.get_read_ptr()
    }
    
    /// Get write pointer for buffer initialization
    pub(crate) fn get_write_ptr(&self, handle: &WasmBufferHandle) -> *mut u8 {
        if handle.initialized {
            panic!("Attempt to write to already initialized buffer");
        }
        handle.ptr
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

    /// Get current memory usage statistics
    #[wasm_bindgen]
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
    
    /// Mark a buffer as initialized after writing to it
    pub(crate) fn mark_buffer_initialized(&self, handle: &mut WasmBufferHandle) {
        handle.mark_initialized();
    }
    
    /// Increment reference count for a buffer
    pub(crate) fn increment_ref_count(&self, buffer_id: BufferId) {
        if let Some(buffer_info) = self.active_buffers.get(&buffer_id) {
            buffer_info.ref_count.fetch_add(1, Ordering::AcqRel);
        }
    }
    
    /// Compact pools by deallocating excess buffers
    pub fn compact_pools(&mut self) {
        for pool in &mut self.pools {
            // Only compact if we have significantly more than max pool size
            let compact_threshold = pool.max_pool_size + (pool.max_pool_size / 4); // 25% buffer
            
            while pool.available_buffers.len() > compact_threshold {
                if let Some(ptr) = pool.available_buffers.pop() {
                    let size = pool.size_class.actual_size();
                    let aligned_size = align_size(size, MEMORY_ALIGNMENT);
                    
                    let layout = std::alloc::Layout::from_size_align(aligned_size, MEMORY_ALIGNMENT)
                        .expect("Invalid layout for buffer deallocation");
                    
                    unsafe { std::alloc::dealloc(ptr, layout) };
                    TOTAL_ALLOCATED.fetch_sub(aligned_size, Ordering::Relaxed);
                    pool.allocated_count -= 1;
                }
            }
        }
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
        let handle = manager.create_buffer_with_data(&data);
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
        let handle1 = manager.create_buffer_with_data(&data1);
        let ptr1 = handle1.ptr;
        manager.release_buffer(handle1);
        
        // Create another buffer of the same size - should reuse the same memory
        let data2 = vec![5u8, 6, 7, 8];
        let handle2 = manager.create_buffer_with_data(&data2);
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
        let (handle, write_ptr) = manager.create_empty_buffer(100);
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
        let handle1 = manager.create_buffer_with_data(&data);
        let handle2 = manager.create_buffer_with_data(&data);
        
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