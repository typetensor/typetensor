use wasm_bindgen::prelude::*;
use std::alloc::{Layout, alloc, dealloc};

/// Debug function to test basic memory allocation
#[wasm_bindgen]
pub fn test_basic_allocation() -> String {
    let mut results = Vec::new();
    
    // Test sizes from small to large
    let test_sizes = [16, 64, 256, 1024, 4096];
    
    for size in test_sizes {
        let layout = match Layout::from_size_align(size, 8) {
            Ok(layout) => layout,
            Err(e) => {
                results.push(format!("Size {}: Layout error - {}", size, e));
                continue;
            }
        };
        
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            results.push(format!("Size {}: FAILED - allocation returned null", size));
        } else {
            results.push(format!("Size {}: SUCCESS - allocated at {:p}", size, ptr));
            unsafe { dealloc(ptr, layout) };
        }
    }
    
    results.join("\n")
}

/// Debug function to test our memory manager initialization
#[wasm_bindgen]
pub fn test_memory_manager_init() -> String {
    use crate::memory::WasmMemoryManager;
    
    match std::panic::catch_unwind(|| {
        let manager = WasmMemoryManager::new();
        format!("Memory manager created successfully")
    }) {
        Ok(msg) => msg,
        Err(_) => "Memory manager creation panicked".to_string(),
    }
}

/// Debug function to test buffer creation
#[wasm_bindgen]
pub fn test_buffer_creation() -> String {
    use crate::memory::WasmMemoryManager;
    
    let mut manager = WasmMemoryManager::new();
    let test_data = vec![1u8, 2, 3, 4, 5];
    
    match manager.create_buffer_with_data(&test_data) {
        Ok(handle) => format!("Buffer created successfully: id={}, size={}", handle.id(), handle.size()),
        Err(e) => format!("Buffer creation failed: {}", e),
    }
}