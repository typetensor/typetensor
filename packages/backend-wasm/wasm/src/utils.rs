/*!
 * Utility functions for WebAssembly operations
 * 
 * Provides helper functions for:
 * - Performance timing and profiling
 * - Error handling and debugging
 * - Memory management helpers
 * - SIMD feature detection
 */

use wasm_bindgen::prelude::*;

/// Simple console logging for debugging
pub fn console_log(message: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::console::log_1(&message.into());
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("{}", message);
    }
}

/// Log a message to the browser console with timing information
pub fn log_with_timing(message: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        // Try to use web_sys for performance timing, fallback to simple logging if not available
        if let Some(window) = web_sys::window() {
            if let Some(performance) = window.performance() {
                let timestamp = performance.now();
                web_sys::console::log_1(&format!("[{:.3}ms] {}", timestamp, message).into());
                return;
            }
        }
        
        // Fallback: simple console logging without timing (Node.js/test environment)
        web_sys::console::log_1(&format!("[WASM] {}", message).into());
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        // For native tests, just print to stdout
        println!("[TEST] {}", message);
    }
}

/// Check if WebAssembly SIMD is supported
#[wasm_bindgen]
pub fn has_simd_support() -> bool {
    // This is a runtime check for SIMD support
    // In actual implementation, we'd use feature detection
    cfg!(target_feature = "simd128")
}

/// Check if SharedArrayBuffer is available
#[wasm_bindgen]
pub fn has_shared_memory_support() -> bool {
    // This would check for SharedArrayBuffer support in the JS environment
    // For now, we'll return false as a conservative default
    false
}

/// Get the optimal number of threads for operations
#[wasm_bindgen]
pub fn get_optimal_thread_count() -> usize {
    // In a real implementation, this would query navigator.hardwareConcurrency
    // For now, return a sensible default
    4
}

/// Calculate optimal chunk size for processing large arrays
pub fn calculate_chunk_size(total_size: usize, num_threads: usize) -> usize {
    // Ensure minimum chunk size for efficiency
    let min_chunk_size = 1024;
    let calculated = (total_size + num_threads - 1) / num_threads;
    calculated.max(min_chunk_size)
}

/// Align a size to the nearest multiple of alignment
pub fn align_size(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

/// Check if two memory ranges overlap
pub fn ranges_overlap(start1: usize, len1: usize, start2: usize, len2: usize) -> bool {
    let end1 = start1 + len1;
    let end2 = start2 + len2;
    start1 < end2 && start2 < end1
}

/// Safe division with overflow protection
pub fn safe_div_f32(a: f32, b: f32) -> f32 {
    if b == 0.0 {
        if a > 0.0 { f32::INFINITY } else if a < 0.0 { f32::NEG_INFINITY } else { f32::NAN }
    } else {
        a / b
    }
}

/// Safe division with overflow protection  
pub fn safe_div_f64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        if a > 0.0 { f64::INFINITY } else if a < 0.0 { f64::NEG_INFINITY } else { f64::NAN }
    } else {
        a / b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_size() {
        assert_eq!(align_size(10, 8), 16);
        assert_eq!(align_size(16, 8), 16);
        assert_eq!(align_size(17, 8), 24);
    }

    #[test]
    fn test_ranges_overlap() {
        assert!(ranges_overlap(0, 10, 5, 10));
        assert!(!ranges_overlap(0, 5, 10, 5));
        assert!(ranges_overlap(10, 5, 0, 15));
    }

    #[test]
    fn test_safe_div() {
        assert_eq!(safe_div_f32(1.0, 2.0), 0.5);
        assert!(safe_div_f32(1.0, 0.0).is_infinite());
        assert!(safe_div_f32(-1.0, 0.0).is_infinite());
        assert!(safe_div_f32(0.0, 0.0).is_nan());
    }
}