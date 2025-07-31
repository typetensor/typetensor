
use wasm_bindgen::prelude::*;

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

pub fn log_with_timing(message: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        if let Some(window) = web_sys::window() {
            if let Some(performance) = window.performance() {
                let timestamp = performance.now();
                web_sys::console::log_1(&format!("[{:.3}ms] {}", timestamp, message).into());
                return;
            }
        }
        
        web_sys::console::log_1(&format!("[WASM] {}", message).into());
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        println!("[TEST] {}", message);
    }
}

#[wasm_bindgen]
pub fn has_simd_support() -> bool {
    cfg!(target_feature = "simd128")
}

#[wasm_bindgen]
pub fn has_shared_memory_support() -> bool {
    false
}

#[wasm_bindgen]
pub fn get_optimal_thread_count() -> usize {
    4
}

pub fn calculate_chunk_size(total_size: usize, num_threads: usize) -> usize {
    let min_chunk_size = 1024;
    let calculated = (total_size + num_threads - 1) / num_threads;
    calculated.max(min_chunk_size)
}

pub fn align_size(size: usize, alignment: usize) -> usize {
    (size + alignment - 1) & !(alignment - 1)
}

pub fn ranges_overlap(start1: usize, len1: usize, start2: usize, len2: usize) -> bool {
    let end1 = start1 + len1;
    let end2 = start2 + len2;
    start1 < end2 && start2 < end1
}

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