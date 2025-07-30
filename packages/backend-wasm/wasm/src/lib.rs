/*!
 * TypeTensor WebAssembly Backend
 *
 * High-performance tensor operations compiled to WebAssembly
 * 
 * This module provides:
 * - Memory-efficient tensor data management
 * - SIMD-optimized operations where available  
 * - Multi-threading support via Web Workers
 * - Zero-copy operations where possible
 */

mod utils;
mod memory;
mod operations;
mod types;

use wasm_bindgen::prelude::*;

// Use the default allocator for better compatibility and performance
// wee_alloc was causing conflicts with direct std::alloc usage

// This is like the `extern` block, except it's for JS.
#[wasm_bindgen]
extern "C" {
    // Bind the `console.log` function from the JS side
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    // Bind performance.now() for high-precision timing
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}

// Define a macro for easier console logging
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
pub fn greet() {
    console_log!("Hello from TypeTensor WASM backend!");
}

#[wasm_bindgen]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Initialize the WASM module
/// This should be called once when the module is loaded
#[wasm_bindgen(start)]
pub fn init() {
    // Set panic hook for better error reporting in development
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    console_log!("TypeTensor WASM backend initialized - version {}", get_version());
}

// Export the core types and functions
pub use memory::*;
pub use operations::*;
pub use types::*;
pub use utils::*;