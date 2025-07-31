
mod utils;
mod memory;
mod operations;
mod types;
mod debug;
mod simd;
mod fast_math;
mod view_descriptor;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = performance, js_name = now)]
    fn performance_now() -> f64;
}

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

/// Check if SIMD128 is supported at compile time
#[wasm_bindgen]
pub fn has_simd128_support() -> bool {
    cfg!(target_feature = "simd128")
}

/// Check if bulk memory operations are supported
#[wasm_bindgen]
pub fn has_bulk_memory_support() -> bool {
    cfg!(target_feature = "bulk-memory")
}

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Get the WebAssembly memory object
/// Note: wasm-bindgen automatically exports memory, so we provide this helper
/// to access it from our module if needed
pub fn get_wasm_memory() -> JsValue {
    wasm_bindgen::memory()
}

pub use memory::*;
pub use operations::*;
pub use types::*;
pub use utils::*;
pub use debug::*;
pub use simd::*;