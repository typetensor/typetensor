// Integration tests for WebAssembly module
use wasm_bindgen_test::*;

// Tests run in Node.js by default, no configuration needed
// Use wasm_bindgen_test_configure!(run_in_browser) for browser tests

#[wasm_bindgen_test]
fn test_wasm_module_loads() {
    // If we get here, the WASM module loaded successfully
    assert!(true);
}

// The rest of the tests are in unit tests within the Rust code
// since wasm-bindgen has limitations on what can be tested from outside