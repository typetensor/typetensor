//! Test suite for the Web and headless browsers.

#![cfg(target_arch = "wasm32")]

extern crate wasm_bindgen_test;
use wasm_bindgen_test::*;

// This runs the tests in a headless browser
wasm_bindgen_test_configure!(run_in_browser);