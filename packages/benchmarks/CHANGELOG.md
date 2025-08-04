# @typetensor/benchmarks

## 0.2.1

### Patch Changes

- Updated dependencies
  - @typetensor/core@0.2.1
  - @typetensor/backend-cpu@0.2.1
  - @typetensor/backend-wasm@0.2.1

## 0.2.0

### Minor Changes

- b5c6c1d: # Extended Einops support and enhanced WASM backend

  This release adds comprehensive Einops operations support and significantly enhances the WebAssembly backend with advanced memory management and performance optimizations.

  ## New Features

  ### Core
  - **Extended Einops support**: Added comprehensive `rearrange` and `reduce` operations with full type-safe shape resolution
  - **Enhanced tensor creation**: Improved tensor creation APIs with better type inference
  - **Advanced shape operations**: New expand, tile, and view operations

  ### WASM Backend
  - **Complete WASM backend implementation**: High-performance tensor operations compiled to WebAssembly
  - **Advanced memory management**: Reference counting, buffer lifecycle management, and automatic cleanup
  - **Performance optimizations**: SIMD optimizations, optimized matrix multiplication, and efficient memory allocation
  - **Comprehensive test coverage**: Memory safety, pressure testing, and zero-copy operations

  ### WebGPU Backend
  - **Initial WebGPU backend**: Foundation for GPU-accelerated tensor operations
  - **Native WGPU integration**: FFI bindings and TypeGPU integration

  ### Benchmarking
  - **Comprehensive benchmark suite**: Performance comparison across backends
  - **WASM-specific benchmarks**: Detailed performance metrics for WebAssembly operations

  ## Improvements
  - **Enhanced type safety**: Better TypeScript inference and compile-time shape validation
  - **Memory efficiency**: Optimized memory usage patterns and automatic cleanup
  - **Cross-platform compatibility**: Testing across Ubuntu, macOS, and Windows
  - **Developer experience**: Improved error messages and debugging capabilities

  ## Breaking Changes
  - Some internal APIs have been restructured for better performance and type safety
  - Memory management patterns may require updates for direct WASM usage

### Patch Changes

- Updated dependencies [b5c6c1d]
  - @typetensor/backend-wasm@0.2.0
  - @typetensor/backend-cpu@0.1.2
  - @typetensor/core@0.2.0
