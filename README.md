<div align="center">
  <img src="media/favicon/web-app-manifest-192x192.png" alt="TypeTensor Logo" width="128" height="128">
  
  # TypeTensor

  **TypeScript's compile-time tensor library - catch shape errors before runtime**
</div>

<br>

[![CI](https://github.com/typetensor/typetensor/actions/workflows/ci.yml/badge.svg)](https://github.com/typetensor/typetensor/actions/workflows/ci.yml)
[![npm version](https://img.shields.io/npm/v/@typetensor/core.svg)](https://www.npmjs.com/package/@typetensor/core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9+-blue.svg)](https://www.typescriptlang.org/)

Traditional tensor libraries catch shape errors at runtime. TypeTensor catches them at **compile time** using TypeScript's type system, preventing bugs before your code runs.

**Quick Install:**

```bash
npm install @typetensor/core @typetensor/backend-cpu
```

## Key Features

### 🔍 **Compile-Time Shape Safety**

- Tensor shapes are validated at compile time using TypeScript's type system
- Incompatible operations are caught before your code runs
- IntelliSense shows resulting tensor shapes as you type

### 🧮 **Complete Type System**

- Full numeric type support: `bool`, `int8/16/32/64`, `uint8/16/32/64`, `float32/64`
- NumPy-compatible type promotion rules
- Safe type conversion with overflow/precision loss detection

### 📐 **Broadcasting & Shape Operations**

- NumPy-compatible broadcasting with compile-time validation
- Rich shape manipulation: reshape, transpose, squeeze, unsqueeze
- Matrix multiplication with automatic shape inference

### 🔀 **Complete Einops Support**

- **Rearrange**: `"h w c -> c h w"`, `"batch seq -> batch seq 1"`
- **Reduce**: `"h w c -> h w"` (sum, mean, max, min operations)
- **Repeat**: `"h w -> h w c"`, `"batch -> batch repeat"`
- Advanced patterns with explicit dimension sizes
- Compile-time validation of all einops patterns

### ⚡ **Pluggable Backends**

- Modular backend system for different compute targets
- CPU, GPU (CUDA, WebGPU, Metal), and WebAssembly backends
- SIMD-optimized WebAssembly with arena-based memory management
- Zero-copy operations where possible

## What Makes TypeTensor Different

- **Compile-Time Validation**: Shape mismatches caught by TypeScript before runtime
- **Zero Runtime Overhead**: Type checking happens entirely at compile time
- **Familiar API**: NumPy-like interface that JavaScript developers expect
- **Complete Einops**: Full rearrange, reduce, and repeat operations
- **High Performance**: SIMD-optimized WebAssembly and efficient CPU backends

## Architecture

TypeTensor is designed with a clear separation of concerns:

```
┌─────────────────────────────────────┐
│            @typetensor/core         │  ← Type system, shapes, tensor API
├─────────────────────────────────────┤
│  Backend Interface (Device/Ops)     │  ← Abstract execution layer
├─────────────────────────────────────┤
│     Concrete Backend Packages       │  ← Actual computation
│  • backend-cpu                      │
│  • backend-cuda                     │
│  • backend-webgpu                   │
│  • backend-metal                    │
│  • backend-wasm                     │
└─────────────────────────────────────┘
```

### Core Design Principles

1. **Type Safety First**: Every operation is validated at compile time
2. **Zero Runtime Overhead**: Type checking happens entirely at compile time
3. **Familiar API**: NumPy-like interface that JavaScript/TypeScript developers expect
4. **Modular Architecture**: Use only the backends you need
5. **Mathematical Correctness**: Proper handling of broadcasting, type promotion, and numerical precision

## Packages

| Package | NPM | Status | Description |
| ------- | --- | ------ | ----------- |
| [`@typetensor/core`](./packages/core) | [![npm](https://img.shields.io/npm/v/@typetensor/core.svg)](https://www.npmjs.com/package/@typetensor/core) | ✅ **Alpha** | Core tensor operations and type system |
| [`@typetensor/backend-cpu`](./packages/backend-cpu) | [![npm](https://img.shields.io/npm/v/@typetensor/backend-cpu.svg)](https://www.npmjs.com/package/@typetensor/backend-cpu) | ✅ **Alpha** | CPU backend implementation |
| [`@typetensor/backend-cuda`](./packages/backend-cuda) | [![npm](https://img.shields.io/npm/v/@typetensor/backend-cuda.svg)](https://www.npmjs.com/package/@typetensor/backend-cuda) | 🚧 **TODO** | NVIDIA CUDA GPU backend |
| [`@typetensor/backend-webgpu`](./packages/backend-webgpu) | [![npm](https://img.shields.io/npm/v/@typetensor/backend-webgpu.svg)](https://www.npmjs.com/package/@typetensor/backend-webgpu) | 🚧 **TODO** | WebGPU backend for browsers |
| [`@typetensor/backend-metal`](./packages/backend-metal) | [![npm](https://img.shields.io/npm/v/@typetensor/backend-metal.svg)](https://www.npmjs.com/package/@typetensor/backend-metal) | 🚧 **TODO** | Apple Metal GPU backend |
| [`@typetensor/backend-vulkan`](./packages/backend-vulkan) | [![npm](https://img.shields.io/npm/v/@typetensor/backend-vulkan.svg)](https://www.npmjs.com/package/@typetensor/backend-vulkan) | 🚧 **TODO** | Vulkan GPU backend |
| [`@typetensor/backend-wasm`](./packages/backend-wasm) | [![npm](https://img.shields.io/npm/v/@typetensor/backend-wasm.svg)](https://www.npmjs.com/package/@typetensor/backend-wasm) | ✅ **Alpha** | High-performance WebAssembly backend with SIMD |

## Core Capabilities

### **Tensor Operations**

- **Creation**: Tensors from data, zeros, ones, identity matrices
- **Element-wise**: Arithmetic, trigonometric, and exponential functions
- **Linear Algebra**: Matrix multiplication with automatic broadcasting
- **Reductions**: Sum, mean, max, min along specified axes
- **Shape Manipulation**: Reshape, transpose, slice, and permute operations

### **Advanced Features**

- **Broadcasting**: NumPy-compatible shape alignment
- **Type System**: Numeric type support with automatic promotion
- **Memory Views**: Tensor views without data copying
- **Activation Functions**: Softmax, log-softmax

## Einstein Notation (Einops)

TypeTensor provides complete einops support with three core operations:

### **Rearrange Operations**
- **Format Conversions**: `"h w c -> c h w"` (HWC ↔ CHW)
- **Dimension Manipulation**: `"h w -> 1 h w"` (add batch dimension)
- **Flattening**: `"h w c -> (h w) c"` (combine spatial dimensions)
- **Splitting**: `"(h w) c -> h w c"` with explicit dimension sizes

### **Reduce Operations**
- **Spatial Reduction**: `"h w c -> c"` (sum across spatial dimensions)
- **Channel Reduction**: `"h w c -> h w"` (reduce channels)
- **Selective Reduction**: `"b h w c -> b c"` (reduce height and width)
- Supports: `sum`, `mean`, `max`, `min`, `prod` operations

### **Repeat Operations**
- **Broadcasting**: `"h w -> h w c"` (repeat across new channel dimension)
- **Batch Expansion**: `"h w c -> b h w c"` (add batch dimension)
- **Element Repetition**: `"h w -> (h 2) (w 2)"` (repeat elements)

All operations include compile-time pattern validation and comprehensive error handling.

## Getting Started

- **[Examples](./examples)** - Hands-on demos of key features
- **[Documentation](#)** - Full API reference and guides _(coming soon)_
- **[Contributing](CONTRIBUTING.md)** - How to contribute to the project

## Development Status

TypeTensor is in early development. The core type system and CPU backend are functional, but the project is not yet ready for production use.

**Current Status:**

- ✅ Complete type system with compile-time validation
- ✅ Full tensor operations (arithmetic, views, reshaping, reductions)
- ✅ Complete einops support (rearrange, reduce, repeat)
- ✅ CPU backend with comprehensive operations
- ✅ High-performance WASM backend with SIMD optimizations
- 🚧 Advanced operations (convolution, pooling, etc.)
- 🚧 Additional GPU backends (CUDA, WebGPU, Metal)
- 🚧 Framework integrations and ecosystem

## Contributing

We welcome contributions! Please see our [contribution guidelines](CONTRIBUTING.md) for details.

## Prior Art & Inspiration

TypeTensor builds on ideas from:

- **[PyTorch](https://github.com/pytorch/pytorch)** - Dynamic neural networks and tensor operations
- **[Candle](https://github.com/huggingface/candle/)** - Rust tensor library (contributor experience with tensor implementations)
- **[Burn](https://github.com/tracel-ai/burn)** - Compile-time shape safety in Rust
- **[ArkType](https://github.com/arktypeio/arktype)** - Type-safe validation (inspiration for einops parsing patterns)

## License

MIT © [Thomas Santerre](https://github.com/tomsanbear)
