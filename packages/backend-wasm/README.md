# @typetensor/backend-wasm

[![npm version](https://img.shields.io/npm/v/@typetensor/backend-wasm.svg)](https://www.npmjs.com/package/@typetensor/backend-wasm)

**High-performance WebAssembly backend for TypeTensor tensor operations**

> Part of [TypeTensor](https://github.com/typetensor/typetensor)

## Install

```bash
npm install @typetensor/backend-wasm
```

## Features

- **High Performance**: Rust-compiled WebAssembly with SIMD optimizations
- **Complete Operations**: Full support for unary, binary, matrix, reduction, and view operations
- **Memory Safety**: Arena-based memory management with automatic cleanup
- **Cross-Platform**: Works in browsers and Node.js with WebAssembly support
- **Pattern Optimization**: Intelligent operation caching and optimization

## Supported Operations

### **Unary Operations**
- Arithmetic: `neg`, `abs`, `square`
- Trigonometric: `sin`, `cos`
- Exponential: `exp`, `log`, `sqrt`

### **Binary Operations**
- Element-wise: `add`, `sub`, `mul`, `div`
- Broadcasting support for compatible shapes

### **Matrix Operations**
- `matmul` - Matrix multiplication with optimized algorithms

### **View Operations**
- Shape manipulation: `reshape`, `flatten`, `transpose`, `permute`
- Slicing: `slice` with stride-aware implementations
- Dimension operations: `squeeze`, `unsqueeze`, `expand`, `tile`

### **Reduction Operations**
- Statistical: `sum`, `mean`, `max`, `min`, `prod`
- Axis-specific reductions with keepdims support

### **Activation Functions**
- `softmax`, `log_softmax` with numerical stability

## Usage

```typescript
import { getWASMDevice } from '@typetensor/backend-wasm';
import { tensor } from '@typetensor/core';

// Get WASM device instance
const device = await getWASMDevice();

// Create tensors on WASM device
const a = tensor([[1, 2], [3, 4]], { device });
const b = tensor([[5, 6], [7, 8]], { device });

// Perform operations
const result = await a.add(b);
console.log(await result.toArray()); // [[6, 8], [10, 12]]
```

## Performance Features

### **SIMD Optimizations**
- Vectorized operations using WebAssembly SIMD instructions
- Automatic fallback for environments without SIMD support

### **Arena Memory Management**
- Fast allocation and bulk deallocation
- Automatic cleanup prevents memory leaks
- Scoped memory management for complex operations

### **Pattern Caching**
- Operation pattern recognition and optimization
- Reduced overhead for repeated operation sequences

## Memory Management

```typescript
const device = await getWASMDevice();

// Automatic cleanup with scopes
const result = device.withScope(() => {
  const temp1 = tensor([1, 2, 3], { device });
  const temp2 = tensor([4, 5, 6], { device });
  return temp1.add(temp2); // Only result survives scope
});

// Manual scope management
const checkpoint = device.beginScope();
// ... perform operations ...
device.endScope(checkpoint); // Clean up all temporaries
```

## Requirements

- **Modern Browser** or **Node.js** with WebAssembly support
- **SIMD Support** (optional) - for optimal performance

## Configuration

```typescript
import { createWASMDevice } from '@typetensor/backend-wasm';

// Create device with custom settings
const device = await createWASMDevice({
  simdEnabled: true,           // Enable SIMD optimizations
  patternCaching: true,        // Enable operation pattern caching
  maxPatterns: 1000,          // Maximum cached patterns
  maxMemoryMB: 512,           // Memory limit in MB
});
```

## Performance Characteristics

- **Best For**: CPU-intensive operations in browsers, cross-platform compatibility
- **Memory**: Efficient arena-based allocation with configurable limits
- **Threading**: Single-threaded with potential for future Web Workers support
- **Precision**: Full support for all TypeTensor numeric types

## Contributing

Interested in improving the WebAssembly backend? See the [contribution guidelines](https://github.com/typetensor/typetensor/blob/main/CONTRIBUTING.md).

## License

MIT