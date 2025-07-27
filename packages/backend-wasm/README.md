# @typetensor/backend-wasm

WebAssembly backend for TypedTensor, providing high-performance tensor operations in browsers and Node.js.

## Installation

```bash
npm install @typetensor/backend-wasm
```

## Features

- High-performance tensor operations compiled to WebAssembly
- Cross-platform support (browsers and Node.js)
- SIMD optimizations where available
- Smaller bundle size compared to GPU backends

## Usage

```typescript
import { tensor } from '@typetensor/core';
import { WASMBackend } from '@typetensor/backend-wasm';

// Initialize WASM backend
const backend = await WASMBackend.create();

// Create tensors that will run on WASM
const a = tensor([1, 2, 3, 4], { shape: [2, 2], backend });
const b = tensor([5, 6, 7, 8], { shape: [2, 2], backend });

// Operations run in WebAssembly
const result = a.add(b);
```

## Requirements

- Modern browser or Node.js with WebAssembly support
- TypeScript 5.9.0 or higher

## Status

⚠️ This package is currently a placeholder and under development.

## License

MIT