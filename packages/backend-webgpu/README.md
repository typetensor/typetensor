# @typetensor/backend-webgpu

WebGPU backend for TypedTensor, providing GPU-accelerated tensor operations in modern browsers.

## Installation

```bash
npm install @typetensor/backend-webgpu
```

## Features

- GPU-accelerated tensor operations using WebGPU
- Automatic fallback to CPU when WebGPU is unavailable
- Efficient memory management with GPU buffers
- Support for all core tensor operations

## Usage

```typescript
import { tensor } from '@typetensor/core';
import { WebGPUBackend } from '@typetensor/backend-webgpu';

// Initialize WebGPU backend
const backend = await WebGPUBackend.create();

// Create tensors that will run on GPU
const a = tensor([1, 2, 3, 4], { shape: [2, 2], backend });
const b = tensor([5, 6, 7, 8], { shape: [2, 2], backend });

// Operations automatically run on GPU
const result = a.add(b);
```

## Requirements

- Browser with WebGPU support (Chrome 113+, Edge 113+, etc.)
- TypeScript 5.9.0 or higher

## Status

⚠️ This package is currently a placeholder and under development.

## License

MIT