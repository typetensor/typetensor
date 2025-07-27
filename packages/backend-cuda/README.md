# @typetensor/backend-cuda

NVIDIA CUDA backend for TypedTensor, providing GPU-accelerated tensor operations in modern browsers.

## Installation

```bash
npm install @typetensor/backend-cuda
```

## Features

- Hardware-accelerated tensor operations using NVIDIA CUDA
- Optimized kernels for common operations
- Efficient memory management
- Support for all core tensor operations

## Usage

```typescript
import { tensor } from '@typetensor/core';
import { CUDABackend } from '@typetensor/backend-cuda';

// Initialize NVIDIA CUDA backend
const backend = await CUDABackend.create();

// Create tensors that will run on CUDA
const a = tensor([1, 2, 3, 4], { shape: [2, 2], backend });
const b = tensor([5, 6, 7, 8], { shape: [2, 2], backend });

// Operations automatically run on CUDA
const result = a.add(b);
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA runtime and drivers
- TypeScript 5.9.0 or higher

## Status

⚠️ This package is currently a placeholder and under development.

## License

MIT