# @typetensor/backend-metal

Apple Metal backend for TypedTensor, providing GPU-accelerated tensor operations on Apple devices.

## Installation

```bash
npm install @typetensor/backend-metal
```

## Features

- Hardware-accelerated tensor operations using Apple Metal
- Optimized kernels for Apple Silicon
- Efficient memory management
- Support for all core tensor operations

## Usage

```typescript
import { tensor } from '@typetensor/core';
import { MetalBackend } from '@typetensor/backend-metal';

// Initialize Metal backend
const backend = await MetalBackend.create();

// Create tensors that will run on Metal GPU
const a = tensor([1, 2, 3, 4], { shape: [2, 2], backend });
const b = tensor([5, 6, 7, 8], { shape: [2, 2], backend });

// Operations automatically run on Metal GPU
const result = a.add(b);
```

## Requirements

- Apple Silicon Mac or iOS device
- macOS 11+ or iOS 14+
- TypeScript 5.9.0 or higher

## Status

⚠️ This package is currently a placeholder and under development.

## License

MIT