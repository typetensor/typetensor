# @typetensor/backend-vulkan

Vulkan backend for TypedTensor, providing GPU-accelerated tensor operations across platforms.

## Installation

```bash
npm install @typetensor/backend-vulkan
```

## Features

- Hardware-accelerated tensor operations using Vulkan
- Cross-platform GPU support
- Efficient memory management
- Support for all core tensor operations

## Usage

```typescript
import { tensor } from '@typetensor/core';
import { VulkanBackend } from '@typetensor/backend-vulkan';

// Initialize Vulkan backend
const backend = await VulkanBackend.create();

// Create tensors that will run on Vulkan GPU
const a = tensor([1, 2, 3, 4], { shape: [2, 2], backend });
const b = tensor([5, 6, 7, 8], { shape: [2, 2], backend });

// Operations automatically run on Vulkan GPU
const result = a.add(b);
```

## Requirements

- GPU with Vulkan support
- Vulkan drivers installed
- TypeScript 5.9.0 or higher

## Status

⚠️ This package is currently a placeholder and under development.

## License

MIT