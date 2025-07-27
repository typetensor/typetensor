# @typetensor/core

Core tensor operations library with compile-time type safety for TypeScript.

## Installation

```bash
npm install @typetensor/core
```

## Features

- Type-safe tensor operations with compile-time shape checking
- Support for various data types (float32, int32, etc.)
- Broadcasting operations
- Shape manipulation and views
- Zero runtime overhead for type checking

## Usage

```typescript
import { tensor } from '@typetensor/core';

// Create tensors with automatic shape inference
const a = tensor([1, 2, 3, 4], { shape: [2, 2] });
const b = tensor([5, 6, 7, 8], { shape: [2, 2] });

// Operations are type-safe - shapes must match
const result = a.add(b);
```

## License

MIT