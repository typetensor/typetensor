# @typetensor/backend-cpu

CPU backend implementation for TypeTensor.

## Installation

```bash
npm install @typetensor/backend-cpu
# or
yarn add @typetensor/backend-cpu
# or
pnpm add @typetensor/backend-cpu
```

## Usage

The CPU backend is automatically registered when you import it:

```typescript
import { tensor, zeros, ones, float32 } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';

// Create tensors on CPU
const a = await tensor([[1, 2], [3, 4]], { device: cpu, dtype: float32 });
const b = await zeros([2, 2], { device: cpu, dtype: float32 });

// Perform operations
const c = await a.add(b);
console.log(await c.toArray()); // [[1, 2], [3, 4]]
```

## Features

- **Full dtype support**: All TypeTensor data types (int8, uint8, int16, uint16, int32, uint32, float32, float64, int64, uint64, bool)
- **Broadcasting**: Automatic broadcasting for binary operations following NumPy rules
- **View operations**: Efficient reshape, flatten, and view operations without copying data
- **Pure JavaScript**: No native dependencies, works in any JavaScript environment

## Operations

### Unary Operations
- `neg()` - Negation
- `abs()` - Absolute value
- `square()` - Element-wise square
- `sqrt()` - Square root (outputs float)
- `exp()` - Exponential (outputs float)
- `log()` - Natural logarithm (outputs float)
- `sin()` - Sine (outputs float)
- `cos()` - Cosine (outputs float)

### Binary Operations
- `add()` - Addition with broadcasting
- `sub()` - Subtraction with broadcasting  
- `mul()` - Multiplication with broadcasting
- `div()` - Division with broadcasting

### View Operations
- `reshape()` - Change tensor shape without copying
- `flatten()` - Convert to 1D tensor
- `view()` - Create view with dimension inference (-1)

## Performance

The CPU backend uses TypedArrays for efficient numerical computation. Operations are optimized for:
- Contiguous memory access when possible
- Fast path for operations without broadcasting
- Minimal memory allocation

## License

MIT
