# @typetensor/einops

Type-safe einops operations for TypeTensor.

## Installation

```bash
npm install @typetensor/einops @typetensor/core
# or
bun add @typetensor/einops @typetensor/core
```

## Usage

```typescript
import { tensor } from '@typetensor/core';
import { rearrange, reduce, repeat } from '@typetensor/einops';

// Rearrange dimensions
const input = await tensor([[1, 2, 3], [4, 5, 6]]);
const transposed = await rearrange(input, 'h w -> w h');

// Reduce operations
const sum = await reduce(input, 'h w -> h', 'sum');

// Repeat operations
const repeated = await repeat(input, 'h w -> h w c', { c: 2 });
```

## Features

- **Type-safe patterns**: Compile-time validation of einops patterns
- **Full einops syntax**: Support for all einops operations (rearrange, reduce, repeat)
- **Zero runtime overhead**: Pattern validation happens at compile time
- **Excellent error messages**: Clear, actionable error messages for invalid patterns

## Pattern Syntax

### Rearrange
```typescript
// Simple transpose
await rearrange(tensor, 'h w -> w h');

// Flatten
await rearrange(tensor, 'h w c -> (h w) c');

// Add/remove singleton dimensions
await rearrange(tensor, 'h w -> h w 1');

// Complex patterns with named axes
await rearrange(tensor, 'batch (seq head) dim -> batch seq head dim', { seq: 10 });
```

### Reduce
```typescript
// Sum along dimension
await reduce(tensor, 'h w c -> h c', 'sum');

// Mean with keepdims
await reduce(tensor, 'h w c -> h 1 c', 'mean', { keepdims: true });

// Max pooling pattern
await reduce(tensor, '(h h2) (w w2) c -> h w c', 'max', { axes: { h2: 2, w2: 2 } });
```

### Repeat
```typescript
// Add new dimension
await repeat(tensor, 'h w -> h w c', { c: 3 });

// Repeat along existing dimension
await repeat(tensor, 'h w -> (h h2) w', { h2: 2 });

// Complex upsampling
await repeat(tensor, 'h w c -> (h h2) (w w2) c', { h2: 2, w2: 2 });
```

## License

MIT