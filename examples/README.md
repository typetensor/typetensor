# TypeTensor Examples

This directory contains a comprehensive example demonstrating the key features of the TypeTensor library, with a focus on TypeScript's compile-time shape safety.

## Important: Build First

Since this example is within the monorepo, you need to build the packages first:

```bash
# From the project root
bun run build
```

## Running the Example

Due to Bun's workspace behavior (which prefers TypeScript source files), we need to run the example using the built JavaScript:

```bash
# From the examples directory
bun run type-safe-tensors.js

# This will transpile the TypeScript example to JavaScript using the built packages
bun build type-safe-tensors.ts --outfile type-safe-tensors.js --target node
bun run type-safe-tensors.js
```

## For Real-World Usage

In a real project outside this monorepo, you would simply:

1. Install the packages:
   ```bash
   npm install @typetensor/core @typetensor/backend-cpu
   ```

2. Import and use normally:
   ```typescript
   import { tensor, float32 } from '@typetensor/core';
   import { cpu } from '@typetensor/backend-cpu';
   ```

## Examples

### 01-simple-operations.ts
Basic tensor creation, arithmetic, and shape safety demonstrations.

### 02-shape-errors.ts
Examples of compile-time shape error detection.

### 03-view-operations.ts
Memory-efficient tensor views and reshaping.

### 04-chaining.ts
Chaining tensor operations with type safety.

### 05-einops-rearrange.ts ⭐ **NEW**
Simple tensor rearrangement using Einstein notation:
- Transpose: `"h w -> w h"`
- Format conversion: `"c h w -> h w c"`
- Add dimensions: `"h w -> 1 h w"`
- Split dimensions: `"(h w) c -> h w c"`
- Multi-head attention preparation

## What the Examples Demonstrate

- **Type-safe tensor creation**: See how TypeScript infers and validates tensor shapes at compile time
- **Shape safety**: Examples of operations that TypeScript prevents due to incompatible shapes
- **Broadcasting**: Type-safe broadcasting operations with automatic shape inference
- **Common operations**: Basic tensor operations like reshape, view, and arithmetic
- **Einops patterns**: Elegant tensor manipulations using Einstein notation
- **Error prevention**: Commented examples showing what TypeScript catches at compile time

## Key Features Highlighted

The example showcases TypeTensor's main advantage: catching shape mismatches and invalid operations at compile time rather than runtime. This helps prevent common tensor operation errors before your code even runs.