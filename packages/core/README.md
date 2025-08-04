# @typetensor/core

[![npm version](https://img.shields.io/npm/v/@typetensor/core.svg)](https://www.npmjs.com/package/@typetensor/core)

**Core tensor operations with compile-time type safety for TypeScript**

> Part of [TypeTensor](https://github.com/typetensor/typetensor)

## Install

```bash
npm install @typetensor/core
```

**Key Features:**
- Compile-time shape validation using TypeScript's type system
- NumPy-compatible broadcasting and operations  
- Complete Einstein notation (einops) support: rearrange, reduce, repeat
- Full numeric type system with automatic promotion
- Zero runtime overhead for type checking

**Core Operations:**
- **Creation**: Tensor from data, zeros, ones, identity matrices
- **Arithmetic**: Element-wise operations with automatic broadcasting
- **Linear Algebra**: Matrix multiplication with shape inference
- **Shape Operations**: reshape, transpose, slice, permute, squeeze, unsqueeze
- **Reductions**: sum, mean, max, min, prod along specified axes
- **Einops**: rearrange, reduce, repeat with pattern validation
- **Activations**: softmax, log-softmax with numerical stability

**[See examples →](https://github.com/typetensor/typetensor/tree/main/examples)**

## License

MIT
