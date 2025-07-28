/**
 * Type-only tensor storage system with compile-time shape and dtype propagation
 *
 * This module provides a purely type-level storage system that propagates
 * tensor metadata (shape, dtype, strides) through operations without any
 * runtime implementation coupling.
 */

// Re-export layout and base storage types
export type * from './layout';

// Re-export unary operations
export type * from './unary';

// Re-export binary operations
export type * from './binary';

// Re-export view operations
export type * from './view';

// Re-export matrix multiplication operations
export type * from './matmul';

// Re-export softmax operations
export type * from './softmax';

// Re-export reduction operations
export type * from './reduction';

// Re-export einops operations
export type * from './einops';
