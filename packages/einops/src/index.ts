/**
 * Einops operations for TypeTensor
 *
 * This module provides einops-style tensor operations with compile-time
 * pattern validation and type-safe transformations.
 */

// Export main operations
export { rearrange } from './rearrange';
export type { RearrangeOptions } from './rearrange';

export { reduce } from './reduce';
export type { ReduceOptions, ReductionOp } from './reduce';

export { repeat } from './repeat';
export type { RepeatOptions } from './repeat';

// Export storage transformation types
export type { RearrangeOp, ReduceEinopsOp, RepeatOp } from './storage-types';