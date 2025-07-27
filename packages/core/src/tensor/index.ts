/**
 * Tensor module exports
 *
 * @module tensor
 *
 * This module provides the main Tensor class and creation functions
 * for the tensor library. Tensors combine type-safe metadata from
 * the storage layer with runtime execution via backends.
 */

export { Tensor } from './tensor';
export { tensor, zeros, ones, eye } from './creation';
export type { TensorOptions, NestedArray, InferShape, FlattenArray, DTypeValue } from './types';

// Re-export commonly used utilities
export { nestedArrayToBuffer, bufferToNestedArray, inferShape } from './types';
