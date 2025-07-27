/**
 * Shape module exports
 *
 * @module shape
 *
 * This module provides a comprehensive shape system for tensor operations with:
 * - Type-level shape validation and inference
 * - Runtime shape validation and manipulation
 * - NumPy-compatible broadcasting
 * - Symbolic shape resolution with constraints
 *
 * ## System Limits
 * - Maximum tensor rank: 8 dimensions
 * - Maximum tensor size: Number.MAX_SAFE_INTEGER (2^53 - 1) elements
 * - Type recursion depth: ~1000 iterations for complex operations
 *
 * ## Broadcasting Rules
 * Two shapes are compatible for broadcasting when:
 * 1. They have the same number of dimensions, OR
 * 2. One shape can be prepended with 1s to match the other's rank
 * 3. For each dimension, the sizes must be equal OR one must be 1
 *
 * ## Common Patterns
 * ```typescript
 * // Scalar broadcasting
 * [2, 3] + [] => [2, 3]
 *
 * // Vector broadcasting
 * [2, 1] + [1, 3] => [2, 3]
 *
 * // Matrix multiplication
 * [M, K] @ [K, N] => [M, N]
 *
 * // Batched operations
 * [B, M, K] @ [B, K, N] => [B, M, N]
 * ```
 */

export type {
  // Core shape types
  Shape,
  DynamicShape,
  SymbolicShape,
  ResolvedShape,
  PartialShape,

  // Shape operations
  Product,
  Length,
  IsEmpty,
  Head,
  Tail,
  Last,
  LastDim,
  AllButLast,
  SecondToLast,
  BatchDims,
  Init,
  Concat,
  Reverse,
  Take,
  Drop,
  Permute,
  At,
  Transpose,
  Squeeze,
  Unsqueeze,
  Reshape,

  // Broadcasting
  CanBroadcast,
  BroadcastShapes,
  IncompatibleShapes,

  // Shape comparisons
  Equals,
  IsAssignableTo,
  CanReshape,
  IsMatMulCompatible,
  HasMatchingInnerDims,
  MatMulShape,

  // Error types
  ShapeError,
  ShapeMismatchError,
  ShapeToString,

  // Common shapes
  Shape1D,
  Shape2D,
  Shape3D,
  Shape4D,
  BatchShape,
  ImageShape,
  SequenceShape,
  AttentionShape,

  // Utility types
  TupleOf,
  SymbolicDim,
} from './types';

// Runtime exports
export {
  // Classes
  RuntimeShape,

  // Validation
  isValidShape,
  isStaticShape,
  hasSymbolicDimensions,
  assertValidShape,
  assertShapesCompatible,

  // Shape creation
  createShape,
  reshape,
  SCALAR_SHAPE,
  SHAPE_PATTERNS,

  // Formatting utilities
  formatShape,

  // Constants
  MAX_TENSOR_SIZE,
  MAX_TENSOR_RANK,
} from './runtime';

// Broadcasting exports
export {
  BroadcastManager,
  BinaryBroadcaster,
  ReductionBroadcaster,
  canBroadcast,
  broadcastShapes,
  broadcastBinaryOp,
} from './broadcasting';

// Symbolic shape exports
export { resolveSymbolicShape, LayerShapeResolver } from './symbolic';
