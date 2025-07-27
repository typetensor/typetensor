/**
 * Runtime shape management and validation
 *
 * This module provides runtime classes and utilities for working with tensor shapes,
 * bridging the gap between compile-time type safety and runtime execution.
 */

import type { Shape, DynamicShape, SymbolicShape } from './types.js';

// =============================================================================
// Configuration and Constants
// =============================================================================

/**
 * Maximum number of elements allowed in a tensor
 * Default: Number.MAX_SAFE_INTEGER (2^53 - 1 ≈ 9 quadrillion)
 */
export const MAX_TENSOR_SIZE = Number.MAX_SAFE_INTEGER;

/**
 * Maximum rank (number of dimensions) for tensors
 */
export const MAX_TENSOR_RANK = 8;

// =============================================================================
// Runtime Shape Class
// =============================================================================

/**
 * Runtime representation of a tensor shape with computed properties
 * Provides efficient access to shape metadata and validation
 */
export class RuntimeShape<S extends Shape = Shape> {
  readonly dims: S;
  private _rank: number;
  private _size: number;
  private _strides: number[];

  constructor(dims: S) {
    this.dims = dims;
    this._rank = dims.length;
    this._size = this.computeSize();
    this._strides = this.computeStrides();

    // Validate shape at construction
    this.validate();
  }

  /**
   * Number of dimensions (rank) of the tensor
   */
  get rank(): number {
    return this._rank;
  }

  /**
   * Total number of elements in the tensor
   */
  get size(): number {
    return this._size;
  }

  /**
   * Strides for each dimension (row-major order)
   * Used for efficient memory access calculations
   */
  get strides(): readonly number[] {
    return this._strides;
  }

  /**
   * Check if this shape represents a scalar (0 dimensions)
   */
  get isScalar(): boolean {
    return this._rank === 0;
  }

  /**
   * Check if this shape represents a vector (1 dimension)
   */
  get isVector(): boolean {
    return this._rank === 1;
  }

  /**
   * Check if this shape represents a matrix (2 dimensions)
   */
  get isMatrix(): boolean {
    return this._rank === 2;
  }

  /**
   * Get a dimension size by index, with support for negative indexing
   */
  dim(index: number): number {
    const normalizedIndex = index < 0 ? this._rank + index : index;

    if (normalizedIndex < 0 || normalizedIndex >= this._rank) {
      throw new Error(
        `Dimension index ${index.toString()} out of bounds for rank ${this._rank.toString()} tensor`,
      );
    }

    const dimension = this.dims[normalizedIndex];
    if (dimension === undefined) {
      throw new Error(`Dimension at index ${normalizedIndex.toString()} is undefined`);
    }
    return dimension;
  }

  /**
   * Check if this shape is compatible with another shape for broadcasting
   */
  canBroadcastWith(other: RuntimeShape): boolean {
    return RuntimeShape.canBroadcast(this.dims, other.dims);
  }

  /**
   * Compute the broadcasted shape with another shape
   */
  broadcastWith(other: RuntimeShape): RuntimeShape {
    const result = RuntimeShape.broadcastShapes(this.dims, other.dims);
    return new RuntimeShape(result);
  }

  /**
   * Check if this shape is exactly equal to another
   */
  equals(other: RuntimeShape): boolean {
    if (this._rank !== other._rank) {
      return false;
    }

    for (let i = 0; i < this._rank; i++) {
      if (this.dims[i] !== other.dims[i]) {
        return false;
      }
    }

    return true;
  }

  /**
   * Create a new shape with dimensions squeezed (size-1 dimensions removed)
   */
  squeeze(axes?: number[]): RuntimeShape {
    if (axes === undefined) {
      // Remove all size-1 dimensions
      const newDims = this.dims.filter((dim) => dim !== 1);
      return new RuntimeShape(newDims as Shape);
    }

    // Remove specific axes
    const normalizedAxes = axes.map((axis) => (axis < 0 ? this._rank + axis : axis));

    // Validate axes
    for (const axis of normalizedAxes) {
      if (axis < 0 || axis >= this._rank) {
        throw new Error(
          `Axis ${axis.toString()} out of bounds for rank ${this._rank.toString()} tensor`,
        );
      }
      if (this.dims[axis] !== 1) {
        const dim = this.dims[axis];
        if (dim === undefined) {
          throw new Error(`Dimension at index ${axis.toString()} is undefined`);
        }
        throw new Error(
          `Cannot squeeze dimension ${axis.toString()} with size ${dim.toString()} (must be 1)`,
        );
      }
    }

    const newDims = this.dims.filter((_, index) => !normalizedAxes.includes(index));
    return new RuntimeShape(newDims as Shape);
  }

  /**
   * Create a new shape with a size-1 dimension added at the specified axis
   */
  unsqueeze(axis: number): RuntimeShape {
    const normalizedAxis = axis < 0 ? this._rank + axis + 1 : axis;

    if (normalizedAxis < 0 || normalizedAxis > this._rank) {
      throw new Error(
        `Unsqueeze axis ${axis.toString()} out of bounds for rank ${this._rank.toString()} tensor`,
      );
    }

    const newDims = [...this.dims.slice(0, normalizedAxis), 1, ...this.dims.slice(normalizedAxis)];

    return new RuntimeShape(newDims as Shape);
  }

  /**
   * Create a new shape by transposing dimensions
   */
  transpose(axes?: number[]): RuntimeShape {
    if (axes === undefined) {
      // Default: reverse all dimensions
      return new RuntimeShape([...this.dims].reverse() as Shape);
    }

    if (axes.length !== this._rank) {
      throw new Error(
        `Transpose axes length ${axes.length.toString()} must match tensor rank ${this._rank.toString()}`,
      );
    }

    // Validate and normalize axes
    const normalizedAxes = axes.map((axis) => (axis < 0 ? this._rank + axis : axis));
    const axisSet = new Set(normalizedAxes);

    if (axisSet.size !== this._rank) {
      throw new Error('Transpose axes must be unique');
    }

    for (const axis of normalizedAxes) {
      if (axis < 0 || axis >= this._rank) {
        throw new Error(
          `Transpose axis ${axis.toString()} out of bounds for rank ${this._rank.toString()} tensor`,
        );
      }
    }

    const newDims = normalizedAxes.map((axis) => this.dims[axis]);
    return new RuntimeShape(newDims as Shape);
  }

  /**
   * Check if this shape can be reshaped to another shape
   */
  canReshapeTo(newShape: number[]): boolean {
    const newSize = newShape.reduce((prod, dim) => {
      if (dim < 0) {
        throw new Error('Reshape dimensions must be non-negative');
      }
      return prod * dim;
    }, 1);

    return newSize === this._size;
  }

  /**
   * Check if this shape is compatible with another shape for matrix multiplication
   */
  canMatMulWith(other: RuntimeShape): boolean {
    // Matrix multiplication requires at least 1D tensors
    if (this._rank === 0 || other._rank === 0) {
      return false;
    }

    // For 1D tensors, treat as row vector x column vector
    if (this._rank === 1 && other._rank === 1) {
      return this.dims[0] === other.dims[0];
    }

    // For 1D x 2D: [K] x [K, N] -> [N]
    if (this._rank === 1 && other._rank === 2) {
      return this.dims[0] === other.dims[0];
    }

    // For 2D x 1D: [M, K] x [K] -> [M]
    if (this._rank === 2 && other._rank === 1) {
      return this.dims[1] === other.dims[0];
    }

    // For general case, check last dim of this matches second-to-last dim of other
    const thisInnerDim = this.dims[this._rank - 1];
    const otherInnerDim = other.dims[other._rank - 2];

    if (thisInnerDim !== otherInnerDim) {
      return false;
    }

    // For batched operations, batch dimensions must be broadcastable
    if (this._rank > 2 || other._rank > 2) {
      const thisBatchDims = this.dims.slice(0, -2);
      const otherBatchDims = other.dims.slice(0, -2);
      return RuntimeShape.canBroadcast(thisBatchDims, otherBatchDims);
    }

    return true;
  }

  /**
   * Compute the result shape of matrix multiplication with another shape
   */
  matMul(other: RuntimeShape): RuntimeShape {
    if (!this.canMatMulWith(other)) {
      throw new Error(
        `Cannot perform matrix multiplication between shapes [${this.dims.join(', ')}] and [${other.dims.join(', ')}]`,
      );
    }

    // Handle different cases
    if (this._rank === 1 && other._rank === 1) {
      // Vector dot product: [K] x [K] -> []
      return new RuntimeShape([] as Shape);
    }

    if (this._rank === 1 && other._rank === 2) {
      // Vector x matrix: [K] x [K, N] -> [N]
      return new RuntimeShape([other.dims[1]] as Shape);
    }

    if (this._rank === 2 && other._rank === 1) {
      // Matrix x vector: [M, K] x [K] -> [M]
      return new RuntimeShape([this.dims[0]] as Shape);
    }

    // General case: [..., M, K] x [..., K, N] -> [..., M, N]
    const thisBatchDims = this.dims.slice(0, -2);
    const otherBatchDims = other.dims.slice(0, -2);
    const broadcastedBatch = RuntimeShape.broadcastShapes(thisBatchDims, otherBatchDims);

    const resultDims = [
      ...broadcastedBatch,
      this.dims[this._rank - 2], // M
      other.dims[other._rank - 1], // N
    ];

    return new RuntimeShape(resultDims as Shape);
  }

  /**
   * Validate the tensor shape
   */
  private validate(): void {
    if (this._rank > MAX_TENSOR_RANK) {
      throw new Error(
        `Tensor rank ${this._rank.toString()} exceeds maximum supported rank of ${MAX_TENSOR_RANK.toString()}`,
      );
    }

    for (let i = 0; i < this._rank; i++) {
      const dim = this.dims[i];
      if (dim === undefined || !Number.isInteger(dim) || dim < 0) {
        const dimStr = dim?.toString();
        if (dimStr === undefined) {
          throw new Error(`Dimension at index ${i.toString()} is undefined`);
        }
        throw new Error(
          `Invalid dimension ${dimStr} at index ${i.toString()}: dimensions must be non-negative integers`,
        );
      }
    }

    // Validate total size doesn't exceed safe limits
    if (this._size > MAX_TENSOR_SIZE) {
      throw new Error(
        `Tensor size ${this._size.toString()} exceeds maximum safe size of ${MAX_TENSOR_SIZE.toString()}. ` +
          `Shape: [${this.dims.join(', ')}]`,
      );
    }
  }

  /**
   * Compute the total number of elements
   */
  private computeSize(): number {
    return this.dims.reduce((prod, dim) => prod * dim, 1);
  }

  /**
   * Compute strides for row-major (C-style) memory layout
   */
  private computeStrides(): number[] {
    const strides = new Array<number>(this._rank);
    let stride = 1;

    // Compute strides from right to left (row-major)
    for (let i = this._rank - 1; i >= 0; i--) {
      strides[i] = stride;
      const dim = this.dims[i];
      if (dim === undefined) {
        throw new Error(`Dimension at index ${i.toString()} is undefined`);
      }
      stride *= dim;
    }

    return strides;
  }

  /**
   * Convert linear index to multi-dimensional indices
   */
  unravel(index: number): number[] {
    if (index < 0 || index >= this._size) {
      throw new Error(
        `Index ${index.toString()} out of bounds for tensor with ${this._size.toString()} elements`,
      );
    }

    const indices = new Array<number>(this._rank);
    let remaining = index;

    for (let i = 0; i < this._rank; i++) {
      const stride = this._strides[i];
      if (stride === undefined) {
        throw new Error(`Stride at index ${i.toString()} is undefined`);
      }
      indices[i] = Math.floor(remaining / stride);
      remaining %= stride;
    }

    return indices;
  }

  /**
   * Convert multi-dimensional indices to linear index
   */
  ravel(indices: number[]): number {
    if (indices.length !== this._rank) {
      throw new Error(
        `Expected ${this._rank.toString()} indices, got ${indices.length.toString()}`,
      );
    }

    let index = 0;
    for (let i = 0; i < this._rank; i++) {
      const idx = indices[i];
      const dim = this.dims[i];

      if (idx === undefined) {
        throw new Error(`Index at position ${i.toString()} is undefined`);
      }
      if (dim === undefined) {
        throw new Error(`Dimension at index ${i.toString()} is undefined`);
      }

      // Handle negative indexing
      const normalizedIdx = idx < 0 ? dim + idx : idx;

      if (normalizedIdx < 0 || normalizedIdx >= dim) {
        throw new Error(
          `Index ${idx.toString()} out of bounds for dimension ${i.toString()} with size ${dim.toString()}`,
        );
      }

      const stride = this._strides[i];
      if (stride === undefined) {
        throw new Error(`Stride at index ${i.toString()} is undefined`);
      }
      index += normalizedIdx * stride;
    }

    return index;
  }

  /**
   * Get default strides for this shape (row-major)
   */
  defaultStrides(): number[] {
    return [...this._strides];
  }

  /**
   * String representation for debugging
   */
  toString(): string {
    return `Shape[${this.dims.join(', ')}]`;
  }

  // =============================================================================
  // Static Utility Methods
  // =============================================================================

  /**
   * Create a RuntimeShape from array-like input
   */
  static from<S extends Shape>(dims: S): RuntimeShape<S> {
    return new RuntimeShape(dims);
  }

  /**
   * Check if two shapes can be broadcast together
   */
  static canBroadcast(shape1: Shape, shape2: Shape): boolean {
    const maxRank = Math.max(shape1.length, shape2.length);

    // Pad shapes with 1s on the left
    const padded1 = this.padLeft(shape1, maxRank);
    const padded2 = this.padLeft(shape2, maxRank);

    // Check each dimension
    for (let i = 0; i < maxRank; i++) {
      const d1 = padded1[i];
      const d2 = padded2[i];

      // Standard broadcasting rules: dimensions must be equal, or one must be 1
      // Special case: 0 can only broadcast with 0 or 1, not with other positive numbers
      if (d1 !== d2 && d1 !== 1 && d2 !== 1) {
        return false;
      }
    }

    return true;
  }

  /**
   * Check if two shapes can be matrix multiplied
   * 
   * Matrix multiplication is valid when:
   * - Both tensors have at least 1 dimension
   * - The last dimension of the first tensor equals the second-to-last dimension of the second
   * - All batch dimensions (if any) match exactly
   * 
   * @param shape1 - First tensor shape
   * @param shape2 - Second tensor shape
   * @returns true if shapes can be matrix multiplied
   */
  static canMatmul(shape1: Shape, shape2: Shape): boolean {
    // Both must be at least 1D
    if (shape1.length === 0 || shape2.length === 0) {
      return false;
    }

    // For 1D x 1D, dimensions must match (dot product)
    if (shape1.length === 1 && shape2.length === 1) {
      return shape1[0] === shape2[0];
    }

    // Get dimensions to compare
    const dim1 = shape1[shape1.length - 1]; // Last of first
    let dim2: number;
    
    if (shape2.length === 1) {
      // 1D second operand uses its only dimension
      dim2 = shape2[0]!;
    } else {
      // 2D+ second operand uses second-to-last dimension
      dim2 = shape2[shape2.length - 2]!;
    }

    // Inner dimensions must match
    if (dim1 !== dim2) {
      return false;
    }

    // Check batch dimensions can broadcast
    if (shape1.length > 2 || shape2.length > 2) {
      const batchDims1 = shape1.length > 2 ? shape1.slice(0, -2) : [];
      const batchDims2 = shape2.length > 2 ? shape2.slice(0, -2) : [];
      
      // Apply broadcasting rules: align from the right, pad with 1s on the left
      const maxBatchLength = Math.max(batchDims1.length, batchDims2.length);
      const paddedBatch1 = new Array(maxBatchLength - batchDims1.length).fill(1).concat(batchDims1);
      const paddedBatch2 = new Array(maxBatchLength - batchDims2.length).fill(1).concat(batchDims2);
      
      // Check each batch dimension for broadcasting compatibility
      for (let i = 0; i < maxBatchLength; i++) {
        const dim1 = paddedBatch1[i];
        const dim2 = paddedBatch2[i];
        // Broadcasting rule: dimensions must be equal OR one must be 1
        if (dim1 !== dim2 && dim1 !== 1 && dim2 !== 1) {
          return false;
        }
      }
    }

    return true;
  }

  /**
   * Compute the output shape of matrix multiplication
   * 
   * @param shape1 - First tensor shape
   * @param shape2 - Second tensor shape
   * @returns Output shape or null if shapes are incompatible
   */
  static matmulShape(shape1: Shape, shape2: Shape): Shape | null {
    if (!this.canMatmul(shape1, shape2)) {
      return null;
    }

    // 1D x 1D -> scalar
    if (shape1.length === 1 && shape2.length === 1) {
      return [];
    }

    // Helper function to broadcast batch dimensions
    const broadcastBatchDims = (batchA: number[], batchB: number[]): number[] => {
      const maxLength = Math.max(batchA.length, batchB.length);
      const paddedA = new Array(maxLength - batchA.length).fill(1).concat(batchA);
      const paddedB = new Array(maxLength - batchB.length).fill(1).concat(batchB);
      
      const result: number[] = [];
      for (let i = 0; i < maxLength; i++) {
        const dimA = paddedA[i]!;
        const dimB = paddedB[i]!;
        // Broadcasting rule: take the larger dimension (assuming validation passed)
        result.push(Math.max(dimA, dimB));
      }
      return result;
    };

    // Extract batch dimensions and matrix dimensions
    const batchA = shape1.length > 2 ? shape1.slice(0, -2) : [];
    const batchB = shape2.length > 2 ? shape2.slice(0, -2) : [];

    // 1D x 2D -> [n]
    if (shape1.length === 1 && shape2.length === 2) {
      return [shape2[1]!];
    }

    // 1D x ND -> [...broadcast_batch, n]
    if (shape1.length === 1 && shape2.length > 2) {
      const broadcastedBatch = broadcastBatchDims([], batchB);
      return [...broadcastedBatch, shape2[shape2.length - 1]!];
    }

    // 2D x 1D -> [m]
    if (shape1.length === 2 && shape2.length === 1) {
      return [shape1[0]!];
    }

    // ND x 1D -> [...broadcast_batch, m]
    if (shape1.length > 2 && shape2.length === 1) {
      const broadcastedBatch = broadcastBatchDims(batchA, []);
      return [...broadcastedBatch, shape1[shape1.length - 2]!];
    }

    // General case: [...broadcast_batch, m, n]
    const broadcastedBatch = broadcastBatchDims(batchA, batchB);
    const m = shape1[shape1.length - 2]!;
    const n = shape2[shape2.length - 1]!;

    return [...broadcastedBatch, m, n];
  }

  /**
   * Compute the result shape when broadcasting two shapes
   */
  static broadcastShapes(shape1: Shape, shape2: Shape): Shape {
    if (!this.canBroadcast(shape1, shape2)) {
      throw new Error(
        `Cannot broadcast shapes [${shape1.join(',')}] and [${shape2.join(',')}]: ` +
          'dimensions must be equal or one of them must be 1',
      );
    }

    const maxRank = Math.max(shape1.length, shape2.length);
    const result: number[] = new Array<number>(maxRank);

    const padded1 = this.padLeft(shape1, maxRank);
    const padded2 = this.padLeft(shape2, maxRank);

    for (let i = 0; i < maxRank; i++) {
      const dim1 = padded1[i];
      const dim2 = padded2[i];
      if (dim1 === undefined || dim2 === undefined) {
        throw new Error(`Dimension at index ${i.toString()} is undefined`);
      }
      // Special broadcasting rule for 0: 0 always wins over 1, otherwise take max
      if (dim1 === 0 || dim2 === 0) {
        result[i] = 0;
      } else {
        result[i] = Math.max(dim1, dim2);
      }
    }

    return result as Shape;
  }

  /**
   * Pad a shape with 1s on the left to reach target length
   * @internal
   */
  static padLeft(shape: Shape, targetLength: number): number[] {
    const padding = Math.max(0, targetLength - shape.length);
    return [...Array<number>(padding).fill(1), ...shape];
  }

  /**
   * Validate that a shape is well-formed
   */
  static validate(dims: number[]): boolean {
    if (dims.length > MAX_TENSOR_RANK) {
      return false;
    }

    const isValidDims = dims.every((dim) => Number.isInteger(dim) && dim >= 0);
    if (!isValidDims) {
      return false;
    }

    // Check total size
    const totalSize = dims.reduce((prod, dim) => prod * dim, 1);
    return totalSize <= MAX_TENSOR_SIZE;
  }

  /**
   * Check if two shapes are exactly equal
   */
  static equals(shape1: Shape, shape2: Shape): boolean {
    if (shape1.length !== shape2.length) {
      return false;
    }

    for (let i = 0; i < shape1.length; i++) {
      if (shape1[i] !== shape2[i]) {
        return false;
      }
    }

    return true;
  }

  /**
   * Compute the product of all dimensions in a shape
   */
  static product(shape: Shape): number {
    return shape.reduce((prod, dim) => prod * dim, 1);
  }

  /**
   * Infer shape from nested array structure
   */
  static inferFromNestedArray(data: unknown): Shape {
    function getShape(arr: unknown): number[] {
      if (!Array.isArray(arr)) {
        return []; // Scalar
      }

      if (arr.length === 0) {
        return [0]; // Empty array
      }

      const firstElement = arr[0] as unknown;
      if (!Array.isArray(firstElement)) {
        return [arr.length]; // 1D array
      }

      // Recursively get shape from first element
      const subShape = getShape(firstElement);
      return [arr.length, ...subShape];
    }

    return getShape(data) as Shape;
  }
}

// =============================================================================
// Shape Validation and Type Guards
// =============================================================================

/**
 * Type guard to check if a value is a valid shape
 */
export function isValidShape(value: unknown): value is Shape {
  return (
    Array.isArray(value) &&
    value.length <= MAX_TENSOR_RANK &&
    value.every((dim) => typeof dim === 'number' && Number.isInteger(dim) && dim >= 0) &&
    RuntimeShape.validate(value as number[])
  );
}

/**
 * Type guard to check if a shape is static (no dynamic dimensions)
 */
export function isStaticShape(shape: DynamicShape): shape is Shape {
  return shape.every((dim) => typeof dim === 'number' && dim >= 0);
}

/**
 * Type guard to check if a shape contains symbolic dimensions
 */
export function hasSymbolicDimensions(shape: SymbolicShape): boolean {
  // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
  return shape.some((dim) => typeof dim === 'object' && dim !== null && '__symbolic' in dim);
}

/**
 * Assertion function for shape validation
 */
export function assertValidShape(value: unknown, message?: string): asserts value is Shape {
  if (!isValidShape(value)) {
    if (message === undefined) {
      throw new Error(`Invalid shape: ${JSON.stringify(value)}`);
    }
    throw new Error(message);
  }
}

/**
 * Assertion function for shape compatibility
 */
export function assertShapesCompatible(shape1: Shape, shape2: Shape, operation: string): void {
  if (!RuntimeShape.canBroadcast(shape1, shape2)) {
    const shape1Str = formatShape(shape1);
    const shape2Str = formatShape(shape2);
    const suggestion = getBroadcastingSuggestion(shape1, shape2);

    throw new Error(
      `Shapes ${shape1Str} and ${shape2Str} are not compatible for ${operation}.\n` +
        `Broadcasting rule: dimensions must be equal or one of them must be 1.\n` +
        suggestion,
    );
  }
}

/**
 * Format a shape for display in error messages
 */
export function formatShape(shape: Shape): string {
  if (shape.length === 0) {
    return 'scalar []';
  }
  return `[${shape.join(', ')}]`;
}

/**
 * Get a helpful suggestion for fixing broadcasting errors
 */
function getBroadcastingSuggestion(shape1: Shape, shape2: Shape): string {
  const maxRank = Math.max(shape1.length, shape2.length);
  const padded1 = RuntimeShape.padLeft(shape1, maxRank);
  const padded2 = RuntimeShape.padLeft(shape2, maxRank);

  const mismatches: number[] = [];
  for (let i = 0; i < maxRank; i++) {
    const d1 = padded1[i];
    const d2 = padded2[i];
    if (d1 !== d2 && d1 !== 1 && d2 !== 1) {
      mismatches.push(i);
    }
  }

  if (mismatches.length === 0) {
    return '';
  }

  const suggestions: string[] = [];
  for (const idx of mismatches) {
    const dim1 = padded1[idx];
    const dim2 = padded2[idx];
    if (dim1 === undefined || dim2 === undefined) {
      throw new Error(`Dimension at index ${idx.toString()} is undefined`);
    }
    suggestions.push(
      `  - Dimension ${idx.toString()}: ${dim1.toString()} vs ${dim2.toString()} (consider reshaping or using squeeze/unsqueeze)`,
    );
  }

  return 'Incompatible dimensions:\n' + suggestions.join('\n');
}

// =============================================================================
// Common Shape Utilities
// =============================================================================

/**
 * Create a shape representing a scalar (0 dimensions)
 */
export const SCALAR_SHAPE: Shape = [] as const;

/**
 * Common shape patterns
 */
export const SHAPE_PATTERNS = {
  scalar: (): Shape => [],
  vector: (length: number): Shape => [length],
  matrix: (rows: number, cols: number): Shape => [rows, cols],
  image2d: (height: number, width: number, channels = 3): Shape => [height, width, channels],
  batch: (batchSize: number, ...shape: number[]): Shape => [batchSize, ...shape],
  sequence: (batchSize: number, seqLength: number): Shape => [batchSize, seqLength],
} as const;

/**
 * Utility to create common shapes with validation
 */
export function createShape(...dims: number[]): RuntimeShape {
  return new RuntimeShape(dims as Shape);
}

/**
 * Utility to reshape while preserving total elements
 */
export function reshape(currentShape: Shape, newShape: number[]): Shape {
  const currentSize = RuntimeShape.product(currentShape);

  // Validate newShape dimensions before processing
  for (let i = 0; i < newShape.length; i++) {
    const dim = newShape[i];
    if (dim === undefined) {
      throw new Error(`Dimension at index ${i.toString()} is undefined`);
    }
    if (dim !== -1 && (!Number.isInteger(dim) || dim < 0)) {
      throw new Error(
        `Invalid dimension ${dim.toString()} at index ${i.toString()}: must be a non-negative integer or -1`,
      );
    }
  }

  // Create a copy to avoid mutating the input
  const resultShape = [...newShape];

  // Handle -1 dimension (infer size)
  const inferIndex = resultShape.findIndex((dim) => dim === -1);
  if (inferIndex !== -1) {
    const knownSize = resultShape.reduce(
      (prod, dim, i) => (i === inferIndex ? prod : prod * dim),
      1,
    );

    if (knownSize === 0) {
      throw new Error('Cannot infer dimension when other dimensions have size 0');
    }

    if (currentSize % knownSize !== 0) {
      throw new Error(
        `Cannot reshape tensor of size ${currentSize.toString()} to shape with known dimensions ${knownSize.toString()}`,
      );
    }

    resultShape[inferIndex] = currentSize / knownSize;
  }

  const newSize = resultShape.reduce((prod, dim) => prod * dim, 1);
  if (newSize !== currentSize) {
    throw new Error(
      `Cannot reshape tensor: total size changed from ${currentSize.toString()} to ${newSize.toString()}`,
    );
  }

  return resultShape as Shape;
}

// =============================================================================
// Matrix Multiplication Utilities
// =============================================================================

/**
 * Check if two shapes can be matrix multiplied
 * 
 * @param shape1 - First tensor shape
 * @param shape2 - Second tensor shape  
 * @returns true if shapes can be matrix multiplied
 */
export function canMatmul(shape1: Shape, shape2: Shape): boolean {
  return RuntimeShape.canMatmul(shape1, shape2);
}

/**
 * Compute the output shape of matrix multiplication
 * 
 * @param shape1 - First tensor shape
 * @param shape2 - Second tensor shape
 * @returns Output shape
 * @throws {Error} If shapes are incompatible
 */
export function matmulShape(shape1: Shape, shape2: Shape): Shape {
  const result = RuntimeShape.matmulShape(shape1, shape2);
  if (result === null) {
    throw new Error(
      `Cannot multiply tensors with shapes ${formatShape(shape1)} and ${formatShape(shape2)}. ` +
      `Matrix multiplication requires compatible inner dimensions.`
    );
  }
  return result;
}

/**
 * Assert that two shapes can be matrix multiplied
 * 
 * @param shape1 - First tensor shape
 * @param shape2 - Second tensor shape
 * @throws {Error} If shapes are incompatible with detailed error message
 */
export function assertMatmulCompatible(shape1: Shape, shape2: Shape): void {
  // Check for scalars
  if (shape1.length === 0 || shape2.length === 0) {
    throw new Error(
      'Cannot multiply scalar tensors. Matrix multiplication requires at least 1D tensors.'
    );
  }

  // Check inner dimensions
  const dim1 = shape1[shape1.length - 1];
  const dim2 = shape2.length === 1 ? shape2[0] : shape2[shape2.length - 2];

  if (dim1 !== dim2) {
    const shape1Str = formatShape(shape1);
    const shape2Str = formatShape(shape2);
    throw new Error(
      `Cannot multiply tensors with shapes ${shape1Str} and ${shape2Str}. ` +
      `Matrix multiplication requires the last dimension of the first tensor (${dim1}) ` +
      `to match the ${shape2.length === 1 ? 'dimension' : 'second-to-last dimension'} ` +
      `of the second tensor (${dim2}).`
    );
  }

  // Check batch dimensions using broadcasting rules
  if (shape1.length > 2 || shape2.length > 2) {
    const batchDims1 = shape1.length > 2 ? shape1.slice(0, -2) : [];
    const batchDims2 = shape2.length > 2 ? shape2.slice(0, -2) : [];
    
    // Apply broadcasting rules: align from the right, pad with 1s on the left
    const maxBatchLength = Math.max(batchDims1.length, batchDims2.length);
    const paddedBatch1 = new Array(maxBatchLength - batchDims1.length).fill(1).concat(batchDims1);
    const paddedBatch2 = new Array(maxBatchLength - batchDims2.length).fill(1).concat(batchDims2);
    
    // Check each batch dimension for broadcasting compatibility
    for (let i = 0; i < maxBatchLength; i++) {
      const dim1 = paddedBatch1[i];
      const dim2 = paddedBatch2[i];
      // Broadcasting rule: dimensions must be equal OR one must be 1
      if (dim1 !== dim2 && dim1 !== 1 && dim2 !== 1) {
        const shape1Str = formatShape(shape1);
        const shape2Str = formatShape(shape2);
        throw new Error(
          `Cannot broadcast batch dimensions for matrix multiplication. ` +
          `Tensors with shapes ${shape1Str} and ${shape2Str} have incompatible ` +
          `batch dimension ${i}: ${dim1} vs ${dim2}. ` +
          `Dimensions must be equal or one must be 1 for broadcasting.`
        );
      }
    }
  }
}
