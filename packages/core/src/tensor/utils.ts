/**
 * Internal utilities for tensor operations
 */

import type { SliceIndex } from '../shape/types';

/**
 * Compute strides for a shape in C-order (row-major)
 */
export function computeStrides(shape: readonly number[]): number[] {
  const strides: number[] = [];
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides.unshift(stride);
    const dim = shape[i];
    if (dim === undefined) {
      throw new Error(`Invalid shape dimension at index ${i.toString()}`);
    }
    stride *= dim;
  }
  return strides;
}

/**
 * Compute the product of a shape (total number of elements)
 */
export function computeSize(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * Normalize a slice index, handling negative values and defaults
 */
export function normalizeSliceIndex(index: number, dim: number): number {
  if (index < 0) {
    return Math.max(0, dim + index);
  }
  return Math.min(index, dim);
}

/**
 * Compute the shape after slicing
 */
export function computeSlicedShape(
  shape: readonly number[],
  indices: readonly SliceIndex[],
): number[] {
  const result: number[] = [];

  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i];
    if (dim === undefined) {
      throw new Error(`Invalid shape dimension at index ${i}`);
    }

    // If we've run out of indices, keep remaining dimensions
    if (i >= indices.length) {
      result.push(dim);
      continue;
    }

    const index = indices[i];

    if (index === null) {
      // null means keep the whole dimension
      result.push(dim);
    } else if (typeof index === 'number') {
      // Integer index removes the dimension
      // We don't push anything to result
    } else if (index && typeof index === 'object') {
      // SliceSpec
      const start = index.start !== undefined ? normalizeSliceIndex(index.start, dim) : 0;
      const stop = index.stop !== undefined ? normalizeSliceIndex(index.stop, dim) : dim;
      const step = index.step ?? 1;

      if (step === 0) {
        throw new Error('Slice step cannot be zero');
      }

      let size: number;
      if (step > 0) {
        size = Math.max(0, Math.ceil((stop - start) / step));
      } else {
        // Negative step - swap start/stop logic
        const adjStart =
          index.start !== undefined ? normalizeSliceIndex(index.start, dim) : dim - 1;
        // For negative step, stop of -1 means go to beginning (index 0)
        // stop undefined means go to beginning as well
        let adjStop: number;
        if (index.stop === undefined) {
          adjStop = -1; // Go all the way to beginning
        } else if (index.stop === -1) {
          adjStop = -1; // Explicitly specified -1, go to beginning
        } else {
          adjStop = normalizeSliceIndex(index.stop, dim);
        }
        size = Math.max(0, Math.ceil((adjStart - adjStop) / Math.abs(step)));
      }

      result.push(size);
    }
  }

  return result;
}

/**
 * Compute the strides after slicing
 */
export function computeSlicedStrides(
  strides: readonly number[],
  indices: readonly SliceIndex[],
): number[] {
  const result: number[] = [];

  for (let i = 0; i < strides.length; i++) {
    const stride = strides[i];
    if (stride === undefined) {
      throw new Error(`Invalid stride at index ${i}`);
    }

    // If we've run out of indices, keep remaining strides
    if (i >= indices.length) {
      result.push(stride);
      continue;
    }

    const index = indices[i];

    if (index === null) {
      // null means keep the stride
      result.push(stride);
    } else if (typeof index === 'number') {
      // Integer index removes the stride
      // We don't push anything to result
    } else if (index && typeof index === 'object') {
      // SliceSpec
      const step = index.step ?? 1;
      // Multiply stride by step
      result.push(stride * step);
    }
  }

  return result;
}

/**
 * Validate slice indices against shape
 */
export function validateSliceIndices(
  shape: readonly number[],
  indices: readonly SliceIndex[],
): void {
  if (indices.length > shape.length) {
    throw new Error(
      `Too many indices for tensor of dimension ${shape.length}: got ${indices.length} indices`,
    );
  }

  for (let i = 0; i < indices.length; i++) {
    const index = indices[i];
    const dim = shape[i];

    if (dim === undefined) {
      throw new Error(`Invalid shape dimension at index ${i}`);
    }

    if (typeof index === 'number') {
      // Validate integer index
      const normalized = index < 0 ? dim + index : index;
      if (normalized < 0 || normalized >= dim) {
        throw new Error(`Index ${index} is out of bounds for dimension ${i} with size ${dim}`);
      }
    }
  }
}

/**
 * Compute strides after transpose (swap last two dimensions)
 *
 * For tensors with rank < 2, returns strides unchanged.
 * For higher rank tensors, swaps the last two stride values.
 *
 * @param shape - Original shape
 * @param strides - Original strides
 * @returns New strides after transpose
 *
 * @example
 * computeTransposedStrides([2, 3, 4], [12, 4, 1]) // Returns [12, 1, 4]
 * computeTransposedStrides([3, 4], [4, 1]) // Returns [1, 4]
 * computeTransposedStrides([10], [1]) // Returns [1]
 */
export function computeTransposedStrides(
  shape: readonly number[],
  strides: readonly number[],
): number[] {
  if (shape.length < 2) {
    // For scalars and 1D tensors, return unchanged
    return [...strides];
  }

  // Swap last two strides
  const newStrides = [...strides];
  const lastIdx = strides.length - 1;
  const secondLastIdx = lastIdx - 1;

  const temp = newStrides[lastIdx];
  const secondLastStride = newStrides[secondLastIdx];
  if (secondLastStride === undefined) {
    throw new Error(`Invalid stride at index ${secondLastIdx}`);
  }
  newStrides[lastIdx] = secondLastStride;

  if (temp === undefined) {
    throw new Error(`Invalid stride at index ${lastIdx}`);
  }
  newStrides[secondLastIdx] = temp;

  return newStrides;
}

/**
 * Compute shape after transpose (swap last two dimensions)
 *
 * @param shape - Original shape
 * @returns New shape after transpose
 *
 * @example
 * computeTransposedShape([2, 3, 4]) // Returns [2, 4, 3]
 * computeTransposedShape([3, 4]) // Returns [4, 3]
 * computeTransposedShape([10]) // Returns [10]
 */
export function computeTransposedShape(shape: readonly number[]): number[] {
  if (shape.length < 2) {
    return [...shape];
  }

  const newShape = [...shape];
  const lastIdx = shape.length - 1;
  const secondLastIdx = lastIdx - 1;

  const temp = newShape[lastIdx];
  newShape[lastIdx] = newShape[secondLastIdx]!;
  newShape[secondLastIdx] = temp!;

  return newShape;
}

/**
 * Compute strides after permutation
 *
 * Rearranges strides according to the given axes permutation.
 *
 * @param strides - Original strides
 * @param axes - Permutation axes (each axis 0 to rank-1 must appear exactly once)
 * @returns New strides after permutation
 *
 * @example
 * computePermutedStrides([12, 4, 1], [2, 0, 1]) // Returns [1, 12, 4]
 * computePermutedStrides([20, 1], [1, 0]) // Returns [1, 20]
 */
export function computePermutedStrides(
  strides: readonly number[],
  axes: readonly number[],
): number[] {
  const newStrides = new Array<number>(axes.length);
  for (let i = 0; i < axes.length; i++) {
    const axis = axes[i];
    if (axis === undefined) {
      throw new Error(`Invalid axis at index ${i}`);
    }
    const stride = strides[axis];
    if (stride === undefined) {
      throw new Error(`Invalid stride for axis ${axis}`);
    }
    newStrides[i] = stride;
  }
  return newStrides;
}

/**
 * Compute shape after permutation
 *
 * Rearranges shape dimensions according to the given axes permutation.
 *
 * @param shape - Original shape
 * @param axes - Permutation axes (each axis 0 to rank-1 must appear exactly once)
 * @returns New shape after permutation
 *
 * @example
 * computePermutedShape([2, 3, 4], [2, 0, 1]) // Returns [4, 2, 3]
 * computePermutedShape([10, 20], [1, 0]) // Returns [20, 10]
 */
export function computePermutedShape(shape: readonly number[], axes: readonly number[]): number[] {
  const newShape = new Array<number>(axes.length);
  for (let i = 0; i < axes.length; i++) {
    const axis = axes[i];
    if (axis === undefined) {
      throw new Error(`Invalid axis at index ${i}`);
    }
    const dim = shape[axis];
    if (dim === undefined) {
      throw new Error(`Invalid dimension for axis ${axis}`);
    }
    newShape[i] = dim;
  }
  return newShape;
}

/**
 * Validate permutation axes
 *
 * Ensures that:
 * 1. The number of axes matches the tensor rank
 * 2. Each axis is within valid range [0, rank) or [-rank, -1]
 * 3. Each axis appears exactly once (no duplicates)
 *
 * @param rank - Tensor rank
 * @param axes - Permutation axes to validate
 * @throws Error if axes are invalid
 *
 * @example
 * validatePermutationAxes(3, [2, 0, 1]) // Valid
 * validatePermutationAxes(3, [0, 1, 2]) // Valid (identity)
 * validatePermutationAxes(3, [-1, 0, 1]) // Valid (using negative index)
 * validatePermutationAxes(3, [0, 0, 1]) // Throws: duplicate axis
 * validatePermutationAxes(3, [0, 1]) // Throws: wrong length
 * validatePermutationAxes(3, [0, 1, 3]) // Throws: out of bounds
 */
export function validatePermutationAxes(rank: number, axes: readonly number[]): void {
  if (axes.length !== rank) {
    throw new Error(`Permutation axes length ${axes.length} must match tensor rank ${rank}`);
  }

  const seen = new Set<number>();
  for (let i = 0; i < axes.length; i++) {
    const axis = axes[i];
    if (axis === undefined) {
      throw new Error(`Axis at index ${i} is undefined`);
    }

    // Handle negative indexing
    const normalizedAxis = axis < 0 ? rank + axis : axis;

    if (normalizedAxis < 0 || normalizedAxis >= rank) {
      throw new Error(`Axis ${axis} is out of bounds for tensor of rank ${rank}`);
    }

    if (seen.has(normalizedAxis)) {
      throw new Error(`Duplicate axis ${axis} in permutation`);
    }
    seen.add(normalizedAxis);
  }
}

/**
 * Normalize permutation axes (convert negative indices to positive)
 *
 * @param axes - Permutation axes (may contain negative indices)
 * @param rank - Tensor rank
 * @returns Normalized axes (all positive indices)
 *
 * @example
 * normalizePermutationAxes([0, -1, 1], 3) // Returns [0, 2, 1]
 * normalizePermutationAxes([-3, -2, -1], 3) // Returns [0, 1, 2]
 */
export function normalizePermutationAxes(axes: readonly number[], rank: number): number[] {
  return axes.map((axis) => (axis < 0 ? rank + axis : axis));
}

// =============================================================================
// Matrix Multiplication Utilities
// =============================================================================

/**
 * Assert that two shapes can be matrix multiplied
 * 
 * This is a more specific version for tensor operations that provides
 * detailed error messages for debugging.
 * 
 * @param shapeA - First tensor shape
 * @param shapeB - Second tensor shape
 * @throws {Error} If shapes are incompatible
 */
export function assertMatmulCompatible(
  shapeA: readonly number[],
  shapeB: readonly number[],
): void {
  // Check for scalars
  if (shapeA.length === 0 || shapeB.length === 0) {
    throw new Error(
      'Cannot perform matrix multiplication on scalar tensors. Both tensors must be at least 1D.',
    );
  }

  // Get inner dimensions
  const innerA = shapeA[shapeA.length - 1];
  const innerB = shapeB.length === 1 ? shapeB[0] : shapeB[shapeB.length - 2];

  if (innerA === undefined || innerB === undefined) {
    throw new Error('Invalid tensor shape: missing dimensions');
  }

  // Check inner dimensions match
  if (innerA !== innerB) {
    throw new Error(
      `Matrix multiplication inner dimensions must match: ${innerA} != ${innerB}. ` +
        `Got shapes [${shapeA.join(', ')}] and [${shapeB.join(', ')}].`,
    );
  }

  // Validate batch dimensions
  validateBatchDimensions(shapeA, shapeB);
}

/**
 * Compute the output shape for matrix multiplication
 * 
 * Handles all cases: 1D×1D (dot product), 1D×2D, 2D×1D, 2D×2D, and batched operations.
 * 
 * @param shapeA - First tensor shape
 * @param shapeB - Second tensor shape
 * @returns Output shape
 */
export function computeMatmulOutputShape(
  shapeA: readonly number[],
  shapeB: readonly number[],
): number[] {
  // 1D × 1D → scalar
  if (shapeA.length === 1 && shapeB.length === 1) {
    return [];
  }

  // Extract batch dimensions
  const batchShape = extractBatchShape(shapeA, shapeB);

  // 1D × 2D → [n]
  if (shapeA.length === 1 && shapeB.length === 2) {
    const n = shapeB[1];
    if (n === undefined) throw new Error('Invalid shape dimension');
    return [n];
  }

  // 1D × ND → [...batch, n]
  if (shapeA.length === 1 && shapeB.length > 2) {
    const n = shapeB[shapeB.length - 1];
    if (n === undefined) throw new Error('Invalid shape dimension');
    return [...batchShape, n];
  }

  // 2D × 1D → [m]
  if (shapeA.length === 2 && shapeB.length === 1) {
    const m = shapeA[0];
    if (m === undefined) throw new Error('Invalid shape dimension');
    return [m];
  }

  // ND × 1D → [...batch, m]
  if (shapeA.length > 2 && shapeB.length === 1) {
    const m = shapeA[shapeA.length - 2];
    if (m === undefined) throw new Error('Invalid shape dimension');
    return [...batchShape, m];
  }

  // General case: [...batch, m, n]
  const m = shapeA[shapeA.length - 2];
  const n = shapeB[shapeB.length - 1];
  if (m === undefined || n === undefined) {
    throw new Error('Invalid shape dimensions');
  }

  return [...batchShape, m, n];
}

/**
 * Expand dimensions for 1D tensors in matrix multiplication
 * 
 * Following NumPy convention:
 * - 1D as first operand: prepend 1 → [n] becomes [1, n]
 * - 1D as second operand: append 1 → [n] becomes [n, 1]
 * 
 * @param shape - Original shape
 * @param isFirstOperand - Whether this is the first operand in matmul
 * @returns Expanded shape
 */
export function expandDimsForMatmul(
  shape: readonly number[],
  isFirstOperand: boolean,
): number[] {
  if (shape.length !== 1) {
    return [...shape];
  }

  const n = shape[0];
  if (n === undefined) {
    throw new Error('Invalid 1D shape');
  }

  return isFirstOperand ? [1, n] : [n, 1];
}

/**
 * Squeeze result dimensions after matrix multiplication
 * 
 * Removes dimensions that were added for 1D operands:
 * - If both were 1D: return scalar (empty shape)
 * - If first was 1D: remove first dimension
 * - If second was 1D: remove last dimension
 * 
 * @param shape - Result shape from matmul
 * @param operandAWas1D - Whether first operand was originally 1D
 * @param operandBWas1D - Whether second operand was originally 1D
 * @returns Squeezed shape
 */
export function squeezeMatmulResult(
  shape: readonly number[],
  operandAWas1D: boolean,
  operandBWas1D: boolean,
): number[] {
  // Both were 1D → scalar
  if (operandAWas1D && operandBWas1D) {
    return [];
  }

  let result = [...shape];

  // First was 1D → remove first dimension
  if (operandAWas1D && result.length > 0 && result[0] === 1) {
    result = result.slice(1);
  }

  // Second was 1D → remove last dimension
  if (operandBWas1D && result.length > 0 && result[result.length - 1] === 1) {
    result = result.slice(0, -1);
  }

  return result;
}

/**
 * Validate that batch dimensions match for matrix multiplication
 * 
 * Batch dimensions must match exactly (no broadcasting in batch dims).
 * 
 * @param shapeA - First tensor shape
 * @param shapeB - Second tensor shape
 * @throws {Error} If batch dimensions don't match
 */
export function validateBatchDimensions(
  shapeA: readonly number[],
  shapeB: readonly number[],
): void {
  const batchDimsA = shapeA.length > 2 ? shapeA.slice(0, -2) : [];
  const batchDimsB = shapeB.length > 2 ? shapeB.slice(0, -2) : [];

  if (batchDimsA.length !== batchDimsB.length) {
    return; // Different ranks are OK, we'll use the shorter one
  }

  for (let i = 0; i < batchDimsA.length; i++) {
    if (batchDimsA[i] !== batchDimsB[i]) {
      throw new Error(
        `Batch dimension ${i} does not match: ${batchDimsA[i]} != ${batchDimsB[i]}`,
      );
    }
  }
}

/**
 * Extract common batch dimensions from two shapes
 * 
 * Returns the batch dimensions that are common to both tensors.
 * For tensors with different numbers of batch dimensions, returns
 * the dimensions from the tensor with more batch dims.
 * 
 * @param shapeA - First tensor shape
 * @param shapeB - Second tensor shape
 * @returns Batch dimensions
 */
export function extractBatchShape(
  shapeA: readonly number[],
  shapeB: readonly number[],
): number[] {
  const batchDimsA = shapeA.length > 2 ? shapeA.slice(0, -2) : [];
  const batchDimsB = shapeB.length > 2 ? shapeB.slice(0, -2) : [];

  // Return the longer batch dimensions (handles broadcasting-like behavior)
  return batchDimsA.length >= batchDimsB.length ? [...batchDimsA] : [...batchDimsB];
}
