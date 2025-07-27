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
