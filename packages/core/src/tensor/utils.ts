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
