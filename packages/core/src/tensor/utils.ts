/**
 * Internal utilities for tensor operations
 */

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
