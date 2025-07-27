/**
 * Utility functions for CPU backend operations
 *
 * Provides helpers for typed array creation, broadcasting, and index calculations.
 */

import type { AnyDType } from '@typetensor/core';

/**
 * Create a typed array from an ArrayBuffer based on dtype
 *
 * @param buffer - Raw data buffer
 * @param dtype - Data type descriptor (compile-time or runtime)
 * @returns Typed array view of the buffer
 */
export function createTypedArray(
  buffer: ArrayBuffer,
  dtype?: AnyDType,
):
  | Int8Array
  | Uint8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array
  | BigInt64Array
  | BigUint64Array {
  if (!dtype) {
    throw new Error('dtype is required to create typed array');
  }

  const TypedArrayConstructor = dtype.__typedArray;
  return new TypedArrayConstructor(buffer);
}

/**
 * Compute the flat index for a multi-dimensional position
 *
 * @param indices - Position in each dimension
 * @param strides - Memory strides for each dimension
 * @returns Flat array index
 */
export function computeFlatIndex(indices: readonly number[], strides: readonly number[]): number {
  let flatIndex = 0;
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    const stride = strides[i];
    if (idx !== undefined && stride !== undefined) {
      flatIndex += idx * stride;
    }
  }
  return flatIndex;
}

/**
 * Convert flat index to multi-dimensional indices
 *
 * @param flatIndex - Flat array index
 * @param shape - Tensor shape
 * @returns Position in each dimension
 */
export function unravelIndex(flatIndex: number, shape: readonly number[]): number[] {
  const indices: number[] = [];
  let remaining = flatIndex;

  for (let i = 0; i < shape.length; i++) {
    const stride = shape.slice(i + 1).reduce((a, b) => a * b, 1);
    indices[i] = Math.floor(remaining / stride);
    remaining %= stride;
  }

  return indices;
}

/**
 * Broadcast shapes to find the output shape
 *
 * @param shapeA - First shape
 * @param shapeB - Second shape
 * @returns Broadcasted output shape
 * @throws {Error} If shapes cannot broadcast
 */
export function broadcastShapes(shapeA: readonly number[], shapeB: readonly number[]): number[] {
  const ndimA = shapeA.length;
  const ndimB = shapeB.length;
  const ndimOut = Math.max(ndimA, ndimB);
  const outputShape: number[] = [];

  // Pad shapes with 1s on the left
  const paddedA: number[] = new Array<number>(ndimOut - ndimA).fill(1).concat(Array.from(shapeA));
  const paddedB: number[] = new Array<number>(ndimOut - ndimB).fill(1).concat(Array.from(shapeB));

  // Check broadcasting rules
  for (let i = 0; i < ndimOut; i++) {
    const dimA = paddedA[i];
    const dimB = paddedB[i];

    if (dimA === undefined || dimB === undefined) {
      throw new Error('Invalid padded shape');
    }

    if (dimA === dimB) {
      outputShape[i] = dimA;
    } else if (dimA === 1) {
      outputShape[i] = dimB;
    } else if (dimB === 1) {
      outputShape[i] = dimA;
    } else {
      throw new Error(
        `Cannot broadcast shapes [${shapeA.join(', ')}] and [${shapeB.join(', ')}]: ` +
          `dimension ${i} has sizes ${dimA} and ${dimB}`,
      );
    }
  }

  return outputShape;
}

/**
 * Iterator for broadcasting indices
 *
 * @param outputShape - Shape to iterate over
 * @param inputShapes - Input shapes to broadcast
 * @yields Indices for output and corresponding indices for each input
 */
export function* broadcastIndices(
  outputShape: readonly number[],
  inputShapes: readonly (readonly number[])[],
): Generator<{
  outputIndex: number[];
  inputIndices: number[][];
}> {
  const ndimOut = outputShape.length;
  const totalSize = outputShape.reduce((a, b) => a * b, 1);

  for (let flatIndex = 0; flatIndex < totalSize; flatIndex++) {
    const outputIndex = unravelIndex(flatIndex, outputShape);
    const inputIndices: number[][] = [];

    for (const inputShape of inputShapes) {
      const ndimIn = inputShape.length;
      const inputIndex: number[] = [];

      // Map output index to input index considering broadcasting
      // Broadcasting aligns dimensions from the right
      for (let i = 0; i < ndimIn; i++) {
        const outDim = ndimOut - ndimIn + i;
        const inDim = inputShape[i];
        const outIdx = outputIndex[outDim];
        if (inDim === undefined || outIdx === undefined) {
          throw new Error(
            `Invalid dimension mapping: outDim=${outDim}, inDim=${inDim}, outIdx=${outIdx}`,
          );
        }

        // If input dimension is 1, always use index 0 (broadcasting)
        inputIndex[i] = inDim === 1 ? 0 : outIdx;
      }

      inputIndices.push(inputIndex);
    }

    yield { outputIndex, inputIndices };
  }
}

/**
 * Check if a shape is contiguous (C-order)
 *
 * @param shape - Tensor shape
 * @param strides - Memory strides
 * @returns True if tensor is C-contiguous
 */
export function isContiguous(shape: readonly number[], strides: readonly number[]): boolean {
  let expectedStride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    if (strides[i] !== expectedStride) {
      return false;
    }
    const dim = shape[i];
    if (dim !== undefined) {
      expectedStride *= dim;
    }
  }
  return true;
}

/**
 * Compute C-order (row-major) strides for a shape
 *
 * @param shape - Tensor shape
 * @returns Strides array
 */
export function computeStrides(shape: readonly number[]): number[] {
  const strides: number[] = [];
  let stride = 1;

  for (let i = shape.length - 1; i >= 0; i--) {
    strides.unshift(stride);
    const dim = shape[i];
    if (dim !== undefined) {
      stride *= dim;
    }
  }

  return strides;
}
