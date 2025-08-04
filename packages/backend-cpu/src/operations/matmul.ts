/**
 * Matrix multiplication operations for CPU backend
 *
 * Implements matrix multiplication following NumPy/PyTorch conventions:
 * - 1D × 1D → scalar (dot product)
 * - 1D × 2D → 1D (vector-matrix multiply)
 * - 2D × 1D → 1D (matrix-vector multiply)
 * - 2D × 2D → 2D (matrix-matrix multiply)
 * - ND × ND → ND (batched matrix multiply)
 */

import type { Device, DeviceData, AnyStorageTransformation } from '@typetensor/core';
import type { CPUDeviceData } from '../data';
import { createTypedArray, computeStrides } from '../utils';

/**
 * Execute matrix multiplication on CPU
 *
 * @param backend - CPU backend instance
 * @param op - Operation descriptor
 * @param inputA - First input tensor data
 * @param inputB - Second input tensor data
 * @param output - Optional pre-allocated output
 * @returns Result tensor data
 */
export async function executeMatmulOp(
  backend: Device,
  op: AnyStorageTransformation,
  inputA: DeviceData,
  inputB: DeviceData,
  output?: DeviceData,
): Promise<DeviceData> {
  const cpuInputA = inputA as CPUDeviceData;
  const cpuInputB = inputB as CPUDeviceData;

  // Get operation metadata
  const inputMetaA = op.__inputs[0];
  const inputMetaB = op.__inputs[1];
  if (!inputMetaA || !inputMetaB) {
    throw new Error('Matrix multiplication requires two inputs');
  }

  const outputMeta = op.__output;
  const outputShape = outputMeta.__shape;
  const outputDtype = outputMeta.__dtype;
  const outputSize = outputMeta.__size;

  // Create output if not provided
  output ??= backend.createData(outputSize * outputDtype.__byteSize);
  const cpuOutput = output as CPUDeviceData;

  // Create typed arrays
  const arrayA = createTypedArray(cpuInputA.buffer, inputMetaA.__dtype);
  const arrayB = createTypedArray(cpuInputB.buffer, inputMetaB.__dtype);
  const arrayOut = createTypedArray(cpuOutput.buffer, outputDtype);

  // Get shapes
  const shapeA = Array.from(inputMetaA.__shape);
  const shapeB = Array.from(inputMetaB.__shape);
  const shapeOut = Array.from(outputShape);

  // Get strides
  const stridesA = Array.from(inputMetaA.__strides);
  const stridesB = Array.from(inputMetaB.__strides);

  // Handle different rank cases
  const rankA = shapeA.length;
  const rankB = shapeB.length;

  if (rankA === 1 && rankB === 1) {
    // 1D × 1D → scalar (dot product)
    executeVectorDotProduct(arrayA, arrayB, arrayOut, shapeA[0]!);
  } else if (rankA === 1 && rankB === 2) {
    // 1D × 2D → 1D (vector-matrix multiply)
    executeVectorMatrixMultiply(
      arrayA,
      arrayB,
      arrayOut,
      shapeA[0]!,
      shapeB[0]!,
      shapeB[1]!,
      stridesA,
      stridesB,
    );
  } else if (rankA === 2 && rankB === 1) {
    // 2D × 1D → 1D (matrix-vector multiply)
    executeMatrixVectorMultiply(
      arrayA,
      arrayB,
      arrayOut,
      shapeA[0]!,
      shapeA[1]!,
      shapeB[0]!,
      stridesA,
      stridesB,
    );
  } else if (rankA === 2 && rankB === 2) {
    // 2D × 2D → 2D (matrix-matrix multiply)
    executeMatrixMatrixMultiply(
      arrayA,
      arrayB,
      arrayOut,
      shapeA[0]!,
      shapeA[1]!,
      shapeB[0]!,
      shapeB[1]!,
      stridesA,
      stridesB,
    );
  } else {
    // ND × ND → ND (batched matrix multiply)
    executeBatchedMatrixMultiply(
      arrayA,
      arrayB,
      arrayOut,
      shapeA,
      shapeB,
      shapeOut,
      stridesA,
      stridesB,
    );
  }

  return output;
}

// OPTIMIZED: Inline helper functions removed - using direct array access with type branching

/**
 * 1D × 1D → scalar (dot product) - OPTIMIZED
 */
function executeVectorDotProduct(
  arrayA: ArrayLike<number | bigint>,
  arrayB: ArrayLike<number | bigint>,
  arrayOut:
    | Int8Array
    | Uint8Array
    | Int16Array
    | Uint16Array
    | Int32Array
    | Uint32Array
    | Float32Array
    | Float64Array
    | BigInt64Array
    | BigUint64Array,
  length: number,
): void {
  // OPTIMIZED: Branch once for array types, then tight loop
  const isBigIntA = arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array;
  const isBigIntB = arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array;
  const isBigIntOut = arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array;

  let sum = 0;

  if (isBigIntA && isBigIntB) {
    // Both inputs are BigInt
    const bigA = arrayA as BigInt64Array | BigUint64Array;
    const bigB = arrayB as BigInt64Array | BigUint64Array;
    for (let i = 0; i < length; i++) {
      sum += Number(bigA[i]!) * Number(bigB[i]!);
    }
  } else if (isBigIntA) {
    // A is BigInt, B is numeric
    const bigA = arrayA as BigInt64Array | BigUint64Array;
    const numB = arrayB as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;
    for (let i = 0; i < length; i++) {
      sum += Number(bigA[i]!) * numB[i]!;
    }
  } else if (isBigIntB) {
    // A is numeric, B is BigInt
    const numA = arrayA as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;
    const bigB = arrayB as BigInt64Array | BigUint64Array;
    for (let i = 0; i < length; i++) {
      sum += numA[i]! * Number(bigB[i]!);
    }
  } else {
    // Both inputs are numeric - fastest path
    const numA = arrayA as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;
    const numB = arrayB as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;
    for (let i = 0; i < length; i++) {
      sum += numA[i]! * numB[i]!;
    }
  }

  // Set output value
  if (isBigIntOut) {
    (arrayOut as BigInt64Array | BigUint64Array)[0] = BigInt(Math.trunc(sum));
  } else {
    (arrayOut as any)[0] = sum;
  }
}

/**
 * 1D × 2D → 1D (vector-matrix multiply) - OPTIMIZED
 * Computes: out[j] = sum(a[k] * b[k, j])
 */
function executeVectorMatrixMultiply(
  arrayA: ArrayLike<number | bigint>,
  arrayB: ArrayLike<number | bigint>,
  arrayOut:
    | Int8Array
    | Uint8Array
    | Int16Array
    | Uint16Array
    | Int32Array
    | Uint32Array
    | Float32Array
    | Float64Array
    | BigInt64Array
    | BigUint64Array,
  K: number, // length of vector A and rows of matrix B
  _M: number, // rows of matrix B (unused, but kept for clarity)
  N: number, // columns of matrix B
  stridesA: number[],
  stridesB: number[],
): void {
  // OPTIMIZED: Branch once for array types, precompute strides
  const isBigIntA = arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array;
  const isBigIntB = arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array;
  const isBigIntOut = arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array;

  const strideA = stridesA[0] ?? 1;
  const strideBRow = stridesB[0] ?? N;
  const strideBCol = stridesB[1] ?? 1;

  if (isBigIntA && isBigIntB) {
    // Both inputs are BigInt
    const bigA = arrayA as BigInt64Array | BigUint64Array;
    const bigB = arrayB as BigInt64Array | BigUint64Array;
    const bigOut = arrayOut as BigInt64Array | BigUint64Array;

    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const idxA = k * strideA;
        const idxB = k * strideBRow + j * strideBCol;
        sum += Number(bigA[idxA]!) * Number(bigB[idxB]!);
      }
      if (isBigIntOut) {
        bigOut[j] = BigInt(Math.trunc(sum));
      } else {
        (arrayOut as any)[j] = sum;
      }
    }
  } else if (isBigIntA) {
    // A is BigInt, B is numeric
    const bigA = arrayA as BigInt64Array | BigUint64Array;
    const numB = arrayB as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;

    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const idxA = k * strideA;
        const idxB = k * strideBRow + j * strideBCol;
        sum += Number(bigA[idxA]!) * numB[idxB]!;
      }
      if (isBigIntOut) {
        (arrayOut as BigInt64Array | BigUint64Array)[j] = BigInt(Math.trunc(sum));
      } else {
        (arrayOut as any)[j] = sum;
      }
    }
  } else if (isBigIntB) {
    // A is numeric, B is BigInt
    const numA = arrayA as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;
    const bigB = arrayB as BigInt64Array | BigUint64Array;

    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const idxA = k * strideA;
        const idxB = k * strideBRow + j * strideBCol;
        sum += numA[idxA]! * Number(bigB[idxB]!);
      }
      if (isBigIntOut) {
        (arrayOut as BigInt64Array | BigUint64Array)[j] = BigInt(Math.trunc(sum));
      } else {
        (arrayOut as any)[j] = sum;
      }
    }
  } else {
    // Both inputs are numeric - fastest path
    const numA = arrayA as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;
    const numB = arrayB as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;

    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const idxA = k * strideA;
        const idxB = k * strideBRow + j * strideBCol;
        sum += numA[idxA]! * numB[idxB]!;
      }
      if (isBigIntOut) {
        (arrayOut as BigInt64Array | BigUint64Array)[j] = BigInt(Math.trunc(sum));
      } else {
        (arrayOut as any)[j] = sum;
      }
    }
  }
}

/**
 * 2D × 1D → 1D (matrix-vector multiply) - OPTIMIZED
 * Computes: out[i] = sum(a[i, k] * b[k])
 */
function executeMatrixVectorMultiply(
  arrayA: ArrayLike<number | bigint>,
  arrayB: ArrayLike<number | bigint>,
  arrayOut:
    | Int8Array
    | Uint8Array
    | Int16Array
    | Uint16Array
    | Int32Array
    | Uint32Array
    | Float32Array
    | Float64Array
    | BigInt64Array
    | BigUint64Array,
  M: number, // rows of matrix A
  K: number, // columns of matrix A and length of vector B
  _N: number, // length of vector B (same as K, but kept for clarity)
  stridesA: number[],
  stridesB: number[],
): void {
  // OPTIMIZED: Branch once for array types
  const isBigIntA = arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array;
  const isBigIntB = arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array;
  const isBigIntOut = arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array;

  const strideARow = stridesA[0] ?? K;
  const strideACol = stridesA[1] ?? 1;
  const strideB = stridesB[0] ?? 1;

  if (isBigIntA && isBigIntB) {
    // Both inputs are BigInt
    const bigA = arrayA as BigInt64Array | BigUint64Array;
    const bigB = arrayB as BigInt64Array | BigUint64Array;

    for (let i = 0; i < M; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const idxA = i * strideARow + k * strideACol;
        const idxB = k * strideB;
        sum += Number(bigA[idxA]!) * Number(bigB[idxB]!);
      }
      if (isBigIntOut) {
        (arrayOut as BigInt64Array | BigUint64Array)[i] = BigInt(Math.trunc(sum));
      } else {
        (arrayOut as any)[i] = sum;
      }
    }
  } else if (isBigIntA) {
    // A is BigInt, B is numeric
    const bigA = arrayA as BigInt64Array | BigUint64Array;
    const numB = arrayB as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;

    for (let i = 0; i < M; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const idxA = i * strideARow + k * strideACol;
        const idxB = k * strideB;
        sum += Number(bigA[idxA]!) * numB[idxB]!;
      }
      if (isBigIntOut) {
        (arrayOut as BigInt64Array | BigUint64Array)[i] = BigInt(Math.trunc(sum));
      } else {
        (arrayOut as any)[i] = sum;
      }
    }
  } else if (isBigIntB) {
    // A is numeric, B is BigInt
    const numA = arrayA as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;
    const bigB = arrayB as BigInt64Array | BigUint64Array;

    for (let i = 0; i < M; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const idxA = i * strideARow + k * strideACol;
        const idxB = k * strideB;
        sum += numA[idxA]! * Number(bigB[idxB]!);
      }
      if (isBigIntOut) {
        (arrayOut as BigInt64Array | BigUint64Array)[i] = BigInt(Math.trunc(sum));
      } else {
        (arrayOut as any)[i] = sum;
      }
    }
  } else {
    // Both inputs are numeric - fastest path
    const numA = arrayA as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;
    const numB = arrayB as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;

    for (let i = 0; i < M; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const idxA = i * strideARow + k * strideACol;
        const idxB = k * strideB;
        sum += numA[idxA]! * numB[idxB]!;
      }
      if (isBigIntOut) {
        (arrayOut as BigInt64Array | BigUint64Array)[i] = BigInt(Math.trunc(sum));
      } else {
        (arrayOut as any)[i] = sum;
      }
    }
  }
}

/**
 * 2D × 2D → 2D (matrix-matrix multiply) - OPTIMIZED
 * Computes: out[i, j] = sum(a[i, k] * b[k, j])
 */
function executeMatrixMatrixMultiply(
  arrayA: ArrayLike<number | bigint>,
  arrayB: ArrayLike<number | bigint>,
  arrayOut:
    | Int8Array
    | Uint8Array
    | Int16Array
    | Uint16Array
    | Int32Array
    | Uint32Array
    | Float32Array
    | Float64Array
    | BigInt64Array
    | BigUint64Array,
  M: number, // rows of matrix A
  K: number, // columns of matrix A / rows of matrix B
  _K2: number, // rows of matrix B (same as K)
  N: number, // columns of matrix B
  stridesA: number[],
  stridesB: number[],
): void {
  // OPTIMIZED: Branch once for array types
  const isBigIntA = arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array;
  const isBigIntB = arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array;
  const isBigIntOut = arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array;

  const strideARow = stridesA[0] ?? K;
  const strideACol = stridesA[1] ?? 1;
  const strideBRow = stridesB[0] ?? N;
  const strideBCol = stridesB[1] ?? 1;

  // Output is always C-contiguous
  let outIdx = 0;

  if (isBigIntA && isBigIntB) {
    // Both inputs are BigInt
    const bigA = arrayA as BigInt64Array | BigUint64Array;
    const bigB = arrayB as BigInt64Array | BigUint64Array;

    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          const idxA = i * strideARow + k * strideACol;
          const idxB = k * strideBRow + j * strideBCol;
          sum += Number(bigA[idxA]!) * Number(bigB[idxB]!);
        }
        if (isBigIntOut) {
          (arrayOut as BigInt64Array | BigUint64Array)[outIdx++] = BigInt(Math.trunc(sum));
        } else {
          (arrayOut as any)[outIdx++] = sum;
        }
      }
    }
  } else if (isBigIntA) {
    // A is BigInt, B is numeric
    const bigA = arrayA as BigInt64Array | BigUint64Array;
    const numB = arrayB as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;

    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          const idxA = i * strideARow + k * strideACol;
          const idxB = k * strideBRow + j * strideBCol;
          sum += Number(bigA[idxA]!) * numB[idxB]!;
        }
        if (isBigIntOut) {
          (arrayOut as BigInt64Array | BigUint64Array)[outIdx++] = BigInt(Math.trunc(sum));
        } else {
          (arrayOut as any)[outIdx++] = sum;
        }
      }
    }
  } else if (isBigIntB) {
    // A is numeric, B is BigInt
    const numA = arrayA as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;
    const bigB = arrayB as BigInt64Array | BigUint64Array;

    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          const idxA = i * strideARow + k * strideACol;
          const idxB = k * strideBRow + j * strideBCol;
          sum += numA[idxA]! * Number(bigB[idxB]!);
        }
        if (isBigIntOut) {
          (arrayOut as BigInt64Array | BigUint64Array)[outIdx++] = BigInt(Math.trunc(sum));
        } else {
          (arrayOut as any)[outIdx++] = sum;
        }
      }
    }
  } else {
    // Both inputs are numeric - fastest path
    const numA = arrayA as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;
    const numB = arrayB as
      | Float32Array
      | Float64Array
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array;

    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          const idxA = i * strideARow + k * strideACol;
          const idxB = k * strideBRow + j * strideBCol;
          sum += numA[idxA]! * numB[idxB]!;
        }
        if (isBigIntOut) {
          (arrayOut as BigInt64Array | BigUint64Array)[outIdx++] = BigInt(Math.trunc(sum));
        } else {
          (arrayOut as any)[outIdx++] = sum;
        }
      }
    }
  }
}

/**
 * ND × ND → ND (batched matrix multiply) - OPTIMIZED
 * Handles arbitrary dimensional tensors with batch dimensions
 */
function executeBatchedMatrixMultiply(
  arrayA: ArrayLike<number | bigint>,
  arrayB: ArrayLike<number | bigint>,
  arrayOut:
    | Int8Array
    | Uint8Array
    | Int16Array
    | Uint16Array
    | Int32Array
    | Uint32Array
    | Float32Array
    | Float64Array
    | BigInt64Array
    | BigUint64Array,
  shapeA: number[],
  shapeB: number[],
  shapeOut: number[],
  stridesA: number[],
  stridesB: number[],
): void {
  // OPTIMIZED: Branch once for array types
  const isBigIntA = arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array;
  const isBigIntB = arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array;
  const isBigIntOut = arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array;

  const rankA = shapeA.length;
  const rankB = shapeB.length;
  const rankOut = shapeOut.length;

  // Extract matrix dimensions
  const M = shapeA[rankA - 2] ?? 1;
  const K = shapeA[rankA - 1] ?? 1;
  const N = shapeB[rankB - 1] ?? 1;

  // Calculate batch size
  let batchSize = 1;
  for (let i = 0; i < rankOut - 2; i++) {
    batchSize *= shapeOut[i] ?? 1;
  }

  // Strides for the matrix dimensions
  const strideARow = stridesA[rankA - 2] ?? K;
  const strideACol = stridesA[rankA - 1] ?? 1;
  const strideBRow = stridesB[rankB - 2] ?? N;
  const strideBCol = stridesB[rankB - 1] ?? 1;

  // Compute strides for batch dimensions
  const batchStridesOut = computeStrides(shapeOut.slice(0, -2));

  // For each batch
  for (let batch = 0; batch < batchSize; batch++) {
    // Compute batch indices
    const batchIndices: number[] = [];
    let temp = batch;
    for (let i = batchStridesOut.length - 1; i >= 0; i--) {
      const stride = batchStridesOut[i]!;
      batchIndices[i] = Math.floor(temp / stride);
      temp %= stride;
    }

    // Compute offsets for this batch
    let offsetA = 0;
    let offsetB = 0;

    // Map batch indices to input tensors (handle broadcasting)
    for (let i = 0; i < batchIndices.length; i++) {
      const batchIdx = batchIndices[i]!;

      // For A: map from output batch dims to A's batch dims
      const dimIdxA = rankA - rankOut + i;
      if (dimIdxA >= 0 && (shapeA[dimIdxA] ?? 0) > 1) {
        offsetA += batchIdx * (stridesA[dimIdxA] ?? 0);
      }

      // For B: map from output batch dims to B's batch dims
      const dimIdxB = rankB - rankOut + i;
      if (dimIdxB >= 0 && (shapeB[dimIdxB] ?? 0) > 1) {
        offsetB += batchIdx * (stridesB[dimIdxB] ?? 0);
      }
    }

    // Perform matrix multiplication for this batch
    const baseOutIdx = batch * M * N;

    for (let i = 0; i < M; i++) {
      for (let j = 0; j < N; j++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          const idxA = offsetA + i * strideARow + k * strideACol;
          const idxB = offsetB + k * strideBRow + j * strideBCol;

          // OPTIMIZED: Direct array access with type handling
          const valA = isBigIntA
            ? Number((arrayA as BigInt64Array | BigUint64Array)[idxA]!)
            : (arrayA as any)[idxA]!;
          const valB = isBigIntB
            ? Number((arrayB as BigInt64Array | BigUint64Array)[idxB]!)
            : (arrayB as any)[idxB]!;
          sum += valA * valB;
        }
        const outIdx = baseOutIdx + i * N + j;

        // OPTIMIZED: Direct output setting with type handling
        if (isBigIntOut) {
          (arrayOut as BigInt64Array | BigUint64Array)[outIdx] = BigInt(Math.trunc(sum));
        } else {
          (arrayOut as any)[outIdx] = sum;
        }
      }
    }
  }
}
