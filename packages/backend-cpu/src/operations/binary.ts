/**
 * Binary operations for CPU backend
 *
 * Implements element-wise binary operations with broadcasting support.
 */

import type { Device, DeviceData, AnyStorageTransformation } from '@typetensor/core';
import type { CPUDeviceData } from '../data';
import { createTypedArray, broadcastIndices, computeFlatIndex, computeStrides } from '../utils';

const getBigIntValue = (arr: ArrayLike<number | bigint>, index: number): bigint | undefined => {
  const val = arr[index];
  if (val === undefined) {
    return undefined;
  }
  return typeof val === 'bigint' ? val : BigInt(Math.trunc(val));
};

// Helper function to get numeric value (handles BigInt conversion)
const getNumberValue = (arr: ArrayLike<number | bigint>, index: number): number => {
  const val = arr[index];
  if (val === undefined) {
    return NaN;
  }
  return typeof val === 'bigint' ? Number(val) : val;
};

/**
 * Execute a binary operation on CPU with broadcasting
 *
 * @param backend - CPU backend instance
 * @param op - Operation descriptor
 * @param inputA - First input tensor data
 * @param inputB - Second input tensor data
 * @param output - Optional pre-allocated output
 * @returns Result tensor data
 */
export async function executeBinaryOp(
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
    throw new Error('Binary operation requires two inputs');
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

  // Get shapes and strides
  const shapeA = Array.from(inputMetaA.__shape);
  const shapeB = Array.from(inputMetaB.__shape);
  const stridesA = Array.from(inputMetaA.__strides);
  const stridesB = Array.from(inputMetaB.__strides);

  // Check if we can use fast path (no broadcasting needed)
  const sameSizeNoBC =
    inputMetaA.__size === inputMetaB.__size &&
    inputMetaA.__size === outputSize &&
    shapeA.length === shapeB.length &&
    shapeA.every((dim, i) => dim === shapeB[i]);

  if (
    sameSizeNoBC &&
    inputMetaA.__layout.c_contiguous === true &&
    inputMetaB.__layout.c_contiguous === true
  ) {
    // Fast path: both inputs are contiguous and same shape
    executeBinaryOpFast(op.__op, arrayA, arrayB, arrayOut);
  } else {
    // Slow path: handle broadcasting and/or non-contiguous arrays
    executeBinaryOpBroadcast(
      op.__op,
      arrayA,
      arrayB,
      arrayOut,
      shapeA,
      shapeB,
      Array.from(outputShape),
      stridesA,
      stridesB,
      computeStrides(outputShape),
    );
  }

  return output;
}

/**
 * Fast path for binary operations when no broadcasting is needed
 */
function executeBinaryOpFast(
  opType: string,
  arrayA:
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
  arrayB:
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
): void {
  const length = arrayA.length;

  // Handle bigint arrays separately (only if BOTH input AND output arrays are bigint)
  if (
    (arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array) &&
    (arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array) &&
    (arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array)
  ) {
    const bigA = arrayA as BigInt64Array | BigUint64Array;
    const bigB = arrayB as BigInt64Array | BigUint64Array;
    const bigOut = arrayOut as BigInt64Array | BigUint64Array;

    switch (opType) {
      case 'add':
        for (let i = 0; i < length; i++) {
          const a = bigA[i];
          const b = bigB[i];
          if (a !== undefined && b !== undefined) {
            bigOut[i] = a + b;
          }
        }
        break;
      case 'sub':
        for (let i = 0; i < length; i++) {
          const a = bigA[i];
          const b = bigB[i];
          if (a !== undefined && b !== undefined) {
            bigOut[i] = a - b;
          }
        }
        break;
      case 'mul':
        for (let i = 0; i < length; i++) {
          const a = bigA[i];
          const b = bigB[i];
          if (a !== undefined && b !== undefined) {
            bigOut[i] = a * b;
          }
        }
        break;
      case 'div':
        for (let i = 0; i < length; i++) {
          const a = bigA[i];
          const b = bigB[i];
          if (a !== undefined && b !== undefined) {
            if (b === 0n) {
              // Handle division by zero for BigInt (similar to regular float behavior)
              bigOut[i] = a > 0n ? 9223372036854775807n : -9223372036854775808n; // Max/min BigInt values
            } else {
              bigOut[i] = a / b;
            }
          }
        }
        break;
      default:
        throw new Error(`Unsupported binary operation: ${opType}`);
    }
  } else if (arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array) {
    // Handle cases where output is BigInt but inputs might not be BigInt
    const bigOut = arrayOut as BigInt64Array | BigUint64Array;

    switch (opType) {
      case 'add':
        for (let i = 0; i < length; i++) {
          const a = getBigIntValue(arrayA, i);
          const b = getBigIntValue(arrayB, i);
          if (a !== undefined && b !== undefined) {
            bigOut[i] = a + b;
          }
        }
        break;
      case 'sub':
        for (let i = 0; i < length; i++) {
          const a = getBigIntValue(arrayA, i);
          const b = getBigIntValue(arrayB, i);
          if (a !== undefined && b !== undefined) {
            bigOut[i] = a - b;
          }
        }
        break;
      case 'mul':
        for (let i = 0; i < length; i++) {
          const a = getBigIntValue(arrayA, i);
          const b = getBigIntValue(arrayB, i);
          if (a !== undefined && b !== undefined) {
            bigOut[i] = a * b;
          }
        }
        break;
      case 'div':
        for (let i = 0; i < length; i++) {
          const a = getBigIntValue(arrayA, i);
          const b = getBigIntValue(arrayB, i);
          if (a !== undefined && b !== undefined) {
            if (b === 0n) {
              // Handle division by zero for BigInt (similar to regular float behavior)
              bigOut[i] = a > 0n ? 9223372036854775807n : -9223372036854775808n; // Max/min BigInt values
            } else {
              bigOut[i] = a / b;
            }
          }
        }
        break;
      default:
        throw new Error(`Unsupported binary operation: ${opType}`);
    }
  } else {
    // Handle numeric arrays (regular number types)
    const numOut = arrayOut as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;

    switch (opType) {
      case 'add':
        for (let i = 0; i < length; i++) {
          const a = getNumberValue(arrayA, i);
          const b = getNumberValue(arrayB, i);
          numOut[i] = a + b;
        }
        break;
      case 'sub':
        for (let i = 0; i < length; i++) {
          const a = getNumberValue(arrayA, i);
          const b = getNumberValue(arrayB, i);
          numOut[i] = a - b;
        }
        break;
      case 'mul':
        for (let i = 0; i < length; i++) {
          const a = getNumberValue(arrayA, i);
          const b = getNumberValue(arrayB, i);
          numOut[i] = a * b;
        }
        break;
      case 'div':
        for (let i = 0; i < length; i++) {
          const a = getNumberValue(arrayA, i);
          const b = getNumberValue(arrayB, i);
          numOut[i] = a / b;
        }
        break;
      default:
        throw new Error(`Unsupported binary operation: ${opType}`);
    }
  }
}

/**
 * Slow path for binary operations with broadcasting
 */
function executeBinaryOpBroadcast(
  opType: string,
  arrayA:
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
  arrayB:
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
  _stridesOut: number[],
): void {
  // Use broadcasting iterator
  let outIdx = 0;
  for (const { inputIndices } of broadcastIndices(shapeOut, [shapeA, shapeB])) {
    const indicesA = inputIndices[0];
    const indicesB = inputIndices[1];
    if (indicesA === undefined || indicesB === undefined) {
      continue;
    }
    const idxA = computeFlatIndex(indicesA, stridesA);
    const idxB = computeFlatIndex(indicesB, stridesB);

    // Handle bigint arrays separately (all BigInt inputs and output)
    if (
      (arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array) &&
      (arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array) &&
      (arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array)
    ) {
      const bigA = arrayA as BigInt64Array | BigUint64Array;
      const bigB = arrayB as BigInt64Array | BigUint64Array;
      const bigOut = arrayOut as BigInt64Array | BigUint64Array;
      const valA = bigA[idxA];
      const valB = bigB[idxB];
      if (valA === undefined || valB === undefined) {
        continue;
      }

      switch (opType) {
        case 'add':
          bigOut[outIdx] = valA + valB;
          break;
        case 'sub':
          bigOut[outIdx] = valA - valB;
          break;
        case 'mul':
          bigOut[outIdx] = valA * valB;
          break;
        case 'div':
          if (valB === 0n) {
            // Handle division by zero for BigInt (similar to regular float behavior)
            bigOut[outIdx] = valA > 0n ? 9223372036854775807n : -9223372036854775808n; // Max/min BigInt values
          } else {
            bigOut[outIdx] = valA / valB;
          }
          break;
        default:
          throw new Error(`Unsupported binary operation: ${opType}`);
      }
    } else if (arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array) {
      // Handle cases where output is BigInt but inputs might not be BigInt
      const bigOut = arrayOut as BigInt64Array | BigUint64Array;

      // Helper function to get BigInt value (handles Number -> BigInt conversion)
      const getBigIntValue = (
        arr: ArrayLike<number | bigint>,
        index: number,
      ): bigint | undefined => {
        const val = arr[index];
        if (val === undefined) {
          return undefined;
        }
        return typeof val === 'bigint' ? val : BigInt(Math.trunc(val));
      };

      const valA = getBigIntValue(arrayA, idxA);
      const valB = getBigIntValue(arrayB, idxB);
      if (valA === undefined || valB === undefined) {
        continue;
      }

      switch (opType) {
        case 'add':
          bigOut[outIdx] = valA + valB;
          break;
        case 'sub':
          bigOut[outIdx] = valA - valB;
          break;
        case 'mul':
          bigOut[outIdx] = valA * valB;
          break;
        case 'div':
          if (valB === 0n) {
            // Handle division by zero for BigInt (similar to regular float behavior)
            bigOut[outIdx] = valA > 0n ? 9223372036854775807n : -9223372036854775808n; // Max/min BigInt values
          } else {
            bigOut[outIdx] = valA / valB;
          }
          break;
        default:
          throw new Error(`Unsupported binary operation: ${opType}`);
      }
    } else {
      // Handle numeric arrays (regular number types)
      const numOut = arrayOut as
        | Int8Array
        | Uint8Array
        | Int16Array
        | Uint16Array
        | Int32Array
        | Uint32Array
        | Float32Array
        | Float64Array;

      // Helper function to get numeric value (handles BigInt conversion)
      const getValue = (arr: ArrayLike<number | bigint>, index: number): number => {
        const val = arr[index];
        if (val === undefined) {
          return NaN;
        }
        return typeof val === 'bigint' ? Number(val) : val;
      };

      const valA = getValue(arrayA, idxA);
      const valB = getValue(arrayB, idxB);

      switch (opType) {
        case 'add':
          numOut[outIdx] = valA + valB;
          break;
        case 'sub':
          numOut[outIdx] = valA - valB;
          break;
        case 'mul':
          numOut[outIdx] = valA * valB;
          break;
        case 'div':
          numOut[outIdx] = valA / valB;
          break;
        default:
          throw new Error(`Unsupported binary operation: ${opType}`);
      }
    }

    outIdx++;
  }
}
