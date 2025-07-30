/**
 * Binary operations for CPU backend
 *
 * Implements element-wise binary operations with broadcasting support.
 */

import type { Device, DeviceData, AnyStorageTransformation } from '@typetensor/core';
import type { CPUDeviceData } from '../data';
import { createTypedArray, broadcastIndices, computeFlatIndex, computeStrides } from '../utils';

// OPTIMIZED: Inline helper functions removed - using direct array access instead

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
 * Fast path for binary operations when no broadcasting is needed (OPTIMIZED)
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

  // OPTIMIZED: Branch once for type checking, then tight loops without redundant checks
  if (
    (arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array) &&
    (arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array) &&
    (arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array)
  ) {
    // Pure BigInt path - all arrays are BigInt types
    const bigA = arrayA as BigInt64Array | BigUint64Array;
    const bigB = arrayB as BigInt64Array | BigUint64Array;
    const bigOut = arrayOut as BigInt64Array | BigUint64Array;

    switch (opType) {
      case 'add':
        for (let i = 0; i < length; i++) {
          bigOut[i] = bigA[i]! + bigB[i]!;
        }
        break;
      case 'sub':
        for (let i = 0; i < length; i++) {
          bigOut[i] = bigA[i]! - bigB[i]!;
        }
        break;
      case 'mul':
        for (let i = 0; i < length; i++) {
          bigOut[i] = bigA[i]! * bigB[i]!;
        }
        break;
      case 'div':
        for (let i = 0; i < length; i++) {
          const a = bigA[i]!;
          const b = bigB[i]!;
          if (b === 0n) {
            bigOut[i] = a > 0n ? 9223372036854775807n : -9223372036854775808n;
          } else {
            bigOut[i] = a / b;
          }
        }
        break;
      default:
        throw new Error(`Unsupported binary operation: ${opType}`);
    }
  } else if (arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array) {
    // Mixed types with BigInt output - convert inputs to BigInt
    const bigOut = arrayOut as BigInt64Array | BigUint64Array;

    if (arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array) {
      // arrayA is BigInt, convert arrayB
      const bigA = arrayA as BigInt64Array | BigUint64Array;
      switch (opType) {
        case 'add':
          for (let i = 0; i < length; i++) {
            const b = arrayB[i];
            bigOut[i] = bigA[i]! + (typeof b === 'bigint' ? b : BigInt(Math.trunc(b!)));
          }
          break;
        case 'sub':
          for (let i = 0; i < length; i++) {
            const b = arrayB[i];
            bigOut[i] = bigA[i]! - (typeof b === 'bigint' ? b : BigInt(Math.trunc(b!)));
          }
          break;
        case 'mul':
          for (let i = 0; i < length; i++) {
            const b = arrayB[i];
            bigOut[i] = bigA[i]! * (typeof b === 'bigint' ? b : BigInt(Math.trunc(b!)));
          }
          break;
        case 'div':
          for (let i = 0; i < length; i++) {
            const a = bigA[i]!;
            const b =
              typeof arrayB[i] === 'bigint'
                ? (arrayB[i] as bigint)
                : BigInt(Math.trunc(arrayB[i] as number));
            if (b === 0n) {
              bigOut[i] = a > 0n ? 9223372036854775807n : -9223372036854775808n;
            } else {
              bigOut[i] = a / b;
            }
          }
          break;
        default:
          throw new Error(`Unsupported binary operation: ${opType}`);
      }
    } else {
      // arrayB is BigInt, convert arrayA
      const bigB = arrayB as BigInt64Array | BigUint64Array;
      switch (opType) {
        case 'add':
          for (let i = 0; i < length; i++) {
            const a = arrayA[i];
            bigOut[i] = (typeof a === 'bigint' ? a : BigInt(Math.trunc(a!))) + bigB[i]!;
          }
          break;
        case 'sub':
          for (let i = 0; i < length; i++) {
            const a = arrayA[i];
            bigOut[i] = (typeof a === 'bigint' ? a : BigInt(Math.trunc(a!))) - bigB[i]!;
          }
          break;
        case 'mul':
          for (let i = 0; i < length; i++) {
            const a = arrayA[i];
            bigOut[i] = (typeof a === 'bigint' ? a : BigInt(Math.trunc(a!))) * bigB[i]!;
          }
          break;
        case 'div':
          for (let i = 0; i < length; i++) {
            const a =
              typeof arrayA[i] === 'bigint'
                ? (arrayA[i] as unknown as bigint)
                : BigInt(Math.trunc(arrayA[i]!));
            const b = bigB[i]!;
            if (b === 0n) {
              bigOut[i] = a > 0n ? 9223372036854775807n : -9223372036854775808n;
            } else {
              bigOut[i] = a / b;
            }
          }
          break;
        default:
          throw new Error(`Unsupported binary operation: ${opType}`);
      }
    }
  } else {
    // Pure numeric path - all operations on numbers
    const numOut = arrayOut as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;

    // OPTIMIZED: Handle BigInt inputs by converting them once per array
    if (arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array) {
      if (arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array) {
        // Both inputs are BigInt, convert to numbers
        const bigA = arrayA as BigInt64Array | BigUint64Array;
        const bigB = arrayB as BigInt64Array | BigUint64Array;
        switch (opType) {
          case 'add':
            for (let i = 0; i < length; i++) {
              numOut[i] = Number(bigA[i]!) + Number(bigB[i]!);
            }
            break;
          case 'sub':
            for (let i = 0; i < length; i++) {
              numOut[i] = Number(bigA[i]!) - Number(bigB[i]!);
            }
            break;
          case 'mul':
            for (let i = 0; i < length; i++) {
              numOut[i] = Number(bigA[i]!) * Number(bigB[i]!);
            }
            break;
          case 'div':
            for (let i = 0; i < length; i++) {
              numOut[i] = Number(bigA[i]!) / Number(bigB[i]!);
            }
            break;
          default:
            throw new Error(`Unsupported binary operation: ${opType}`);
        }
      } else {
        // arrayA is BigInt, arrayB is numeric
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
        switch (opType) {
          case 'add':
            for (let i = 0; i < length; i++) {
              numOut[i] = Number(bigA[i]!) + numB[i]!;
            }
            break;
          case 'sub':
            for (let i = 0; i < length; i++) {
              numOut[i] = Number(bigA[i]!) - numB[i]!;
            }
            break;
          case 'mul':
            for (let i = 0; i < length; i++) {
              numOut[i] = Number(bigA[i]!) * numB[i]!;
            }
            break;
          case 'div':
            for (let i = 0; i < length; i++) {
              numOut[i] = Number(bigA[i]!) / numB[i]!;
            }
            break;
          default:
            throw new Error(`Unsupported binary operation: ${opType}`);
        }
      }
    } else if (arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array) {
      // arrayA is numeric, arrayB is BigInt
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
      switch (opType) {
        case 'add':
          for (let i = 0; i < length; i++) {
            numOut[i] = numA[i]! + Number(bigB[i]!);
          }
          break;
        case 'sub':
          for (let i = 0; i < length; i++) {
            numOut[i] = numA[i]! - Number(bigB[i]!);
          }
          break;
        case 'mul':
          for (let i = 0; i < length; i++) {
            numOut[i] = numA[i]! * Number(bigB[i]!);
          }
          break;
        case 'div':
          for (let i = 0; i < length; i++) {
            numOut[i] = numA[i]! / Number(bigB[i]!);
          }
          break;
        default:
          throw new Error(`Unsupported binary operation: ${opType}`);
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
      switch (opType) {
        case 'add':
          for (let i = 0; i < length; i++) {
            numOut[i] = numA[i]! + numB[i]!;
          }
          break;
        case 'sub':
          for (let i = 0; i < length; i++) {
            numOut[i] = numA[i]! - numB[i]!;
          }
          break;
        case 'mul':
          for (let i = 0; i < length; i++) {
            numOut[i] = numA[i]! * numB[i]!;
          }
          break;
        case 'div':
          for (let i = 0; i < length; i++) {
            numOut[i] = numA[i]! / numB[i]!;
          }
          break;
        default:
          throw new Error(`Unsupported binary operation: ${opType}`);
      }
    }
  }
}

/**
 * Slow path for binary operations with broadcasting (OPTIMIZED)
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
  // OPTIMIZED: Branch once for type checking, avoid repeated instanceof checks
  const isBigIntA = arrayA instanceof BigInt64Array || arrayA instanceof BigUint64Array;
  const isBigIntB = arrayB instanceof BigInt64Array || arrayB instanceof BigUint64Array;
  const isBigIntOut = arrayOut instanceof BigInt64Array || arrayOut instanceof BigUint64Array;

  let outIdx = 0;
  for (const { inputIndices } of broadcastIndices(shapeOut, [shapeA, shapeB])) {
    const indicesA = inputIndices[0];
    const indicesB = inputIndices[1];
    if (indicesA === undefined || indicesB === undefined) {
      continue;
    }
    const idxA = computeFlatIndex(indicesA, stridesA);
    const idxB = computeFlatIndex(indicesB, stridesB);

    if (isBigIntA && isBigIntB && isBigIntOut) {
      // Pure BigInt path
      const bigA = arrayA as BigInt64Array | BigUint64Array;
      const bigB = arrayB as BigInt64Array | BigUint64Array;
      const bigOut = arrayOut as BigInt64Array | BigUint64Array;
      const valA = bigA[idxA]!;
      const valB = bigB[idxB]!;

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
            bigOut[outIdx] = valA > 0n ? 9223372036854775807n : -9223372036854775808n;
          } else {
            bigOut[outIdx] = valA / valB;
          }
          break;
        default:
          throw new Error(`Unsupported binary operation: ${opType}`);
      }
    } else if (isBigIntOut) {
      // Mixed types with BigInt output
      const bigOut = arrayOut as BigInt64Array | BigUint64Array;

      // OPTIMIZED: Direct type conversion without helper functions
      const valA = isBigIntA
        ? (arrayA as BigInt64Array | BigUint64Array)[idxA]!
        : BigInt(Math.trunc((arrayA as any)[idxA]!));
      const valB = isBigIntB
        ? (arrayB as BigInt64Array | BigUint64Array)[idxB]!
        : BigInt(Math.trunc((arrayB as any)[idxB]!));

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
            bigOut[outIdx] = valA > 0n ? 9223372036854775807n : -9223372036854775808n;
          } else {
            bigOut[outIdx] = valA / valB;
          }
          break;
        default:
          throw new Error(`Unsupported binary operation: ${opType}`);
      }
    } else {
      // Numeric output path
      const numOut = arrayOut as
        | Int8Array
        | Uint8Array
        | Int16Array
        | Uint16Array
        | Int32Array
        | Uint32Array
        | Float32Array
        | Float64Array;

      // OPTIMIZED: Direct type conversion without helper functions
      const valA = isBigIntA
        ? Number((arrayA as BigInt64Array | BigUint64Array)[idxA]!)
        : (arrayA as any)[idxA]!;
      const valB = isBigIntB
        ? Number((arrayB as BigInt64Array | BigUint64Array)[idxB]!)
        : (arrayB as any)[idxB]!;

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
