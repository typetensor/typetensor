/**
 * Unary operations for CPU backend
 *
 * Implements element-wise unary operations like negation, absolute value, etc.
 */

import type { Device, DeviceData, AnyStorageTransformation } from '@typetensor/core';
import type { CPUDeviceData } from '../data';
import { createTypedArray } from '../utils';

/**
 * Execute a unary operation on CPU
 *
 * @param backend - CPU backend instance
 * @param op - Operation descriptor
 * @param input - Input tensor data
 * @param output - Optional pre-allocated output
 * @returns Result tensor data
 */
export async function executeUnaryOp(
  backend: Device,
  op: AnyStorageTransformation,
  input: DeviceData,
  output?: DeviceData,
): Promise<DeviceData> {
  const inputData = input as CPUDeviceData;
  const dtype = op.__output.__dtype;
  const size = op.__output.__size;

  // Create output if not provided
  output ??= backend.createData(size * dtype.__byteSize);
  const outputData = output as CPUDeviceData;

  // Create typed arrays for input and output
  const inputArray = createTypedArray(inputData.buffer, op.__inputs[0]?.__dtype);
  const outputArray = createTypedArray(outputData.buffer, dtype);

  // Dispatch to specific operation
  switch (op.__op) {
    case 'neg':
      executeNeg(inputArray, outputArray);
      break;
    case 'abs':
      executeAbs(inputArray, outputArray);
      break;
    case 'square':
      executeSquare(inputArray, outputArray);
      break;
    case 'sqrt':
      executeSqrt(inputArray, outputArray);
      break;
    case 'exp':
      executeExp(inputArray, outputArray);
      break;
    case 'log':
      executeLog(inputArray, outputArray);
      break;
    case 'sin':
      executeSin(inputArray, outputArray);
      break;
    case 'cos':
      executeCos(inputArray, outputArray);
      break;
    default:
      throw new Error(`Unsupported unary operation: ${op.__op}`);
  }

  return output;
}

/**
 * Negation operation (OPTIMIZED)
 */
function executeNeg(
  input:
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
  output:
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
  // OPTIMIZED: Branch once, then tight loop without checks
  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    const bigInput = input as BigInt64Array | BigUint64Array;
    const bigOutput = output as BigInt64Array | BigUint64Array;
    for (let i = 0; i < bigInput.length; i++) {
      bigOutput[i] = -bigInput[i]!;
    }
  } else {
    // OPTIMIZED: Direct negation without redundant checks or casting
    const numInput = input as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    const numOutput = output as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    for (let i = 0; i < numInput.length; i++) {
      numOutput[i] = -numInput[i]!;
    }
  }
}

/**
 * Absolute value operation (OPTIMIZED)
 */
function executeAbs(
  input:
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
  output:
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
  // OPTIMIZED: Branch once, then tight loop
  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    const bigInput = input as BigInt64Array | BigUint64Array;
    const bigOutput = output as BigInt64Array | BigUint64Array;
    for (let i = 0; i < bigInput.length; i++) {
      const value = bigInput[i]!;
      bigOutput[i] = value < 0n ? -value : value;
    }
  } else {
    // OPTIMIZED: Use Math.abs in tight loop
    const numInput = input as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    const numOutput = output as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    for (let i = 0; i < numInput.length; i++) {
      numOutput[i] = Math.abs(numInput[i]!);
    }
  }
}

/**
 * Square operation (OPTIMIZED)
 */
function executeSquare(
  input:
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
  output:
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
  // OPTIMIZED: Branch once, then tight loop
  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    const bigInput = input as BigInt64Array | BigUint64Array;
    const bigOutput = output as BigInt64Array | BigUint64Array;
    for (let i = 0; i < bigInput.length; i++) {
      const value = bigInput[i]!;
      bigOutput[i] = value * value;
    }
  } else {
    // OPTIMIZED: Direct multiplication in tight loop
    const numInput = input as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    const numOutput = output as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    for (let i = 0; i < numInput.length; i++) {
      const value = numInput[i]!;
      numOutput[i] = value * value;
    }
  }
}

/**
 * Square root operation (OPTIMIZED - only for float types)
 */
function executeSqrt(
  input:
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
  output:
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
  // OPTIMIZED: Single type check, tight loops
  const floatOutput = output as Float32Array | Float64Array;

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    // Convert bigint to number for sqrt
    for (let i = 0; i < input.length; i++) {
      floatOutput[i] = Math.sqrt(Number(input[i]!));
    }
  } else {
    const numInput = input as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    for (let i = 0; i < numInput.length; i++) {
      floatOutput[i] = Math.sqrt(numInput[i]!);
    }
  }
}

/**
 * Exponential operation (OPTIMIZED - only for float types)
 */
function executeExp(
  input:
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
  output:
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
  // OPTIMIZED: Single cast, tight loops
  const floatOutput = output as Float32Array | Float64Array;

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    for (let i = 0; i < input.length; i++) {
      floatOutput[i] = Math.exp(Number(input[i]!));
    }
  } else {
    const numInput = input as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    for (let i = 0; i < numInput.length; i++) {
      floatOutput[i] = Math.exp(numInput[i]!);
    }
  }
}

/**
 * Natural logarithm operation (OPTIMIZED - only for float types)
 */
function executeLog(
  input:
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
  output:
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
  // OPTIMIZED: Single cast, tight loops
  const floatOutput = output as Float32Array | Float64Array;

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    for (let i = 0; i < input.length; i++) {
      floatOutput[i] = Math.log(Number(input[i]!));
    }
  } else {
    const numInput = input as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    for (let i = 0; i < numInput.length; i++) {
      floatOutput[i] = Math.log(numInput[i]!);
    }
  }
}

/**
 * Sine operation (OPTIMIZED - only for float types)
 */
function executeSin(
  input:
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
  output:
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
  // OPTIMIZED: Single cast, tight loops
  const floatOutput = output as Float32Array | Float64Array;

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    for (let i = 0; i < input.length; i++) {
      floatOutput[i] = Math.sin(Number(input[i]!));
    }
  } else {
    const numInput = input as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    for (let i = 0; i < numInput.length; i++) {
      floatOutput[i] = Math.sin(numInput[i]!);
    }
  }
}

/**
 * Cosine operation (OPTIMIZED - only for float types)
 */
function executeCos(
  input:
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
  output:
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
  // OPTIMIZED: Single cast, tight loops
  const floatOutput = output as Float32Array | Float64Array;

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    for (let i = 0; i < input.length; i++) {
      floatOutput[i] = Math.cos(Number(input[i]!));
    }
  } else {
    const numInput = input as Float32Array | Float64Array | Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array;
    for (let i = 0; i < numInput.length; i++) {
      floatOutput[i] = Math.cos(numInput[i]!);
    }
  }
}
