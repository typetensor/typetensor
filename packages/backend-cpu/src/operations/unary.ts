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
 * Negation operation
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
  // Handle bigint arrays separately
  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    const bigInput = input as BigInt64Array | BigUint64Array;
    const bigOutput = output as BigInt64Array | BigUint64Array;
    for (let i = 0; i < bigInput.length; i++) {
      const val = bigInput[i];
      if (val !== undefined) {
        bigOutput[i] = -val;
      }
    }
  } else {
    // Handle regular number arrays
    const numInput = input as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    const numOutput = output as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    for (let i = 0; i < numInput.length; i++) {
      const val = numInput[i];
      if (val !== undefined) {
        numOutput[i] = -val;
      }
    }
  }
}

/**
 * Absolute value operation
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
  // Handle bigint arrays separately
  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    const bigInput = input as BigInt64Array | BigUint64Array;
    const bigOutput = output as BigInt64Array | BigUint64Array;
    for (let i = 0; i < bigInput.length; i++) {
      const value = bigInput[i];
      if (value !== undefined) {
        bigOutput[i] = value < 0n ? -value : value;
      }
    }
  } else {
    // Handle regular number arrays
    const numInput = input as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    const numOutput = output as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    for (let i = 0; i < numInput.length; i++) {
      const val = numInput[i];
      if (val !== undefined) {
        numOutput[i] = Math.abs(val);
      }
    }
  }
}

/**
 * Square operation
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
  // Handle bigint arrays separately
  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    const bigInput = input as BigInt64Array | BigUint64Array;
    const bigOutput = output as BigInt64Array | BigUint64Array;
    for (let i = 0; i < bigInput.length; i++) {
      const value = bigInput[i];
      if (value !== undefined) {
        bigOutput[i] = value * value;
      }
    }
  } else {
    // Handle regular number arrays
    const numInput = input as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    const numOutput = output as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    for (let i = 0; i < numInput.length; i++) {
      const value = numInput[i];
      if (value !== undefined) {
        numOutput[i] = value * value;
      }
    }
  }
}

/**
 * Square root operation (only for float types)
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
  if (!(output instanceof Float32Array || output instanceof Float64Array)) {
    throw new Error('sqrt operation requires float output type');
  }

  // Input can be any numeric type, but output must be float
  const floatOutput = output as Float32Array | Float64Array;

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    // Convert bigint to number for sqrt
    for (let i = 0; i < input.length; i++) {
      const val = input[i];
      if (val !== undefined) {
        floatOutput[i] = Math.sqrt(Number(val));
      }
    }
  } else {
    const numInput = input as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    for (let i = 0; i < numInput.length; i++) {
      const val = numInput[i];
      if (val !== undefined) {
        floatOutput[i] = Math.sqrt(val);
      }
    }
  }
}

/**
 * Exponential operation (only for float types)
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
  if (!(output instanceof Float32Array || output instanceof Float64Array)) {
    throw new Error('exp operation requires float output type');
  }

  const floatOutput = output as Float32Array | Float64Array;

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    for (let i = 0; i < input.length; i++) {
      const val = input[i];
      if (val !== undefined) {
        floatOutput[i] = Math.exp(Number(val));
      }
    }
  } else {
    const numInput = input as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    for (let i = 0; i < numInput.length; i++) {
      const val = numInput[i];
      if (val !== undefined) {
        floatOutput[i] = Math.exp(val);
      }
    }
  }
}

/**
 * Natural logarithm operation (only for float types)
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
  if (!(output instanceof Float32Array || output instanceof Float64Array)) {
    throw new Error('log operation requires float output type');
  }

  const floatOutput = output as Float32Array | Float64Array;

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    for (let i = 0; i < input.length; i++) {
      const val = input[i];
      if (val !== undefined) {
        floatOutput[i] = Math.log(Number(val));
      }
    }
  } else {
    const numInput = input as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    for (let i = 0; i < numInput.length; i++) {
      const val = numInput[i];
      if (val !== undefined) {
        floatOutput[i] = Math.log(val);
      }
    }
  }
}

/**
 * Sine operation (only for float types)
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
  if (!(output instanceof Float32Array || output instanceof Float64Array)) {
    throw new Error('sin operation requires float output type');
  }

  const floatOutput = output as Float32Array | Float64Array;

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    for (let i = 0; i < input.length; i++) {
      const val = input[i];
      if (val !== undefined) {
        floatOutput[i] = Math.sin(Number(val));
      }
    }
  } else {
    const numInput = input as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    for (let i = 0; i < numInput.length; i++) {
      const val = numInput[i];
      if (val !== undefined) {
        floatOutput[i] = Math.sin(val);
      }
    }
  }
}

/**
 * Cosine operation (only for float types)
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
  if (!(output instanceof Float32Array || output instanceof Float64Array)) {
    throw new Error('cos operation requires float output type');
  }

  const floatOutput = output as Float32Array | Float64Array;

  if (input instanceof BigInt64Array || input instanceof BigUint64Array) {
    for (let i = 0; i < input.length; i++) {
      const val = input[i];
      if (val !== undefined) {
        floatOutput[i] = Math.cos(Number(val));
      }
    }
  } else {
    const numInput = input as
      | Int8Array
      | Uint8Array
      | Int16Array
      | Uint16Array
      | Int32Array
      | Uint32Array
      | Float32Array
      | Float64Array;
    for (let i = 0; i < numInput.length; i++) {
      const val = numInput[i];
      if (val !== undefined) {
        floatOutput[i] = Math.cos(val);
      }
    }
  }
}
