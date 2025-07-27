/**
 * Softmax operations for CPU backend
 *
 * Implements softmax and log-softmax operations with numerical stability.
 * These operations apply normalization along a specified axis.
 */

import type { Device, DeviceData, AnyStorageTransformation } from '@typetensor/core';
import type { CPUDeviceData } from '../data';
import { createTypedArray } from '../utils';

/**
 * Execute softmax operation on CPU
 *
 * @param backend - CPU backend instance
 * @param op - Operation descriptor with softmax axis metadata
 * @param input - Input tensor data
 * @param output - Optional pre-allocated output
 * @returns Result tensor data
 */
export async function executeSoftmaxOp(
  backend: Device,
  op: AnyStorageTransformation & { __softmaxAxis: number },
  input: DeviceData,
  output?: DeviceData,
): Promise<DeviceData> {
  const inputData = input as CPUDeviceData;
  const dtype = op.__output.__dtype;
  const size = op.__output.__size;
  const shape = op.__output.__shape;
  const axis = op.__softmaxAxis;

  // Create output if not provided
  output ??= backend.createData(size * dtype.__byteSize);
  const outputData = output as CPUDeviceData;

  // Create typed arrays for input and output
  const inputArray = createTypedArray(inputData.buffer, op.__inputs[0]?.__dtype);
  const outputArray = createTypedArray(outputData.buffer, dtype);

  if (!inputArray || !outputArray) {
    throw new Error('Failed to create typed arrays for softmax operation');
  }

  // Execute softmax with numerical stability
  executeSoftmax(inputArray as ArrayLike<number>, outputArray as { [index: number]: number; length: number }, shape, axis);

  return outputData;
}

/**
 * Execute log-softmax operation on CPU
 *
 * @param backend - CPU backend instance
 * @param op - Operation descriptor with log-softmax axis metadata
 * @param input - Input tensor data
 * @param output - Optional pre-allocated output
 * @returns Result tensor data
 */
export async function executeLogSoftmaxOp(
  backend: Device,
  op: AnyStorageTransformation & { __logSoftmaxAxis: number },
  input: DeviceData,
  output?: DeviceData,
): Promise<DeviceData> {
  const inputData = input as CPUDeviceData;
  const dtype = op.__output.__dtype;
  const size = op.__output.__size;
  const shape = op.__output.__shape;
  const axis = op.__logSoftmaxAxis;

  // Create output if not provided
  output ??= backend.createData(size * dtype.__byteSize);
  const outputData = output as CPUDeviceData;

  // Create typed arrays for input and output
  const inputArray = createTypedArray(inputData.buffer, op.__inputs[0]?.__dtype);
  const outputArray = createTypedArray(outputData.buffer, dtype);

  if (!inputArray || !outputArray) {
    throw new Error('Failed to create typed arrays for log-softmax operation');
  }

  // Execute log-softmax with numerical stability
  executeLogSoftmax(inputArray as ArrayLike<number>, outputArray as { [index: number]: number; length: number }, shape, axis);

  return outputData;
}

/**
 * Execute softmax along specified axis with numerical stability
 * 
 * Uses the numerically stable formulation:
 * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 */
function executeSoftmax(
  input: ArrayLike<number>,
  output: { [index: number]: number; length: number },
  shape: readonly number[],
  axis: number,
): void {
  const rank = shape.length;
  
  // Handle 1D case (simple softmax over entire array)
  if (rank === 1) {
    executeSoftmax1D(input, output);
    return;
  }

  // Calculate stride and dimension info for the specified axis
  const axisSize = shape[axis];
  if (axisSize === undefined) {
    throw new Error(`Invalid axis ${axis} for shape with length ${shape.length}`);
  }
  const strides = computeStrides(shape);
  const axisStride = strides[axis];
  if (axisStride === undefined) {
    throw new Error(`Failed to compute stride for axis ${axis}`);
  }
  
  // Calculate number of softmax operations needed
  const totalSize = input.length;
  const axisElements = axisSize;
  const numOperations = totalSize / axisElements;

  // Iterate through each slice along the axis
  for (let op = 0; op < numOperations; op++) {
    const baseIdx = getBaseIndex(op, shape, axis, strides);
    
    // Find max for numerical stability
    let maxVal = -Infinity;
    for (let i = 0; i < axisElements; i++) {
      const idx = baseIdx + i * axisStride;
      const val = input[idx];
      if (val !== undefined && val > maxVal) {
        maxVal = val;
      }
    }

    // Compute exp(x - max) and sum
    let sum = 0;
    for (let i = 0; i < axisElements; i++) {
      const idx = baseIdx + i * axisStride;
      const val = input[idx];
      if (val !== undefined) {
        const expVal = Math.exp(val - maxVal);
        output[idx] = expVal;
        sum += expVal;
      }
    }

    // Normalize by sum
    if (sum > 0) {
      for (let i = 0; i < axisElements; i++) {
        const idx = baseIdx + i * axisStride;
        const currentValue = output[idx];
        if (currentValue !== undefined) {
          output[idx] = currentValue / sum;
        }
      }
    }
  }
}

/**
 * Execute log-softmax along specified axis with numerical stability
 * 
 * Uses the numerically stable formulation:
 * log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
 */
function executeLogSoftmax(
  input: ArrayLike<number>,
  output: { [index: number]: number; length: number },
  shape: readonly number[],
  axis: number,
): void {
  const rank = shape.length;
  
  // Handle 1D case (simple log-softmax over entire array)
  if (rank === 1) {
    executeLogSoftmax1D(input, output);
    return;
  }

  // Calculate stride and dimension info for the specified axis
  const axisSize = shape[axis];
  if (axisSize === undefined) {
    throw new Error(`Invalid axis ${axis} for shape with length ${shape.length}`);
  }
  const strides = computeStrides(shape);
  const axisStride = strides[axis];
  if (axisStride === undefined) {
    throw new Error(`Failed to compute stride for axis ${axis}`);
  }
  
  // Calculate number of softmax operations needed
  const totalSize = input.length;
  const axisElements = axisSize;
  const numOperations = totalSize / axisElements;

  // Iterate through each slice along the axis
  for (let op = 0; op < numOperations; op++) {
    const baseIdx = getBaseIndex(op, shape, axis, strides);
    
    // Find max for numerical stability
    let maxVal = -Infinity;
    for (let i = 0; i < axisElements; i++) {
      const idx = baseIdx + i * axisStride;
      const val = input[idx];
      if (val !== undefined && val > maxVal) {
        maxVal = val;
      }
    }

    // Compute sum of exp(x - max)
    let sum = 0;
    for (let i = 0; i < axisElements; i++) {
      const idx = baseIdx + i * axisStride;
      const val = input[idx];
      if (val !== undefined) {
        sum += Math.exp(val - maxVal);
      }
    }

    // Compute log_softmax = x - max - log(sum)
    const logSum = Math.log(sum);
    for (let i = 0; i < axisElements; i++) {
      const idx = baseIdx + i * axisStride;
      const val = input[idx];
      if (val !== undefined) {
        output[idx] = val - maxVal - logSum;
      }
    }
  }
}

/**
 * Simple 1D softmax implementation
 */
function executeSoftmax1D(
  input: ArrayLike<number>,
  output: { [index: number]: number; length: number },
): void {
  // Find max for numerical stability
  let maxVal = -Infinity;
  for (let i = 0; i < input.length; i++) {
    const val = input[i];
    if (val !== undefined && val > maxVal) {
      maxVal = val;
    }
  }

  // Compute exp(x - max) and sum
  let sum = 0;
  for (let i = 0; i < input.length; i++) {
    const val = input[i];
    if (val !== undefined) {
      const expVal = Math.exp(val - maxVal);
      output[i] = expVal;
      sum += expVal;
    }
  }

  // Normalize by sum
  if (sum > 0) {
    for (let i = 0; i < input.length; i++) {
      const currentValue = output[i];
      if (currentValue !== undefined) {
        output[i] = currentValue / sum;
      }
    }
  }
}

/**
 * Simple 1D log-softmax implementation
 */
function executeLogSoftmax1D(
  input: ArrayLike<number>,
  output: { [index: number]: number; length: number },
): void {
  // Find max for numerical stability
  let maxVal = -Infinity;
  for (let i = 0; i < input.length; i++) {
    const val = input[i];
    if (val !== undefined && val > maxVal) {
      maxVal = val;
    }
  }

  // Compute sum of exp(x - max)
  let sum = 0;
  for (let i = 0; i < input.length; i++) {
    const val = input[i];
    if (val !== undefined) {
      sum += Math.exp(val - maxVal);
    }
  }

  // Compute log_softmax = x - max - log(sum)
  const logSum = Math.log(sum);
  for (let i = 0; i < input.length; i++) {
    const val = input[i];
    if (val !== undefined) {
      output[i] = val - maxVal - logSum;
    }
  }
}

/**
 * Compute strides for row-major (C-order) layout
 */
function computeStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length);
  let stride = 1;
  
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    const dim = shape[i];
    if (dim === undefined) {
      throw new Error(`Shape dimension at index ${i} is undefined`);
    }
    stride *= dim;
  }
  
  return strides;
}

/**
 * Get the base index for a slice operation along the specified axis
 */
function getBaseIndex(
  operationIndex: number,
  shape: readonly number[],
  axis: number,
  strides: number[],
): number {
  const rank = shape.length;
  let remaining = operationIndex;
  let baseIdx = 0;

  for (let dim = 0; dim < rank; dim++) {
    if (dim === axis) {
      continue; // Skip the axis dimension
    }
    
    // Calculate stride for this reduced dimension
    const dimSize = shape[dim];
    const stride = strides[dim];
    if (dimSize === undefined || stride === undefined) {
      throw new Error(`Invalid dimension or stride at index ${dim}`);
    }
    
    const coord = remaining % dimSize;
    remaining = Math.floor(remaining / dimSize);
    baseIdx += coord * stride;
  }

  return baseIdx;
}