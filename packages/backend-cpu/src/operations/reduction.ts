/**
 * Reduction operations for CPU backend
 *
 * Implements sum and mean operations that reduce tensor dimensions.
 * These operations compute aggregates along specified axes.
 */

import type { Device, DeviceData, AnyStorageTransformation } from '@typetensor/core';
import type { CPUDeviceData } from '../data';
import { createTypedArray } from '../utils';

/**
 * Execute sum reduction operation
 *
 * @param device - CPU device instance
 * @param op - Sum operation descriptor with metadata
 * @param input - Input tensor data
 * @param output - Optional pre-allocated output buffer
 * @returns Result tensor data containing sum
 */
export async function executeSumOp(
  device: Device,
  op: AnyStorageTransformation & {
    __sumAxes: readonly number[] | undefined;
    __keepDims: boolean;
  },
  input: DeviceData,
  output?: DeviceData,
): Promise<DeviceData> {
  const inputStorage = op.__inputs[0];
  if (!inputStorage) {
    throw new Error('Sum operation requires input storage');
  }
  const inputShape = inputStorage.__shape;
  const outputShape = op.__output.__shape;
  const axes = op.__sumAxes;

  // Create output buffer if not provided
  const result = output || device.createData(op.__output.__size * op.__output.__dtype.__byteSize);
  const resultData = result as CPUDeviceData;
  const inputData = input as CPUDeviceData;

  // Get typed array views
  const inputView = createTypedArray(inputData.buffer, inputStorage.__dtype);
  const outputView = createTypedArray(resultData.buffer, op.__output.__dtype);

  // Handle different reduction cases
  if (axes === undefined) {
    // Global sum - sum all elements
    performGlobalSum(inputView, outputView);
  } else if (axes.length === 0) {
    // Empty axes - copy input to output
    performCopy(inputView, outputView);
  } else {
    // Reduction along specific axes
    performAxisReduction(inputView, outputView, inputShape, outputShape, axes, false);
  }

  return result;
}

/**
 * Execute mean reduction operation
 *
 * @param device - CPU device instance
 * @param op - Mean operation descriptor with metadata
 * @param input - Input tensor data
 * @param output - Optional pre-allocated output buffer
 * @returns Result tensor data containing mean
 */
export async function executeMeanOp(
  device: Device,
  op: AnyStorageTransformation & {
    __meanAxes: readonly number[] | undefined;
    __keepDims: boolean;
  },
  input: DeviceData,
  output?: DeviceData,
): Promise<DeviceData> {
  const inputStorage = op.__inputs[0];
  if (!inputStorage) {
    throw new Error('Mean operation requires input storage');
  }
  const inputShape = inputStorage.__shape;
  const outputShape = op.__output.__shape;
  const axes = op.__meanAxes;

  // Create output buffer if not provided
  const result = output || device.createData(op.__output.__size * op.__output.__dtype.__byteSize);
  const resultData = result as CPUDeviceData;
  const inputData = input as CPUDeviceData;

  // Get typed array views
  const inputView = createTypedArray(inputData.buffer, inputStorage.__dtype);
  const outputView = createTypedArray(resultData.buffer, op.__output.__dtype);

  // Calculate the number of elements being averaged
  let numElements: number;
  if (axes === undefined) {
    numElements = inputView.length;
  } else if (axes.length === 0) {
    numElements = 1;
  } else {
    numElements = 1;
    for (const axis of axes) {
      const normalizedAxis = axis < 0 ? inputShape.length + axis : axis;
      const dimSize = inputShape[normalizedAxis];
      if (dimSize === undefined) {
        throw new Error(`Invalid axis ${axis} for shape with ${inputShape.length} dimensions`);
      }
      numElements *= dimSize;
    }
  }

  // Handle different reduction cases
  if (axes === undefined) {
    // Global mean - average all elements
    performGlobalMean(inputView, outputView, numElements);
  } else if (axes.length === 0) {
    // Empty axes - copy input to output
    performCopy(inputView, outputView);
  } else {
    // Reduction along specific axes
    performAxisReduction(inputView, outputView, inputShape, outputShape, axes, true, numElements);
  }

  return result;
}

/**
 * Perform global sum reduction
 */
function performGlobalSum(
  inputView: ArrayLike<number | bigint>,
  outputView: ArrayLike<number | bigint> & { [index: number]: number | bigint },
): void {
  let sum: number | bigint = 0;
  for (let i = 0; i < inputView.length; i++) {
    const val = inputView[i];
    if (val !== undefined) {
      sum = typeof sum === 'bigint' || typeof val === 'bigint' ? 
        BigInt(sum) + BigInt(val) : Number(sum) + Number(val);
    }
  }
  outputView[0] = sum;
}

/**
 * Perform global mean reduction
 */
function performGlobalMean(
  inputView: ArrayLike<number | bigint>,
  outputView: ArrayLike<number | bigint> & { [index: number]: number | bigint },
  numElements: number,
): void {
  let sum: number | bigint = 0;
  for (let i = 0; i < inputView.length; i++) {
    const val = inputView[i];
    if (val !== undefined) {
      sum = typeof sum === 'bigint' || typeof val === 'bigint' ? 
        BigInt(sum) + BigInt(val) : Number(sum) + Number(val);
    }
  }
  outputView[0] = typeof sum === 'bigint' ? sum / BigInt(numElements) : sum / numElements;
}

/**
 * Perform copy operation
 */
function performCopy(
  inputView: ArrayLike<number | bigint>,
  outputView: ArrayLike<number | bigint> & { [index: number]: number | bigint },
): void {
  for (let i = 0; i < inputView.length; i++) {
    const val = inputView[i];
    if (val !== undefined) {
      outputView[i] = val;
    }
  }
}

/**
 * Perform reduction along specific axes
 *
 * @param inputView - Input typed array view
 * @param outputView - Output typed array view
 * @param inputShape - Input tensor shape
 * @param outputShape - Output tensor shape
 * @param axes - Axes to reduce along
 * @param isMean - Whether to compute mean (divide by count) or sum
 * @param meanDivisor - For mean operations, the total number of elements per output element
 */
function performAxisReduction(
  inputView: ArrayLike<number | bigint>,
  outputView: ArrayLike<number | bigint> & { [index: number]: number | bigint },
  inputShape: readonly number[],
  outputShape: readonly number[],
  axes: readonly number[],
  isMean: boolean = false,
  meanDivisor?: number,
): void {
  // Normalize negative axes
  const normalizedAxes = axes.map(axis => axis < 0 ? inputShape.length + axis : axis);
  const axisSet = new Set(normalizedAxes);

  // Initialize output to zero
  for (let i = 0; i < outputView.length; i++) {
    outputView[i] = 0;
  }

  // Compute input and output strides for efficient indexing
  const inputStrides = computeStrides(inputShape);
  const outputStrides = computeStrides(outputShape);

  // Iterate through all input elements
  const inputSize = inputView.length;
  for (let flatIdx = 0; flatIdx < inputSize; flatIdx++) {
    // Convert flat index to multi-dimensional coordinates
    const inputCoords = flatIndexToCoords(flatIdx, inputStrides);
    
    // Map to output coordinates by removing reduced dimensions
    const outputCoords: number[] = [];
    for (let dim = 0; dim < inputCoords.length; dim++) {
      if (!axisSet.has(dim)) {
        const coord = inputCoords[dim];
        if (coord !== undefined) {
          outputCoords.push(coord);
        }
      }
    }

    // Convert output coordinates to flat index
    const outputIdx = coordsToFlatIndex(outputCoords, outputStrides);
    
    // Accumulate the value
    const inputVal = inputView[flatIdx];
    const currentOutput = outputView[outputIdx];
    if (inputVal !== undefined && currentOutput !== undefined) {
      outputView[outputIdx] = typeof currentOutput === 'bigint' || typeof inputVal === 'bigint' ?
        BigInt(currentOutput) + BigInt(inputVal) : Number(currentOutput) + Number(inputVal);
    }
  }

  // Convert sums to means if needed
  if (isMean && meanDivisor !== undefined) {
    for (let i = 0; i < outputView.length; i++) {
      const val = outputView[i];
      if (val !== undefined) {
        outputView[i] = typeof val === 'bigint' ? val / BigInt(meanDivisor) : val / meanDivisor;
      }
    }
  }
}

/**
 * Compute strides for a given shape (C-order/row-major)
 */
function computeStrides(shape: readonly number[]): number[] {
  const strides: number[] = new Array(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    const dim = shape[i];
    if (dim !== undefined) {
      stride *= dim;
    }
  }
  return strides;
}

/**
 * Convert flat index to multi-dimensional coordinates
 */
function flatIndexToCoords(flatIdx: number, strides: number[]): number[] {
  const coords: number[] = new Array(strides.length);
  let remaining = flatIdx;
  
  for (let i = 0; i < strides.length; i++) {
    const stride = strides[i];
    if (stride !== undefined) {
      coords[i] = Math.floor(remaining / stride);
      remaining %= stride;
    }
  }
  
  return coords;
}

/**
 * Convert multi-dimensional coordinates to flat index
 */
function coordsToFlatIndex(coords: number[], strides: number[]): number {
  let flatIdx = 0;
  for (let i = 0; i < coords.length; i++) {
    const coord = coords[i];
    const stride = strides[i];
    if (coord !== undefined && stride !== undefined) {
      flatIdx += coord * stride;
    }
  }
  return flatIdx;
}