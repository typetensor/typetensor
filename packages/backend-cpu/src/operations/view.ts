/**
 * View operations for CPU backend
 *
 * Implements operations that create different views of tensor data.
 * - View operations (reshape, flatten, view): no copying, just metadata changes
 * - Slice operations: creates new data with copied sliced elements
 */

import type { Device, DeviceData, AnyStorageTransformation } from '@typetensor/core';
import type { SliceIndex } from '@typetensor/core';
import { createTypedArray, computeFlatIndex } from '../utils';

/**
 * Execute a view operation on CPU
 *
 * View operations don't create new data, they just return the input
 * with different metadata. The actual view logic is handled by the
 * tensor class at the storage transformation level.
 *
 * @param backend - CPU backend instance
 * @param op - Operation descriptor
 * @param input - Input tensor data
 * @returns Same tensor data (views share underlying buffer)
 */
export async function executeViewOp(
  _device: Device,
  op: AnyStorageTransformation,
  input: DeviceData,
): Promise<DeviceData> {
  // Validate operation type
  if (op.__op !== 'reshape' && op.__op !== 'flatten' && op.__op !== 'view') {
    throw new Error(`Invalid view operation: ${op.__op}`);
  }

  // View operations return the same data handle
  // The tensor class handles the metadata transformation
  return input;
}

/**
 * Execute a slice operation on CPU
 *
 * Slice operations create new data by copying elements from the input tensor
 * based on the slice indices. Unlike view operations, this requires actual
 * data copying to create a contiguous result.
 *
 * @param device - CPU device instance
 * @param op - Slice operation descriptor
 * @param input - Input tensor data
 * @returns New tensor data with sliced elements
 */
export async function executeSliceOp(
  device: Device,
  op: AnyStorageTransformation & { __op: 'slice' },
  input: DeviceData,
): Promise<DeviceData> {
  // Validate operation type
  if (op.__op !== 'slice') {
    throw new Error(`Invalid slice operation: ${op.__op}`);
  }

  // Extract slice information from operation metadata
  const sliceIndices = (op.__output as any).__sliceIndices as SliceIndex[];
  const inputStorage = op.__inputs[0];
  if (!inputStorage) {
    throw new Error('Slice operation missing input storage metadata');
  }

  const inputShape = inputStorage.__shape;
  const inputStrides = inputStorage.__strides;
  const dtype = inputStorage.__dtype;

  // Output metadata
  const outputShape = op.__output.__shape;
  const outputSize = op.__output.__size;

  // Create output buffer
  const outputByteLength = outputSize * dtype.__byteSize;
  const outputData = device.createData(outputByteLength);

  // Copy sliced data
  await copySlicedData(
    device,
    input,
    outputData,
    sliceIndices,
    inputShape,
    inputStrides,
    outputShape,
    dtype,
  );

  return outputData;
}

/**
 * Copy sliced data from input to output buffer
 *
 * This function handles the core slicing logic by iterating through the output
 * tensor elements and copying the corresponding input elements based on the
 * slice indices.
 *
 * @param device - CPU device for data access
 * @param input - Input tensor data
 * @param output - Output tensor data (pre-allocated)
 * @param sliceIndices - Array of slice indices for each dimension
 * @param inputShape - Shape of input tensor
 * @param inputStrides - Strides of input tensor
 * @param outputShape - Shape of output tensor
 * @param dtype - Data type
 */
async function copySlicedData(
  device: Device,
  input: DeviceData,
  output: DeviceData,
  sliceIndices: SliceIndex[],
  inputShape: readonly number[],
  inputStrides: readonly number[],
  outputShape: readonly number[],
  dtype: any,
): Promise<void> {
  // Read input and create output buffers
  const inputBuffer = await device.readData(input);
  const outputBuffer = new ArrayBuffer(output.byteLength);

  // Create typed arrays for efficient data access
  const inputArray = createTypedArray(inputBuffer, dtype);
  const outputArray = createTypedArray(outputBuffer, dtype);

  // Iterate through all output positions and copy corresponding input elements
  const totalOutputElements = outputShape.reduce((a, b) => a * b, 1);

  for (let outputFlatIndex = 0; outputFlatIndex < totalOutputElements; outputFlatIndex++) {
    // Convert output flat index to multi-dimensional indices
    const outputIndices = flatIndexToIndices(outputFlatIndex, outputShape);
    
    // Map output indices to input indices using slice specifications
    const inputIndices = mapOutputToInputIndices(outputIndices, sliceIndices, inputShape);
    
    // Convert input indices to flat index
    const inputFlatIndex = computeFlatIndex(inputIndices, inputStrides);
    
    // Copy the element - arrays are properly sized so access is safe
    (outputArray as any)[outputFlatIndex] = (inputArray as any)[inputFlatIndex];
  }

  // Write the result back to the output device data
  await device.writeData(output, outputBuffer);
}

/**
 * Convert flat index to multi-dimensional indices
 *
 * @param flatIndex - Flat array index
 * @param shape - Tensor shape
 * @returns Multi-dimensional indices
 */
function flatIndexToIndices(flatIndex: number, shape: readonly number[]): number[] {
  const indices: number[] = [];
  let remaining = flatIndex;

  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i];
    if (dim === undefined) {
      throw new Error(`Invalid shape dimension at index ${i}`);
    }
    const stride = shape.slice(i + 1).reduce((a, b) => a * b, 1);
    indices[i] = Math.floor(remaining / stride);
    remaining %= stride;
  }

  return indices;
}

/**
 * Map output indices to input indices using slice specifications
 *
 * @param outputIndices - Position in output tensor
 * @param sliceIndices - Slice specifications for each dimension
 * @param inputShape - Shape of input tensor
 * @returns Corresponding position in input tensor
 */
function mapOutputToInputIndices(
  outputIndices: number[],
  sliceIndices: SliceIndex[],
  inputShape: readonly number[],
): number[] {
  const inputIndices: number[] = [];
  let outputDim = 0;

  for (let inputDim = 0; inputDim < inputShape.length; inputDim++) {
    const sliceIndex = inputDim < sliceIndices.length ? sliceIndices[inputDim] : null;
    const inputSize = inputShape[inputDim];
    
    if (inputSize === undefined) {
      throw new Error(`Invalid input shape dimension at index ${inputDim}`);
    }

    if (typeof sliceIndex === 'number') {
      // Integer index: use the specified index directly
      const normalizedIndex = sliceIndex < 0 ? inputSize + sliceIndex : sliceIndex;
      inputIndices[inputDim] = normalizedIndex;
      // Don't increment outputDim - this dimension was removed
    } else if (sliceIndex === null) {
      // null: keep entire dimension, direct mapping
      const outputIndex = outputIndices[outputDim];
      if (outputIndex === undefined) {
        throw new Error(`Missing output index for dimension ${outputDim}`);
      }
      inputIndices[inputDim] = outputIndex;
      outputDim++;
    } else {
      // SliceSpec: apply start/stop/step transformation
      const outputIndex = outputIndices[outputDim];
      if (outputIndex === undefined) {
        throw new Error(`Missing output index for dimension ${outputDim}`);
      }
      
      if (sliceIndex && typeof sliceIndex === 'object') {
        const start = sliceIndex.start !== undefined ? 
          (sliceIndex.start < 0 ? inputSize + sliceIndex.start : sliceIndex.start) : 0;
        const step = sliceIndex.step ?? 1;
        
        inputIndices[inputDim] = start + outputIndex * step;
      } else {
        throw new Error(`Invalid slice index type for SliceSpec: ${typeof sliceIndex}`);
      }
      outputDim++;
    }
  }

  return inputIndices;
}
