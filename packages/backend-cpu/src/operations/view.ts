/**
 * View operations for CPU backend
 *
 * Implements operations that create different views of tensor data
 * without copying (reshape, flatten, view).
 */

import type { Device, DeviceData, AnyStorageTransformation } from '@typetensor/core';

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
