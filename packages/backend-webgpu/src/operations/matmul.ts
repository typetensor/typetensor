/**
 * Matrix multiplication operation for WebGPU backend
 */

import type { DeviceData, AnyStorageTransformation } from '@typetensor/core';
import type { WebGPUDevice } from '../device';

/**
 * Execute matrix multiplication on WebGPU
 */
export async function executeMatmulOp(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  device: WebGPUDevice,
  op: AnyStorageTransformation,
  inputA: DeviceData,
  inputB: DeviceData,
  output?: DeviceData,
): Promise<DeviceData> {
  // TODO: Implement matrix multiplication with tiled WGSL shader
  throw new Error('Matrix multiplication not yet implemented for WebGPU backend');
}