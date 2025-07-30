/**
 * Softmax operations for WebGPU backend
 */

import type { DeviceData, AnyStorageTransformation } from '@typetensor/core';
import type { WebGPUDevice } from '../device';

/**
 * Execute softmax operation
 */
export async function executeSoftmaxOp(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  device: WebGPUDevice,
  op: AnyStorageTransformation & { __softmaxAxis: number },
  input: DeviceData,
  output?: DeviceData,
): Promise<DeviceData> {
  // TODO: Implement softmax with proper WGSL shader
  throw new Error('Softmax operation not yet implemented for WebGPU backend');
}

/**
 * Execute log-softmax operation
 */
export async function executeLogSoftmaxOp(
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  device: WebGPUDevice,
  op: AnyStorageTransformation & { __logSoftmaxAxis: number },
  input: DeviceData,
  output?: DeviceData,
): Promise<DeviceData> {
  // TODO: Implement log-softmax with proper WGSL shader
  throw new Error('Log-softmax operation not yet implemented for WebGPU backend');
}