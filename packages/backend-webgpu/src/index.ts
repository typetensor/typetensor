/**
 * WebGPU device for TypeTensor
 *
 * @module @typetensor/backend-webgpu
 *
 * This module provides a WebGPU-based device implementation for tensor operations.
 * It supports both browser and Node.js environments through appropriate WebGPU APIs.
 */

import { WebGPUDevice } from './device';
import { isWebGPUAvailable } from './utils';

// Export the device class
export { WebGPUDevice } from './device';
export { WebGPUDeviceData } from './data';

// Export utilities
export { isWebGPUAvailable } from './utils';

// Export shader utilities for advanced usage
export { ShaderCache } from './shaders/cache';
export { ShaderGenerator } from './shaders/generator';

/**
 * Create and initialize a WebGPU device
 * 
 * @returns Promise that resolves to a WebGPU device instance
 * @throws Error if WebGPU is not available in the current environment
 */
export async function webgpu(): Promise<WebGPUDevice> {
  if (!isWebGPUAvailable()) {
    throw new Error(
      'WebGPU is not available in this environment. ' +
      'Make sure you are using a WebGPU-enabled browser or have installed the "webgpu" package for Node.js.',
    );
  }

  return WebGPUDevice.create();
}

// For backwards compatibility
export const WebGPUBackend = WebGPUDevice;