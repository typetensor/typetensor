/**
 * CPU device for TypeTensor
 *
 * @module @typetensor/backend-cpu
 *
 * This module provides a CPU-based device implementation for tensor operations.
 */

import { CPUDevice } from './device';

// Export the device class
export { CPUDevice } from './device';
export { CPUDeviceData } from './data';

// Export the singleton CPU device instance
export const cpu = new CPUDevice();

// Export utils for testing or advanced usage
export * from './utils';
