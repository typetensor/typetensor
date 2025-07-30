/**
 * Device abstraction for tensor computation
 *
 * This module provides interfaces for compute devices that can execute
 * tensor operations. Devices handle both identification and computation.
 */

import type { AnyStorageTransformation, AllOperationTypes } from '../storage/layout';

// =============================================================================
// Operation Validation Helpers
// =============================================================================

/**
 * Utility type to validate that a device supports all required operations
 *
 * This helps catch missing operation implementations at compile time.
 * If a device doesn't support an operation, TypeScript will show which ones are missing.
 *
 * @template DeviceOps - The operations that the device claims to support
 *
 * @example
 * // This will show a compile error listing missing operations
 * type Validation = ValidateDeviceOperations<'add' | 'sub'>;
 * // Error: __missing_operations: 'create' | 'neg' | 'abs' | ...
 */
export type ValidateDeviceOperations<DeviceOps extends AllOperationTypes> =
  AllOperationTypes extends DeviceOps
    ? true
    : {
        __missing_operations: Exclude<AllOperationTypes, DeviceOps>;
        __error: 'Device implementation missing operations';
        __help: "Add cases for missing operations in your device's execute() method";
      };

// =============================================================================
// Device Interface
// =============================================================================

/**
 * Device interface for tensor computation
 *
 * Represents a compute device (CPU, GPU, etc.) that can execute tensor operations.
 * Devices are responsible for:
 * - Executing tensor operations described by StorageTransformation
 * - Managing device memory allocation and deallocation
 * - Transferring data between host and device
 *
 * @example
 * class CPUDevice implements Device {
 *   readonly id = 'cpu:0';
 *   readonly type = 'cpu';
 *
 *   async execute(op, inputs) {
 *     switch (op.__op) {
 *       case 'neg': return this.executeNeg(op, inputs[0]);
 *       case 'add': return this.executeAdd(op, inputs[0], inputs[1]);
 *       // ... more operations
 *     }
 *   }
 * }
 */
export interface Device {
  /** Unique identifier for this device instance */
  readonly id: string;

  /** Device type identifier (e.g., 'cpu', 'webgpu', 'cuda') */
  readonly type: string;

  /**
   * Execute a tensor operation
   *
   * Takes a StorageTransformation describing the operation and input data,
   * executes the computation, and returns the result data.
   *
   * @param op - Operation metadata from storage layer
   * @param inputs - Input tensor data (must be on this device)
   * @param output - Optional pre-allocated output buffer
   * @returns Promise resolving to result data
   *
   * @throws {Error} If operation is not supported
   * @throws {Error} If inputs are not on the correct device
   */
  execute<T extends AnyStorageTransformation>(
    op: T,
    inputs: DeviceData[],
    output?: DeviceData,
  ): Promise<DeviceData>;

  /**
   * Allocate data on this device
   *
   * @param byteLength - Number of bytes to allocate
   * @returns Newly allocated data handle
   *
   * @throws {Error} If allocation fails (e.g., out of memory)
   */
  createData(byteLength: number): DeviceData;

  /**
   * Create data directly with a buffer (optional optimization)
   *
   * @param buffer - Pre-existing ArrayBuffer to use
   * @returns Data handle wrapping the buffer
   */
  createDataWithBuffer?(buffer: ArrayBuffer): DeviceData;

  /**
   * Free device memory
   *
   * @param data - Data to dispose
   */
  disposeData(data: DeviceData): void;

  /**
   * Read data from device to host memory
   *
   * @param data - Device data to read
   * @returns Promise resolving to data as ArrayBuffer
   *
   * @throws {Error} If data is not from this device
   */
  readData(data: DeviceData): Promise<ArrayBuffer>;

  /**
   * Write data from host to device memory
   *
   * @param data - Device data handle to write to
   * @param buffer - Source data as ArrayBuffer
   * @returns Promise resolving when write is complete
   *
   * @throws {Error} If data is not from this device
   * @throws {Error} If buffer size doesn't match data size
   */
  writeData(data: DeviceData, buffer: ArrayBuffer): Promise<void>;

  /**
   * Check if this device supports non-contiguous tensors for a specific operation
   *
   * @param op - The operation type to check
   * @returns True if the operation can handle non-contiguous inputs, false otherwise
   *
   * @example
   * // CPU device might not support non-contiguous for most ops
   * cpuDevice.supportsNonContiguous('matmul'); // false
   * cpuDevice.supportsNonContiguous('neg'); // false
   *
   * // GPU device might support non-contiguous for all ops
   * gpuDevice.supportsNonContiguous('matmul'); // true
   * gpuDevice.supportsNonContiguous('neg'); // true
   */
  supportsNonContiguous(op: AllOperationTypes): boolean;
}

/**
 * Data handle for device memory
 *
 * Represents tensor data residing on a specific device. The actual
 * data representation is device-specific and opaque to the core library.
 *
 * @example
 * // CPU device might store ArrayBuffer
 * // WebGPU device might store GPUBuffer
 * // CUDA device might store device pointer
 */
export interface DeviceData {
  /** Device that manages this data */
  readonly device: Device;

  /** Size of the data in bytes */
  readonly byteLength: number;
}
