/**
 * CPU device implementation
 *
 * This module provides the main CPU device that executes tensor operations
 * on the CPU using JavaScript/TypeScript typed arrays.
 */

import type {
  Device,
  DeviceData,
  AnyStorageTransformation,
  ValidateDeviceOperations,
} from '@typetensor/core';
import { CPUDeviceData } from './data';
import { executeOperation } from './operations';

/**
 * CPU computation device
 *
 * Implements tensor operations on the CPU using TypedArrays for efficient
 * numerical computation in JavaScript/TypeScript.
 */
export class CPUDevice implements Device {
  /**
   * Device identifiers
   */
  readonly id = 'cpu:0';
  readonly type = 'cpu';

  /**
   * Compile-time validation that all operations are implemented
   * This will cause a TypeScript error if any operation is missing from executeOperation()
   */
  // @ts-expect-error: This property is used for compile-time validation only
  private _operationValidation: ValidateDeviceOperations<
    | 'create'
    | 'neg'
    | 'abs'
    | 'sin'
    | 'cos'
    | 'exp'
    | 'log'
    | 'sqrt'
    | 'square' // Unary ops
    | 'add'
    | 'sub'
    | 'mul'
    | 'div' // Binary ops
    | 'reshape'
    | 'view'
    | 'slice'
    | 'flatten' // View ops
    | 'matmul' // Matrix ops
  > = true;

  /**
   * Execute a tensor operation
   *
   * Dispatches to specific operation implementations based on the operation type.
   */
  async execute<T extends AnyStorageTransformation>(
    op: T,
    inputs: DeviceData[],
    output?: DeviceData,
  ): Promise<DeviceData> {
    // Validate inputs are from this device
    for (const input of inputs) {
      if (input.device.id !== this.id) {
        throw new Error(`Input tensor is on device ${input.device.id}, expected ${this.id}`);
      }
    }

    // Validate output if provided
    if (output && output.device.id !== this.id) {
      throw new Error(`Output tensor is on device ${output.device.id}, expected ${this.id}`);
    }

    // Execute the operation
    return executeOperation(this, op, inputs, output);
  }

  /**
   * Allocate data on the CPU
   */
  createData(byteLength: number): DeviceData {
    return new CPUDeviceData(this, byteLength);
  }

  /**
   * Free CPU memory
   *
   * For CPU backend, this is mostly a no-op as JavaScript handles GC,
   * but we can clear the buffer reference to help GC.
   */
  disposeData(data: DeviceData): void {
    if (data.device.id !== this.id) {
      throw new Error(`Cannot dispose data from device ${data.device.id} on ${this.id}`);
    }

    // Clear buffer reference to help GC
    const cpuData = data as CPUDeviceData;
    cpuData.buffer = new ArrayBuffer(0);
  }

  /**
   * Read data from CPU memory
   *
   * For CPU backend, this returns the buffer directly.
   */
  async readData(data: DeviceData): Promise<ArrayBuffer> {
    if (data.device.id !== this.id) {
      throw new Error(`Cannot read data from device ${data.device.id} on ${this.id}`);
    }

    const cpuData = data as CPUDeviceData;
    // Return a copy to prevent external modifications
    return cpuData.buffer.slice(0);
  }

  /**
   * Write data to CPU memory
   */
  async writeData(data: DeviceData, buffer: ArrayBuffer): Promise<void> {
    if (data.device.id !== this.id) {
      throw new Error(`Cannot write data to device ${data.device.id} on ${this.id}`);
    }

    if (buffer.byteLength !== data.byteLength) {
      throw new Error(
        `Buffer size mismatch: expected ${data.byteLength} bytes, got ${buffer.byteLength} bytes`,
      );
    }

    const cpuData = data as CPUDeviceData;
    // Copy the buffer to prevent external modifications
    cpuData.buffer = buffer.slice(0);
  }

  /**
   * Check if CPU backend supports non-contiguous tensors for a specific operation
   *
   * For the CPU demo backend, we don't support non-contiguous tensors for any operations
   * to keep the implementation simple. All operations expect contiguous data.
   */
  supportsNonContiguous(_op: AnyStorageTransformation['__op']): boolean {
    // CPU backend doesn't support non-contiguous tensors for any operation
    // This is a simplification for the demo backend
    return false;
  }
}
