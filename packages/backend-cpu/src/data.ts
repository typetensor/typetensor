/**
 * CPU device data management
 *
 * This module provides the CPUDeviceData implementation that stores
 * tensor data directly in ArrayBuffers in host memory.
 */

import type { DeviceData, Device } from '@typetensor/core';

/**
 * CPU-specific device data handle
 *
 * Stores tensor data directly in an ArrayBuffer with metadata
 * for efficient access and memory management.
 */
export class CPUDeviceData implements DeviceData {
  /**
   * Unique identifier for this data allocation
   */
  readonly id: string;

  /**
   * The actual data buffer
   * This is the raw memory containing tensor elements
   */
  buffer: ArrayBuffer;

  constructor(
    public readonly device: Device,
    public readonly byteLength: number,
    buffer?: ArrayBuffer,
  ) {
    this.id = `cpu-data-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.buffer = buffer ?? new ArrayBuffer(byteLength);

    // Validate buffer size if provided
    if (buffer && buffer.byteLength !== byteLength) {
      throw new Error(
        `Buffer size mismatch: expected ${byteLength} bytes, got ${buffer.byteLength} bytes`,
      );
    }
  }

  /**
   * Create a clone of this data with a new buffer
   */
  clone(): CPUDeviceData {
    const newBuffer = this.buffer.slice(0);
    return new CPUDeviceData(this.device, this.byteLength, newBuffer);
  }
}
