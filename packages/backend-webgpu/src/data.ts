/**
 * WebGPU device data management
 */

import type { DeviceData, Device } from '@typetensor/core';

/**
 * WebGPU-specific device data handle
 * 
 * Manages GPU buffers and provides async data transfer capabilities
 */
export class WebGPUDeviceData implements DeviceData {
  /**
   * Unique identifier for this data allocation
   */
  readonly id: string;

  /**
   * The GPU buffer containing the tensor data
   */
  private _buffer: GPUBuffer;

  /**
   * Cached array buffer for faster repeated reads
   */
  private _cachedArrayBuffer: ArrayBuffer | undefined;

  constructor(
    public readonly device: Device,
    public readonly byteLength: number,
    buffer: GPUBuffer,
  ) {
    this.id = `webgpu-data-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this._buffer = buffer;
  }

  /**
   * Get the underlying GPU buffer
   */
  get buffer(): GPUBuffer {
    return this._buffer;
  }

  /**
   * Clear any cached data
   */
  clearCache(): void {
    this._cachedArrayBuffer = undefined;
  }

  /**
   * Mark the buffer as destroyed
   */
  destroy(): void {
    this._buffer.destroy();
    this._cachedArrayBuffer = undefined;
  }

  /**
   * Check if the buffer is destroyed
   */
  get isDestroyed(): boolean {
    // GPUBuffer doesn't have a direct way to check if destroyed,
    // but accessing properties of destroyed buffers throws
    try {
      // eslint-disable-next-line @typescript-eslint/no-unused-expressions
      this._buffer.size;
      return false;
    } catch {
      return true;
    }
  }
}