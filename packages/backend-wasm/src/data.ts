/**
 * WASM device data implementation
 *
 * Provides the DeviceData interface for WASM backend with:
 * - Reference counting for memory management
 * - Integration with WASM memory pools
 * - Efficient memory transfers
 */

import type { DeviceData, Device } from '@typetensor/core';
import type { WASMModule } from './types';
import { getLoadedWASMModule } from './loader';

/**
 * WASM-specific device data implementation
 *
 * Manages tensor data stored in WebAssembly linear memory with
 * reference counting and efficient memory operations.
 */
export class WASMDeviceData implements DeviceData {
  readonly id: string;
  readonly device: Device;
  readonly byteLength: number;

  private wasmHandle: unknown; // WasmBufferHandle from Rust
  private wasmModule: WASMModule;
  private disposed = false;

  constructor(device: Device, byteLength: number, wasmHandle: unknown, wasmModule: WASMModule) {
    this.device = device;
    this.byteLength = byteLength;
    this.wasmHandle = wasmHandle;
    this.wasmModule = wasmModule;
    this.id = `wasm-data-${(wasmHandle as any).id}`;
  }

  /**
   * Create a clone of this data (shares the same buffer with reference counting)
   */
  clone(): WASMDeviceData {
    if (this.disposed) {
      throw new Error('Cannot clone disposed WASMDeviceData');
    }

    // Use the device's operation dispatcher to properly clone the handle
    // This increments the reference count for the underlying buffer
    const device = this.device as any;
    const clonedHandle = device.operationDispatcher.clone_buffer_handle(this.wasmHandle);

    return new WASMDeviceData(this.device, this.byteLength, clonedHandle, this.wasmModule);
  }

  /**
   * Dispose of this data handle
   *
   * Marks this handle as disposed. The actual buffer release should be done
   * by the device when disposeData is called.
   */
  dispose(): void {
    if (this.disposed) {
      return;
    }

    this.disposed = true;
    
    // Note: The actual buffer release is handled by device.disposeData()
    // to avoid circular dependencies. This just marks the handle as disposed.
  }

  /**
   * Get the current reference count (legacy compatibility)
   * In the new buffer architecture, this just returns 1 for active, 0 for disposed
   */
  getRefCount(): number {
    return this.disposed ? 0 : 1;
  }

  /**
   * Get the WASM handle (internal use)
   */
  getWASMHandle(): unknown {
    if (this.disposed) {
      throw new Error('WASMDeviceData has been disposed');
    }
    return this.wasmHandle;
  }

  /**
   * Get the size of the data in bytes
   */
  getByteLength(): number {
    return this.byteLength;
  }

  /**
   * Check if this data handle is disposed
   */
  isDisposed(): boolean {
    return this.disposed;
  }

  /**
   * Get debug information about this data handle
   */
  getDebugInfo(): Record<string, unknown> {
    return {
      id: this.id,
      byteLength: this.byteLength,
      active: !this.disposed,
      disposed: this.disposed,
      deviceId: this.device.id,
      wasmHandleId: this.disposed ? null : (this.wasmHandle as any).id(),
    };
  }

  /**
   * Create a typed array view of the data (for direct access)
   *
   * WARNING: This provides direct access to WASM memory. Use with caution.
   * The returned view may become invalid if the data is disposed or moved.
   */
  createTypedArrayView<T extends ArrayBufferView>(
    _constructor: new (buffer: ArrayBuffer, byteOffset?: number, length?: number) => T,
    _elementSize: number,
  ): T {
    if (this.disposed) {
      throw new Error('Cannot create view of disposed WASMDeviceData');
    }

    // Get pointer to WASM memory
    // This would require additional Rust exports to get the actual memory pointer
    // For now, this is a placeholder for the concept
    throw new Error('Direct memory access not yet implemented');
  }

  /**
   * Copy data to a JavaScript ArrayBuffer
   *
   * This creates a copy of the data that can be safely used outside of WASM.
   */
  async copyToArrayBuffer(): Promise<ArrayBuffer> {
    if (this.disposed) {
      throw new Error('Cannot copy from disposed WASMDeviceData');
    }

    // This would require the device to implement data reading
    return await this.device.readData(this);
  }

  /**
   * Copy data from a JavaScript ArrayBuffer
   *
   * Updates the WASM memory with data from the provided buffer.
   */
  async copyFromArrayBuffer(buffer: ArrayBuffer): Promise<void> {
    if (this.disposed) {
      throw new Error('Cannot copy to disposed WASMDeviceData');
    }

    if (buffer.byteLength !== this.byteLength) {
      throw new Error(
        `Buffer size mismatch: expected ${this.byteLength}, got ${buffer.byteLength}`,
      );
    }

    // This would require the device to implement data writing
    await this.device.writeData(this, buffer);
  }

  /**
   * Compare data handles for equality
   */
  equals(other: WASMDeviceData): boolean {
    if (!(other instanceof WASMDeviceData)) {
      return false;
    }

    return (
      this.id === other.id &&
      this.device.id === other.device.id &&
      !this.disposed &&
      !other.disposed
    );
  }

  /**
   * String representation for debugging
   */
  toString(): string {
    const status = this.disposed ? 'disposed' : `refs=${this.getRefCount()}`;
    return `WASMDeviceData(id=${this.id}, size=${this.byteLength}, ${status})`;
  }

  /**
   * Symbol.dispose implementation for explicit resource management
   */
  [Symbol.dispose](): void {
    this.dispose();
  }
}

/**
 * Factory function to create WASM device data
 *
 * @param device The WASM device
 * @param byteLength Size in bytes
 * @param wasmHandle Handle from WASM memory manager
 * @returns New WASMDeviceData instance
 */
export function createWASMDeviceData(
  device: Device,
  byteLength: number,
  wasmHandle: unknown,
): WASMDeviceData {
  const wasmModule = getLoadedWASMModule();
  return new WASMDeviceData(device, byteLength, wasmHandle, wasmModule);
}
