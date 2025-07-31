import type { DeviceData, Device } from '@typetensor/core';
import type { WASMModule } from './types';
import { getLoadedWASMModule } from './loader';

// Global cleanup registry for automatic buffer disposal - TEMPORARILY DISABLED
const cleanupRegistry: FinalizationRegistry<any> | null = null;
// TODO: Re-enable after fixing BorrowMutError
// const cleanupRegistry =
//   typeof FinalizationRegistry !== 'undefined'
//     ? new FinalizationRegistry((cleanup: { device: any; wasmHandle: any }) => {
//         try {
//           // This runs when WASMDeviceData is garbage collected
//           if (cleanup.device && cleanup.device.operationDispatcher && cleanup.wasmHandle) {
//             cleanup.device.operationDispatcher.release_buffer(cleanup.wasmHandle);
//           }
//         } catch {}
//       })
//     : null;

export class WASMDeviceData implements DeviceData {
  readonly id: string;
  readonly device: Device;
  readonly byteLength: number;

  #wasmHandle: unknown;  // Private field
  #wasmModule: WASMModule;
  #disposed = false;
  #cleanupToken?: object; // Reference for unregistering cleanup

  constructor(device: Device, byteLength: number, wasmHandle: unknown, wasmModule: WASMModule) {
    this.device = device;
    this.byteLength = byteLength;
    this.#wasmHandle = wasmHandle;
    this.#wasmModule = wasmModule;
    this.id = `wasm-data-${(wasmHandle as any).id}`;

    // Register for automatic cleanup when this object is garbage collected
    if (cleanupRegistry) {
      this.#cleanupToken = {};
      cleanupRegistry.register(this, { device, wasmHandle }, this.#cleanupToken);
    }
  }

  clone(): WASMDeviceData {
    if (this.#disposed) {
      throw new Error('Cannot clone disposed WASMDeviceData');
    }

    const device = this.device as any;
    const clonedHandle = device.operationDispatcher.clone_buffer_handle(this.#wasmHandle);

    return new WASMDeviceData(this.device, this.byteLength, clonedHandle, this.#wasmModule);
  }

  dispose(): void {
    if (this.#disposed) {
      return;
    }

    this.#disposed = true;

    // Unregister from automatic cleanup since we're manually disposing
    if (cleanupRegistry && this.#cleanupToken) {
      cleanupRegistry.unregister(this.#cleanupToken);
    }

    // Release the WASM buffer back to the pool
    try {
      const device = this.device as any;
      if (device.operationDispatcher && this.#wasmHandle) {
        const released = device.operationDispatcher.release_buffer(this.#wasmHandle);
        if (!released) {
          console.warn(`Failed to release WASM buffer ${this.id}`);
        }
      }
    } catch (error) {
      console.warn(`Error disposing WASM buffer ${this.id}:`, error);
    }

    // Clear references
    this.#wasmHandle = null;
    this.#cleanupToken = undefined;
  }

  getRefCount(): number {
    return this.#disposed ? 0 : 1;
  }

  getWASMHandle(): unknown {
    if (this.#disposed) {
      throw new Error('WASMDeviceData has been disposed');
    }
    return this.#wasmHandle;
  }

  getByteLength(): number {
    return this.byteLength;
  }

  isDisposed(): boolean {
    return this.#disposed;
  }

  getDebugInfo(): Record<string, unknown> {
    return {
      id: this.id,
      byteLength: this.byteLength,
      active: !this.#disposed,
      disposed: this.#disposed,
      deviceId: this.device.id,
      wasmHandleId: this.#disposed ? null : (this.#wasmHandle as any).id(),
    };
  }

  createTypedArrayView<T extends ArrayBufferView>(
    _constructor: new (buffer: ArrayBuffer, byteOffset?: number, length?: number) => T,
    _elementSize: number,
  ): T {
    if (this.#disposed) {
      throw new Error('Cannot create view of disposed WASMDeviceData');
    }

    throw new Error('Direct memory access not yet implemented');
  }

  async copyToArrayBuffer(): Promise<ArrayBuffer> {
    if (this.#disposed) {
      throw new Error('Cannot copy from disposed WASMDeviceData');
    }

    return await this.device.readData(this);
  }

  async copyFromArrayBuffer(buffer: ArrayBuffer): Promise<void> {
    if (this.#disposed) {
      throw new Error('Cannot copy to disposed WASMDeviceData');
    }

    if (buffer.byteLength !== this.byteLength) {
      throw new Error(
        `Buffer size mismatch: expected ${this.byteLength}, got ${buffer.byteLength}`,
      );
    }

    await this.device.writeData(this, buffer);
  }

  equals(other: WASMDeviceData): boolean {
    if (!(other instanceof WASMDeviceData)) {
      return false;
    }

    return (
      this.id === other.id &&
      this.device.id === other.device.id &&
      !this.#disposed &&
      !other.#disposed
    );
  }

  toString(): string {
    const status = this.#disposed ? 'disposed' : `refs=${this.getRefCount()}`;
    return `WASMDeviceData(id=${this.id}, size=${this.byteLength}, ${status})`;
  }

  updateHandle(newHandle: unknown): void {
    if (this.#disposed) {
      throw new Error('Cannot update handle of disposed WASMDeviceData');
    }
    
    // Clean up old handle if needed and valid
    const device = this.device as any;
    if (this.#wasmHandle && device.operationDispatcher && this.#wasmHandle !== null) {
      try {
        const released = device.operationDispatcher.release_buffer(this.#wasmHandle);
        if (!released) {
          console.warn(`Failed to release old buffer handle ${this.id}`);
        }
      } catch (error) {
        // Handle cleanup errors gracefully - old handle might already be invalid
        console.warn(`Error releasing old buffer handle ${this.id}:`, error);
      }
    }
    
    this.#wasmHandle = newHandle;
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

export function createWASMDeviceData(
  device: Device,
  byteLength: number,
  wasmHandle: unknown,
): WASMDeviceData {
  const wasmModule = getLoadedWASMModule();
  return new WASMDeviceData(device, byteLength, wasmHandle, wasmModule);
}
