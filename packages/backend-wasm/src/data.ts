import type { DeviceData, Device } from '@typetensor/core';
import type { WASMModule } from './types';
import { getLoadedWASMModule } from './loader';

export class WASMDeviceData implements DeviceData {
  readonly id: string;
  readonly device: Device;
  readonly byteLength: number;

  private wasmHandle: unknown;
  private wasmModule: WASMModule;
  private disposed = false;

  constructor(device: Device, byteLength: number, wasmHandle: unknown, wasmModule: WASMModule) {
    this.device = device;
    this.byteLength = byteLength;
    this.wasmHandle = wasmHandle;
    this.wasmModule = wasmModule;
    this.id = `wasm-data-${(wasmHandle as any).id}`;
  }

  clone(): WASMDeviceData {
    if (this.disposed) {
      throw new Error('Cannot clone disposed WASMDeviceData');
    }

    const device = this.device as any;
    const clonedHandle = device.operationDispatcher.clone_buffer_handle(this.wasmHandle);

    return new WASMDeviceData(this.device, this.byteLength, clonedHandle, this.wasmModule);
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }

    this.disposed = true;
    
  }

  getRefCount(): number {
    return this.disposed ? 0 : 1;
  }

  getWASMHandle(): unknown {
    if (this.disposed) {
      throw new Error('WASMDeviceData has been disposed');
    }
    return this.wasmHandle;
  }

  getByteLength(): number {
    return this.byteLength;
  }

  isDisposed(): boolean {
    return this.disposed;
  }

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

  createTypedArrayView<T extends ArrayBufferView>(
    _constructor: new (buffer: ArrayBuffer, byteOffset?: number, length?: number) => T,
    _elementSize: number,
  ): T {
    if (this.disposed) {
      throw new Error('Cannot create view of disposed WASMDeviceData');
    }

    throw new Error('Direct memory access not yet implemented');
  }

  async copyToArrayBuffer(): Promise<ArrayBuffer> {
    if (this.disposed) {
      throw new Error('Cannot copy from disposed WASMDeviceData');
    }

    return await this.device.readData(this);
  }

  async copyFromArrayBuffer(buffer: ArrayBuffer): Promise<void> {
    if (this.disposed) {
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
      !this.disposed &&
      !other.disposed
    );
  }

  toString(): string {
    const status = this.disposed ? 'disposed' : `refs=${this.getRefCount()}`;
    return `WASMDeviceData(id=${this.id}, size=${this.byteLength}, ${status})`;
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
