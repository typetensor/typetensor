import type { DeviceData, Device } from '@typetensor/core';
import type { WASMModule } from './types';
import { getLoadedWASMModule } from './loader';
import { WASMErrorHandler, WASMCleanupFinalizationError } from './errors';

// Shared reference counting for coordinated cleanup
const sharedBufferRefs = new Map<string, { 
  refCount: number; 
  device: any; 
  wasmHandle: any; 
  cleanupToken?: object;
}>();

// Global cleanup registry for automatic buffer disposal with coordination
const cleanupRegistry =
  typeof FinalizationRegistry !== 'undefined'
    ? new FinalizationRegistry((bufferId: string) => {
        try {
          // This runs when a WASMDeviceData is garbage collected
          const bufferRef = sharedBufferRefs.get(bufferId);
          if (bufferRef) {
            bufferRef.refCount--;
            
            // Only cleanup when all references are gone
            if (bufferRef.refCount <= 0) {
              if (bufferRef.device && bufferRef.device.operationDispatcher && bufferRef.wasmHandle) {
                bufferRef.device.operationDispatcher.release_buffer(bufferRef.wasmHandle);
              }
              sharedBufferRefs.delete(bufferId);
            }
          }
        } catch (error) {
          // Handle finalization errors properly
          const cleanupError = new WASMCleanupFinalizationError(
            error instanceof Error ? error.message : String(error),
            { bufferId }
          );
          WASMErrorHandler.handle(cleanupError);
        }
      })
    : null;

export class WASMDeviceData implements DeviceData {
  readonly id: string;
  readonly device: Device;
  readonly byteLength: number;

  #wasmHandle: unknown;  // Private field
  #wasmModule: WASMModule;
  #disposed = false;
  #cleanupToken?: object | undefined; // Reference for unregistering cleanup

  constructor(device: Device, byteLength: number, wasmHandle: unknown, wasmModule: WASMModule, isClone = false) {
    this.device = device;
    this.byteLength = byteLength;
    this.#wasmHandle = wasmHandle;
    this.#wasmModule = wasmModule;
    this.id = `wasm-data-${(wasmHandle as any).id}`;

    // Register for coordinated automatic cleanup
    if (cleanupRegistry) {
      if (!isClone) {
        // First instance - create shared reference entry
        sharedBufferRefs.set(this.id, {
          refCount: 1,
          device,
          wasmHandle
        });
      } else {
        // Clone - increment existing reference count
        const bufferRef = sharedBufferRefs.get(this.id);
        if (bufferRef) {
          bufferRef.refCount++;
        } else {
          // Fallback: create new entry if original was already cleaned up
          sharedBufferRefs.set(this.id, {
            refCount: 1,
            device,
            wasmHandle
          });
        }
      }
      
      // Register this instance for cleanup (passes buffer ID, not cleanup data)
      this.#cleanupToken = {};
      cleanupRegistry.register(this, this.id, this.#cleanupToken);
    }
  }

  clone(): WASMDeviceData {
    if (this.#disposed) {
      throw new Error('Cannot clone disposed WASMDeviceData');
    }

    const device = this.device as any;
    const clonedHandle = device.operationDispatcher.clone_buffer_handle(this.#wasmHandle);

    // Create clone with coordinated reference counting
    return new WASMDeviceData(this.device, this.byteLength, clonedHandle, this.#wasmModule, true);
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

    // Invalidate all views for this buffer BEFORE releasing it
    const device = this.device as any;
    if (device.memoryViewManager) {
      device.memoryViewManager.invalidateBuffer(this.id);
    }

    // Coordinate with shared reference counting
    const bufferRef = sharedBufferRefs.get(this.id);
    if (bufferRef) {
      bufferRef.refCount--;
      
      // Only release WASM buffer when all references are gone
      if (bufferRef.refCount <= 0) {
        try {
          if (device.operationDispatcher && this.#wasmHandle) {
            const released = device.operationDispatcher.release_buffer(this.#wasmHandle);
            if (!released) {
              // Buffer release failure during disposal is a cleanup error
              const error = WASMErrorHandler.createBufferReleaseError(
                this.id,
                released,
                { operation: 'dispose', deviceId: this.device.id }
              );
              WASMErrorHandler.handle(error);
            }
          }
        } catch (error) {
          // Handle disposal errors as cleanup errors
          const cleanupError = WASMErrorHandler.createBufferReleaseError(
            this.id,
            false,
            { 
              operation: 'dispose', 
              deviceId: this.device.id,
              originalError: error instanceof Error ? error.message : String(error)
            }
          );
          WASMErrorHandler.handle(cleanupError);
        }
        
        // Clean up shared reference entry
        sharedBufferRefs.delete(this.id);
      }
      // If refCount > 0, other clones still exist - don't release WASM buffer yet
    } else {
      // Fallback: no shared ref entry, release directly (shouldn't happen in normal flow)
      try {
        const device = this.device as any;
        if (device.operationDispatcher && this.#wasmHandle) {
          device.operationDispatcher.release_buffer(this.#wasmHandle);
        }
      } catch (error) {
        const cleanupError = WASMErrorHandler.createBufferReleaseError(
          this.id,
          false,
          { 
            operation: 'dispose_fallback', 
            deviceId: this.device.id,
            originalError: error instanceof Error ? error.message : String(error)
          }
        );
        WASMErrorHandler.handle(cleanupError);
      }
    }

    // Clear references
    this.#wasmHandle = null;
    this.#cleanupToken = undefined;
  }

  getRefCount(): number {
    if (this.#disposed) {
      return 0;
    }
    
    const bufferRef = sharedBufferRefs.get(this.id);
    return bufferRef ? bufferRef.refCount : 1;
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
    
    // Check if we're trying to update with the same handle
    if (this.#wasmHandle === newHandle) {
      // No-op: same handle, nothing to update
      return;
    }
    
    const oldId = this.id;
    const device = this.device as any;
    
    // When updating handle, we're breaking sharing (copy-on-write)
    // Decrement old buffer's reference count
    const oldBufferRef = sharedBufferRefs.get(oldId);
    if (oldBufferRef) {
      oldBufferRef.refCount--;
      
      // If this was the last reference to the old buffer, clean it up
      if (oldBufferRef.refCount <= 0) {
        try {
          if (device.operationDispatcher && this.#wasmHandle) {
            device.operationDispatcher.release_buffer(this.#wasmHandle);
          }
        } catch (error) {
          // Handle cleanup errors gracefully
          const cleanupError = WASMErrorHandler.createBufferReleaseError(
            oldId,
            false,
            { 
              operation: 'updateHandle_cleanup', 
              deviceId: this.device.id,
              originalError: error instanceof Error ? error.message : String(error)
            }
          );
          WASMErrorHandler.handle(cleanupError);
        }
        sharedBufferRefs.delete(oldId);
      }
    }
    
    // Update to new handle
    this.#wasmHandle = newHandle;
    
    // Create new ID for the new handle and new shared reference entry
    const newId = `wasm-data-${(newHandle as any).id}`;
    (this as any).id = newId; // Update the ID (casting to any since id is readonly)
    
    // Create new shared reference entry for this buffer
    sharedBufferRefs.set(newId, {
      refCount: 1,
      device: this.device,
      wasmHandle: newHandle
    });
    
    // Re-register with cleanup registry under new ID
    if (cleanupRegistry && this.#cleanupToken) {
      cleanupRegistry.unregister(this.#cleanupToken);
      this.#cleanupToken = {};
      cleanupRegistry.register(this, newId, this.#cleanupToken);
    }
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
