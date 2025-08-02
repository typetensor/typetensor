/**
 * View Manager - coordinates memory view creation and lifecycle
 *
 * This abstraction layer decouples memory view management from the Device class,
 * providing proper lifetime coordination and safety guarantees.
 */

import type { DType, DeviceData } from '@typetensor/core';
import { WASMDeviceData } from './data';
import { MemoryViewManager } from './memory-views';
import { getDTypeByteSize } from './utils/dtype-helpers';
import type { WasmOperationDispatcher, WasmBufferHandle } from './types/wasm-bindings';

export interface ViewCreationOptions {
  dtype: DType<any, any, any>;
  offset?: number;
  length?: number;
  byteOffset?: number;
  byteLength?: number;
}

export interface ViewInfo {
  isValid: boolean;
  bufferId: string;
  byteOffset: number;
  byteLength: number;
  elementCount: number;
  dtype: string;
}

export interface ViewManagerStats {
  activeViews: number;
  invalidatedViews: number;
  memorySize: number;
  generation: number;
}

/**
 * Manages creation and lifecycle of memory views with proper safety guarantees
 */
export class ViewManager {
  private memoryViewManager: MemoryViewManager;
  private operationDispatcher: WasmOperationDispatcher;
  private deviceId: string;

  constructor(
    memoryViewManager: MemoryViewManager,
    operationDispatcher: WasmOperationDispatcher,
    deviceId: string,
  ) {
    this.memoryViewManager = memoryViewManager;
    this.operationDispatcher = operationDispatcher;
    this.deviceId = deviceId;
  }

  /**
   * Create a zero-copy view into WASM memory with lifetime tracking
   */
  createView(data: DeviceData, options: ViewCreationOptions): ArrayBufferView {
    this.validateData(data);

    const wasmData = data as WASMDeviceData;
    const wasmHandle = wasmData.getWASMHandle() as WasmBufferHandle;

    // Get buffer info: [ptr, size, initialized]
    const info = this.operationDispatcher.get_buffer_view_info(wasmHandle);
    const ptr = info[0]!;
    const size = info[1]!;
    const initialized = info[2]!;

    if (!initialized) {
      throw new Error('Cannot create view of uninitialized buffer');
    }

    // Calculate view parameters
    const elementSize = getDTypeByteSize(options.dtype);
    const elementCount = options.length ?? size / elementSize;
    const byteOffset = options.byteOffset ?? (options.offset ?? 0) * elementSize;

    // Validate view bounds
    if (byteOffset + elementCount * elementSize > size) {
      throw new Error(
        `View bounds exceed buffer: offset=${byteOffset}, ` +
          `elementCount=${elementCount}, elementSize=${elementSize}, bufferSize=${size}`,
      );
    }

    // Create safe zero-copy view with lifetime tracking
    const bufferId = wasmData.id;
    return this.memoryViewManager.createSafeView(
      bufferId,
      ptr + byteOffset,
      elementCount,
      options.dtype,
    );
  }

  /**
   * Check if a view is still valid
   */
  isViewValid(view: ArrayBufferView): boolean {
    return this.memoryViewManager.isViewValid(view);
  }

  /**
   * Get information about a view
   */
  getViewInfo(view: ArrayBufferView): ViewInfo | null {
    // This is a simplified implementation - in practice we'd need to track more metadata
    const isValid = this.isViewValid(view);

    if (!isValid) {
      return null;
    }

    return {
      isValid: true,
      bufferId: 'unknown', // Would need view tracking to get this
      byteOffset: view.byteOffset,
      byteLength: view.byteLength,
      elementCount: (view as any).length ?? 0,
      dtype: this.getViewDType(view),
    };
  }

  /**
   * Invalidate all views for a specific buffer
   */
  invalidateBufferViews(bufferId: string): void {
    this.memoryViewManager.invalidateBuffer(bufferId);
  }

  /**
   * Invalidate all views (called on memory growth or device reset)
   */
  invalidateAllViews(): void {
    this.memoryViewManager.invalidateAllViews();
  }

  /**
   * Create a safe copy if zero-copy is not possible
   */
  createSafeCopy(view: ArrayBufferView): ArrayBuffer {
    return MemoryViewManager.createSafeCopy(view);
  }

  /**
   * Get view manager statistics
   */
  getStats(): ViewManagerStats {
    const stats = this.memoryViewManager.getStats();
    return {
      activeViews: stats.trackedViews,
      invalidatedViews: 0, // Would need tracking to implement this
      memorySize: stats.memorySize,
      generation: stats.generation,
    };
  }

  /**
   * Batch create multiple views for efficiency
   */
  createViews(
    requests: Array<{
      data: DeviceData;
      options: ViewCreationOptions;
    }>,
  ): ArrayBufferView[] {
    // Validate all requests first
    for (const request of requests) {
      this.validateData(request.data);
    }

    // Create all views
    return requests.map((request) => this.createView(request.data, request.options));
  }

  /**
   * Create a read-only view (TypeScript typing only - WASM doesn't enforce this)
   */
  createReadOnlyView(data: DeviceData, options: ViewCreationOptions): Readonly<ArrayBufferView> {
    return this.createView(data, options) as Readonly<ArrayBufferView>;
  }

  /**
   * Create a typed view with specific ArrayBufferView constructor
   */
  createTypedView<T extends ArrayBufferView>(
    data: DeviceData,
    _ViewConstructor: new (buffer: ArrayBuffer, byteOffset?: number, length?: number) => T,
    _options: Omit<ViewCreationOptions, 'dtype'>,
  ): T {
    this.validateData(data);

    const wasmData = data as WASMDeviceData;
    const wasmHandle = wasmData.getWASMHandle() as WasmBufferHandle;

    const info = this.operationDispatcher.get_buffer_view_info(wasmHandle);
    const initialized = info[2]!;

    if (!initialized) {
      throw new Error('Cannot create view of uninitialized buffer');
    }

    // For now, delegate to the memory view manager's generic method
    // In a full implementation, we'd create the specific typed view directly
    throw new Error(
      'createTypedView not fully implemented - use createView with appropriate dtype',
    );
  }

  /**
   * Validate that data belongs to this device
   */
  private validateData(data: DeviceData): void {
    if (data.device.id !== this.deviceId) {
      throw new Error(
        `Cannot create view for data from device ${data.device.id} on ${this.deviceId}`,
      );
    }

    if (!(data instanceof WASMDeviceData)) {
      throw new Error('ViewManager can only create views for WASMDeviceData');
    }

    if ((data as WASMDeviceData).isDisposed()) {
      throw new Error('Cannot create view of disposed buffer');
    }
  }

  /**
   * Get dtype string from ArrayBufferView
   */
  private getViewDType(view: ArrayBufferView): string {
    if (view instanceof Float32Array) return 'float32';
    if (view instanceof Float64Array) return 'float64';
    if (view instanceof Int32Array) return 'int32';
    if (view instanceof Uint32Array) return 'uint32';
    if (view instanceof Int16Array) return 'int16';
    if (view instanceof Uint16Array) return 'uint16';
    if (view instanceof Int8Array) return 'int8';
    if (view instanceof Uint8Array) return 'uint8';
    return 'unknown';
  }

}
