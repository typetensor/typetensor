/**
 * Zero-copy memory view management for WASM backend
 *
 * Provides safe typed array views into WebAssembly memory without copying data.
 */

import type { DType } from '@typetensor/core';
import { float32, float64, int32, int16, int8, uint32, uint16, uint8 } from '@typetensor/core';

export interface MemoryView {
  readonly buffer: ArrayBuffer;
  readonly byteOffset: number;
  readonly byteLength: number;
  readonly isValid: boolean;
}

export interface TrackedView {
  readonly view: ArrayBufferView;
  readonly bufferId: string;
  readonly generation: number;
  readonly byteOffset: number;
  readonly byteLength: number;
}

export class MemoryViewManager {
  private memory: WebAssembly.Memory;
  private views: Map<string, WeakRef<MemoryView>> = new Map();
  private generation = 0;
  
  // New tracking for buffer lifetimes
  private bufferGenerations = new Map<string, number>();
  private activeViews = new Map<string, Set<WeakRef<TrackedView>>>();
  private viewRegistry: FinalizationRegistry<string> | null = null;

  constructor(memory: WebAssembly.Memory) {
    this.memory = memory;

    // Listen for memory growth to invalidate views
    if (typeof FinalizationRegistry !== 'undefined') {
      // Track view lifecycle for debugging
      this.viewRegistry = new FinalizationRegistry((viewId: string) => {
        // Clean up weak references when views are GC'd
        this.cleanupView(viewId);
      });
    }
  }

  private cleanupView(viewId: string): void {
    // Remove from all tracking structures
    for (const [bufferId, viewSet] of this.activeViews.entries()) {
      const toRemove: WeakRef<TrackedView>[] = [];
      for (const viewRef of viewSet) {
        const view = viewRef.deref();
        if (!view || view.bufferId === viewId) {
          toRemove.push(viewRef);
        }
      }
      toRemove.forEach(ref => viewSet.delete(ref));
      
      // Clean up empty sets
      if (viewSet.size === 0) {
        this.activeViews.delete(bufferId);
      }
    }
  }

  /**
   * Create a zero-copy typed array view into WASM memory
   * @deprecated Use createTrackedView for safe buffer lifetime tracking
   */
  createView(ptr: number, size: number, dtype: DType<any, any, any>): ArrayBufferView {
    // Validate pointer and size
    if (ptr < 0 || size < 0) {
      throw new Error(`Invalid pointer (${ptr}) or size (${size})`);
    }

    const buffer = this.memory.buffer;
    const elementSize = dtype.__byteSize;
    const byteOffset = ptr;
    const byteLength = size * elementSize;

    // Check bounds
    if (byteOffset + byteLength > buffer.byteLength) {
      throw new Error(
        `View would exceed memory bounds: offset=${byteOffset}, length=${byteLength}, bufferSize=${buffer.byteLength}`,
      );
    }

    // Create appropriate typed array view
    switch (dtype) {
      case float32:
        return new Float32Array(buffer, byteOffset, size);
      case float64:
        return new Float64Array(buffer, byteOffset, size);
      case int32:
        return new Int32Array(buffer, byteOffset, size);
      case uint32:
        return new Uint32Array(buffer, byteOffset, size);
      case int16:
        return new Int16Array(buffer, byteOffset, size);
      case uint16:
        return new Uint16Array(buffer, byteOffset, size);
      case int8:
        return new Int8Array(buffer, byteOffset, size);
      case uint8:
        return new Uint8Array(buffer, byteOffset, size);
      default:
        throw new Error(`Unsupported dtype for view: ${dtype.__dtype}`);
    }
  }

  /**
   * Create a safe tracked view with buffer lifetime management
   */
  createSafeView(
    bufferId: string,
    ptr: number,
    size: number,
    dtype: DType<any, any, any>
  ): ArrayBufferView {
    // Initialize buffer generation if not exists
    if (!this.bufferGenerations.has(bufferId)) {
      this.bufferGenerations.set(bufferId, 0);
    }
    const generation = this.bufferGenerations.get(bufferId)!;
    
    // Create the raw view
    const rawView = this.createView(ptr, size, dtype);
    
    // Create tracked view metadata
    const trackedView: TrackedView = {
      view: rawView,
      bufferId,
      generation,
      byteOffset: ptr,
      byteLength: size * dtype.__byteSize
    };
    
    // Track this view
    if (!this.activeViews.has(bufferId)) {
      this.activeViews.set(bufferId, new Set());
    }
    this.activeViews.get(bufferId)!.add(new WeakRef(trackedView));
    
    // Create a unique ID for this view
    const viewId = `${bufferId}-${Date.now()}-${Math.random()}`;
    
    // Register for cleanup
    if (this.viewRegistry) {
      this.viewRegistry.register(trackedView, viewId);
    }
    
    // Return a proxied view that checks validity on access
    return this.createProxiedView(rawView, trackedView);
  }

  private createProxiedView(
    rawView: ArrayBufferView,
    trackedView: TrackedView
  ): ArrayBufferView {
    const self = this;
    
    return new Proxy(rawView, {
      get(target, prop, receiver) {
        // Check validity before any property access
        if (typeof prop === 'string' && !isNaN(Number(prop))) {
          // Numeric index access - check validity
          if (!self.isTrackedViewValid(trackedView)) {
            throw new Error('View is no longer valid: buffer has been disposed');
          }
        } else if (prop === 'buffer') {
          // Special handling for buffer property
          if (!self.isTrackedViewValid(trackedView)) {
            throw new Error('View is no longer valid: buffer has been disposed');
          }
          return target.buffer;
        } else if (prop === 'subarray') {
          // Handle subarray specially to return a proxied subarray
          return function(...args: any[]) {
            if (!self.isTrackedViewValid(trackedView)) {
              throw new Error('View is no longer valid: buffer has been disposed');
            }
            const subarray = (target as any).subarray(...args);
            // Create a new tracked view for the subarray
            const subTrackedView: TrackedView = {
              ...trackedView,
              view: subarray
            };
            return self.createProxiedView(subarray, subTrackedView);
          };
        }
        
        // Handle other important properties
        if (prop === 'length' || prop === 'byteLength' || prop === 'byteOffset') {
          if (!self.isTrackedViewValid(trackedView)) {
            throw new Error('View is no longer valid: buffer has been disposed');
          }
          return (target as any)[prop];
        }
        
        // For all other properties, check validity if it's a method
        const value = Reflect.get(target, prop, target); // Use target as receiver to avoid issues
        if (typeof value === 'function' && prop !== 'slice') {
          return function(...args: any[]) {
            if (!self.isTrackedViewValid(trackedView)) {
              throw new Error('View is no longer valid: buffer has been disposed');
            }
            return value.apply(target, args);
          };
        }
        
        return value;
      },
      
      set(target, prop, value, receiver) {
        // Check validity before any property write
        if (!self.isTrackedViewValid(trackedView)) {
          throw new Error('View is no longer valid: buffer has been disposed');
        }
        return Reflect.set(target, prop, value, receiver);
      }
    }) as ArrayBufferView;
  }

  private isTrackedViewValid(trackedView: TrackedView): boolean {
    // Check if buffer was disposed
    const currentGeneration = this.bufferGenerations.get(trackedView.bufferId);
    if (currentGeneration === undefined || currentGeneration !== trackedView.generation) {
      return false;
    }
    
    // Check if WASM memory grew
    if (trackedView.view.buffer !== this.memory.buffer) {
      return false;
    }
    
    return true;
  }

  /**
   * Create a tracked view that can be invalidated on memory growth
   */
  createTrackedView(
    id: string,
    ptr: number,
    size: number,
    dtype: DType<any, any, any>,
  ): ArrayBufferView {
    const view = this.createView(ptr, size, dtype);

    // Track the view with metadata
    const viewInfo: MemoryView = {
      buffer: this.memory.buffer,
      byteOffset: ptr,
      byteLength: size * dtype.__byteSize,
      isValid: true,
    };

    // Store weak reference
    this.views.set(id, new WeakRef(viewInfo));

    // Register for cleanup if supported
    const registry = (this as any).cleanupRegistry;
    if (registry) {
      registry.register(viewInfo, id);
    }

    return view;
  }

  /**
   * Check if a view is still valid (memory hasn't grown)
   */
  isViewValid(view: ArrayBufferView): boolean {
    // Check if this is a proxied view by looking for our tracking
    try {
      // Try to access the length property - this will trigger validation
      const len = (view as any).length;
      // If we get here, the view is valid and memory hasn't grown
      return view.buffer === this.memory.buffer;
    } catch (e) {
      // If accessing properties throws, the view is invalid
      return false;
    }
  }
  
  /**
   * Invalidate all views for a specific buffer
   */
  invalidateBuffer(bufferId: string): void {
    // Increment generation to invalidate all views of this buffer
    const currentGen = this.bufferGenerations.get(bufferId) || 0;
    this.bufferGenerations.set(bufferId, currentGen + 1);
    
    // Clean up tracking for this buffer
    const viewSet = this.activeViews.get(bufferId);
    if (viewSet) {
      viewSet.clear();
      this.activeViews.delete(bufferId);
    }
  }

  /**
   * Invalidate all views (called on memory growth)
   */
  invalidateAllViews(): void {
    this.generation++;
    // Views automatically become invalid when memory.buffer changes
  }

  /**
   * Get memory statistics
   */
  getStats() {
    return {
      memorySize: this.memory.buffer.byteLength,
      trackedViews: this.views.size,
      generation: this.generation,
    };
  }

  /**
   * Create a safe copy if zero-copy is not possible
   */
  static createSafeCopy(data: ArrayBufferView): ArrayBuffer {
    const copy = new ArrayBuffer(data.byteLength);
    new Uint8Array(copy).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    return copy;
  }
}
