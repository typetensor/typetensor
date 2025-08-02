/**
 * Buffer Lifecycle Manager - handles creation, disposal, and lifecycle of WASM buffers
 * 
 * This abstraction layer decouples buffer management from the Device class,
 * making the system more modular and testable.
 */

import type { DeviceData } from '@typetensor/core';
import { WASMDeviceData, createWASMDeviceData } from './data';
import type { WasmOperationDispatcher } from './types/wasm-bindings';
import { WASMErrorHandler, WASMAllocationError, WASMBoundsError, WASMMemoryLimitError } from './errors';
import { createDefensiveWASMOperations } from './wasm-defensive-wrapper';

export interface BufferLifecycleManagerConfig {
  maxMemory?: number;
  compactThreshold?: number;
  autoCompact?: boolean;
}

export interface MemoryPressureStats {
  totalAllocated: number;
  maxMemory: number;
  usageRatio: number;
  shouldCompact: boolean;
}

/**
 * Manages the entire lifecycle of WASM buffers from creation to disposal
 */
export class BufferLifecycleManager {
  private config: Required<BufferLifecycleManagerConfig>;
  private operationDispatcher: WasmOperationDispatcher;
  private lastCompactTime = 0;
  private compactInterval = 60000; // 1 minute
  private device: any; // Reference to the device for creating data
  private defensiveOperations: any; // Defensive WASM operations

  constructor(
    operationDispatcher: WasmOperationDispatcher,
    device: any, // Pass the actual device reference
    config: BufferLifecycleManagerConfig = {}
  ) {
    this.operationDispatcher = operationDispatcher;
    this.device = device;
    this.config = {
      maxMemory: config.maxMemory ?? Number.MAX_SAFE_INTEGER,
      compactThreshold: config.compactThreshold ?? 0.8,
      autoCompact: config.autoCompact ?? false,
    };
    
    // Initialize defensive operations
    this.defensiveOperations = createDefensiveWASMOperations(operationDispatcher);
  }

  /**
   * Create a new buffer with specified size
   */
  async createBuffer(byteLength: number, _deviceId: string): Promise<DeviceData> {
    try {
      // Check for reasonable allocation size (e.g., max 1GB)
      const MAX_ALLOCATION_SIZE = 1024 * 1024 * 1024; // 1GB
      if (byteLength > MAX_ALLOCATION_SIZE) {
        throw new WASMBoundsError(
          'buffer allocation',
          byteLength,
          { max: MAX_ALLOCATION_SIZE },
          { requestedSize: byteLength, maxSize: MAX_ALLOCATION_SIZE }
        );
      }
      
      // Check memory pressure before allocation
      await this.checkMemoryPressure(byteLength);
      
      // For large allocations, try to allocate in smaller chunks first to test memory availability
      if (byteLength > 50 * 1024 * 1024) { // > 50MB
        this.testLargeAllocation();
      }
      
      // Use zero-allocation method with defensive wrapper
      const wasmHandle = await this.defensiveOperations.createEmptyBuffer(byteLength);
      return createWASMDeviceData(
        this.device, // Pass the actual device reference
        byteLength, 
        wasmHandle
      );
    } catch (error: any) {
      // Don't re-throw our own errors
      if (error instanceof WASMBoundsError || error instanceof WASMAllocationError) {
        throw error;
      }
      
      // Use error handler to create appropriate error type
      throw WASMErrorHandler.createAllocationError(
        byteLength,
        error instanceof Error ? error : new Error(String(error)),
        { totalAllocated: 0, limit: this.config.maxMemory }
      );
    }
  }

  /**
   * Create buffer with existing data
   */
  async createBufferWithData(buffer: ArrayBuffer, _deviceId: string): Promise<DeviceData> {
    try {
      // Check memory pressure before allocation
      await this.checkMemoryPressure(buffer.byteLength);
      
      // No need to slice - use the buffer directly
      const sourceData = new Uint8Array(buffer);

      // Use defensive wrapper for buffer creation
      const wasmHandle = await this.defensiveOperations.createBufferWithData(sourceData);

      return createWASMDeviceData(
        this.device, // Pass the actual device reference
        buffer.byteLength, 
        wasmHandle
      );
    } catch (error) {
      // Don't re-wrap our own errors
      if (error instanceof WASMAllocationError) {
        throw error;
      }
      
      throw new Error(`Failed to create buffer with data: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  /**
   * Dispose a buffer, handling reference counting and cleanup
   */
  disposeBuffer(data: DeviceData): void {
    if (!(data instanceof WASMDeviceData)) {
      throw new Error('Can only dispose WASMDeviceData through BufferLifecycleManager');
    }

    // Delegate to WASMDeviceData's dispose method which handles reference counting
    data.dispose();
  }

  /**
   * Clone a buffer with proper reference counting
   */
  cloneBuffer(data: DeviceData): DeviceData {
    if (!(data instanceof WASMDeviceData)) {
      throw new Error('Can only clone WASMDeviceData through BufferLifecycleManager');
    }

    return data.clone();
  }

  /**
   * Get memory usage statistics
   */
  async getMemoryStats() {
    try {
      return await this.defensiveOperations.getMemoryStats();
    } catch (error) {
      console.warn('Defensive getMemoryStats failed, using direct call:', error);
      return await this.operationDispatcher.get_memory_stats();
    }
  }

  /**
   * Get memory pressure information
   */
  async getMemoryPressure(): Promise<MemoryPressureStats> {
    const stats = await this.getMemoryStats();
    const totalAllocated = stats.total_allocated_bytes;
    const usageRatio = totalAllocated / this.config.maxMemory;
    
    return {
      totalAllocated,
      maxMemory: this.config.maxMemory,
      usageRatio,
      shouldCompact: usageRatio > this.config.compactThreshold
    };
  }

  /**
   * Perform intensive cleanup
   */
  async performCleanup(): Promise<void> {
    try {
      await this.defensiveOperations.intensiveCleanup();
    } catch (error) {
      console.warn('Defensive intensive cleanup failed, using direct call:', error);
      this.operationDispatcher.intensive_cleanup();
    }

    // Hint to JS garbage collector (if available)
    if (typeof global !== 'undefined' && global.gc) {
      global.gc();
    } else if (typeof window !== 'undefined' && (window as any).gc) {
      (window as any).gc();
    }
  }

  /**
   * Update memory configuration
   */
  updateConfig(newConfig: Partial<BufferLifecycleManagerConfig>): void {
    this.config = {
      ...this.config,
      ...newConfig
    };
  }

  /**
   * Get current configuration
   */
  getConfig(): BufferLifecycleManagerConfig {
    return { ...this.config };
  }

  /**
   * Check memory pressure and compact if needed
   */
  private async checkMemoryPressure(requestedBytes: number): Promise<void> {

    const stats = await this.getMemoryStats();
    const currentUsage = stats.total_allocated_bytes;
    const afterAllocation = currentUsage + requestedBytes;
    
    // Always check if we would exceed the limit
    if (afterAllocation > this.config.maxMemory) {
      if (this.config.autoCompact) {
        // Try compaction first
        this.performCleanup();
        
        // Check again after compaction
        const newStats = await this.getMemoryStats();
        const newUsage = newStats.total_allocated_bytes;
        
        if (newUsage + requestedBytes > this.config.maxMemory) {
          throw new WASMMemoryLimitError(
            newUsage,
            this.config.maxMemory,
            requestedBytes,
            { operation: 'buffer_allocation', phase: 'after_compaction' }
          );
        }
      } else {
        // No compaction enabled, fail immediately
        throw new WASMMemoryLimitError(
          currentUsage,
          this.config.maxMemory,
          requestedBytes,
          { operation: 'buffer_allocation', phase: 'no_compaction' }
        );
      }
    } else if (this.config.autoCompact && currentUsage / this.config.maxMemory > this.config.compactThreshold) {
      // Compact if we're above the threshold and haven't compacted recently
      const now = Date.now();
      if (now - this.lastCompactTime > this.compactInterval) {
        this.performCleanup();
        this.lastCompactTime = now;
      }
    }
  }

  /**
   * Test allocation capability for large buffers
   */
  private testLargeAllocation(): void {
    try {
      // Try a small test allocation first
      const testBuffer = new ArrayBuffer(1024);
      const testHandle = this.operationDispatcher.create_buffer_with_js_data(
        new Uint8Array(testBuffer),
      );
      this.operationDispatcher.release_buffer(testHandle);
    } catch (testError) {
      throw new WASMAllocationError(
        1024, // Test size
        'Memory system not ready for large allocation',
        [
          'Try disposing unused buffers',
          'Consider calling performCleanup() to free memory pools',
          'Break large allocations into smaller chunks'
        ],
        { testError: testError instanceof Error ? testError.message : String(testError) }
      );
    }
  }
}