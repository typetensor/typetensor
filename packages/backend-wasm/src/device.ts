import type {
  Device,
  DeviceData,
  AnyStorageTransformation,
  ValidateDeviceOperations,
  SliceIndex,
  DType,
} from '@typetensor/core';
import { WASMDeviceData, createWASMDeviceData } from './data';
import { loadWASMModule } from './loader';
import { dtypeToWasm, operationToWasm } from './types';
import type { WASMModule, WASMLoadOptions, WASMCapabilities, WASMMemoryStats, WASMMemoryConfig } from './types';
import type { WasmOperationDispatcher, WasmBufferHandle, WasmTensorMeta } from './types/wasm-bindings';
import { MemoryViewManager } from './memory-views';
import { getDTypeByteSize } from './utils/dtype-helpers';
import { 
  WASMErrorHandler, 
  WASMInvalidStateError, 
  WASMOperationError,
  WASMAllocationError,
  WASMBoundsError 
} from './errors';
import { BufferLifecycleManager } from './buffer-lifecycle-manager';
import { OperationOrchestrator } from './operation-orchestrator';
import { ViewManager } from './view-manager';
import { createDefensiveWASMOperations, type WASMDefensiveWrapper } from './wasm-defensive-wrapper';

export class WASMDevice implements Device {
  readonly id: string = 'wasm:0';
  readonly type: string = 'wasm';

  private wasmModule: WASMModule | null = null;
  private operationDispatcher: WasmOperationDispatcher | null = null;
  private capabilities: WASMCapabilities | null = null;
  private initialized = false;
  private memoryViewManager: MemoryViewManager | null = null;
  private memoryConfig: Required<WASMMemoryConfig>;
  
  // New abstraction layers
  private bufferLifecycleManager: BufferLifecycleManager | null = null;
  private operationOrchestrator: OperationOrchestrator | null = null;
  private viewManager: ViewManager | null = null;
  
  // Defensive WASM operations
  private defensiveOperations: any = null;
  private defensiveWrapper: WASMDefensiveWrapper | null = null;

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
    | 'flatten'
    | 'permute'
    | 'transpose'
    | 'squeeze'
    | 'unsqueeze'
    | 'expand'
    | 'tile' // View ops
    | 'matmul' // Matrix ops
    | 'softmax'
    | 'log_softmax' // Activation ops
    | 'sum'
    | 'mean'
    | 'max'
    | 'min'
    | 'prod' // Reduction ops
  > = true;

  private constructor() {
    // Set default memory configuration - no limits by default
    this.memoryConfig = {
      maxMemory: Number.MAX_SAFE_INTEGER, // No limit by default
      compactThreshold: 0.8,
      autoCompact: false, // Disabled by default
    };
  }

  /**
   * Create and initialize a new WASM device
   *
   * @param options Loading options for the WASM module
   * @returns Promise resolving to initialized WASM device
   */
  static async create(options: WASMLoadOptions = {}): Promise<WASMDevice> {
    const device = new WASMDevice();
    await device.initialize(options);
    return device;
  }

  /**
   * Initialize the WASM device
   */
  private async initialize(options: WASMLoadOptions): Promise<void> {
    if (this.initialized) {
      return;
    }

    // Apply memory configuration from options
    if (options.memoryConfig) {
      this.memoryConfig = {
        ...this.memoryConfig,
        ...options.memoryConfig,
      };
    }

    try {
      this.wasmModule = await loadWASMModule(options);

      this.operationDispatcher = new this.wasmModule.WasmOperationDispatcher();

      // Initialize memory view manager
      this.memoryViewManager = new MemoryViewManager(this.wasmModule.memory);

      // Initialize defensive WASM operations
      const defensiveOps = createDefensiveWASMOperations(this.operationDispatcher);
      this.defensiveOperations = defensiveOps;
      this.defensiveWrapper = defensiveOps.wrapper;

      // Initialize abstraction layers
      this.bufferLifecycleManager = new BufferLifecycleManager(
        this.operationDispatcher,
        this, // Pass the device reference
        this.memoryConfig
      );

      this.operationOrchestrator = new OperationOrchestrator({
        deviceId: this.id,
        wasmModule: this.wasmModule,
        operationDispatcher: this.operationDispatcher,
        device: this // Pass the actual device reference
      });

      this.viewManager = new ViewManager(
        this.memoryViewManager,
        this.operationDispatcher,
        this.id
      );

      this.capabilities = {
        simd: this.wasmModule.has_simd_support(),
        sharedMemory: this.wasmModule.has_shared_memory_support(),
        optimalThreadCount: this.wasmModule.get_optimal_thread_count(),
        availableMemory: 256 * 1024 * 1024,
        version: this.wasmModule.get_version(),
      };

      this.initialized = true;
    } catch (error) {
      throw new WASMInvalidStateError(
        'initialize',
        'uninitialized',
        'error',
        { 
          originalError: error instanceof Error ? error.message : String(error),
          wasmModuleLoaded: !!this.wasmModule 
        }
      );
    }
  }

  /**
   * Execute a tensor operation
   */
  async execute<T extends AnyStorageTransformation>(
    op: T,
    inputs: DeviceData[],
    output?: DeviceData,
  ): Promise<DeviceData> {
    this.ensureInitialized();

    // Delegate to OperationOrchestrator
    const result = await this.operationOrchestrator!.execute(op, inputs, output);
    return result.outputData;
  }

  /**
   * Check memory pressure and compact if needed
   */
  private checkMemoryPressure(requestedBytes: number): void {
    if (!this.memoryConfig.autoCompact) {
      return;
    }

    const stats = this.getMemoryStats();
    const currentUsage = stats.totalAllocated;
    const afterAllocation = currentUsage + requestedBytes;
    
    // Check if we would exceed the limit
    if (afterAllocation > this.memoryConfig.maxMemory) {
      // Try compaction first
      this.performIntensiveCleanup();
      
      // Check again after compaction
      const newStats = this.getMemoryStats();
      const newUsage = newStats.totalAllocated;
      
      if (newUsage + requestedBytes > this.memoryConfig.maxMemory) {
        throw new Error(
          `Memory limit exceeded: requested ${requestedBytes} bytes, ` +
          `current usage ${newUsage} bytes, limit ${this.memoryConfig.maxMemory} bytes`
        );
      }
    } else if (currentUsage / this.memoryConfig.maxMemory > this.memoryConfig.compactThreshold) {
      // Compact if we're above the threshold
      this.performIntensiveCleanup();
    }
  }

  /**
   * Allocate data on the WASM device
   */
  createData(byteLength: number): DeviceData {
    this.ensureInitialized();

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

    // For backward compatibility, we need to keep this synchronous
    // We'll create a synchronous version that falls back to direct calls
    try {
      // Try direct call first (synchronous)
      const wasmHandle = this.operationDispatcher!.create_empty_buffer(byteLength);
      return createWASMDeviceData(this, byteLength, wasmHandle);
    } catch (error: any) {
      console.warn('Direct buffer creation failed, this may indicate WASM corruption:', error);
      
      // Convert WASM runtime errors to proper error types
      if (error && error.message) {
        const errorMessage = error.message;
        if (errorMessage.includes('Out of bounds') || errorMessage.includes('bounds')) {
          throw new WASMBoundsError(
            'buffer allocation',
            byteLength,
            { max: MAX_ALLOCATION_SIZE },
            { requestedSize: byteLength, originalError: errorMessage }
          );
        }
        if (errorMessage.includes('out of memory') || errorMessage.includes('allocation failed')) {
          throw WASMErrorHandler.createAllocationError(
            byteLength,
            error,
            { limit: MAX_ALLOCATION_SIZE }
          );
        }
      }
      
      throw error;
    }
  }

  /**
   * Create data with existing buffer
   */
  createDataWithBuffer(buffer: ArrayBuffer): DeviceData {
    this.ensureInitialized();

    // For backward compatibility, keep this synchronous with fallback
    try {
      const sourceData = new Uint8Array(buffer);
      const wasmHandle = this.operationDispatcher!.create_buffer_with_js_data(sourceData);
      return createWASMDeviceData(this, buffer.byteLength, wasmHandle);
    } catch (error) {
      console.warn('Direct buffer creation with data failed, this may indicate WASM corruption:', error);
      throw error;
    }
  }

  /**
   * Dispose device data
   */
  disposeData(data: DeviceData): void {
    if (data.device.id !== this.id) {
      throw new Error(`Cannot dispose data from device ${data.device.id} on ${this.id}`);
    }

    // Delegate to BufferLifecycleManager
    this.bufferLifecycleManager!.disposeBuffer(data);
  }

  /**
   * Read data from WASM device
   */
  async readData(data: DeviceData): Promise<ArrayBuffer> {
    if (data.device.id !== this.id) {
      throw new Error(`Cannot read data from device ${data.device.id} on ${this.id}`);
    }

    this.ensureInitialized();

    const wasmData = data as WASMDeviceData;
    const wasmHandle = wasmData.getWASMHandle();
    
    // Use defensive wrapper for WASM operation
    const uint8Array = await this.defensiveOperations.copyBufferToJs(wasmHandle);

    // If the array is already properly aligned, return its buffer directly
    if (uint8Array.byteOffset === 0 && uint8Array.byteLength === uint8Array.buffer.byteLength) {
      return uint8Array.buffer;
    }
    
    // Only slice if necessary (when the view is a subset of the buffer)
    return uint8Array.buffer.slice(
      uint8Array.byteOffset,
      uint8Array.byteOffset + uint8Array.byteLength,
    );
  }

  /**
   * Read data from WASM device with zero-copy view (when possible)
   * Returns a typed array view directly into WASM memory
   *
   * WARNING: The returned view is only valid until:
   * - The buffer is released/disposed
   * - WASM memory grows (rare but possible)
   *
   * For long-term storage, use readData() instead
   */
  readDataView(data: DeviceData, dtype: DType<any, any, any>): ArrayBufferView {
    this.ensureInitialized();

    // Delegate to ViewManager
    return this.viewManager!.createView(data, { dtype });
  }

  /**
   * Check if a view created by readDataView is still valid
   */
  isViewValid(view: ArrayBufferView): boolean {
    this.ensureInitialized();
    return this.viewManager!.isViewValid(view);
  }

  /**
   * Write data to WASM device
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

    const wasmData = data as WASMDeviceData;
    if (wasmData.isDisposed()) {
      throw new Error('Cannot write to disposed WASMDeviceData');
    }

    // For cloned buffers, we need to decide between:
    // 1. Copy-on-write: Create new buffer, breaking sharing (current approach)
    // 2. In-place update: Modify shared buffer directly
    // 
    // We'll use copy-on-write for safety, but this means clones will diverge after writes
    
    // Invalidate all views before replacing the buffer
    this.memoryViewManager!.invalidateBuffer(wasmData.id);

    // Create the buffer directly with the defensive operation dispatcher
    // We can't use BufferLifecycleManager here because we need the raw handle
    const sourceData = new Uint8Array(buffer);
    const newHandle = await this.defensiveOperations.createBufferWithData(sourceData);

    // updateHandle will handle the old handle cleanup and update reference counting
    wasmData.updateHandle(newHandle);
  }

  /**
   * Check if WASM backend supports non-contiguous tensors for a specific operation
   */
  supportsNonContiguous(op: AnyStorageTransformation['__op']): boolean {
    this.ensureInitialized();
    return this.operationOrchestrator!.supportsNonContiguous(op);
  }

  /**
   * Get device capabilities
   */
  getCapabilities(): WASMCapabilities {
    this.ensureInitialized();
    return this.capabilities!;
  }

  /**
   * Get memory usage statistics
   */
  getMemoryStats(): WASMMemoryStats {
    this.ensureInitialized();
    
    // For backward compatibility, use direct calls for memory stats
    try {
      const wasmStats = this.operationDispatcher!.get_memory_stats();
      
      return {
        totalAllocated: wasmStats.total_allocated_bytes,
        activeBuffers: wasmStats.active_buffers,
        poolSummary: typeof wasmStats.get_pool_summary === 'function' ? wasmStats.get_pool_summary() : 'Pool summary not available',
      };
    } catch (error) {
      console.warn('Direct getMemoryStats failed:', error);
      return {
        totalAllocated: 0,
        activeBuffers: 0,
        poolSummary: 'Memory stats unavailable due to WASM error',
      };
    }
  }

  /**
   * Get current memory configuration
   */
  getMemoryConfig(): WASMMemoryConfig {
    return { ...this.memoryConfig };
  }

  /**
   * Perform intensive cleanup - useful during benchmarks or stress testing
   */
  async performIntensiveCleanup(): Promise<void> {
    this.ensureInitialized();
    
    try {
      // Use defensive wrapper for cleanup
      await this.defensiveOperations.intensiveCleanup();
    } catch (error) {
      console.warn('Defensive intensive cleanup failed, using fallback:', error);
      // Fallback to BufferLifecycleManager
      this.bufferLifecycleManager!.performCleanup();
    }
  }

  /**
   * Check if the device is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get the underlying WASM module (for advanced use cases)
   */
  getWASMModule(): WASMModule {
    this.ensureInitialized();
    return this.wasmModule!;
  }

  /**
   * Get defensive wrapper statistics (for debugging)
   */
  getDefensiveStats() {
    this.ensureInitialized();
    return this.defensiveWrapper?.getStats() || null;
  }

  /**
   * Reset defensive wrapper statistics
   */
  resetDefensiveStats(): void {
    this.ensureInitialized();
    this.defensiveWrapper?.resetStats();
  }

  /**
   * Ensure the device is initialized
   */
  private ensureInitialized(): void {
    if (!this.initialized || !this.wasmModule || !this.operationDispatcher) {
      throw new Error('WASM device not initialized. Call WASMDevice.create() first.');
    }
  }


  toString(): string {
    const status = this.initialized ? 'initialized' : 'not initialized';
    const version = this.capabilities?.version || 'unknown';
    return `WASMDevice(id=${this.id}, version=${version}, ${status})`;
  }
}
