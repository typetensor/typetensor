/**
 * Minimal WASM Device - Direct WasmExecutor integration
 * 
 * This implementation removes all abstraction layers and directly
 * uses WasmExecutor with arena-based memory management.
 */

import type {
  Device,
  DeviceData,
  AnyStorageTransformation,
  ValidateDeviceOperations,
  AnyDType,
} from '@typetensor/core';
import { WASMTensorData, createWASMTensorData } from './data';
import type {
  WasmExecutor,
  WasmTensor,
  WasmDType,
  WasmOperation,
  WASMCapabilities,
  WASMMemoryStats,
  WASMLoadOptions,
} from './types';
import { OPS, DTYPES } from './types';

export class WASMDevice implements Device {
  readonly type = 'wasm';
  readonly id = 'wasm:0';
  
  private executor: WasmExecutor | null = null;
  private initialized = false;
  private useCustomPatternCache?: { maxPatterns: number; maxMemoryMB: number };

  // @ts-expect-error: This property is used for compile-time validation only
  private _operationValidation: ValidateDeviceOperations<
    | 'neg' | 'abs' | 'sin' | 'cos' | 'exp' | 'log' | 'sqrt' | 'square' // Unary ops
    | 'add' | 'sub' | 'mul' | 'div' // Binary ops  
    | 'matmul' // Matrix ops
  > = true;

  /**
   * Create and initialize a new WASM device
   */
  static async create(options: WASMLoadOptions = {}): Promise<WASMDevice> {
    const device = new WASMDevice();
    await device.init(options);
    return device;
  }

  /**
   * Create WASM device with custom pattern cache settings
   */
  static async createWithPatternCache(
    maxPatterns: number,
    maxMemoryMB: number,
    options: WASMLoadOptions = {}
  ): Promise<WASMDevice> {
    const device = new WASMDevice();
    device.useCustomPatternCache = { maxPatterns, maxMemoryMB };
    await device.init(options);
    return device;
  }

  /**
   * Initialize the WASM device
   */
  private async init(_options: WASMLoadOptions): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      // Import WASM module (auto-initializes)
      const wasmModule = await import('../wasm/pkg/typetensor_wasm.js');
      
      // Create WasmExecutor with optional pattern cache settings
      if (this.useCustomPatternCache) {
        this.executor = wasmModule.WasmExecutor.new_with_pattern_cache(
          this.useCustomPatternCache.maxPatterns,
          this.useCustomPatternCache.maxMemoryMB
        );
      } else {
        this.executor = new wasmModule.WasmExecutor();
      }
      
      this.initialized = true;
    } catch (error) {
      throw new Error(`Failed to initialize WASM device: ${error}`);
    }
  }

  /**
   * Execute a tensor operation with direct WasmExecutor calls
   */
  async execute<T extends AnyStorageTransformation>(
    op: T,
    inputs: DeviceData[],
    output?: DeviceData,
  ): Promise<DeviceData> {
    this.ensureInitialized();

    // Convert inputs to WasmTensor
    const wasmInputs = inputs.map(input => (input as WASMTensorData).wasmTensor);
    
    // Allocate output tensor if not provided
    const outputTensor = output ? 
      (output as WASMTensorData).wasmTensor :
      this.allocateOutputTensor(op);

    // Direct operation dispatch based on input count
    const wasmOp = this.mapOperation(op.__op);
    
    try {
      if (inputs.length === 1) {
        // Unary operation
        this.executor!.execute_unary(wasmOp, wasmInputs[0]!, outputTensor);
      } else if (inputs.length === 2) {
        // Binary or matrix operation
        if (op.__op === 'matmul') {
          this.executor!.execute_matmul(wasmInputs[0]!, wasmInputs[1]!, outputTensor);
        } else {
          this.executor!.execute_binary(wasmOp, wasmInputs[0]!, wasmInputs[1]!, outputTensor);
        }
      } else {
        throw new Error(`Unsupported input count: ${inputs.length} for operation ${op.__op}`);
      }

      return createWASMTensorData(this, outputTensor);

    } catch (error) {
      throw new Error(`WASM operation failed: ${error}`);
    }
  }

  /**
   * Create data using arena allocation
   */
  createData(byteLength: number): DeviceData {
    this.ensureInitialized();

    // Default to Float32 and infer shape from byte length
    const elementCount = Math.ceil(byteLength / 4); // 4 bytes per float32
    const shape = new Uint32Array([elementCount]);
    
    const tensor = this.executor!.alloc_temp_tensor(DTYPES.float32, shape);
    return createWASMTensorData(this, tensor);
  }

  /**
   * Create data from existing buffer
   */
  createDataWithBuffer(buffer: ArrayBuffer): DeviceData {
    this.ensureInitialized();

    const data = new Uint8Array(buffer);
    const shape = new Uint32Array([Math.ceil(buffer.byteLength / 4)]); // Assume Float32
    
    const tensor = this.executor!.tensor_from_data(data, DTYPES.float32, shape);
    return createWASMTensorData(this, tensor);
  }

  /**
   * Create data from existing buffer with specific shape and dtype
   */
  createDataWithBufferAndShape(buffer: ArrayBuffer, dtype: AnyDType, tensorShape: readonly number[]): DeviceData {
    this.ensureInitialized();

    const data = new Uint8Array(buffer);
    const shape = new Uint32Array(tensorShape);
    const wasmDType = this.mapDType(dtype);
    
    const tensor = this.executor!.tensor_from_data(data, wasmDType, shape);
    return createWASMTensorData(this, tensor);
  }

  /**
   * Dispose device data (no-op - arena handles cleanup)
   */
  disposeData(data: DeviceData): void {
    // Arena automatically handles cleanup - no manual disposal needed
    if (data.device.id !== this.id) {
      throw new Error(`Cannot dispose data from device ${data.device.id} on ${this.id}`);
    }
  }

  /**
   * Read data from WASM device
   */
  async readData(data: DeviceData): Promise<ArrayBuffer> {
    this.ensureInitialized();
    
    if (data.device.id !== this.id) {
      throw new Error(`Cannot read data from device ${data.device.id} on ${this.id}`);
    }

    const wasmTensorData = data as WASMTensorData;
    
    // Use the new WASM bridge method to copy tensor data to JavaScript
    const uint8Data = this.executor!.copy_tensor_data_to_js(wasmTensorData.wasmTensor);
    
    // Convert Uint8Array to ArrayBuffer
    return uint8Data.buffer.slice(uint8Data.byteOffset, uint8Data.byteOffset + uint8Data.byteLength);
  }

  /**
   * Create typed array view with actual tensor data
   */
  readDataView(data: DeviceData, dtype: AnyDType): ArrayBufferView {
    this.ensureInitialized();
    
    if (data.device.id !== this.id) {
      throw new Error(`Cannot read data from device ${data.device.id} on ${this.id}`);
    }

    const wasmTensorData = data as WASMTensorData;
    const tensor = wasmTensorData.wasmTensor;
    
    // Get the actual tensor data from WASM memory
    const uint8Data = this.executor!.copy_tensor_data_to_js(tensor);
    
    // Calculate element count based on total bytes and requested dtype element size
    const totalBytes = uint8Data.byteLength;
    const dtypeElementSizes: Record<string, number> = {
      'float32': 4,
      'float64': 8,
      'int32': 4,
      'uint8': 1,
    };
    
    const elementSize = dtypeElementSizes[dtype.__dtype];
    if (!elementSize) {
      throw new Error(`Unsupported dtype for view: ${dtype.__dtype}`);
    }
    
    const elementCount = totalBytes / elementSize;
    
    // Create appropriate typed array view based on dtype
    switch (dtype.__dtype) {
      case 'float32':
        return new Float32Array(uint8Data.buffer, uint8Data.byteOffset, elementCount);
      case 'float64':
        return new Float64Array(uint8Data.buffer, uint8Data.byteOffset, elementCount);
      case 'int32':
        return new Int32Array(uint8Data.buffer, uint8Data.byteOffset, elementCount);
      case 'uint8':
        return new Uint8Array(uint8Data.buffer, uint8Data.byteOffset, elementCount);
      default:
        throw new Error(`Unsupported dtype for view: ${dtype.__dtype}`);
    }
  }

  /**
   * Check if view is valid (always true with arena - arena handles safety)
   */
  isViewValid(_view: ArrayBufferView): boolean {
    return true; // Arena ensures views are always safe
  }

  /**
   * Write data to WASM device
   */
  async writeData(data: DeviceData, buffer: ArrayBuffer): Promise<void> {
    this.ensureInitialized();
    
    if (data.device.id !== this.id) {
      throw new Error(`Cannot write data to device ${data.device.id} on ${this.id}`);
    }

    if (buffer.byteLength !== data.byteLength) {
      throw new Error(
        `Buffer size mismatch: expected ${data.byteLength} bytes, got ${buffer.byteLength} bytes`
      );
    }

    const wasmTensorData = data as WASMTensorData;
    const uint8Data = new Uint8Array(buffer);
    
    // Use the new WASM bridge method to copy JavaScript data to tensor
    this.executor!.copy_js_data_to_tensor(wasmTensorData.wasmTensor, uint8Data);
  }

  /**
   * Check if device supports non-contiguous tensors
   */
  supportsNonContiguous(_op: AnyStorageTransformation['__op']): boolean {
    // Most operations support non-contiguous tensors with arena allocation
    return true;
  }

  /**
   * Get device capabilities
   */
  getCapabilities(): WASMCapabilities {
    this.ensureInitialized();
    
    return {
      simd: true, // Assume SIMD support
      sharedMemory: false,
      optimalThreadCount: 1,
      availableMemory: 256 * 1024 * 1024, // 256MB
      version: '0.1.0'
    };
  }

  /**
   * Get memory usage statistics
   */
  getMemoryStats(): WASMMemoryStats {
    this.ensureInitialized();
    
    const wasmStats = this.executor!.memory_stats();
    
    return {
      totalAllocated: wasmStats.arena_used,
      activeBuffers: 0, // Arena doesn't track individual buffers
      poolSummary: `Arena: ${wasmStats.arena_used}/${wasmStats.arena_capacity} bytes`
    };
  }

  /**
   * Check if device is initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Scope management for automatic cleanup
   */
  withScope<T>(fn: () => T): T {
    this.ensureInitialized();
    
    const checkpoint = this.executor!.checkpoint();
    try {
      return fn();
    } finally {
      this.executor!.restore(checkpoint);
    }
  }

  /**
   * Begin a memory scope (returns checkpoint ID)
   */
  beginScope(): number {
    this.ensureInitialized();
    return this.executor!.checkpoint();
  }

  /**
   * End a memory scope (restore to checkpoint)
   */
  endScope(checkpointId: number): void {
    this.ensureInitialized();
    this.executor!.restore(checkpointId);
  }

  /**
   * Garbage collect persistent tensors
   */
  gc(): number {
    this.ensureInitialized();
    return this.executor!.gc();
  }

  /**
   * Get pattern cache statistics for optimization insights
   */
  getPatternCacheStats(): import('../wasm/pkg/typetensor_wasm').PatternCacheStats {
    this.ensureInitialized();
    return this.executor!.pattern_cache_stats();
  }

  /**
   * Enable or disable pattern optimization
   */
  setPatternOptimization(enabled: boolean): void {
    this.ensureInitialized();
    this.executor!.set_pattern_optimization(enabled);
  }

  /**
   * Clear the pattern cache (useful for benchmarking)
   */
  clearPatternCache(): void {
    this.ensureInitialized();
    this.executor!.clear_pattern_cache();
  }

  /**
   * Allocate a persistent tensor (manual memory management)
   */
  createPersistentData(dtype: AnyDType, shape: number[]): DeviceData {
    this.ensureInitialized();
    
    const wasmDType = this.mapDType(dtype);
    const wasmShape = new Uint32Array(shape);
    
    const tensor = this.executor!.alloc_persistent_tensor(wasmDType, wasmShape);
    return createWASMTensorData(this, tensor);
  }

  /**
   * Map TypeTensor operation name to WasmOperation
   */
  private mapOperation(op: string): WasmOperation {
    const wasmOp = OPS[op as keyof typeof OPS];
    if (wasmOp === undefined) {
      throw new Error(`Unsupported operation: ${op}. Available: ${Object.keys(OPS).join(', ')}`);
    }
    return wasmOp;
  }

  /**
   * Map TypeTensor dtype to WasmDType
   */
  private mapDType(dtype: AnyDType): WasmDType {
    const wasmDType = DTYPES[dtype.__dtype as keyof typeof DTYPES];
    if (wasmDType === undefined) {
      throw new Error(`Unsupported dtype: ${dtype.__dtype}. Available: ${Object.keys(DTYPES).join(', ')}`);
    }
    return wasmDType;
  }

  /**
   * Allocate output tensor based on operation metadata
   */
  private allocateOutputTensor(op: AnyStorageTransformation): WasmTensor {
    const shape = new Uint32Array(op.__output.__shape);
    const dtype = this.mapDType(op.__output.__dtype);
    
    return this.executor!.alloc_temp_tensor(dtype, shape);
  }

  /**
   * Ensure device is initialized
   */
  private ensureInitialized(): void {
    if (!this.initialized || !this.executor) {
      throw new Error('WASM device not initialized. Call WASMDevice.create() first.');
    }
  }

  toString(): string {
    const status = this.initialized ? 'initialized' : 'not initialized';
    return `WASMDevice(id=${this.id}, ${status})`;
  }
}