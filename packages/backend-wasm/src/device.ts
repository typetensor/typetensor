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
import { type WASMTensorData, createWASMTensorData } from './data';
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
import {
  WASMBoundsError,
  WASMAllocationError,
  WASMOperationError,
  WASMInvalidStateError,
} from './errors';

export class WASMDevice implements Device {
  readonly type = 'wasm';
  readonly id = 'wasm:0';

  private executor: WasmExecutor | null = null;
  private initialized = false;
  private useCustomPatternCache?: { maxPatterns: number; maxMemoryMB: number };

  // @ts-expect-error: This property is used for compile-time validation only
  private _operationValidation: ValidateDeviceOperations<
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
    options: WASMLoadOptions = {},
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

    // TODO: Pass options.debug to WASM Rust core for enabling logging/debug features

    try {
      // Import and initialize WASM module
      const wasmModule = await import('../wasm/pkg/typetensor_wasm.js');
      await wasmModule.default(); // Initialize the WASM module

      // Create WasmExecutor with optional pattern cache settings
      if (this.useCustomPatternCache) {
        this.executor = wasmModule.WasmExecutor.new_with_pattern_cache(
          this.useCustomPatternCache.maxPatterns,
          this.useCustomPatternCache.maxMemoryMB,
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

    // Handle view metadata from DeviceData
    const wasmInputs = inputs.map((input) => {
      const wasmTensorData = input as WASMTensorData;

      // If this data has view metadata, create a WASM tensor with the correct shape and strides
      if (wasmTensorData.viewMetadata) {
        const viewShape = new Uint32Array(wasmTensorData.viewMetadata.shape);
        const viewStrides = new Uint32Array(wasmTensorData.viewMetadata.strides);
        // Use the new method that preserves strides for non-contiguous views
        return this.executor!.create_view_with_shape_and_strides(
          wasmTensorData.wasmTensor,
          viewShape,
          viewStrides,
        );
      }

      return wasmTensorData.wasmTensor;
    });

    // Allocate output tensor if not provided
    const outputTensor = output
      ? (output as WASMTensorData).wasmTensor
      : this.allocateOutputTensor(op);

    // Direct operation dispatch based on operation type and input count
    const wasmOp = this.mapOperation(op.__op);

    try {
      // Dispatch based on operation category
      if (this.isUnaryOperation(op.__op) && inputs.length === 1) {
        // Unary operations (neg, abs, sin, cos, exp, log, sqrt, square)
        this.executor!.execute_unary(wasmOp, wasmInputs[0]!, outputTensor);
      } else if (this.isBinaryOperation(op.__op) && inputs.length === 2) {
        // Binary operations (add, sub, mul, div)
        this.executor!.execute_binary(wasmOp, wasmInputs[0]!, wasmInputs[1]!, outputTensor);
      } else if (op.__op === 'matmul' && inputs.length === 2) {
        // Matrix multiplication
        this.executor!.execute_matmul(wasmInputs[0]!, wasmInputs[1]!, outputTensor);
      } else if (this.isViewOperation(op.__op) && inputs.length === 1) {
        // View operations (slice, reshape, flatten, transpose, etc.)
        this.executeViewOperation(wasmOp, wasmInputs[0]!, outputTensor, op);
      } else if (this.isReductionOperation(op.__op) && inputs.length === 1) {
        // Reduction operations (sum, mean, max, min, prod)
        this.executeReductionOperation(wasmOp, wasmInputs[0]!, outputTensor, op);
      } else if (this.isSoftmaxOperation(op.__op) && inputs.length === 1) {
        // Softmax operations (softmax, log_softmax)
        this.executeSoftmaxOperation(wasmOp, wasmInputs[0]!, outputTensor, op);
      } else {
        throw new Error(`Unsupported operation: ${op.__op} with ${inputs.length} inputs`);
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

    // Check for reasonable allocation size limits
    const MAX_ALLOCATION_SIZE = 1024 * 1024 * 1024; // 1GB limit
    if (byteLength > MAX_ALLOCATION_SIZE) {
      throw new WASMBoundsError(
        'buffer allocation',
        byteLength,
        { max: MAX_ALLOCATION_SIZE },
        {
          requestedSize: byteLength,
          maxSize: MAX_ALLOCATION_SIZE,
          suggestion: 'Consider using smaller allocations or streaming for large data',
        },
      );
    }

    // Default to Float32 and infer shape from byte length
    const elementCount = Math.ceil(byteLength / 4); // 4 bytes per float32
    const shape = new Uint32Array([elementCount]);

    try {
      const tensor = this.executor!.alloc_temp_tensor(DTYPES.float32, shape);
      return createWASMTensorData(this, tensor);
    } catch (error) {
      // Convert WASM errors to typed errors
      const message = error instanceof Error ? error.message : String(error);

      if (message.includes('Allocation too large')) {
        throw new WASMBoundsError(
          'buffer allocation',
          byteLength,
          { max: MAX_ALLOCATION_SIZE },
          { requestedSize: byteLength, maxSize: MAX_ALLOCATION_SIZE },
        );
      } else if (message.includes('out of memory') || message.includes('allocation failed')) {
        throw new WASMAllocationError(byteLength, message, { requestedSize: byteLength });
      } else {
        throw new WASMOperationError('createData', message, { byteLength });
      }
    }
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
  createDataWithBufferAndShape(
    buffer: ArrayBuffer,
    dtype: AnyDType,
    tensorShape: readonly number[],
  ): DeviceData {
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
    return uint8Data.buffer.slice(
      uint8Data.byteOffset,
      uint8Data.byteOffset + uint8Data.byteLength,
    ) as ArrayBuffer;
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
      bool: 1,
      float32: 4,
      float64: 8,
      int32: 4,
      uint8: 1,
    };

    const elementSize = dtypeElementSizes[dtype.__dtype];
    if (!elementSize) {
      throw new Error(`Unsupported dtype for view: ${dtype.__dtype}`);
    }

    const elementCount = totalBytes / elementSize;

    // Create appropriate typed array view based on dtype
    switch (dtype.__dtype) {
      case 'bool':
        return new Uint8Array(uint8Data.buffer, uint8Data.byteOffset, elementCount);
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
        `Buffer size mismatch: expected ${data.byteLength} bytes, got ${buffer.byteLength} bytes`,
      );
    }

    const wasmTensorData = data as WASMTensorData;
    const uint8Data = new Uint8Array(buffer);

    // Use the new WASM bridge method to copy JavaScript data to tensor
    this.executor!.copy_js_data_to_tensor(wasmTensorData.wasmTensor, uint8Data);
  }

  /**
   * Create a view of existing device data with different shape/strides
   *
   * This creates a zero-copy view by reusing the same underlying WASM tensor
   * but wrapping it with new view metadata.
   */
  createView(
    data: DeviceData,
    shape: readonly number[],
    strides: readonly number[],
    offset: number,
    dtype: { __byteSize: number },
  ): DeviceData {
    this.ensureInitialized();

    const wasmTensorData = data as WASMTensorData;

    // Create new WASMTensorData with view metadata
    // The underlying WASM tensor remains the same (zero-copy)
    return createWASMTensorData(this, wasmTensorData.wasmTensor, {
      shape,
      strides,
      offset,
      dtype,
    });
  }

  /**
   * Check if device supports non-contiguous tensors
   */
  supportsNonContiguous(op: AnyStorageTransformation['__op']): boolean {
    // WASM backend currently assumes C-contiguous data for most operations
    // Operations that don't properly handle stride-based memory access need contiguous data

    // Operations that properly handle non-contiguous data via strides
    const strideAwareOps = new Set([
      // Only simple unary operations that process elements linearly
      // These work because they just transform each element in place
      'neg',
      'abs',
      'sin',
      'cos',
      'exp',
      'log',
      'sqrt',
      'square',
    ]);

    // For all other operations, require contiguous data
    // This includes: tile, expand, slice, transpose, binary ops, reductions, etc.
    return strideAwareOps.has(op);
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
      version: '0.1.0',
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
      poolSummary: `Arena: ${wasmStats.arena_used}/${wasmStats.arena_capacity} bytes`,
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
   * Check if operation is a unary operation
   */
  private isUnaryOperation(op: string): boolean {
    return ['neg', 'abs', 'sin', 'cos', 'exp', 'log', 'sqrt', 'square'].includes(op);
  }

  /**
   * Check if operation is a binary operation
   */
  private isBinaryOperation(op: string): boolean {
    return ['add', 'sub', 'mul', 'div'].includes(op);
  }

  /**
   * Check if operation is a view operation
   */
  private isViewOperation(op: string): boolean {
    return [
      'reshape',
      'view',
      'slice',
      'flatten',
      'permute',
      'transpose',
      'squeeze',
      'unsqueeze',
      'expand',
      'tile',
    ].includes(op);
  }

  /**
   * Check if operation is a reduction operation
   */
  private isReductionOperation(op: string): boolean {
    return ['sum', 'mean', 'max', 'min', 'prod'].includes(op);
  }

  /**
   * Check if operation is a softmax operation
   */
  private isSoftmaxOperation(op: string): boolean {
    return ['softmax', 'log_softmax'].includes(op);
  }

  /**
   * Execute view operation using WASM view operations
   */
  private executeViewOperation(
    wasmOp: WasmOperation,
    input: WasmTensor,
    output: WasmTensor,
    op?: AnyStorageTransformation,
  ): void {
    // Check if this is a slice operation and we have operation metadata
    if (wasmOp === OPS.slice && op?.__output && '__sliceIndices' in op.__output) {
      // Extract slice indices from operation metadata
      const sliceIndices = (op.__output as any).__sliceIndices;

      if (Array.isArray(sliceIndices) && sliceIndices.length > 0) {
        // Extract start offsets for row and column dimensions
        const rowStart = sliceIndices[0]?.start || 0;
        const colStart = sliceIndices[1]?.start || 0;

        // Check if the input tensor is non-contiguous by looking at its layout
        // For non-contiguous tensors, we need to pass stride information
        const inputMeta = input.meta;

        const isContiguous =
          inputMeta.shape.length <= 1 ||
          (inputMeta.strides && this.isContiguousStrides(Array.from(inputMeta.shape), Array.from(inputMeta.strides)));

        if (!isContiguous && inputMeta.strides) {
          // Non-contiguous tensor: pass stride information
          const strides = Array.from(inputMeta.strides);
          this.executor!.execute_slice(input, output, rowStart, colStart, new Uint32Array(strides));
        } else {
          // Contiguous tensor: use fast path without strides
          this.executor!.execute_slice(input, output, rowStart, colStart, undefined);
        }
        return;
      }
    }

    // For all other view operations, use the standard view operation handler
    this.executor!.execute_unary(wasmOp, input, output);
  }

  /**
   * Check if strides represent contiguous memory layout
   */
  private isContiguousStrides(shape: readonly number[], strides: readonly number[]): boolean {
    // C-contiguous: strides should be [product(shape[i+1:]), ..., shape[-2], 1]
    let expectedStride = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      if (strides[i] !== expectedStride) {
        return false;
      }
      expectedStride *= shape[i]!;
    }
    return true;
  }

  /**
   * Execute reduction operation using WASM reduction operations
   */
  private executeReductionOperation(
    wasmOp: WasmOperation,
    input: WasmTensor,
    output: WasmTensor,
    op?: AnyStorageTransformation,
  ): void {
    // Extract axis information based on operation type
    let axis: number[] | null = null;
    let keepDims = false;

    if (op) {
      // Look for axis information in operation metadata
      // Different operations store axis information with different keys
      const opAny = op as any;

      if ('__sumAxes' in opAny) {
        axis = opAny.__sumAxes;
      } else if ('__meanAxes' in opAny) {
        axis = opAny.__meanAxes;
      } else if ('__maxAxes' in opAny) {
        axis = opAny.__maxAxes;
      } else if ('__minAxes' in opAny) {
        axis = opAny.__minAxes;
      } else if ('__prodAxes' in opAny) {
        axis = opAny.__prodAxes;
      }

      keepDims = ('__keepDims' in opAny && opAny.__keepDims) || false;
    }

    if (axis && Array.isArray(axis) && axis.length > 0) {
      // Use the new dedicated reduction method with axis information
      this.executor!.execute_reduction(wasmOp, input, output, new Uint32Array(axis), keepDims);
    } else {
      // No axis specified - reduce all dimensions
      this.executor!.execute_reduction(wasmOp, input, output, null, false);
    }
  }

  /**
   * Execute softmax operation using WASM softmax operations
   */
  private executeSoftmaxOperation(
    wasmOp: WasmOperation,
    input: WasmTensor,
    output: WasmTensor,
    op?: AnyStorageTransformation,
  ): void {
    // Extract axis information from operation metadata
    let axis: number | null = null;

    if (op) {
      const opAny = op as any;

      // Look for softmax axis information in operation metadata
      // Different operations use different field names
      if ('__softmaxAxis' in opAny) {
        axis = opAny.__softmaxAxis;
      } else if ('__logSoftmaxAxis' in opAny) {
        axis = opAny.__logSoftmaxAxis;
      } else if ('__axis' in opAny) {
        axis = opAny.__axis;
      } else if ('__dim' in opAny) {
        axis = opAny.__dim;
      }
    }

    // Use the new dedicated softmax method with axis information
    this.executor!.execute_softmax(wasmOp, input, output, axis);
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
      throw new Error(
        `Unsupported dtype: ${dtype.__dtype}. Available: ${Object.keys(DTYPES).join(', ')}`,
      );
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
      throw new WASMInvalidStateError('use device', 'not initialized', 'initialized', {
        hint: 'Call WASMDevice.create() first.',
      });
    }
  }

  toString(): string {
    const status = this.initialized ? 'initialized' : 'not initialized';
    return `WASMDevice(id=${this.id}, ${status})`;
  }
}
