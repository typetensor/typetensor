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

export class WASMDevice implements Device {
  readonly id: string = 'wasm:0';
  readonly type: string = 'wasm';

  private wasmModule: WASMModule | null = null;
  private operationDispatcher: WasmOperationDispatcher | null = null;
  private capabilities: WASMCapabilities | null = null;
  private initialized = false;
  private memoryViewManager: MemoryViewManager | null = null;
  private memoryConfig: Required<WASMMemoryConfig>;

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

      this.capabilities = {
        simd: this.wasmModule.has_simd_support(),
        sharedMemory: this.wasmModule.has_shared_memory_support(),
        optimalThreadCount: this.wasmModule.get_optimal_thread_count(),
        availableMemory: 256 * 1024 * 1024,
        version: this.wasmModule.get_version(),
      };

      this.initialized = true;
    } catch (error) {
      throw new Error(`Failed to initialize WASM device: ${error}`);
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

    for (const input of inputs) {
      if (input.device.id !== this.id) {
        throw new Error(`Input tensor is on device ${input.device.id}, expected ${this.id}`);
      }
    }

    if (output && output.device.id !== this.id) {
      throw new Error(`Output tensor is on device ${output.device.id}, expected ${this.id}`);
    }

    if (op.__op === 'slice') {
      return this.executeSliceOp(op as any, inputs[0]!);
    }

    let reductionAxes: number[] | null = null;
    let keepDims = false;
    if (
      op.__op === 'sum' ||
      op.__op === 'mean' ||
      op.__op === 'max' ||
      op.__op === 'min' ||
      op.__op === 'prod'
    ) {
      const reductionOp = op as any;
      const axesKey = `__${op.__op}Axes`; // e.g., __sumAxes, __meanAxes, __prodAxes
      reductionAxes =
        reductionOp[axesKey] === undefined ? null : Array.from(reductionOp[axesKey] || []);
      keepDims = reductionOp.__keepDims || false;
    }

    let softmaxAxis: number | null = null;
    if (op.__op === 'softmax' || op.__op === 'log_softmax') {
      const softmaxOp = op as any;
      if (op.__op === 'softmax') {
        softmaxAxis = softmaxOp.__softmaxAxis ?? null;
      } else {
        softmaxAxis = softmaxOp.__logSoftmaxAxis ?? null;
      }
    }

    try {
      const wasmInputs: WasmBufferHandle[] = [];
      const inputHandles: WasmBufferHandle[] = [];

      for (let i = 0; i < inputs.length; i++) {
        const wasmData = inputs[i] as WASMDeviceData;
        const handle = wasmData.getWASMHandle() as WasmBufferHandle;

        inputHandles.push(handle);
        wasmInputs.push(handle);
      }
      const inputMetas = inputs.map(
        (input, i) => this.createTensorMeta([...op.__inputs], input, i),
      );

      const outputMeta = this.createTensorMeta([op.__output], null, 0);
      const wasmOperation = operationToWasm(op.__op);

      // ALWAYS pre-allocate output buffer upfront to avoid nested RefCell borrows
      let outputHandle: WasmBufferHandle;
      if (output) {
        outputHandle = (output as WASMDeviceData).getWASMHandle();
      } else {
        // Pre-allocate output buffer - single RefCell borrow
        const outputSize = op.__output.__size * getDTypeByteSize(op.__output.__dtype);
        const zeroBuffer = new ArrayBuffer(outputSize);
        outputHandle = this.operationDispatcher.create_buffer_with_js_data(
          new Uint8Array(zeroBuffer),
        );
      }

      let resultHandle;
      if (inputs.length === 0) {
        // For 0-input operations, we already have the pre-allocated buffer
        resultHandle = outputHandle;
      } else if (inputs.length === 1) {
        if (reductionAxes !== null) {
          resultHandle = this.operationDispatcher.execute_reduction(
            wasmOperation,
            wasmInputs[0],
            inputMetas[0],
            outputMeta,
            reductionAxes.length > 0 ? reductionAxes : null,
            keepDims,
            outputHandle, // Always provided - never null
          );
        } else if (softmaxAxis !== null) {
          resultHandle = this.operationDispatcher.execute_softmax(
            wasmOperation,
            wasmInputs[0],
            inputMetas[0],
            outputMeta,
            softmaxAxis,
            outputHandle, // Always provided - never null
          );
        } else {
          resultHandle = this.operationDispatcher.execute_unary(
            wasmOperation,
            wasmInputs[0],
            inputMetas[0],
            outputMeta,
            outputHandle, // Always provided - never null
          );
        }
      } else if (inputs.length === 2) {
        resultHandle = this.operationDispatcher.execute_binary(
          wasmOperation,
          wasmInputs[0],
          wasmInputs[1],
          inputMetas[0],
          inputMetas[1],
          outputMeta,
          outputHandle, // Always provided - never null
        );
      } else {
        throw new Error(`Unsupported number of inputs: ${inputs.length}`);
      }

      const resultSize = op.__output.__size * getDTypeByteSize(op.__output.__dtype);
      return createWASMDeviceData(this, resultSize, resultHandle);
    } catch (error) {
      throw new Error(`WASM operation '${op.__op}' failed: ${error}`);
    }
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

    try {
      // Check for reasonable allocation size (e.g., max 1GB)
      const MAX_ALLOCATION_SIZE = 1024 * 1024 * 1024; // 1GB
      if (byteLength > MAX_ALLOCATION_SIZE) {
        throw new Error(`Requested allocation size ${byteLength} exceeds maximum allowed size of ${MAX_ALLOCATION_SIZE} bytes`);
      }
      
      // Check memory pressure before allocation
      this.checkMemoryPressure(byteLength);
      
      // For large allocations, try to allocate in smaller chunks first to test memory availability
      if (byteLength > 50 * 1024 * 1024) { // > 50MB
        try {
          // Try a small test allocation first
          const testBuffer = new ArrayBuffer(1024);
          const testHandle = this.operationDispatcher.create_buffer_with_js_data(
            new Uint8Array(testBuffer),
          );
          this.operationDispatcher.release_buffer(testHandle);
        } catch (testError) {
          throw new Error(`Memory system not ready for large allocation: ${testError}`);
        }
      }
      
      // Use zero-allocation method - no JavaScript buffer needed
      const wasmHandle = this.operationDispatcher.create_empty_buffer(byteLength);
      return createWASMDeviceData(this, byteLength, wasmHandle);
    } catch (error: any) {
      // Handle specific WASM out-of-bounds errors
      if (error.message && error.message.includes('Out of bounds memory access')) {
        throw new Error(
          `Failed to allocate ${byteLength} bytes: WASM memory limit exceeded. ` +
          `Try allocating smaller buffers or increasing WASM memory limit.`
        );
      }
      throw new Error(`Failed to allocate ${byteLength} bytes on WASM device: ${error}`);
    }
  }

  /**
   * Create data with existing buffer
   */
  createDataWithBuffer(buffer: ArrayBuffer): DeviceData {
    this.ensureInitialized();

    try {
      // No need to slice - use the buffer directly
      const sourceData = new Uint8Array(buffer);

      // Handle Result type from WASM
      let wasmHandle;
      try {
        wasmHandle = this.operationDispatcher.create_buffer_with_js_data(sourceData);
      } catch (wasmError) {
        throw new Error(`WASM allocation failed: ${wasmError}`);
      }

      return createWASMDeviceData(this, buffer.byteLength, wasmHandle);
    } catch (error) {
      throw new Error(`Failed to create data with buffer: ${error}`);
    }
  }

  /**
   * Dispose device data
   */
  disposeData(data: DeviceData): void {
    if (data.device.id !== this.id) {
      throw new Error(`Cannot dispose data from device ${data.device.id} on ${this.id}`);
    }

    const wasmData = data as WASMDeviceData;

    if (!wasmData.isDisposed()) {
      // Invalidate all views for this buffer BEFORE releasing it
      this.memoryViewManager!.invalidateBuffer(wasmData.id);

      const wasmHandle = wasmData.getWASMHandle();

      const released = this.operationDispatcher.release_buffer(wasmHandle);

      if (!released) {
        console.warn(`Failed to release buffer ${wasmData.id}`);
      }
    }
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
    const uint8Array = this.operationDispatcher!.copy_buffer_to_js(wasmHandle);

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
    if (data.device.id !== this.id) {
      throw new Error(`Cannot read data from device ${data.device.id} on ${this.id}`);
    }

    this.ensureInitialized();

    const wasmData = data as WASMDeviceData;
    const wasmHandle = wasmData.getWASMHandle() as WasmBufferHandle;

    // Get buffer info: [ptr, size, initialized]
    const info = this.operationDispatcher!.get_buffer_view_info(wasmHandle);
    const ptr = info[0];
    const size = info[1];
    const initialized = info[2];

    if (!initialized) {
      throw new Error('Cannot create view of uninitialized buffer');
    }

    // Create safe zero-copy view with lifetime tracking
    const bufferId = wasmData.id;
    return this.memoryViewManager!.createSafeView(bufferId, ptr, size / getDTypeByteSize(dtype), dtype);
  }

  /**
   * Check if a view created by readDataView is still valid
   */
  isViewValid(view: ArrayBufferView): boolean {
    this.ensureInitialized();
    return this.memoryViewManager!.isViewValid(view);
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
    if (!wasmData.isDisposed()) {
      // Invalidate all views before replacing the buffer
      this.memoryViewManager!.invalidateBuffer(wasmData.id);
    }

    const sourceData = new Uint8Array(buffer);
    const newHandle = this.operationDispatcher.create_buffer_with_js_data(sourceData);

    // updateHandle will handle the old handle cleanup
    wasmData.updateHandle(newHandle);
  }

  /**
   * Check if WASM backend supports non-contiguous tensors for a specific operation
   */
  supportsNonContiguous(_op: AnyStorageTransformation['__op']): boolean {
    return false;
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
    const wasmStats = this.operationDispatcher.get_memory_stats();

    return {
      totalAllocated: wasmStats.total_allocated_bytes,
      activeBuffers: wasmStats.active_buffers,
      poolSummary: wasmStats.get_pool_summary(),
    };
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
  performIntensiveCleanup(): void {
    this.ensureInitialized();
    this.operationDispatcher.intensive_cleanup();

    // Hint to JS garbage collector (if available)
    if (typeof global !== 'undefined' && global.gc) {
      global.gc();
    } else if (typeof window !== 'undefined' && (window as any).gc) {
      (window as any).gc();
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
   * Ensure the device is initialized
   */
  private ensureInitialized(): void {
    if (!this.initialized || !this.wasmModule || !this.operationDispatcher) {
      throw new Error('WASM device not initialized. Call WASMDevice.create() first.');
    }
  }

  /**
   * Create tensor metadata for WASM operations
   */
  private createTensorMeta(tensorInfos: any[], _data: DeviceData | null, index = 0): WasmTensorMeta {
    const tensorInfo = tensorInfos[index];
    if (!tensorInfo) {
      throw new Error(`Missing tensor info at index ${index}`);
    }

    const wasmDtype = dtypeToWasm(tensorInfo.__dtype);
    const shapeArray = tensorInfo.__shape.map((n: any) => Math.floor(Number(n)));
    const stridesArray = tensorInfo.__strides.map((n: any) => Math.floor(Number(n)));
    const size = Math.floor(Number(tensorInfo.__size));
    const offset = Math.floor(Number(tensorInfo.__offset || 0));

    const WasmTensorMeta = this.wasmModule!.WasmTensorMeta;

    const meta = new WasmTensorMeta(wasmDtype, shapeArray, stridesArray, size, offset);

    return meta;
  }

  /**
   * Execute a slice operation on WASM device (TypeScript implementation)
   */
  private async executeSliceOp(
    op: AnyStorageTransformation & { __op: 'slice' },
    input: DeviceData,
  ): Promise<DeviceData> {
    const sliceIndices = (op.__output as any).__sliceIndices as SliceIndex[];
    const inputStorage = op.__inputs[0];
    if (!inputStorage) {
      throw new Error('Slice operation missing input storage metadata');
    }

    const inputShape = inputStorage.__shape;
    const inputStrides = inputStorage.__strides;
    const dtype = inputStorage.__dtype;

    const outputShape = op.__output.__shape;
    const outputBuffer = await this.createSlicedBuffer(
      input,
      sliceIndices,
      inputShape,
      inputStrides,
      outputShape,
      dtype,
    );

    return this.createDataWithBuffer(outputBuffer);
  }

  /**
   * Create sliced buffer from input data
   */
  private async createSlicedBuffer(
    input: DeviceData,
    sliceIndices: SliceIndex[],
    inputShape: readonly number[],
    inputStrides: readonly number[],
    outputShape: readonly number[],
    dtype: DType<any, any, any>,
  ): Promise<ArrayBuffer> {
    // Validate slice indices before processing
    this.validateSliceIndices(sliceIndices, inputShape);
    
    const inputBuffer = await this.readData(input);

    const outputSize = outputShape.reduce((a, b) => a * b, 1);
    const outputBuffer = new ArrayBuffer(outputSize * getDTypeByteSize(dtype));

    const inputArray = this.createTypedArray(inputBuffer, dtype);
    const outputArray = this.createTypedArray(outputBuffer, dtype);
    
    // Add bounds checking
    const inputLength = inputArray.length;
    
    for (let outputFlatIndex = 0; outputFlatIndex < outputSize; outputFlatIndex++) {
      const outputIndices = this.flatIndexToIndices(outputFlatIndex, outputShape);
      const inputIndices = this.mapOutputToInputIndices(outputIndices, sliceIndices, inputShape);

      const inputFlatIndex = this.computeFlatIndex(inputIndices, inputStrides);
      
      // BOUNDS CHECK: Validate index is within bounds
      if (inputFlatIndex < 0 || inputFlatIndex >= inputLength) {
        throw new Error(
          `Index ${inputFlatIndex} out of bounds for array of length ${inputLength}. ` +
          `Input indices: [${inputIndices.join(', ')}], shape: [${inputShape.join(', ')}]`
        );
      }
      
      outputArray[outputFlatIndex] = inputArray[inputFlatIndex];
    }

    return outputBuffer;
  }

  /**
   * Validate slice indices to ensure they are within bounds
   */
  private validateSliceIndices(
    sliceIndices: SliceIndex[],
    shape: readonly number[]
  ): void {
    for (let i = 0; i < sliceIndices.length && i < shape.length; i++) {
      const slice = sliceIndices[i];
      const dimSize = shape[i];
      
      if (dimSize === undefined || dimSize <= 0) {
        throw new Error(`Invalid shape dimension at index ${i}: ${dimSize}`);
      }
      
      if (typeof slice === 'number') {
        const normalizedIndex = slice < 0 ? dimSize + slice : slice;
        if (normalizedIndex < 0 || normalizedIndex >= dimSize) {
          throw new Error(
            `Slice index ${slice} is out of bounds for dimension ${i} of size ${dimSize}`
          );
        }
      } else if (slice && typeof slice === 'object') {
        const start = slice.start ?? 0;
        const stop = slice.stop ?? dimSize;
        const step = slice.step ?? 1;
        
        const normalizedStart = start < 0 ? dimSize + start : start;
        const normalizedStop = stop < 0 ? dimSize + stop : stop;
        
        if (normalizedStart < 0 || normalizedStart > dimSize) {
          throw new Error(
            `Slice start ${start} is out of bounds for dimension ${i} of size ${dimSize}`
          );
        }
        
        if (normalizedStop < 0 || normalizedStop > dimSize) {
          throw new Error(
            `Slice stop ${stop} is out of bounds for dimension ${i} of size ${dimSize}`
          );
        }
        
        if (step === 0) {
          throw new Error(`Slice step cannot be zero for dimension ${i}`);
        }
        
        if (step < 0 && normalizedStart < normalizedStop) {
          throw new Error(
            `Invalid slice range for negative step: start=${start}, stop=${stop}, step=${step}`
          );
        }
        
        if (step > 0 && normalizedStart > normalizedStop) {
          throw new Error(
            `Invalid slice range for positive step: start=${start}, stop=${stop}, step=${step}`
          );
        }
      }
    }
  }

  /**
   * Create typed array from buffer based on dtype
   */
  private createTypedArray(buffer: ArrayBuffer, dtype: DType<any, any, any>): ArrayBufferView {
    switch (dtype.__name || dtype.__dtype) {
      case 'float32':
        return new Float32Array(buffer);
      case 'float64':
        return new Float64Array(buffer);
      case 'int32':
        return new Int32Array(buffer);
      case 'uint32':
        return new Uint32Array(buffer);
      case 'int16':
        return new Int16Array(buffer);
      case 'uint16':
        return new Uint16Array(buffer);
      case 'int8':
        return new Int8Array(buffer);
      case 'uint8':
        return new Uint8Array(buffer);
      default:
        throw new Error(`Unsupported dtype: ${dtype.__name || dtype.__dtype}`);
    }
  }

  /**
   * Convert flat index to multi-dimensional indices
   */
  private flatIndexToIndices(flatIndex: number, shape: readonly number[]): number[] {
    const indices: number[] = [];
    let remaining = flatIndex;

    for (let i = 0; i < shape.length; i++) {
      const dim = shape[i];
      if (dim === undefined) {
        throw new Error(`Invalid shape dimension at index ${i}`);
      }
      const stride = shape.slice(i + 1).reduce((a, b) => a * b, 1);
      indices[i] = Math.floor(remaining / stride);
      remaining %= stride;
    }

    return indices;
  }

  /**
   * Map output indices to input indices using slice specifications
   */
  private mapOutputToInputIndices(
    outputIndices: number[],
    sliceIndices: SliceIndex[],
    inputShape: readonly number[],
  ): number[] {
    const inputIndices: number[] = [];
    let outputDim = 0;

    for (let inputDim = 0; inputDim < inputShape.length; inputDim++) {
      const sliceIndex = inputDim < sliceIndices.length ? sliceIndices[inputDim] : null;
      const inputSize = inputShape[inputDim];

      if (inputSize === undefined) {
        throw new Error(`Invalid input shape dimension at index ${inputDim}`);
      }

      if (typeof sliceIndex === 'number') {
        const normalizedIndex = sliceIndex < 0 ? inputSize + sliceIndex : sliceIndex;
        
        // BOUNDS CHECK: Ensure normalized index is within bounds
        if (normalizedIndex < 0 || normalizedIndex >= inputSize) {
          throw new Error(
            `Normalized index ${normalizedIndex} (from ${sliceIndex}) is out of bounds ` +
            `for dimension ${inputDim} of size ${inputSize}`
          );
        }
        
        inputIndices[inputDim] = normalizedIndex;
      } else if (sliceIndex === null) {
        const outputIndex = outputIndices[outputDim];
        if (outputIndex === undefined) {
          throw new Error(`Missing output index for dimension ${outputDim}`);
        }
        inputIndices[inputDim] = outputIndex;
        outputDim++;
      } else {
        const outputIndex = outputIndices[outputDim];
        if (outputIndex === undefined) {
          throw new Error(`Missing output index for dimension ${outputDim}`);
        }

        if (sliceIndex && typeof sliceIndex === 'object') {
          const start =
            sliceIndex.start !== undefined
              ? sliceIndex.start < 0
                ? inputSize + sliceIndex.start
                : sliceIndex.start
              : 0;
          const step = sliceIndex.step ?? 1;

          const computedIndex = start + outputIndex * step;
          
          // BOUNDS CHECK: Ensure computed index is within bounds
          if (computedIndex < 0 || computedIndex >= inputSize) {
            throw new Error(
              `Computed index ${computedIndex} (start=${start} + ${outputIndex} * step=${step}) ` +
              `is out of bounds for dimension ${inputDim} of size ${inputSize}`
            );
          }
          
          inputIndices[inputDim] = computedIndex;
        } else {
          throw new Error(`Invalid slice index type for SliceSpec: ${typeof sliceIndex}`);
        }
        outputDim++;
      }
    }

    return inputIndices;
  }

  /**
   * Compute flat index from multi-dimensional indices and strides
   */
  private computeFlatIndex(indices: number[], strides: readonly number[]): number {
    let flatIndex = 0;
    for (let i = 0; i < indices.length; i++) {
      const stride = strides[i];
      const index = indices[i];
      if (stride !== undefined && index !== undefined) {
        flatIndex += index * stride;
      }
    }
    return flatIndex;
  }

  toString(): string {
    const status = this.initialized ? 'initialized' : 'not initialized';
    const version = this.capabilities?.version || 'unknown';
    return `WASMDevice(id=${this.id}, version=${version}, ${status})`;
  }
}
