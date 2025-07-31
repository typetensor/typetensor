/**
 * WASM device implementation
 * 
 * Implements the Device interface for WebAssembly backend with:
 * - High-performance tensor operations via WASM
 * - Memory management with arena allocation
 * - SIMD optimizations where available
 * - Cross-platform support (browsers and Node.js)
 */

import type {
  Device,
  DeviceData,
  AnyStorageTransformation,
  ValidateDeviceOperations,
  SliceIndex,
} from '@typetensor/core';
import { WASMDeviceData, createWASMDeviceData } from './data';
import { loadWASMModule } from './loader';
import { dtypeToWasm, operationToWasm } from './types';
import type { WASMModule, WASMLoadOptions, WASMCapabilities, WASMMemoryStats } from './types';

/**
 * WebAssembly device for high-performance tensor operations
 * 
 * Provides near-native performance for tensor computations using WebAssembly
 * with SIMD optimizations and efficient memory management.
 */
export class WASMDevice implements Device {
  readonly id: string = 'wasm:0';
  readonly type: string = 'wasm';

  private wasmModule: WASMModule | null = null;
  private operationDispatcher: any = null; // WasmOperationDispatcher instance  
  private capabilities: WASMCapabilities | null = null;
  private initialized = false;

  /**
   * Compile-time validation that all operations are implemented
   */
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
    // Private constructor - use WASMDevice.create() instead
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

    try {
      // Load the WASM module
      this.wasmModule = await loadWASMModule(options);
      
      // Create the operation dispatcher (which includes its own memory manager)
      this.operationDispatcher = new this.wasmModule.WasmOperationDispatcher();
      
      // Get capabilities
      this.capabilities = {
        simd: this.wasmModule.has_simd_support(),
        sharedMemory: this.wasmModule.has_shared_memory_support(),
        optimalThreadCount: this.wasmModule.get_optimal_thread_count(),
        availableMemory: 256 * 1024 * 1024, // Will be detected properly in full implementation
        version: this.wasmModule.get_version(),
      };
      
      this.initialized = true;
      
      if (options.debug) {
        console.log('[WASMDevice] Initialized successfully');
        console.log('[WASMDevice] Capabilities:', this.capabilities);
      }
    } catch (error) {
      console.error('[WASMDevice] Initialization failed:', error);
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

    // Validate inputs are from this device
    for (const input of inputs) {
      if (input.device.id !== this.id) {
        throw new Error(`Input tensor is on device ${input.device.id}, expected ${this.id}`);
      }
    }

    // Validate output if provided
    if (output && output.device.id !== this.id) {
      throw new Error(`Output tensor is on device ${output.device.id}, expected ${this.id}`);
    }

    // Handle slice operations at TypeScript level (like backend-cpu)
    if (op.__op === 'slice') {
      return this.executeSliceOp(op as any, inputs[0]!);
    }
    
    // Extract reduction axes if this is a reduction operation
    let reductionAxes: number[] | null = null;
    let keepDims = false;
    if (op.__op === 'sum' || op.__op === 'mean' || op.__op === 'max' || op.__op === 'min' || op.__op === 'prod') {
      const reductionOp = op as any;
      const axesKey = `__${op.__op}Axes`; // e.g., __sumAxes, __meanAxes, __prodAxes
      reductionAxes = reductionOp[axesKey] === undefined ? null : Array.from(reductionOp[axesKey] || []);
      keepDims = reductionOp.__keepDims || false;
    }

    try {
      // Convert inputs to WASM handles - hold strong references to prevent GC
      const wasmInputs: any[] = [];
      const inputHandles: any[] = [];
      
      for (let i = 0; i < inputs.length; i++) {
        const wasmData = inputs[i] as WASMDeviceData;
        const handle = wasmData.getWASMHandle() as any;
        
        inputHandles.push(handle);
        wasmInputs.push(handle);
      }
      
      // Create input metadata
      const inputMetas = inputs.map((input, i) => this.createTensorMeta([...op.__inputs], input, i) as any);
      
      // Create output metadata
      const outputMeta = this.createTensorMeta([op.__output], null, 0) as any;
      
      // Convert operation to WASM enum
      const wasmOperation = operationToWasm(op.__op);

      // Execute operation in WASM based on number of inputs
      let resultHandle;
      if (inputs.length === 0) {
        // Create operation - no inputs
        resultHandle = output ? (output as WASMDeviceData).getWASMHandle() : 
          this.operationDispatcher.create_buffer_with_js_data(new Uint8Array(op.__output.__size * op.__output.__dtype.__byteSize));
      } else if (inputs.length === 1) {
        // Check if this is a reduction operation
        if (reductionAxes !== null) {
          // Reduction operation with axis support
          resultHandle = this.operationDispatcher.execute_reduction(
            wasmOperation,
            wasmInputs[0],
            inputMetas[0],
            outputMeta,
            reductionAxes.length > 0 ? reductionAxes : null,
            keepDims,
            output ? (output as WASMDeviceData).getWASMHandle() as any : null
          );
        } else {
          // Regular unary operation
          resultHandle = this.operationDispatcher.execute_unary(
            wasmOperation,
            wasmInputs[0],
            inputMetas[0],
            outputMeta,
            output ? (output as WASMDeviceData).getWASMHandle() as any : null
          );
        }
      } else if (inputs.length === 2) {
        // Binary operation
        resultHandle = this.operationDispatcher.execute_binary(
          wasmOperation,
          wasmInputs[0],
          wasmInputs[1],
          inputMetas[0],
          inputMetas[1],
          outputMeta,
          output ? (output as WASMDeviceData).getWASMHandle() as any : null
        );
      } else {
        throw new Error(`Unsupported number of inputs: ${inputs.length}`);
      }
      
      // Create result DeviceData
      const resultSize = op.__output.__size * op.__output.__dtype.__byteSize;
      return createWASMDeviceData(this, resultSize, resultHandle);
      
    } catch (error) {
      console.error('[WASMDevice] Operation failed:', error);
      throw new Error(`WASM operation '${op.__op}' failed: ${error}`);
    }
  }

  /**
   * Allocate data on the WASM device
   */
  createData(byteLength: number): DeviceData {
    this.ensureInitialized();
    
    try {
      // Create empty buffer with zeros for the new immutable buffer architecture
      const zeroBuffer = new ArrayBuffer(byteLength);
      const wasmHandle = this.operationDispatcher.create_buffer_with_js_data(new Uint8Array(zeroBuffer));
      return createWASMDeviceData(this, byteLength, wasmHandle);
    } catch (error) {
      throw new Error(`Failed to allocate ${byteLength} bytes on WASM device: ${error}`);
    }
  }

  /**
   * Create data with existing buffer
   */
  createDataWithBuffer(buffer: ArrayBuffer): DeviceData {
    this.ensureInitialized();
    
    try {
      // Use new immutable buffer API - create buffer with data atomically
      const sourceData = new Uint8Array(buffer);
      const wasmHandle = this.operationDispatcher.create_buffer_with_js_data(sourceData);
      
      
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
    
    // Only dispose if not already disposed
    if (!wasmData.isDisposed()) {
      const wasmHandle = wasmData.getWASMHandle();
      
      // Release the buffer back to the WASM memory pool
      const released = this.operationDispatcher.release_buffer(wasmHandle);
      
      if (!released) {
        console.warn(`[WASMDevice] Failed to release buffer ${wasmData.id} - may have been already released`);
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
    
    // Use the new immutable buffer API to copy data to JavaScript
    const wasmData = data as WASMDeviceData;
    const wasmHandle = wasmData.getWASMHandle();
    const uint8Array = this.operationDispatcher!.copy_buffer_to_js(wasmHandle);
    
    // Convert Uint8Array to ArrayBuffer
    return uint8Array.buffer.slice(uint8Array.byteOffset, uint8Array.byteOffset + uint8Array.byteLength);
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

    // In the new immutable buffer architecture, buffers cannot be modified after creation
    // This method is not supported for existing data - data must be created with the buffer
    throw new Error(
      'writeData is not supported in the immutable buffer architecture. ' +
      'Use createDataWithBuffer() to create new data with the desired buffer content.'
    );
  }

  /**
   * Check if WASM backend supports non-contiguous tensors for a specific operation
   */
  supportsNonContiguous(_op: AnyStorageTransformation['__op']): boolean {
    // For the initial implementation, we'll be conservative and require contiguous tensors
    // This can be expanded as more operations support non-contiguous layouts
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
    
    // Use new memory stats API with buffer pools
    return {
      totalAllocated: wasmStats.total_allocated_bytes,
      activeBuffers: wasmStats.active_buffers,
      poolSummary: wasmStats.get_pool_summary(),
    };
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
  private createTensorMeta(tensorInfos: any[], _data: DeviceData | null, index = 0): any {
    const tensorInfo = tensorInfos[index];
    if (!tensorInfo) {
      throw new Error(`Missing tensor info at index ${index}`);
    }

    const wasmDtype = dtypeToWasm(tensorInfo.__dtype);
    // wasm-bindgen expects regular JavaScript Arrays for Vec<usize>, not TypedArrays!
    const shapeArray = tensorInfo.__shape.map((n: any) => Math.floor(Number(n)));
    const stridesArray = tensorInfo.__strides.map((n: any) => Math.floor(Number(n)));
    const size = Math.floor(Number(tensorInfo.__size));
    const offset = Math.floor(Number(tensorInfo.__offset || 0));

    // Store constructor in variable to avoid potential parsing issues
    const WasmTensorMeta = this.wasmModule!.WasmTensorMeta;
    
    const meta = new WasmTensorMeta(
      wasmDtype,
      shapeArray,  // Pass regular Array
      stridesArray,  // Pass regular Array
      size,
      offset
    );
    
    
    return meta;
  }

  /**
   * Execute a slice operation on WASM device (TypeScript implementation)
   */
  private async executeSliceOp(
    op: AnyStorageTransformation & { __op: 'slice' },
    input: DeviceData,
  ): Promise<DeviceData> {
    // Extract slice information from operation metadata
    const sliceIndices = (op.__output as any).__sliceIndices as SliceIndex[];
    const inputStorage = op.__inputs[0];
    if (!inputStorage) {
      throw new Error('Slice operation missing input storage metadata');
    }

    const inputShape = inputStorage.__shape;
    const inputStrides = inputStorage.__strides;
    const dtype = inputStorage.__dtype;

    // Output metadata
    const outputShape = op.__output.__shape;

    // Create output buffer with sliced data
    const outputBuffer = await this.createSlicedBuffer(
      input,
      sliceIndices,
      inputShape,
      inputStrides,
      outputShape,
      dtype,
    );

    // Create device data with the sliced buffer (immutable architecture)
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
    dtype: any,
  ): Promise<ArrayBuffer> {
    // Read input buffer
    const inputBuffer = await this.readData(input);
    
    // Calculate output buffer size
    const outputSize = outputShape.reduce((a, b) => a * b, 1);
    const outputBuffer = new ArrayBuffer(outputSize * dtype.__byteSize);

    // Create typed arrays for efficient data access
    const inputArray = this.createTypedArray(inputBuffer, dtype);
    const outputArray = this.createTypedArray(outputBuffer, dtype);

    // Iterate through all output positions and copy corresponding input elements
    for (let outputFlatIndex = 0; outputFlatIndex < outputSize; outputFlatIndex++) {
      // Convert output flat index to multi-dimensional indices
      const outputIndices = this.flatIndexToIndices(outputFlatIndex, outputShape);

      // Map output indices to input indices using slice specifications
      const inputIndices = this.mapOutputToInputIndices(outputIndices, sliceIndices, inputShape);

      // Convert input indices to flat index
      const inputFlatIndex = this.computeFlatIndex(inputIndices, inputStrides);

      // Copy the element
      (outputArray as any)[outputFlatIndex] = (inputArray as any)[inputFlatIndex];
    }

    return outputBuffer;
  }

  /**
   * Create typed array from buffer based on dtype
   */
  private createTypedArray(buffer: ArrayBuffer, dtype: any): ArrayBufferView {
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
        // Integer index: use the specified index directly
        const normalizedIndex = sliceIndex < 0 ? inputSize + sliceIndex : sliceIndex;
        inputIndices[inputDim] = normalizedIndex;
        // Don't increment outputDim - this dimension was removed
      } else if (sliceIndex === null) {
        // null: keep entire dimension, direct mapping
        const outputIndex = outputIndices[outputDim];
        if (outputIndex === undefined) {
          throw new Error(`Missing output index for dimension ${outputDim}`);
        }
        inputIndices[inputDim] = outputIndex;
        outputDim++;
      } else {
        // SliceSpec: apply start/stop/step transformation
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

          inputIndices[inputDim] = start + outputIndex * step;
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

  /**
   * String representation for debugging
   */
  toString(): string {
    const status = this.initialized ? 'initialized' : 'not initialized';
    const version = this.capabilities?.version || 'unknown';
    return `WASMDevice(id=${this.id}, version=${version}, ${status})`;
  }
}