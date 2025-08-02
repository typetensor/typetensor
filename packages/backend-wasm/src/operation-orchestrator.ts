/**
 * Operation Orchestrator - coordinates execution of tensor operations
 * 
 * This abstraction layer separates operation execution logic from the Device class,
 * providing a clean interface for tensor computations.
 */

import type { DeviceData, AnyStorageTransformation } from '@typetensor/core';
import { WASMDeviceData, createWASMDeviceData } from './data';
import type { WasmOperationDispatcher, WasmBufferHandle, WasmTensorMeta } from './types/wasm-bindings';
import { dtypeToWasm, operationToWasm } from './types';
import type { WASMModule } from './types';
import { getDTypeByteSize } from './utils/dtype-helpers';
import { WASMOperationError } from './errors';

export interface OperationContext {
  deviceId: string;
  wasmModule: WASMModule;
  operationDispatcher: WasmOperationDispatcher;
  device?: any; // Reference to the actual device
}

export interface ExecutionResult {
  outputData: DeviceData;
  executionTime?: number;
  memoryUsed?: number;
}

/**
 * Orchestrates tensor operation execution with proper error handling and optimization
 */
export class OperationOrchestrator {
  private context: OperationContext;

  constructor(context: OperationContext) {
    this.context = context;
  }

  /**
   * Execute a tensor operation
   */
  async execute<T extends AnyStorageTransformation>(
    op: T,
    inputs: DeviceData[],
    output?: DeviceData,
  ): Promise<ExecutionResult> {
    // Validate inputs
    this.validateInputs(op, inputs, output);

    // Handle special operations
    if (op.__op === 'slice') {
      return this.executeSliceOperation(op as any, inputs[0]!);
    }

    // Parse operation parameters
    const operationParams = this.parseOperationParameters(op);

    const startTime = performance.now();

    try {
      const wasmInputs = this.prepareWasmInputs(inputs);
      const inputMetas = this.createInputMetas(op, inputs);
      const outputMeta = this.createOutputMeta(op);
      const wasmOperation = operationToWasm(op.__op);

      // Pre-allocate output buffer to avoid nested RefCell borrows
      const outputHandle = await this.prepareOutputBuffer(op, output);

      // Execute based on input count and operation type
      const resultHandle = await this.executeWasmOperation(
        wasmOperation,
        wasmInputs,
        inputMetas,
        outputMeta,
        outputHandle,
        operationParams
      );

      const executionTime = performance.now() - startTime;
      const resultSize = op.__output.__size * getDTypeByteSize(op.__output.__dtype);
      
      const device = this.context.device || { id: this.context.deviceId };
      return {
        outputData: createWASMDeviceData(
          device,
          resultSize,
          resultHandle
        ),
        executionTime,
        memoryUsed: resultSize
      };

    } catch (error) {
      const executionTime = performance.now() - startTime;
      const inputInfo = inputs.map((input, i) => ({
        id: (input as WASMDeviceData).id,
        size: input.byteLength,
        index: i
      }));
      
      throw new WASMOperationError(
        op.__op,
        error instanceof Error ? error.message : String(error),
        {
          inputs: inputInfo,
          outputSize: op.__output.__size,
          outputDtype: (op.__output.__dtype as any).__name || 'unknown',
          executionTime
        }
      );
    }
  }

  /**
   * Check if operation supports non-contiguous tensors
   */
  supportsNonContiguous(_op: AnyStorageTransformation['__op']): boolean {
    // Currently no operations support non-contiguous tensors
    return false;
  }

  /**
   * Get operation capabilities and requirements
   */
  getOperationInfo(op: AnyStorageTransformation['__op']) {
    return {
      supportsNonContiguous: this.supportsNonContiguous(op),
      requiresContiguous: true,
      supportsInPlace: false, // WASM backend doesn't support in-place operations
      memoryRequirement: 'output-size', // Requires output buffer size
      computeComplexity: this.estimateComputeComplexity(op)
    };
  }

  /**
   * Validate operation inputs
   */
  private validateInputs<T extends AnyStorageTransformation>(
    _op: T,
    inputs: DeviceData[],
    output?: DeviceData
  ): void {
    for (const input of inputs) {
      if (input.device.id !== this.context.deviceId) {
        throw new Error(`Input tensor is on device ${input.device.id}, expected ${this.context.deviceId}`);
      }
    }

    if (output && output.device.id !== this.context.deviceId) {
      throw new Error(`Output tensor is on device ${output.device.id}, expected ${this.context.deviceId}`);
    }
  }

  /**
   * Parse operation-specific parameters
   */
  private parseOperationParameters<T extends AnyStorageTransformation>(op: T) {
    let reductionAxes: number[] | null = null;
    let keepDims = false;
    let softmaxAxis: number | null = null;

    // Parse reduction parameters
    if (['sum', 'mean', 'max', 'min', 'prod'].includes(op.__op)) {
      const reductionOp = op as any;
      const axesKey = `__${op.__op}Axes`; // e.g., __sumAxes, __meanAxes, __prodAxes
      reductionAxes =
        reductionOp[axesKey] === undefined ? null : Array.from(reductionOp[axesKey] || []);
      keepDims = reductionOp.__keepDims || false;
    }

    // Parse softmax parameters
    if (op.__op === 'softmax' || op.__op === 'log_softmax') {
      const softmaxOp = op as any;
      if (op.__op === 'softmax') {
        softmaxAxis = softmaxOp.__softmaxAxis ?? null;
      } else {
        softmaxAxis = softmaxOp.__logSoftmaxAxis ?? null;
      }
    }

    return {
      reductionAxes,
      keepDims,
      softmaxAxis
    };
  }

  /**
   * Prepare WASM input handles
   */
  private prepareWasmInputs(inputs: DeviceData[]): WasmBufferHandle[] {
    const wasmInputs: WasmBufferHandle[] = [];
    
    for (const input of inputs) {
      const wasmData = input as WASMDeviceData;
      const handle = wasmData.getWASMHandle() as WasmBufferHandle;
      wasmInputs.push(handle);
    }
    
    return wasmInputs;
  }

  /**
   * Create input metadata for WASM operations
   */
  private createInputMetas<T extends AnyStorageTransformation>(
    op: T,
    inputs: DeviceData[]
  ): WasmTensorMeta[] {
    return inputs.map((input, i) => 
      this.createTensorMeta([...op.__inputs], input, i)
    );
  }

  /**
   * Create output metadata for WASM operations
   */
  private createOutputMeta<T extends AnyStorageTransformation>(op: T): WasmTensorMeta {
    return this.createTensorMeta([op.__output], null, 0);
  }

  /**
   * Prepare output buffer (pre-allocation)
   */
  private async prepareOutputBuffer<T extends AnyStorageTransformation>(
    op: T,
    output?: DeviceData
  ): Promise<WasmBufferHandle> {
    if (output) {
      return (output as WASMDeviceData).getWASMHandle() as WasmBufferHandle;
    } else {
      // Pre-allocate output buffer using device's buffer creation
      const outputSize = op.__output.__size * getDTypeByteSize(op.__output.__dtype);
      
      if (this.context.device && this.context.device.createData) {
        // Use device's buffer creation for proper lifecycle management
        const outputData = this.context.device.createData(outputSize);
        return (outputData as WASMDeviceData).getWASMHandle() as WasmBufferHandle;
      } else {
        // Fallback: direct WASM call (should be avoided)
        const zeroBuffer = new ArrayBuffer(outputSize);
        return this.context.operationDispatcher.create_buffer_with_js_data(
          new Uint8Array(zeroBuffer),
        );
      }
    }
  }

  /**
   * Execute WASM operation based on input count and type
   */
  private async executeWasmOperation(
    wasmOperation: any,
    wasmInputs: WasmBufferHandle[],
    inputMetas: WasmTensorMeta[],
    outputMeta: WasmTensorMeta,
    outputHandle: WasmBufferHandle,
    params: any
  ): Promise<WasmBufferHandle> {
    if (wasmInputs.length === 0) {
      // For 0-input operations, we already have the pre-allocated buffer
      return outputHandle;
    } else if (wasmInputs.length === 1) {
      const input0 = wasmInputs[0];
      const inputMeta0 = inputMetas[0];
      if (!input0 || !inputMeta0) {
        throw new Error('Invalid input data for single-input operation');
      }
      if (params.reductionAxes !== null) {
        return this.context.operationDispatcher.execute_reduction(
          wasmOperation,
          input0,
          inputMeta0,
          outputMeta,
          params.reductionAxes.length > 0 ? params.reductionAxes : null,
          params.keepDims,
          outputHandle,
        );
      } else if (params.softmaxAxis !== null) {
        return this.context.operationDispatcher.execute_softmax(
          wasmOperation,
          input0,
          inputMeta0,
          outputMeta,
          params.softmaxAxis,
          outputHandle,
        );
      } else {
        return this.context.operationDispatcher.execute_unary(
          wasmOperation,
          input0,
          inputMeta0,
          outputMeta,
          outputHandle,
        );
      }
    } else if (wasmInputs.length === 2) {
      const input0 = wasmInputs[0];
      const input1 = wasmInputs[1];
      const inputMeta0 = inputMetas[0];
      const inputMeta1 = inputMetas[1];
      if (!input0 || !input1 || !inputMeta0 || !inputMeta1) {
        throw new Error('Invalid input data for binary operation');
      }
      return this.context.operationDispatcher.execute_binary(
        wasmOperation,
        input0,
        input1,
        inputMeta0,
        inputMeta1,
        outputMeta,
        outputHandle,
      );
    } else {
      throw new Error(`Unsupported number of inputs: ${wasmInputs.length}`);
    }
  }

  /**
   * Execute slice operation (TypeScript implementation)
   */
  private async executeSliceOperation(
    op: AnyStorageTransformation & { __op: 'slice' },
    input: DeviceData,
  ): Promise<ExecutionResult> {
    const startTime = performance.now();

    try {
      const sliceIndices = (op.__output as any).__sliceIndices;
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

      const executionTime = performance.now() - startTime;
      const resultSize = outputBuffer.byteLength;

      return {
        outputData: this.createDataWithBuffer(outputBuffer),
        executionTime,
        memoryUsed: resultSize
      };

    } catch (error) {
      const executionTime = performance.now() - startTime;
      throw new WASMOperationError(
        'slice',
        error instanceof Error ? error.message : String(error),
        {
          inputs: [{ id: (input as WASMDeviceData).id, size: input.byteLength, index: 0 }],
          outputSize: op.__output.__size,
          outputDtype: (op.__output.__dtype as any).__name || 'unknown',
          executionTime
        }
      );
    }
  }

  /**
   * Create data with existing buffer
   */
  private createDataWithBuffer(buffer: ArrayBuffer): DeviceData {
    // Delegate to device's BufferLifecycleManager for proper buffer creation
    if (this.context.device && this.context.device.createDataWithBuffer) {
      return this.context.device.createDataWithBuffer(buffer);
    }
    
    // Fallback: create directly but this should be avoided
    const device = this.context.device || { id: this.context.deviceId };
    const sourceData = new Uint8Array(buffer);
    const wasmHandle = this.context.operationDispatcher.create_buffer_with_js_data(sourceData);
    
    return createWASMDeviceData(
      device,
      buffer.byteLength,
      wasmHandle
    );
  }

  /**
   * Create sliced buffer from input data
   */
  private async createSlicedBuffer(
    input: DeviceData,
    sliceIndices: any[],
    inputShape: readonly number[],
    inputStrides: readonly number[],
    outputShape: readonly number[],
    dtype: any,
  ): Promise<ArrayBuffer> {
    // Validate slice indices before processing
    this.validateSliceIndices(sliceIndices, inputShape);
    
    const inputBuffer = await this.readInputData(input);

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
      
      const inputValue = inputArray[inputFlatIndex];
      if (inputValue !== undefined) {
        outputArray[outputFlatIndex] = inputValue;
      } else {
        throw new Error(`Invalid value at index ${inputFlatIndex}`);
      }
    }

    return outputBuffer;
  }

  /**
   * Read data from input
   */
  private async readInputData(input: DeviceData): Promise<ArrayBuffer> {
    if (input.device.id !== this.context.deviceId) {
      throw new Error(`Cannot read data from device ${input.device.id} on ${this.context.deviceId}`);
    }

    const wasmData = input as WASMDeviceData;
    const wasmHandle = wasmData.getWASMHandle() as WasmBufferHandle;
    const uint8Array = this.context.operationDispatcher.copy_buffer_to_js(wasmHandle);

    // Ensure we return an ArrayBuffer, not ArrayBufferLike
    const buffer = uint8Array.buffer;
    if (buffer instanceof ArrayBuffer) {
      // If the array is already properly aligned, return its buffer directly
      if (uint8Array.byteOffset === 0 && uint8Array.byteLength === buffer.byteLength) {
        return buffer;
      }
      
      // Only slice if necessary (when the view is a subset of the buffer)
      return buffer.slice(
        uint8Array.byteOffset,
        uint8Array.byteOffset + uint8Array.byteLength,
      );
    } else {
      // Handle SharedArrayBuffer case by creating a copy
      const arrayBuffer = new ArrayBuffer(uint8Array.byteLength);
      new Uint8Array(arrayBuffer).set(new Uint8Array(buffer, uint8Array.byteOffset, uint8Array.byteLength));
      return arrayBuffer;
    }
  }

  /**
   * Validate slice indices to ensure they are within bounds
   */
  private validateSliceIndices(
    sliceIndices: any[],
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
            `Slice index ${slice} out of bounds for dimension ${i} of size ${dimSize}`
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
            `Slice start ${start} out of bounds for dimension ${i} of size ${dimSize}`
          );
        }
        
        if (normalizedStop < 0 || normalizedStop > dimSize) {
          throw new Error(
            `Slice stop ${stop} out of bounds for dimension ${i} of size ${dimSize}`
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
  private createTypedArray(buffer: ArrayBuffer, dtype: any): Float32Array | Float64Array | Int32Array | Uint32Array | Int16Array | Uint16Array | Int8Array | Uint8Array {
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
    sliceIndices: any[],
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
            `Mapped output index ${sliceIndex} out of bounds for dimension ${inputDim} of size ${inputSize}`
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
      } else {
        throw new Error(`Invalid stride or index at position ${i}: stride=${stride}, index=${index}`);
      }
    }
    return flatIndex;
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

    const WasmTensorMeta = this.context.wasmModule.WasmTensorMeta;
    return new WasmTensorMeta(wasmDtype, shapeArray, stridesArray, size, offset);
  }

  /**
   * Estimate computational complexity for operation
   */
  private estimateComputeComplexity(op: AnyStorageTransformation['__op']): 'low' | 'medium' | 'high' {
    switch (op) {
      case 'neg':
      case 'abs':
      case 'add':
      case 'sub':
      case 'mul':
        return 'low';
      
      case 'div':
      case 'sqrt':
      case 'square':
        return 'medium';
      
      case 'sin':
      case 'cos':
      case 'exp':
      case 'log':
      case 'matmul':
      case 'softmax':
      case 'log_softmax':
        return 'high';
      
      default:
        return 'medium';
    }
  }
}