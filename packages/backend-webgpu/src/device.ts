/**
 * WebGPU device implementation
 */

import type {
  Device,
  DeviceData,
  AnyStorageTransformation,
  ValidateDeviceOperations,
} from '@typetensor/core';
import { WebGPUDeviceData } from './data';
import { getWebGPUDevice, alignBufferSize } from './utils';
import { executeOperation } from './operations';

/**
 * WebGPU computation device
 * 
 * Implements tensor operations on the GPU using WebGPU API
 */
export class WebGPUDevice implements Device {
  /**
   * Device identifiers
   */
  readonly id: string;
  readonly type = 'webgpu';

  /**
   * The underlying WebGPU device
   */
  private _gpuDevice: GPUDevice;

  /**
   * Staging buffer for data transfers
   */
  private _stagingBuffer: GPUBuffer | null = null;
  private _stagingBufferSize = 0;

  /**
   * Compile-time validation that all operations are implemented
   */
  private _operationValidation?: ValidateDeviceOperations<
    | 'create'
    | 'neg'
    | 'abs'
    | 'sin'
    | 'cos'
    | 'exp'
    | 'log'
    | 'sqrt'
    | 'square'
    | 'add'
    | 'sub'
    | 'mul'
    | 'div'
    | 'reshape'
    | 'view'
    | 'slice'
    | 'flatten'
    | 'matmul'
    | 'transpose'
    | 'permute'
    | 'squeeze'
    | 'unsqueeze'
    | 'expand'
    | 'tile'
    | 'softmax'
    | 'log_softmax'
    | 'sum'
    | 'mean'
    | 'max'
    | 'min'
    | 'prod'
    | 'rearrange'
    | 'reduce'
  > = true;

  constructor(gpuDevice: GPUDevice, id: string) {
    this._gpuDevice = gpuDevice;
    this.id = id;
  }

  /**
   * Create a new WebGPU device instance
   */
  static async create(): Promise<WebGPUDevice> {
    const gpuDevice = await getWebGPUDevice();
    const id = `webgpu:${gpuDevice.label || '0'}`;
    return new WebGPUDevice(gpuDevice, id);
  }

  /**
   * Get the underlying GPU device
   */
  get gpuDevice(): GPUDevice {
    return this._gpuDevice;
  }

  /**
   * Execute a tensor operation
   */
  async execute<T extends AnyStorageTransformation>(
    op: T,
    inputs: DeviceData[],
    output?: DeviceData,
  ): Promise<DeviceData> {
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

    // Execute the operation
    return executeOperation(this, op, inputs, output);
  }

  /**
   * Allocate data on the GPU
   */
  createData(byteLength: number): DeviceData {
    const alignedSize = alignBufferSize(byteLength);
    
    const buffer = this._gpuDevice.createBuffer({
      label: 'TypeTensor Data Buffer',
      size: alignedSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    return new WebGPUDeviceData(this, byteLength, buffer);
  }

  /**
   * Create data with initial values from an ArrayBuffer
   */
  createDataWithBuffer(arrayBuffer: ArrayBuffer): DeviceData {
    const alignedSize = alignBufferSize(arrayBuffer.byteLength);
    
    const buffer = this._gpuDevice.createBuffer({
      label: 'TypeTensor Data Buffer',
      size: alignedSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    // Copy data to the mapped buffer
    const mappedArray = new Uint8Array(buffer.getMappedRange());
    mappedArray.set(new Uint8Array(arrayBuffer));
    buffer.unmap();

    return new WebGPUDeviceData(this, arrayBuffer.byteLength, buffer);
  }

  /**
   * Free GPU memory
   */
  disposeData(data: DeviceData): void {
    if (data.device.id !== this.id) {
      throw new Error(`Cannot dispose data from device ${data.device.id} on ${this.id}`);
    }

    const webgpuData = data as WebGPUDeviceData;
    webgpuData.destroy();
  }

  /**
   * Read data from GPU to host memory
   */
  async readData(data: DeviceData): Promise<ArrayBuffer> {
    if (data.device.id !== this.id) {
      throw new Error(`Cannot read data from device ${data.device.id} on ${this.id}`);
    }

    const webgpuData = data as WebGPUDeviceData;
    const alignedSize = alignBufferSize(data.byteLength);

    // Create or reuse staging buffer
    if (!this._stagingBuffer || this._stagingBufferSize < alignedSize) {
      if (this._stagingBuffer) {
        this._stagingBuffer.destroy();
      }
      
      this._stagingBuffer = this._gpuDevice.createBuffer({
        label: 'TypeTensor Staging Buffer',
        size: alignedSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      this._stagingBufferSize = alignedSize;
    }

    // Copy from GPU buffer to staging buffer
    const commandEncoder = this._gpuDevice.createCommandEncoder({
      label: 'Read Data Command Encoder',
    });
    commandEncoder.copyBufferToBuffer(
      webgpuData.buffer,
      0,
      this._stagingBuffer,
      0,
      alignedSize,
    );
    this._gpuDevice.queue.submit([commandEncoder.finish()]);

    // Map the staging buffer and read the data
    await this._stagingBuffer.mapAsync(GPUMapMode.READ);
    const mappedRange = this._stagingBuffer.getMappedRange();
    
    // Create a copy of the data (only the actual data, not the aligned size)
    const result = new ArrayBuffer(data.byteLength);
    new Uint8Array(result).set(new Uint8Array(mappedRange, 0, data.byteLength));
    
    this._stagingBuffer.unmap();

    return result;
  }

  /**
   * Write data from host to GPU memory
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

    const webgpuData = data as WebGPUDeviceData;
    
    // Write the data to the GPU buffer
    this._gpuDevice.queue.writeBuffer(
      webgpuData.buffer,
      0,
      buffer,
      0,
      buffer.byteLength,
    );

    // Clear any cached data
    webgpuData.clearCache();
  }

  /**
   * Check if WebGPU backend supports non-contiguous tensors for a specific operation
   * 
   * WebGPU can handle non-contiguous tensors efficiently for most operations
   * through proper indexing in shaders
   */
  supportsNonContiguous(op: AnyStorageTransformation['__op']): boolean {
    // View operations don't need data movement
    const viewOps = ['reshape', 'view', 'flatten', 'transpose', 'permute', 'squeeze', 'unsqueeze'];
    if (viewOps.includes(op)) {
      return true;
    }

    // Most compute operations can handle non-contiguous data with proper indexing
    // Only operations that might have specific layout requirements return false
    switch (op) {
      case 'matmul':
        // Matrix multiplication might benefit from contiguous data for performance
        return false;
      default:
        // Most operations support non-contiguous tensors
        return true;
    }
  }

  /**
   * Clean up resources when device is no longer needed
   */
  destroy(): void {
    if (this._stagingBuffer) {
      this._stagingBuffer.destroy();
      this._stagingBuffer = null;
    }
    this._gpuDevice.destroy();
  }
}