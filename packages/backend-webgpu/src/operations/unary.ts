/**
 * Unary operations for WebGPU backend
 */

import type { DeviceData, AnyStorageTransformation } from '@typetensor/core';
import type { WebGPUDevice } from '../device';
import type { WebGPUDeviceData } from '../data';
import { UnaryShaderGenerator } from '../shaders/templates/unary';
import { generateShaderKey, calculateDispatchDimensions } from '../utils';
import { ShaderCache } from '../shaders/cache';

// Global shader cache per device
const shaderCaches = new WeakMap<GPUDevice, ShaderCache>();

function getShaderCache(device: GPUDevice): ShaderCache {
  let cache = shaderCaches.get(device);
  if (!cache) {
    cache = new ShaderCache(device);
    shaderCaches.set(device, cache);
  }
  return cache;
}

/**
 * Execute a unary operation on WebGPU
 */
export async function executeUnaryOp(
  device: WebGPUDevice,
  op: AnyStorageTransformation,
  input: DeviceData,
  output?: DeviceData,
): Promise<DeviceData> {
  const inputData = input as WebGPUDeviceData;
  const dtype = op.__output.__dtype;
  const size = op.__output.__size;

  // Create output if not provided
  output ??= device.createData(size * dtype.__byteSize);
  const outputData = output as WebGPUDeviceData;

  // Generate shader
  const shaderGenerator = new UnaryShaderGenerator();
  const shaderCode = shaderGenerator.generate(
    [op.__inputs[0]!],
    op.__output,
    { operation: op.__op },
  );

  // Get or create pipeline
  const shaderKey = generateShaderKey(
    op.__op,
    [op.__inputs[0]!.__shape],
    op.__output.__shape,
    { operation: op.__op },
  );
  
  const shaderCache = getShaderCache(device.gpuDevice);
  const { pipeline, bindGroupLayout } = await shaderCache.getOrCreatePipeline(
    shaderKey,
    shaderCode,
  );

  // Create bind group
  const bindGroup = device.gpuDevice.createBindGroup({
    label: `${op.__op} bind group`,
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: inputData.buffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: outputData.buffer,
        },
      },
    ],
  });

  // Calculate dispatch size
  const workgroupSize = 64;
  const dispatchDimensions = calculateDispatchDimensions(size, workgroupSize);

  // Create and submit command
  const commandEncoder = device.gpuDevice.createCommandEncoder({
    label: `${op.__op} command encoder`,
  });

  const computePass = commandEncoder.beginComputePass({
    label: `${op.__op} compute pass`,
  });

  computePass.setPipeline(pipeline);
  computePass.setBindGroup(0, bindGroup);
  computePass.dispatchWorkgroups(
    dispatchDimensions.x,
    dispatchDimensions.y,
    dispatchDimensions.z,
  );
  computePass.end();

  device.gpuDevice.queue.submit([commandEncoder.finish()]);

  return outputData;
}