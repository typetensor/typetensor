/**
 * Utility functions for WebGPU backend
 */



/**
 * Get the appropriate WebGPU device
 * Supports both browser and Node.js environments
 */
export async function getWebGPUDevice(): Promise<GPUDevice> {
  let adapter: GPUAdapter | null = null;

  if (typeof navigator !== 'undefined' && navigator.gpu) {
    // Browser environment
    adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
  } else {
    // Node.js/Bun environment - dynamic import to avoid issues in browser
    try {
      const { GPU } = await import('webgpu');
      const gpu = new GPU();
      adapter = await gpu.requestAdapter({
        powerPreference: 'high-performance',
      });
    } catch (error) {
      throw new Error(
        'WebGPU is not available in this environment. ' +
          'In Node.js/Bun, install the "webgpu" package. ' +
          'In browsers, ensure WebGPU is supported and enabled.',
      );
    }
  }

  if (!adapter) {
    throw new Error('Failed to get WebGPU adapter');
  }

  const device = await adapter.requestDevice({
    label: 'TypeTensor WebGPU Device',
  });

  return device;
}

/**
 * Check if WebGPU is available in the current environment
 */
export function isWebGPUAvailable(): boolean {
  if (typeof navigator !== 'undefined' && navigator.gpu) {
    return true;
  }

  // Try to check for Node.js webgpu module
  try {
    require.resolve('webgpu');
    return true;
  } catch {
    return false;
  }
}

/**
 * Get WGSL type string for a given dtype
 */
export function getWGSLType(dtype: any): string {
  switch (dtype.__name) {
    case 'float32':
      return 'f32';
    case 'int32':
      return 'i32';
    case 'uint32':
      return 'u32';
    case 'float16':
      return 'f16';
    case 'int16':
      return 'i16';
    case 'uint16':
      return 'u16';
    case 'int8':
      return 'i8';
    case 'uint8':
      return 'u8';
    default:
      throw new Error(`Unsupported dtype for WebGPU: ${dtype.__name}`);
  }
}

/**
 * Calculate aligned buffer size (WebGPU requires 4-byte alignment minimum)
 */
export function alignBufferSize(size: number, alignment: number = 16): number {
  return Math.ceil(size / alignment) * alignment;
}

/**
 * Create a typed array view for a given dtype
 */
export function createTypedArrayView(
  buffer: ArrayBuffer,
  dtype: any,
  offset: number = 0,
  length?: number,
): ArrayBufferView {
  const TypedArrayConstructor = dtype.__typedArray;
  const elementCount = length ?? buffer.byteLength / dtype.__byteSize - offset / dtype.__byteSize;
  return new TypedArrayConstructor(buffer, offset, elementCount);
}

/**
 * Generate a unique key for shader caching
 */
export function generateShaderKey(
  opType: string,
  inputShapes: ReadonlyArray<ReadonlyArray<number>>,
  outputShape: ReadonlyArray<number>,
  params?: Record<string, any>,
): string {
  const shapesStr = [...inputShapes, outputShape].map(shape => shape.join('x')).join('_');
  const paramsStr = params ? JSON.stringify(params) : '';
  return `${opType}_${shapesStr}_${paramsStr}`;
}

/**
 * Calculate dispatch dimensions for compute shaders
 */
export function calculateDispatchDimensions(
  totalElements: number,
  workgroupSize: number = 64,
): { x: number; y: number; z: number } {
  const workgroups = Math.ceil(totalElements / workgroupSize);
  
  // WebGPU has limits on dispatch dimensions, typically 65535 per dimension
  const maxDimension = 65535;
  
  if (workgroups <= maxDimension) {
    return { x: workgroups, y: 1, z: 1 };
  }
  
  // Split into 2D dispatch if needed
  const x = Math.min(workgroups, maxDimension);
  const y = Math.ceil(workgroups / x);
  
  if (y <= maxDimension) {
    return { x, y, z: 1 };
  }
  
  // Split into 3D dispatch if needed
  const z = Math.ceil(y / maxDimension);
  return { x, y: maxDimension, z };
}

/**
 * Convert a dispatch ID back to a linear index
 */
export function dispatchIdToLinearIndex(
  dispatchId: { x: number; y: number; z: number },
  dispatchDimensions: { x: number; y: number; z: number },
  workgroupSize: number,
): number {
  const workgroupIndex = 
    dispatchId.z * dispatchDimensions.x * dispatchDimensions.y +
    dispatchId.y * dispatchDimensions.x +
    dispatchId.x;
  return workgroupIndex * workgroupSize;
}