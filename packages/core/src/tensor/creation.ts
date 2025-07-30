/**
 * Tensor creation functions
 *
 * This module provides functions for creating tensors from various sources,
 * including nested arrays, scalars, and specialized constructors like zeros/ones.
 */

import { Tensor } from './tensor';
import type { TensorOptions, InferShape } from './types';
import { inferShape, nestedArrayToBuffer } from './types';
import type { Shape, Product as ShapeProduct } from '../shape/types';
import type { AnyDType } from '../dtype/types';
import { type float32 } from '../dtype/constants';
import type {
  TensorStorage,
  CreateOp,
  DefaultLayoutFlags,
  ComputeStrides,
} from '../storage/layout';
import type { Device } from '../device';

/**
 * Maximum supported tensor rank (number of dimensions)
 */
const MAX_TENSOR_RANK = 8;

/**
 * Type-level validation for tensor shapes
 * Ensures shape is const and all dimensions are positive integers
 */
type IsValidShape<S extends readonly number[]> = number extends S['length']
  ? false // Not const
  : S['length'] extends 0
    ? true // Empty shape (scalar) is valid
    : S['length'] extends 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8
      ? ValidateDimensions<S>
      : false; // Exceeds max rank

/**
 * Validate all dimensions in a shape are positive integers
 */
type ValidateDimensions<S extends readonly number[]> = S extends readonly []
  ? true
  : S extends readonly [infer Head, ...infer Tail]
    ? Head extends number
      ? IsPositiveInteger<Head> extends true
        ? Tail extends readonly number[]
          ? ValidateDimensions<Tail>
          : false
        : false
      : false
    : false;

/**
 * Check if a number is a positive integer at type level
 * Note: TypeScript can't truly validate integers at type level,
 * but we can check for common cases and non-negative
 */
type IsPositiveInteger<N extends number> = `${N}` extends `-${string}`
  ? false // Negative number
  : `${N}` extends `${string}.${string}`
    ? false // Decimal number
    : number extends N
      ? false // Generic number type
      : true;

/**
 * Compute strides for a shape in C-order (row-major)
 */
function computeStrides(shape: readonly number[]): number[] {
  const strides: number[] = [];
  let stride = 1;

  for (let i = shape.length - 1; i >= 0; i--) {
    strides.unshift(stride);
    const dim = shape[i];
    if (dim === undefined) {
      throw new Error(`Invalid shape dimension at index ${i.toString()}`);
    }
    stride *= dim;
  }

  return strides;
}

/**
 * Type-safe stride creation for tensor storage
 * Returns strides with proper type branding for compile-time safety
 */
function createStrides<S extends Shape>(shape: S): ComputeStrides<S> {
  return computeStrides(shape) as ComputeStrides<S>;
}

/**
 * Type-safe size computation for tensor storage
 * Returns size with proper type branding for compile-time safety
 */
function createSize<S extends Shape>(shape: S): ShapeProduct<S> {
  return product(shape) as ShapeProduct<S>;
}

/**
 * Compute the product of a shape
 */
function product(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * Get byte size for a dtype
 */
function dtypeByteSize(dtype: AnyDType): number {
  return dtype.__byteSize;
}

/**
 * Type-safe storage builder that ensures all type parameters align correctly
 */
function createTensorStorage<D extends AnyDType, S extends Shape>(
  dtype: D,
  shape: S,
): TensorStorage<D, S, ComputeStrides<S>, DefaultLayoutFlags> {
  return {
    __dtype: dtype,
    __shape: shape,
    __strides: createStrides(shape),
    __size: createSize(shape),
    __layout: {
      c_contiguous: true,
      f_contiguous: false,
      is_view: false,
      writeable: true,
      aligned: true,
    } satisfies DefaultLayoutFlags,
    __offset: 0,
  };
}

/**
 * Create a tensor with shape inference and custom dtype/device
 *
 * @param data - Input data (requires const assertion for precise shape inference)
 * @param options - Configuration with required device, optional dtype (shape is inferred)
 * @returns Promise resolving to new tensor with inferred shape
 *
 * @example
 * const a = await tensor([[1, 2, 3], [4, 5, 6]] as const, { device: cpu });
 * // Type: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 3]>>>
 *
 * @example
 * const b = await tensor([1, 2, 3] as const, { device: cpu, dtype: int32 });
 * // Type: Tensor<CreateOp<TensorStorage<Int32, readonly [3]>>>
 */
export async function tensor<const T, D extends AnyDType, Dev extends Device>(
  data: T,
  options: TensorOptions<D, Dev, never> & { shape?: never; device: Dev },
): Promise<Tensor<CreateOp<TensorStorage<D, InferShape<T>>>>>;

/**
 * Create a tensor with explicit shape specification
 *
 * @param data - Input data (can be any data that will be reshaped)
 * @param options - Configuration with required shape and device, optional dtype
 * @returns Promise resolving to new tensor with explicit shape
 *
 * @example
 * const c = await tensor(runtimeData, { shape: [3, 4] as const, device: cpu, dtype: float32 });
 * // Type: Tensor<CreateOp<TensorStorage<Float32, readonly [3, 4]>>>
 */
export async function tensor<D extends AnyDType, S extends Shape, Dev extends Device>(
  data: unknown,
  options: TensorOptions<D, Dev, S> & { shape: S; device: Dev },
): Promise<Tensor<CreateOp<TensorStorage<D, S>>>>;

/**
 * Implementation for all tensor creation overloads
 */
export async function tensor<const T, D extends AnyDType, S extends Shape, Dev extends Device>(
  data: T,
  options?: TensorOptions<D, Dev, S>,
): Promise<
  | Tensor<CreateOp<TensorStorage<typeof float32, InferShape<T>>>>
  | Tensor<CreateOp<TensorStorage<D, InferShape<T>>>>
  | Tensor<CreateOp<TensorStorage<D, S>>>
> {
  if (!options?.device) {
    throw new Error('Device must be specified in tensor options');
  }
  const device = options.device;

  // Type guard to determine if explicit shape was provided
  function hasExplicitShape<D extends AnyDType, S extends Shape, Dev extends Device>(
    options: TensorOptions<D, Dev, S> | undefined,
  ): options is TensorOptions<D, Dev, S> & { shape: S } {
    return options?.shape !== undefined;
  }

  // Branch 1: Explicit shape provided
  if (hasExplicitShape(options)) {
    const shape = options.shape; // Type: S
    const dtype = options.dtype; // Type: D

    // Use type-safe storage builder
    const storage = createTensorStorage(dtype, shape);
    const createOp: CreateOp<typeof storage> = {
      __op: 'create',
      __output: storage,
      __inputs: [] as const,
    };

    const buffer = nestedArrayToBuffer(data, dtype, shape);
    // OPTIMIZED: Use direct buffer creation if available (no copy)
    const deviceData =
      device.createDataWithBuffer?.(buffer) ??
      (await (async () => {
        const data = device.createData(buffer.byteLength);
        await device.writeData(data, buffer);
        return data;
      })());

    return new Tensor(createOp, deviceData);
  }

  // Branch 2 & 3: Inferred shape (with or without custom dtype)
  const inferredShape = inferShape(data);

  // SAFETY: inferShape at runtime computes the same shape that InferShape<T>
  // computes at compile time. This cast bridges the runtime-compile time gap.
  const typedShape = inferredShape as unknown as InferShape<T>;

  // Branch 2: Inferred shape with dtype
  const dtype = options.dtype; // Type: D

  const storage = createTensorStorage(dtype, typedShape);
  const createOp: CreateOp<typeof storage> = {
    __op: 'create',
    __output: storage,
    __inputs: [] as const,
  };

  const buffer = nestedArrayToBuffer(data, dtype, inferredShape);
  // OPTIMIZED: Use direct buffer creation if available (no copy)
  const deviceData =
    device.createDataWithBuffer?.(buffer) ??
    (await (async () => {
      const data = device.createData(buffer.byteLength);
      await device.writeData(data, buffer);
      return data;
    })());

  return new Tensor(createOp, deviceData) as Tensor<CreateOp<TensorStorage<D, InferShape<T>>>>;
}

/**
 * Create a tensor filled with zeros
 *
 * @param shape - Shape of the tensor (must use 'as const')
 * @param options - Optional dtype and device specification
 * @returns Promise resolving to tensor of zeros
 *
 * @example
 * const a = await zeros([2, 3] as const, { device: cpu }); // 2x3 matrix of zeros
 * const b = await zeros([10] as const, { device: cpu, dtype: Int32 });
 */
export async function zeros<D extends AnyDType, S extends Shape>(
  shape: IsValidShape<S> extends true ? S : never,
  options: TensorOptions<D>,
): Promise<Tensor<CreateOp<TensorStorage<D, S>>>> {
  // Runtime validation with helpful error message
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0 || !Number.isFinite(dim)) {
      throw new Error(
        `Invalid shape dimension: ${dim.toString()}. All dimensions must be positive integers. Did you forget 'as const'?`,
      );
    }
  }

  // Additional runtime check for rank
  if (shape.length > MAX_TENSOR_RANK) {
    throw new Error(`Shape rank ${shape.length} exceeds maximum supported rank ${MAX_TENSOR_RANK}`);
  }

  const device = options.device;
  const dtype = options.dtype;
  const size = product(shape);
  const byteLength = size * dtypeByteSize(dtype);

  const storage: TensorStorage<D, S> = {
    __dtype: dtype,
    __shape: shape,
    __strides: createStrides(shape),
    __size: createSize(shape),
    __layout: {
      c_contiguous: true,
      f_contiguous: false,
      is_view: false,
      writeable: true,
      aligned: true,
    } satisfies DefaultLayoutFlags,
    __offset: 0,
  };

  // Wrap in a create transformation
  const createOp: CreateOp<TensorStorage<D, S>> = {
    __op: 'create',
    __output: storage,
    __inputs: [] as const,
  };

  // Create zero-initialized data using createDataWithBuffer for immutable architecture
  const zerosBuffer = new ArrayBuffer(byteLength);
  const deviceData = device.createDataWithBuffer ? 
    device.createDataWithBuffer(zerosBuffer) : 
    await (async () => {
      const data = device.createData(byteLength);
      await device.writeData(data, zerosBuffer);
      return data;
    })();

  return new Tensor(createOp, deviceData);
}

/**
 * Create a tensor filled with ones
 *
 * @param shape - Shape of the tensor (must use 'as const')
 * @param options - Optional dtype and device specification
 * @returns Promise resolving to tensor of ones
 *
 * @example
 * const a = await ones([2, 3] as const, { device: cpu }); // 2x3 matrix of ones
 */
export async function ones<D extends AnyDType, S extends Shape>(
  shape: IsValidShape<S> extends true ? S : never,
  options: TensorOptions<D>,
): Promise<Tensor<CreateOp<TensorStorage<D, S>>>> {
  // Runtime validation with helpful error message
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0 || !Number.isFinite(dim)) {
      throw new Error(
        `Invalid shape dimension: ${dim.toString()}. All dimensions must be positive integers. Did you forget 'as const'?`,
      );
    }
  }

  // Additional runtime check for rank
  if (shape.length > MAX_TENSOR_RANK) {
    throw new Error(`Shape rank ${shape.length} exceeds maximum supported rank ${MAX_TENSOR_RANK}`);
  }
  const device = options.device;
  const dtype = options.dtype;
  const size = product(shape);
  const byteLength = size * dtypeByteSize(dtype);

  const storage: TensorStorage<D, S> = {
    __dtype: dtype,
    __shape: shape,
    __strides: createStrides(shape),
    __size: createSize(shape),
    __layout: {
      c_contiguous: true,
      f_contiguous: false,
      is_view: false,
      writeable: true,
      aligned: true,
    } satisfies DefaultLayoutFlags,
    __offset: 0,
  };

  // Wrap in a create transformation
  const createOp: CreateOp<TensorStorage<D, S>> = {
    __op: 'create',
    __output: storage,
    __inputs: [] as const,
  };

  // Create buffer filled with ones
  const TypedArrayConstructor = dtype.__typedArray;
  const onesArray = new TypedArrayConstructor(size);

  // Fill based on the dtype's JS type
  if (typeof dtype.__jsType === 'bigint') {
    // For bigint types, fill with 1n
    (onesArray as BigInt64Array | BigUint64Array).fill(1n);
  } else if (typeof dtype.__jsType === 'boolean') {
    // For boolean type (stored as Uint8), fill with 1
    (onesArray as Uint8Array).fill(1);
  } else {
    // For number types, fill with 1
    (
      onesArray as
        | Int8Array
        | Uint8Array
        | Int16Array
        | Uint16Array
        | Int32Array
        | Uint32Array
        | Float32Array
        | Float64Array
    ).fill(1);
  }

  const onesBuffer = onesArray.buffer.slice(
    onesArray.byteOffset,
    onesArray.byteOffset + onesArray.byteLength,
  );

  // Create ones-initialized data using createDataWithBuffer for immutable architecture
  const deviceData = device.createDataWithBuffer ? 
    device.createDataWithBuffer(onesBuffer) : 
    await (async () => {
      const data = device.createData(byteLength);
      await device.writeData(data, onesBuffer);
      return data;
    })();

  return new Tensor(createOp, deviceData);
}

/**
 * Create an identity matrix
 *
 * @param n - Size of the square matrix
 * @param options - Optional dtype and device specification
 * @returns Promise resolving to identity matrix
 *
 * @example
 * const I = await eye(3); // 3x3 identity matrix
 */
export async function eye<D extends AnyDType>(
  n: number,
  options: TensorOptions<D>,
): Promise<Tensor<CreateOp<TensorStorage<D, readonly [number, number]>>>> {
  // Validate n
  if (!Number.isInteger(n) || n < 0 || !Number.isFinite(n)) {
    throw new Error(`Invalid size for identity matrix: ${n.toString()}`);
  }

  const device = options.device;
  const dtype = options.dtype;
  const shape = [n, n] as const;
  const size = n * n;
  const byteLength = size * dtypeByteSize(dtype);

  const storage: TensorStorage<D, readonly [number, number]> = {
    __dtype: dtype,
    __shape: shape,
    __strides: createStrides([n, n] as const),
    __size: createSize(shape),
    __layout: {
      c_contiguous: true,
      f_contiguous: false,
      is_view: false,
      writeable: true,
      aligned: true,
    } satisfies DefaultLayoutFlags,
    __offset: 0,
  };

  // Wrap in a create transformation
  const createOp: CreateOp<TensorStorage<D, readonly [number, number]>> = {
    __op: 'create',
    __output: storage,
    __inputs: [] as const,
  };

  // Create buffer with diagonal ones
  const TypedArrayConstructor = dtype.__typedArray;
  const identityArray = new TypedArrayConstructor(size);
  for (let i = 0; i < n; i++) {
    identityArray[i * n + i] = 1;
  }
  const identityBuffer = identityArray.buffer.slice(
    identityArray.byteOffset,
    identityArray.byteOffset + identityArray.byteLength,
  );

  // Create identity matrix data using createDataWithBuffer for immutable architecture
  const deviceData = device.createDataWithBuffer ? 
    device.createDataWithBuffer(identityBuffer) : 
    await (async () => {
      const data = device.createData(byteLength);
      await device.writeData(data, identityBuffer);
      return data;
    })();

  return new Tensor(createOp, deviceData);
}
