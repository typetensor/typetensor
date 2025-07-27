/**
 * Type utilities for tensor operations
 *
 * This module provides type-level utilities for tensor creation
 * and manipulation, including nested array types and option interfaces.
 */

import type { Shape } from '../shape/types';
import type { AnyDType, DTypeValue } from '../dtype/types';
import type { Device } from '../device/types';

// Re-export DTypeValue so tensor/index.ts can export it
export type { DTypeValue } from '../dtype/types';

/**
 * Options for tensor creation
 *
 * @template D - Data type
 * @template Dev - Device type
 * @template S - Shape type (for explicit shape specification)
 */
export interface TensorOptions<
  D extends AnyDType = AnyDType,
  Dev extends Device = Device,
  S extends Shape = never,
> {
  /** Data type for the tensor (required) */
  dtype: D;

  /** Device to place the tensor on (required) */
  device: Dev;

  /** Explicit shape specification (for cases where shape is known but data is dynamic) */
  shape?: S;
}

/**
 * Nested array type based on shape
 *
 * Converts a shape tuple into the corresponding nested array type.
 * Used for type-safe tensor creation from JavaScript arrays.
 *
 * @template T - Element type
 * @template S - Shape tuple
 *
 * @example
 * type Scalar = NestedArray<number, readonly []>;           // number
 * type Vector = NestedArray<number, readonly [3]>;          // [number, number, number]
 * type Matrix = NestedArray<number, readonly [2, 3]>;       // [[number, number, number], [number, number, number]]
 */
export type NestedArray<T, S extends Shape> = S extends readonly []
  ? T
  : S extends readonly [infer N, ...infer Rest]
    ? N extends number
      ? Rest extends Shape
        ? TupleOf<NestedArray<T, Rest>, N>
        : never
      : never
    : never;

/**
 * Create a tuple of specific length
 *
 * Helper type for creating fixed-length tuples.
 *
 * @template T - Element type
 * @template N - Tuple length
 */
type TupleOf<T, N extends number> = N extends N
  ? number extends N
    ? T[]
    : _TupleOf<T, N, []>
  : never;

type _TupleOf<T, N extends number, R extends unknown[]> = R['length'] extends N
  ? R
  : _TupleOf<T, N, [...R, T]>;

/**
 * Extract shape from nested array type
 *
 * Inverse of NestedArray - given a nested array, extract its shape.
 *
 * @template A - Nested array type
 *
 * @example
 * type S1 = InferShape<number>;                    // readonly []
 * type S2 = InferShape<[number, number, number]>;  // readonly [3]
 * type S3 = InferShape<[[number, number]]>;        // readonly [1, 2]
 */
export type InferShape<A> = A extends readonly (infer T)[]
  ? A extends readonly [unknown, ...unknown[]]
    ? readonly [A['length'], ...InferShape<T>]
    : InferShape<T> extends readonly []
      ? readonly [number]
      : readonly [number, ...InferShape<T>]
  : readonly [];

/**
 * Flatten nested array to 1D
 *
 * Recursively flattens a nested array structure into a single-dimensional array.
 *
 * @template A - Nested array type
 */
export type FlattenArray<A> = A extends readonly (infer T)[]
  ? readonly [...FlattenArray<T>]
  : readonly [A];

/**
 * Convert nested array to ArrayBuffer based on dtype
 *
 * @param data - Nested array of numbers
 * @param dtype - Data type for conversion
 * @param shape - Expected shape (for validation)
 * @returns ArrayBuffer containing the data
 */
export function nestedArrayToBuffer<D extends AnyDType>(
  data: unknown,
  dtype: D,
  shape: readonly number[],
): ArrayBuffer {
  // Flatten the nested array
  const flat = flattenNestedArray(data);

  // Validate element count
  const expectedSize = shape.reduce((a, b) => a * b, 1);
  if (flat.length !== expectedSize) {
    throw new Error(
      `Data size ${flat.length.toString()} doesn't match shape [${shape.join(', ')}]`,
    );
  }

  // Create typed array based on dtype
  const TypedArrayConstructor = dtype.__typedArray;
  // Create typed array from flat data
  let typedArray:
    | Float32Array
    | Float64Array
    | Int8Array
    | Uint8Array
    | Int16Array
    | Uint16Array
    | Int32Array
    | Uint32Array
    | BigInt64Array
    | BigUint64Array;

  // Check if it's a bigint type by checking the constructor type
  if (TypedArrayConstructor === BigInt64Array || TypedArrayConstructor === BigUint64Array) {
    // Handle bigint types - need to convert numbers to bigints
    const bigIntFlat = flat.map((v) => BigInt(v));
    typedArray = new TypedArrayConstructor(bigIntFlat) as BigInt64Array | BigUint64Array;
  } else {
    // Handle regular number types
    typedArray = new (TypedArrayConstructor as typeof Float32Array)(flat as number[]);
  }

  return typedArray.buffer.slice(
    typedArray.byteOffset,
    typedArray.byteOffset + typedArray.byteLength,
  ) as ArrayBuffer;
}

/**
 * Flatten a nested array to 1D
 *
 * @param arr - Nested array to flatten
 * @returns Flattened array
 */
function flattenNestedArray(arr: unknown): number[] {
  const result: number[] = [];

  function flatten(item: unknown): void {
    if (Array.isArray(item)) {
      for (const element of item) {
        flatten(element);
      }
    } else if (typeof item === 'number') {
      result.push(item);
    } else if (typeof item === 'bigint') {
      result.push(Number(item));
    } else if (typeof item === 'boolean') {
      result.push(item ? 1 : 0);
    } else {
      throw new Error(`Invalid element type: ${typeof item}`);
    }
  }

  flatten(arr);
  return result;
}

/**
 * Convert ArrayBuffer to nested array based on shape and dtype
 *
 * @param buffer - Raw data buffer
 * @param shape - Target shape
 * @param dtype - Data type
 * @returns Nested array matching the shape
 */
export function bufferToNestedArray<D extends AnyDType, S extends Shape>(
  buffer: ArrayBuffer,
  shape: S,
  dtype: D,
): NestedArray<DTypeValue<D>, S> {
  const TypedArrayConstructor = dtype.__typedArray;
  const typedArray = new TypedArrayConstructor(buffer);

  // Convert to regular array with proper type handling
  let flat: DTypeValue<D>[];
  if (TypedArrayConstructor === BigInt64Array || TypedArrayConstructor === BigUint64Array) {
    // Handle bigint arrays
    flat = Array.from(typedArray as BigInt64Array | BigUint64Array) as DTypeValue<D>[];
  } else if (dtype.__dtype === 'bool') {
    // Handle boolean arrays - convert Uint8Array values back to boolean
    flat = Array.from(typedArray as Uint8Array).map((v) => v !== 0) as DTypeValue<D>[];
  } else {
    // Handle number arrays
    flat = Array.from(
      typedArray as
        | Float32Array
        | Float64Array
        | Int8Array
        | Uint8Array
        | Int16Array
        | Uint16Array
        | Int32Array
        | Uint32Array,
    ) as DTypeValue<D>[];
  }

  // Reshape to nested array
  return reshapeFlatArray(flat, shape) as NestedArray<DTypeValue<D>, S>;
}

/**
 * Reshape a flat array into nested array based on shape
 *
 * @param flat - Flat array
 * @param shape - Target shape
 * @returns Nested array
 */
function reshapeFlatArray<T>(flat: T[], shape: readonly number[]): unknown {
  if (shape.length === 0) {
    return flat[0];
  }

  if (shape.length === 1) {
    return flat.slice(0, shape[0]);
  }

  const result: unknown[] = [];
  const innerSize = shape.slice(1).reduce((a, b) => a * b, 1);

  const firstDim = shape[0];
  if (firstDim === undefined) {
    throw new Error('Invalid shape: first dimension is undefined');
  }
  for (let i = 0; i < firstDim; i++) {
    const start = i * innerSize;
    const end = start + innerSize;
    result.push(reshapeFlatArray(flat.slice(start, end), shape.slice(1)));
  }

  return result;
}

/**
 * Infer shape from a nested array at runtime
 *
 * @param data - Nested array
 * @returns Shape tuple
 */
export function inferShape(data: unknown): number[] {
  if (!Array.isArray(data)) {
    return [];
  }

  // Handle empty arrays - create tensor with shape [0]
  if (data.length === 0) {
    return [0];
  }

  const shape: number[] = [data.length];

  if (Array.isArray(data[0])) {
    const innerShape = inferShape(data[0]);
    shape.push(...innerShape);

    // Validate consistency
    for (let i = 1; i < data.length; i++) {
      const currentShape = inferShape(data[i]);
      if (!arraysEqual(innerShape, currentShape)) {
        throw new Error('Inconsistent array dimensions');
      }
    }
  }

  return shape;
}

/**
 * Check if two arrays are equal
 */
function arraysEqual(a: unknown[], b: unknown[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}
