/**
 * TypedArray integration layer for DType system
 *
 * This module provides type-safe wrappers around JavaScript's TypedArray
 * system, enabling zero-copy operations, memory alignment validation,
 * and seamless integration with our branded DType system.
 */

import type { AnyDType, JSTypeOf, ArrayConstructorOf } from './types.js';
import { type RuntimeDType, getTypedArrayDType } from './runtime.js';
import { convertValue, convertArray, type ConversionOptions } from './conversion.js';

// =============================================================================
// TypedArray Wrapper Interface
// =============================================================================

/**
 * Type-safe wrapper around JavaScript TypedArrays
 * Preserves the branded DType information while providing
 * zero-copy operations and memory alignment validation
 */
export interface DTypedArray<T extends AnyDType> {
  /** The DType this array represents */
  readonly dtype: RuntimeDType<T>;

  /** Underlying TypedArray instance */
  readonly array: InstanceType<ArrayConstructorOf<T>>;

  /** Underlying ArrayBuffer */
  readonly buffer: ArrayBuffer;

  /** Byte offset within the buffer */
  readonly byteOffset: number;

  /** Total bytes occupied by this array */
  readonly byteLength: number;

  /** Number of elements in the array */
  readonly length: number;

  /** Whether this array is read-only */
  readonly readonly: boolean;

  // Accessors
  get(index: number): JSTypeOf<T>;
  set(index: number, value: JSTypeOf<T>): void;

  // Subarray operations
  subarray(begin?: number, end?: number): DTypedArray<T>;
  slice(begin?: number, end?: number): DTypedArray<T>;

  // Copy operations
  copyWithin(target: number, start: number, end?: number): DTypedArray<T>;
  fill(value: JSTypeOf<T>, start?: number, end?: number): DTypedArray<T>;

  // Conversion operations
  toArray(): readonly JSTypeOf<T>[];

  // Iteration
  forEach(callback: (value: JSTypeOf<T>, index: number, array: DTypedArray<T>) => void): void;
  map<U extends AnyDType>(
    callback: (value: JSTypeOf<T>, index: number, array: DTypedArray<T>) => JSTypeOf<U>,
    targetDType: RuntimeDType<U>,
  ): DTypedArray<U>;

  // View operations
  createView<U extends AnyDType>(
    targetDType: RuntimeDType<U>,
    byteOffset?: number,
    length?: number,
  ): DTypedArray<U>;
}

/**
 * Internal implementation of DTypedArray interface
 */
class TypedArrayWrapper<T extends AnyDType> implements DTypedArray<T> {
  constructor(
    public readonly dtype: RuntimeDType<T>,
    public readonly array: InstanceType<ArrayConstructorOf<T>>,
    public readonly readonly = false,
  ) {}

  get buffer(): ArrayBuffer {
    return this.array.buffer;
  }

  get byteOffset(): number {
    return this.array.byteOffset;
  }

  get byteLength(): number {
    return this.array.byteLength;
  }

  get length(): number {
    return this.array.length;
  }

  get(index: number): JSTypeOf<T> {
    if (index < 0 || index >= this.length) {
      throw new Error(
        `Index ${index.toString()} out of bounds for array of length ${this.length.toString()}`,
      );
    }

    // Convert 0/1 to boolean for bool dtype
    if (this.dtype.jsType === 'boolean') {
      return (this.array[index] !== 0) as JSTypeOf<T>;
    }

    return this.array[index] as JSTypeOf<T>;
  }

  set(index: number, value: JSTypeOf<T>): void {
    if (this.readonly) {
      throw new Error('Cannot modify readonly TypedArray');
    }
    if (index < 0 || index >= this.length) {
      throw new Error(
        `Index ${index.toString()} out of bounds for array of length ${this.length.toString()}`,
      );
    }
    if (!this.dtype.isValidValue(value)) {
      throw new Error(`Invalid value for ${this.dtype.name}: ${String(value)}`);
    }

    // Convert boolean to 0/1 for storage
    if (this.dtype.jsType === 'boolean') {
      this.array[index] = (value ? 1 : 0) as never;
    } else {
      this.array[index] = value as never; // TypeScript limitation with generic array access
    }
  }

  subarray(begin?: number, end?: number): DTypedArray<T> {
    const subArray = this.array.subarray(begin, end) as InstanceType<ArrayConstructorOf<T>>;
    return new TypedArrayWrapper(this.dtype, subArray, this.readonly);
  }

  slice(begin?: number, end?: number): DTypedArray<T> {
    const slicedArray = this.array.slice(begin, end) as InstanceType<ArrayConstructorOf<T>>;
    return new TypedArrayWrapper(this.dtype, slicedArray, this.readonly);
  }

  copyWithin(target: number, start: number, end?: number): DTypedArray<T> {
    if (this.readonly) {
      throw new Error('Cannot modify readonly TypedArray');
    }
    this.array.copyWithin(target, start, end);
    return this;
  }

  fill(value: JSTypeOf<T>, start?: number, end?: number): DTypedArray<T> {
    if (this.readonly) {
      throw new Error('Cannot modify readonly TypedArray');
    }
    if (!this.dtype.isValidValue(value)) {
      throw new Error(`Invalid fill value for ${this.dtype.name}: ${String(value)}`);
    }

    // Convert boolean to 0/1 for storage
    if (this.dtype.jsType === 'boolean') {
      this.array.fill((value ? 1 : 0) as never, start, end);
    } else {
      this.array.fill(value as never, start, end);
    }
    return this;
  }

  toArray(): readonly JSTypeOf<T>[] {
    // Handle bigint arrays separately from number arrays
    if (this.dtype.jsType === 'bigint') {
      const result: bigint[] = [];
      const bigintArray = this.array as BigInt64Array | BigUint64Array;
      for (const value of bigintArray) {
        result.push(value);
      }
      return result as readonly JSTypeOf<T>[];
    } else if (this.dtype.jsType === 'boolean') {
      // Convert 0/1 back to boolean values
      const result: boolean[] = [];
      for (let i = 0; i < this.length; i++) {
        result.push(this.array[i] !== 0);
      }
      return result as readonly JSTypeOf<T>[];
    } else {
      return Array.from(
        this.array as Exclude<InstanceType<ArrayConstructorOf<T>>, BigInt64Array | BigUint64Array>,
      ) as readonly JSTypeOf<T>[];
    }
  }

  forEach(callback: (value: JSTypeOf<T>, index: number, array: DTypedArray<T>) => void): void {
    for (let i = 0; i < this.length; i++) {
      callback(this.get(i), i, this);
    }
  }

  map<U extends AnyDType>(
    callback: (value: JSTypeOf<T>, index: number, array: DTypedArray<T>) => JSTypeOf<U>,
    targetDType: RuntimeDType<U>,
  ): DTypedArray<U> {
    const resultArray = targetDType.createTypedArray(this.length);
    const result = new TypedArrayWrapper(targetDType, resultArray);

    for (let i = 0; i < this.length; i++) {
      const mappedValue = callback(this.get(i), i, this);
      result.set(i, mappedValue);
    }

    return result;
  }

  createView<U extends AnyDType>(
    targetDType: RuntimeDType<U>,
    byteOffset?: number,
    length?: number,
  ): DTypedArray<U> {
    const actualByteOffset = (byteOffset ?? 0) + this.byteOffset;

    // Validate alignment
    if (actualByteOffset % targetDType.byteSize !== 0) {
      throw new Error(
        `Buffer alignment error: offset ${actualByteOffset.toString()} not aligned to ${targetDType.byteSize.toString()} bytes for ${targetDType.name}`,
      );
    }

    // Calculate length if not provided
    const remainingBytes = this.buffer.byteLength - actualByteOffset;
    const maxElements = Math.floor(remainingBytes / targetDType.byteSize);

    // If length is specified, validate it fits in the buffer
    if (length !== undefined && length > maxElements) {
      throw new Error('Invalid length for typed array view');
    }

    const actualLength = length !== undefined ? length : maxElements;

    if (actualLength < 0) {
      throw new Error('Invalid length for typed array view');
    }

    const viewArray = targetDType.createTypedArrayFromBuffer(
      this.buffer,
      actualByteOffset,
      actualLength,
    );

    return new TypedArrayWrapper(targetDType, viewArray, this.readonly);
  }
}

// =============================================================================
// Factory Functions
// =============================================================================

/**
 * Create a DTypedArray from a length
 *
 * @example
 * const float32Array = createTypedArray(getDType('float32'), 1000);
 * console.log(float32Array.length); // 1000
 * console.log(float32Array.dtype.name); // 'float32'
 */
export function createTypedArray<T extends AnyDType>(
  dtype: RuntimeDType<T>,
  length: number,
): DTypedArray<T> {
  if (length < 0 || !Number.isInteger(length)) {
    throw new Error(`Invalid array length: ${length.toString()}`);
  }

  const array = dtype.createTypedArray(length);
  return new TypedArrayWrapper(dtype, array);
}

/**
 * Create a DTypedArray from an existing ArrayBuffer
 *
 * @example
 * const buffer = new ArrayBuffer(1024);
 * const int32Array = createTypedArrayFromBuffer(getDType('int32'), buffer, 0, 256);
 */
export function createTypedArrayFromBuffer<T extends AnyDType>(
  dtype: RuntimeDType<T>,
  buffer: ArrayBuffer,
  byteOffset = 0,
  length?: number,
): DTypedArray<T> {
  // Validate alignment
  if (byteOffset % dtype.byteSize !== 0) {
    throw new Error(
      `Buffer alignment error: offset ${byteOffset.toString()} not aligned to ${dtype.byteSize.toString()} bytes for ${dtype.name}`,
    );
  }

  // Validate buffer size
  const remainingBytes = buffer.byteLength - byteOffset;
  if (remainingBytes < 0) {
    throw new Error(
      `Byte offset ${byteOffset.toString()} exceeds buffer size ${buffer.byteLength.toString()}`,
    );
  }

  const maxElements = Math.floor(remainingBytes / dtype.byteSize);

  // If length is specified, validate it fits in the buffer
  if (length !== undefined && length > maxElements) {
    throw new Error('Invalid length for typed array');
  }

  const actualLength = length !== undefined ? length : maxElements;

  if (actualLength < 0) {
    throw new Error('Invalid length for typed array');
  }

  const array = dtype.createTypedArrayFromBuffer(buffer, byteOffset, actualLength);
  return new TypedArrayWrapper(dtype, array);
}

/**
 * Create a DTypedArray from array-like data with validation
 *
 * @example
 * const data = [1, 2, 3, 4, 5];
 * const int32Array = createTypedArrayFromData(getDType('int32'), data);
 */
export function createTypedArrayFromData<T extends AnyDType>(
  dtype: RuntimeDType<T>,
  data: ArrayLike<JSTypeOf<T>>,
): DTypedArray<T> {
  const array = dtype.createTypedArrayFromData(data);
  return new TypedArrayWrapper(dtype, array);
}

/**
 * Create a read-only DTypedArray wrapper
 *
 * @example
 * const mutableArray = createTypedArray(getDType('float32'), 100);
 * const readonlyArray = createReadonlyTypedArray(mutableArray);
 * // readonlyArray.set(0, 1); // Throws error
 */
export function createReadonlyTypedArray<T extends AnyDType>(
  source: DTypedArray<T>,
): DTypedArray<T> {
  return new TypedArrayWrapper(source.dtype, source.array, true);
}

// =============================================================================
// Conversion and Casting Functions
// =============================================================================

/**
 * Convert a DTypedArray to a different DType
 *
 * @example
 * const int32Array = createTypedArrayFromData(getDType('int32'), [1, 2, 3]);
 * const float32Array = convertTypedArray(int32Array, getDType('float32'));
 */
export function convertTypedArray<From extends AnyDType, To extends AnyDType>(
  source: DTypedArray<From>,
  targetDType: RuntimeDType<To>,
  options?: ConversionOptions,
): DTypedArray<To> {
  const sourceData = source.toArray();
  const conversionResult = convertArray(sourceData, source.dtype, targetDType, options);

  if (!conversionResult.success) {
    throw new Error(`Array conversion failed: ${conversionResult.errors.join(', ')}`);
  }

  if (!conversionResult.values) {
    throw new Error('Conversion result values are undefined');
  }

  return createTypedArrayFromData(targetDType, conversionResult.values);
}

/**
 * Create a zero-copy view of a DTypedArray with a different DType
 * The target DType must have compatible alignment and size
 *
 * @example
 * const float32Array = createTypedArray(getDType('float32'), 256);
 * const uint32View = createTypedArrayView(float32Array, getDType('uint32'));
 * // Both arrays share the same underlying buffer
 */
export function createTypedArrayView<From extends AnyDType, To extends AnyDType>(
  source: DTypedArray<From>,
  targetDType: RuntimeDType<To>,
  byteOffset = 0,
  length?: number,
): DTypedArray<To> {
  return source.createView(targetDType, byteOffset, length);
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Wrap an existing JavaScript TypedArray with type safety
 *
 * @example
 * const jsArray = new Float32Array([1, 2, 3, 4]);
 * const wrappedArray = wrapTypedArray(jsArray);
 * console.log(wrappedArray.dtype.name); // 'float32'
 */
export function wrapTypedArray(typedArray: ArrayBufferView): DTypedArray<AnyDType> | null {
  const dtype = getTypedArrayDType(typedArray);
  if (!dtype) {
    return null;
  }

  return new TypedArrayWrapper(dtype, typedArray as never);
}

/**
 * Check if two DTypedArray instances share the same underlying buffer
 *
 * @example
 * const array1 = createTypedArray(getDType('int32'), 100);
 * const array2 = array1.subarray(10, 20);
 * console.log(sharesSameBuffer(array1, array2)); // true
 */
export function sharesSameBuffer<A extends AnyDType, B extends AnyDType>(
  a: DTypedArray<A>,
  b: DTypedArray<B>,
): boolean {
  return a.buffer === b.buffer;
}

/**
 * Check if two DTypedArray instances overlap in memory
 *
 * @example
 * const array1 = createTypedArray(getDType('int32'), 100);
 * const array2 = array1.subarray(50, 150); // Would be clamped to available length
 * console.log(hasMemoryOverlap(array1, array2)); // true
 */
export function hasMemoryOverlap<A extends AnyDType, B extends AnyDType>(
  a: DTypedArray<A>,
  b: DTypedArray<B>,
): boolean {
  if (!sharesSameBuffer(a, b)) {
    return false;
  }

  const aStart = a.byteOffset;
  const aEnd = a.byteOffset + a.byteLength;
  const bStart = b.byteOffset;
  const bEnd = b.byteOffset + b.byteLength;

  return aStart < bEnd && bStart < aEnd;
}

/**
 * Calculate the total memory usage of a DTypedArray including metadata
 *
 * @example
 * const array = createTypedArray(getDType('float64'), 1000);
 * const usage = calculateMemoryUsage(array);
 * console.log(`Array uses ${usage.totalBytes} bytes`);
 */
export function calculateMemoryUsage<T extends AnyDType>(
  array: DTypedArray<T>,
): {
  dataBytes: number;
  metadataBytes: number;
  totalBytes: number;
  elementsPerMB: number;
} {
  const dataBytes = array.byteLength;
  const metadataBytes =
    8 + // Object overhead
    8 + // dtype reference
    8 + // array reference
    4 + // readonly flag
    4; // padding

  const totalBytes = dataBytes + metadataBytes;
  const elementsPerMB = Math.floor((1024 * 1024) / array.dtype.byteSize);

  return {
    dataBytes,
    metadataBytes,
    totalBytes,
    elementsPerMB,
  };
}

/**
 * Validate that a DTypedArray has the expected properties
 *
 * @example
 * const array = createTypedArray(getDType('int32'), 100);
 * const isValid = validateTypedArray(array, {
 *   minLength: 50,
 *   maxLength: 200,
 *   alignment: 4
 * });
 */
export function validateTypedArray<T extends AnyDType>(
  array: DTypedArray<T>,
  constraints: {
    minLength?: number;
    maxLength?: number;
    alignment?: number;
    dtype?: RuntimeDType<T>;
  } = {},
): boolean {
  const { minLength, maxLength, alignment, dtype } = constraints;

  if (minLength !== undefined && array.length < minLength) {
    return false;
  }

  if (maxLength !== undefined && array.length > maxLength) {
    return false;
  }

  if (alignment !== undefined && array.byteOffset % alignment !== 0) {
    return false;
  }

  if (dtype !== undefined && array.dtype !== dtype) {
    return false;
  }

  return true;
}

// =============================================================================
// Performance Utilities
// =============================================================================

/**
 * Efficiently copy data between DTypedArray instances
 * Uses the fastest available copy method based on compatibility
 *
 * @example
 * const source = createTypedArrayFromData(getDType('float32'), [1, 2, 3, 4]);
 * const target = createTypedArray(getDType('float32'), 4);
 * copyTypedArrayData(source, target);
 */
export function copyTypedArrayData<From extends AnyDType, To extends AnyDType>(
  source: DTypedArray<From>,
  target: DTypedArray<To>,
  sourceOffset = 0,
  targetOffset = 0,
  length?: number,
): void {
  const actualLength =
    length ?? Math.min(source.length - sourceOffset, target.length - targetOffset);

  if (actualLength <= 0) {
    return;
  }

  // Validate bounds
  if (sourceOffset + actualLength > source.length) {
    throw new Error('Source copy bounds exceed array length');
  }

  if (targetOffset + actualLength > target.length) {
    throw new Error('Target copy bounds exceed array length');
  }

  // Same DType - use fast typed array copy
  if (source.dtype.name === target.dtype.name) {
    const sourceSubarray = source.array.subarray(sourceOffset, sourceOffset + actualLength);

    // Type-narrow based on whether it's a bigint array
    if (source.dtype.jsType === 'bigint') {
      // Both are bigint arrays since they have the same dtype
      const targetBigInt = target.array as BigInt64Array | BigUint64Array;
      const sourceBigInt = sourceSubarray as BigInt64Array | BigUint64Array;
      targetBigInt.set(sourceBigInt, targetOffset);
    } else {
      // Both are number arrays
      const targetNumber = target.array as Exclude<
        InstanceType<ArrayConstructorOf<AnyDType>>,
        BigInt64Array | BigUint64Array
      >;
      const sourceNumber = sourceSubarray as Exclude<
        InstanceType<ArrayConstructorOf<AnyDType>>,
        BigInt64Array | BigUint64Array
      >;
      targetNumber.set(sourceNumber, targetOffset);
    }
    return;
  }

  // Different DTypes - convert element by element
  for (let i = 0; i < actualLength; i++) {
    const sourceValue = source.get(sourceOffset + i);
    const conversionResult = convertValue(sourceValue, source.dtype, target.dtype);

    if (!conversionResult.success) {
      throw new Error(
        `Conversion failed at index ${i.toString()}: ${conversionResult.errors.join(', ')}`,
      );
    }

    target.set(targetOffset + i, conversionResult.value);
  }
}

/**
 * Create a memory-aligned buffer for optimal performance
 *
 * @example
 * const buffer = createAlignedBuffer(1024, 64); // 1KB buffer, 64-byte aligned
 * const array = createTypedArrayFromBuffer(getDType('float32'), buffer);
 */
export function createAlignedBuffer(size: number, alignment = 32): ArrayBuffer {
  if (alignment <= 0 || (alignment & (alignment - 1)) !== 0) {
    throw new Error('Alignment must be a positive power of 2');
  }

  // Create a slightly larger buffer to allow for alignment adjustment
  const buffer = new ArrayBuffer(size + alignment - 1);

  // Check if the buffer is already aligned
  // Note: We can't get the actual memory address in JavaScript
  const alignedSize = Math.floor(size / alignment) * alignment;

  // Return a view of the aligned portion
  // Note: In JavaScript, we can't control memory alignment directly,
  // but this function provides the interface for when it becomes available
  return buffer.slice(0, Math.max(size, alignedSize));
}

// =============================================================================
// Error Classes
// =============================================================================

/**
 * Error thrown when TypedArray operations fail
 */
export class TypedArrayError extends Error {
  constructor(
    message: string,
    public readonly dtype?: RuntimeDType,
    public readonly arrayLength?: number,
  ) {
    super(message);
    this.name = 'TypedArrayError';
  }
}

/**
 * Error thrown when buffer alignment is invalid
 */
export class AlignmentError extends TypedArrayError {
  constructor(
    message: string,
    public readonly requiredAlignment: number,
    public readonly actualOffset: number,
  ) {
    super(message);
    this.name = 'AlignmentError';
  }
}

/**
 * Error thrown when array bounds are exceeded
 */
export class BoundsError extends TypedArrayError {
  constructor(
    message: string,
    public readonly index: number,
    public override readonly arrayLength: number,
  ) {
    super(message, undefined, arrayLength);
    this.name = 'BoundsError';
  }
}
