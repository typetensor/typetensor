/**
 * Runtime DType management and validation
 *
 * This module provides runtime implementations for DType operations,
 * including type validation, range checking, and integration with
 * JavaScript's TypedArray system. It bridges compile-time branded
 * types with runtime type safety.
 */

import type {
  AnyDType,
  DTypeName,
  DTypeFromName,
  JSTypeOf,
  ArrayConstructorOf,
  TypedArrayConstructor,
  Bool,
  Float32,
  Int16,
  Int32,
  Int8,
  Uint16,
  Uint32,
  Uint8,
  Uint64,
  Int64,
  Float64,
} from './types.js';

// =============================================================================
// Runtime DType Class
// =============================================================================

/**
 * Runtime representation of a DType with validation and metadata
 * This class provides the bridge between compile-time branded types
 * and runtime type checking and validation.
 */
export class RuntimeDType<T extends AnyDType = AnyDType> {
  public readonly name: T['__dtype'];
  public readonly jsType: 'number' | 'boolean' | 'bigint';
  public readonly typedArrayConstructor: TypedArrayConstructor;
  public readonly byteSize: number;
  public readonly signed: boolean;
  public readonly isInteger: boolean;
  public readonly minValue: number | bigint;
  public readonly maxValue: number | bigint;

  constructor(
    name: T['__dtype'],
    jsType: 'number' | 'boolean' | 'bigint',
    typedArrayConstructor: TypedArrayConstructor,
    byteSize: number,
    signed: boolean,
    isInteger: boolean,
    minValue: number | bigint,
    maxValue: number | bigint,
  ) {
    this.name = name;
    this.jsType = jsType;
    this.typedArrayConstructor = typedArrayConstructor;
    this.byteSize = byteSize;
    this.signed = signed;
    this.isInteger = isInteger;
    this.minValue = minValue;
    this.maxValue = maxValue;
  }

  /**
   * Type guard to check if a value matches this DType
   * Includes range validation for numeric types
   */
  isValidValue(value: unknown): value is JSTypeOf<T> {
    // Check JavaScript type first
    if (typeof value !== this.jsType) {
      return false;
    }

    // Boolean type is always valid if typeof matches
    if (this.jsType === 'boolean') {
      return true;
    }

    // BigInt type validation
    if (this.jsType === 'bigint') {
      const bigintValue = value as bigint;
      return bigintValue >= this.minValue && bigintValue <= this.maxValue;
    }

    // Number type validation with range checking
    const numValue = value as number;

    // Check for NaN and Infinity
    if (!Number.isFinite(numValue)) {
      // Allow NaN and Infinity only for floating-point types
      return !this.isInteger;
    }

    // Integer types must be whole numbers
    if (this.isInteger && !Number.isInteger(numValue)) {
      return false;
    }

    // Range validation
    return numValue >= this.minValue && numValue <= this.maxValue;
  }

  /**
   * Validate and potentially convert a value to this DType's JavaScript type
   * Returns the validated value or throws an error
   */
  validateValue(value: unknown): JSTypeOf<T> {
    if (this.isValidValue(value)) {
      return value;
    }

    throw new Error(`Invalid value for ${this.name}: ${String(value)}`);
  }

  /**
   * Create a TypedArray instance of this DType
   */
  createTypedArray(length: number): InstanceType<ArrayConstructorOf<T>> {
    return new this.typedArrayConstructor(length) as InstanceType<ArrayConstructorOf<T>>;
  }

  /**
   * Create a TypedArray from an existing buffer
   */
  createTypedArrayFromBuffer(
    buffer: ArrayBuffer,
    byteOffset?: number,
    length?: number,
  ): InstanceType<ArrayConstructorOf<T>> {
    return new this.typedArrayConstructor(buffer, byteOffset, length) as InstanceType<
      ArrayConstructorOf<T>
    >;
  }

  /**
   * Create a TypedArray from array-like data
   */
  createTypedArrayFromData(data: ArrayLike<JSTypeOf<T>>): InstanceType<ArrayConstructorOf<T>> {
    // Validate data before creating array for better error messages
    for (let i = 0; i < data.length; i++) {
      if (!this.isValidValue(data[i])) {
        throw new Error(`Invalid value at index ${i} for ${this.name}: ${String(data[i])}`);
      }
    }

    // Handle different JavaScript types properly for TypedArray construction
    if (this.jsType === 'bigint') {
      return new (this.typedArrayConstructor as
        | BigInt64ArrayConstructor
        | BigUint64ArrayConstructor)(data as ArrayLike<bigint>) as InstanceType<
        ArrayConstructorOf<T>
      >;
    } else {
      // TypeScript needs explicit casting for non-bigint typed arrays
      const NumberArrayConstructor = this.typedArrayConstructor as
        | Int8ArrayConstructor
        | Uint8ArrayConstructor
        | Int16ArrayConstructor
        | Uint16ArrayConstructor
        | Int32ArrayConstructor
        | Uint32ArrayConstructor
        | Float32ArrayConstructor
        | Float64ArrayConstructor;
      return new NumberArrayConstructor(data as ArrayLike<number>) as InstanceType<
        ArrayConstructorOf<T>
      >;
    }
  }

  /**
   * Check if this DType is compatible with another for operations
   */
  isCompatibleWith(other: RuntimeDType): boolean {
    // Same type is always compatible
    if (this.name === other.name) {
      return true;
    }

    // Boolean is compatible with any numeric type
    if (this.name === 'bool' || other.name === 'bool') {
      return true;
    }

    // Different JavaScript types are not directly compatible
    if (this.jsType !== other.jsType) {
      return false;
    }

    // Same JavaScript type means they can be promoted
    return true;
  }

  /**
   * Get information about this DType for debugging
   */
  getInfo(): {
    name: string;
    jsType: string;
    byteSize: number;
    signed: boolean;
    isInteger: boolean;
    minValue: number | bigint;
    maxValue: number | bigint;
    typedArrayName: string;
  } {
    return {
      name: this.name,
      jsType: this.jsType,
      byteSize: this.byteSize,
      signed: this.signed,
      isInteger: this.isInteger,
      minValue: this.minValue,
      maxValue: this.maxValue,
      typedArrayName: this.typedArrayConstructor.name,
    };
  }

  /**
   * String representation for debugging
   */
  toString(): string {
    return `RuntimeDType(${this.name})`;
  }

  /**
   * JSON representation for serialization
   */
  toJSON(): { name: string; byteSize: number; signed: boolean; isInteger: boolean } {
    return {
      name: this.name,
      byteSize: this.byteSize,
      signed: this.signed,
      isInteger: this.isInteger,
    };
  }
}

// =============================================================================
// DType Registry and Factory
// =============================================================================

/**
 * Precomputed DType instances for efficient lookup
 * Uses singleton pattern to ensure consistent instances
 *
 * NOTE: Order matters for getTypedArrayDType when multiple dtypes use the same
 * TypedArray constructor. uint8 comes before bool so that raw Uint8Array
 * instances default to uint8 (the more common interpretation).
 */
export const DTYPES = {
  int8: new RuntimeDType(
    'int8',
    'number',
    Int8Array,
    1,
    true,
    true,
    -128,
    127,
  ) as RuntimeDType<Int8>,
  uint8: new RuntimeDType(
    'uint8',
    'number',
    Uint8Array,
    1,
    false,
    true,
    0,
    255,
  ) as RuntimeDType<Uint8>,
  bool: new RuntimeDType('bool', 'boolean', Uint8Array, 1, false, true, 0, 1) as RuntimeDType<Bool>,
  int16: new RuntimeDType(
    'int16',
    'number',
    Int16Array,
    2,
    true,
    true,
    -32768,
    32767,
  ) as RuntimeDType<Int16>,
  uint16: new RuntimeDType(
    'uint16',
    'number',
    Uint16Array,
    2,
    false,
    true,
    0,
    65535,
  ) as RuntimeDType<Uint16>,
  int32: new RuntimeDType(
    'int32',
    'number',
    Int32Array,
    4,
    true,
    true,
    -2147483648,
    2147483647,
  ) as RuntimeDType<Int32>,
  uint32: new RuntimeDType(
    'uint32',
    'number',
    Uint32Array,
    4,
    false,
    true,
    0,
    4294967295,
  ) as RuntimeDType<Uint32>,
  float32: new RuntimeDType(
    'float32',
    'number',
    Float32Array,
    4,
    true,
    false,
    -3.4028235e38,
    3.4028235e38,
  ) as RuntimeDType<Float32>,
  float64: new RuntimeDType(
    'float64',
    'number',
    Float64Array,
    8,
    true,
    false,
    -Number.MAX_VALUE,
    Number.MAX_VALUE,
  ) as RuntimeDType<Float64>,
  int64: new RuntimeDType(
    'int64',
    'bigint',
    BigInt64Array,
    8,
    true,
    true,
    -9223372036854775808n,
    9223372036854775807n,
  ) as RuntimeDType<Int64>,
  uint64: new RuntimeDType(
    'uint64',
    'bigint',
    BigUint64Array,
    8,
    false,
    true,
    0n,
    18446744073709551615n,
  ) as RuntimeDType<Uint64>,
} as const;

/**
 * Type-safe runtime DType lookup
 * Maps DType names to their runtime instances
 */
export type DTypeRegistry = typeof DTYPES;

/**
 * Get a RuntimeDType instance by name with proper typing
 *
 * @example
 * const float32DType = getDType('float32'); // RuntimeDType<Float32>
 * const boolDType = getDType('bool'); // RuntimeDType<Bool>
 */
export function getDType<N extends DTypeName>(name: N): RuntimeDType<DTypeFromName<N>> {
  const dtype = DTYPES[name];
  // eslint-disable-next-line
  if (!dtype) {
    throw new Error(`Unknown DType: ${name}`);
  }
  return dtype as RuntimeDType<DTypeFromName<N>>;
}

/**
 * Get all available DType names
 */
export function getDTypeNames(): readonly DTypeName[] {
  return Object.keys(DTYPES) as readonly DTypeName[];
}

/**
 * Check if a string is a valid DType name
 */
export function isValidDTypeName(name: string): name is DTypeName {
  return name in DTYPES;
}

/**
 * Get default DType for a JavaScript value
 */
export function getDefaultDType(value: unknown): RuntimeDType {
  if (typeof value === 'boolean') {
    return DTYPES.bool;
  }
  if (typeof value === 'bigint') {
    return value >= 0n ? DTYPES.uint64 : DTYPES.int64;
  }
  if (typeof value === 'number') {
    if (Number.isInteger(value)) {
      if (value >= -2147483648 && value <= 2147483647) {
        return DTYPES.int32;
      }
      return DTYPES.float64; // Large integers go to float64
    }
    return DTYPES.float32; // Default for floating-point numbers
  }
  throw new Error(`Cannot determine DType for value: ${String(value)}`);
}

// =============================================================================
// Type Guards and Validation
// =============================================================================

/**
 * Type guard to check if a value is a RuntimeDType
 */
export function isRuntimeDType(value: unknown): value is RuntimeDType {
  return value instanceof RuntimeDType;
}

/**
 * Validate that a TypedArray matches the expected DType
 */
export function validateTypedArrayDType<T extends AnyDType>(
  array: ArrayLike<unknown>,
  expectedDType: RuntimeDType<T>,
): array is InstanceType<ArrayConstructorOf<T>> {
  // Check if it's actually a TypedArray
  if (!ArrayBuffer.isView(array)) {
    return false;
  }

  // Check if the constructor matches
  return array.constructor === expectedDType.typedArrayConstructor;
}

/**
 * Get the DType of a TypedArray
 *
 * NOTE: This function cannot distinguish between dtypes that use the same
 * TypedArray constructor (e.g., bool and uint8 both use Uint8Array).
 * It returns the first matching dtype in the registry order.
 * For Uint8Array, this returns uint8 by default.
 */
export function getTypedArrayDType(array: ArrayBufferView): RuntimeDType | null {
  for (const dtype of Object.values(DTYPES)) {
    if (array.constructor === dtype.typedArrayConstructor) {
      return dtype;
    }
  }
  return null;
}

/**
 * Validate that an array-like object contains values compatible with a DType
 */
export function validateArrayData<T extends AnyDType>(
  data: ArrayLike<unknown>,
  dtype: RuntimeDType<T>,
): readonly JSTypeOf<T>[] {
  const result: JSTypeOf<T>[] = [];

  for (let i = 0; i < data.length; i++) {
    const value = data[i];
    if (!dtype.isValidValue(value)) {
      throw new Error(`Invalid value at index ${i} for ${dtype.name}: ${String(value)}`);
    }
    result.push(value);
  }

  return result;
}

// =============================================================================
// Memory and Performance Utilities
// =============================================================================

/**
 * Calculate the total byte size needed for an array of a given DType
 */
export function calculateByteSize(length: number, dtype: RuntimeDType): number {
  return length * dtype.byteSize;
}

/**
 * Check if a buffer has the correct size for a DType array
 */
export function validateBufferSize(
  buffer: ArrayBuffer,
  length: number,
  dtype: RuntimeDType,
  byteOffset = 0,
): boolean {
  const requiredBytes = length * dtype.byteSize;
  const availableBytes = buffer.byteLength - byteOffset;
  return availableBytes >= requiredBytes;
}

/**
 * Check if a buffer offset is properly aligned for a DType
 */
export function isAligned(byteOffset: number, dtype: RuntimeDType): boolean {
  return byteOffset % dtype.byteSize === 0;
}

/**
 * Get the next aligned offset for a DType
 */
export function getAlignedOffset(byteOffset: number, dtype: RuntimeDType): number {
  const alignment = dtype.byteSize;
  return Math.ceil(byteOffset / alignment) * alignment;
}

// =============================================================================
// Error Classes
// =============================================================================

/**
 * Error thrown when DType operations fail
 */
export class DTypeError extends Error {
  constructor(
    message: string,
    public readonly dtype?: RuntimeDType,
    public readonly value?: unknown,
  ) {
    super(message);
    this.name = 'DTypeError';
  }
}

/**
 * Error thrown when type validation fails
 */
export class DTypeValidationError extends DTypeError {
  constructor(value: unknown, expectedDType: RuntimeDType) {
    super(
      `Value ${String(value)} is not valid for DType ${expectedDType.name}`,
      expectedDType,
      value,
    );
    this.name = 'DTypeValidationError';
  }
}

/**
 * Error thrown when buffer operations fail
 */
export class DTypeBufferError extends DTypeError {
  constructor(message: string, dtype: RuntimeDType) {
    super(message, dtype);
    this.name = 'DTypeBufferError';
  }
}
