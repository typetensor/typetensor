/**
 * Runtime dtype constants for use in tensor operations
 *
 * These constants provide runtime values that satisfy the compile-time
 * dtype interfaces, allowing them to be passed to functions that expect
 * dtype parameters.
 */

import type {
  Bool,
  Int8,
  Uint8,
  Int16,
  Uint16,
  Int32,
  Uint32,
  Int64,
  Uint64,
  Float32,
  Float64,
} from './types';

/**
 * Boolean dtype constant
 */
export const bool = Object.freeze({
  __dtype: 'bool',
  __jsType: false as boolean,
  __typedArray: Uint8Array,
  __byteSize: 1 as const,
  __signed: false as const,
  __isInteger: true as const,
} as const) satisfies Bool;

/**
 * 8-bit signed integer dtype constant
 */
export const int8 = Object.freeze({
  __dtype: 'int8',
  __jsType: 0 as number,
  __typedArray: Int8Array,
  __byteSize: 1 as const,
  __signed: true as const,
  __isInteger: true as const,
} as const) satisfies Int8;

/**
 * 8-bit unsigned integer dtype constant
 */
export const uint8 = Object.freeze({
  __dtype: 'uint8',
  __jsType: 0 as number,
  __typedArray: Uint8Array,
  __byteSize: 1 as const,
  __signed: false as const,
  __isInteger: true as const,
} as const) satisfies Uint8;

/**
 * 16-bit signed integer dtype constant
 */
export const int16 = Object.freeze({
  __dtype: 'int16',
  __jsType: 0 as number,
  __typedArray: Int16Array,
  __byteSize: 2 as const,
  __signed: true as const,
  __isInteger: true as const,
} as const) satisfies Int16;

/**
 * 16-bit unsigned integer dtype constant
 */
export const uint16 = Object.freeze({
  __dtype: 'uint16',
  __jsType: 0 as number,
  __typedArray: Uint16Array,
  __byteSize: 2 as const,
  __signed: false as const,
  __isInteger: true as const,
} as const) satisfies Uint16;

/**
 * 32-bit signed integer dtype constant
 */
export const int32 = Object.freeze({
  __dtype: 'int32',
  __jsType: 0 as number,
  __typedArray: Int32Array,
  __byteSize: 4 as const,
  __signed: true as const,
  __isInteger: true as const,
} as const) satisfies Int32;

/**
 * 32-bit unsigned integer dtype constant
 */
export const uint32 = Object.freeze({
  __dtype: 'uint32',
  __jsType: 0 as number,
  __typedArray: Uint32Array,
  __byteSize: 4 as const,
  __signed: false as const,
  __isInteger: true as const,
} as const) satisfies Uint32;

/**
 * 64-bit signed integer dtype constant
 */
export const int64 = Object.freeze({
  __dtype: 'int64',
  __jsType: 0n as bigint,
  __typedArray: BigInt64Array,
  __byteSize: 8 as const,
  __signed: true as const,
  __isInteger: true as const,
} as const) satisfies Int64;

/**
 * 64-bit unsigned integer dtype constant
 */
export const uint64 = Object.freeze({
  __dtype: 'uint64',
  __jsType: 0n as bigint,
  __typedArray: BigUint64Array,
  __byteSize: 8 as const,
  __signed: false as const,
  __isInteger: true as const,
} as const) satisfies Uint64;

/**
 * 32-bit floating point dtype constant
 */
export const float32 = Object.freeze({
  __dtype: 'float32',
  __jsType: 0 as number,
  __typedArray: Float32Array,
  __byteSize: 4 as const,
  __signed: true as const,
  __isInteger: false as const,
} as const) satisfies Float32;

/**
 * 64-bit floating point dtype constant
 */
export const float64 = Object.freeze({
  __dtype: 'float64',
  __jsType: 0 as number,
  __typedArray: Float64Array,
  __byteSize: 8 as const,
  __signed: true as const,
  __isInteger: false as const,
} as const) satisfies Float64;

/**
 * Registry mapping dtype names to their corresponding constants
 *
 * This provides a centralized lookup for dtype constants by name,
 * useful for operations that need to convert from runtime dtype names
 * to the appropriate branded dtype constants.
 */
export const DTYPE_CONSTANTS_MAP = Object.freeze({
  bool: bool,
  int8: int8,
  uint8: uint8,
  int16: int16,
  uint16: uint16,
  int32: int32,
  uint32: uint32,
  int64: int64,
  uint64: uint64,
  float32: float32,
  float64: float64,
} as const);

/**
 * Get a dtype constant by name
 *
 * @param name The dtype name
 * @returns The corresponding dtype constant
 */
export function getDTypeConstant<T extends keyof typeof DTYPE_CONSTANTS_MAP>(
  name: T,
): (typeof DTYPE_CONSTANTS_MAP)[T] {
  return DTYPE_CONSTANTS_MAP[name];
}

/**
 * Convert a dtype to its floating point equivalent at runtime
 * - Float types remain unchanged (preserve precision)
 * - Integer and boolean types convert to Float32
 *
 * This matches the compile-time ToFloat type behavior
 *
 * @param dtype The input dtype
 * @returns The floating point dtype
 */
export function toFloatDType<T extends import('./types').AnyDType>(
  dtype: T,
): T extends import('./types').Float32
  ? typeof float32
  : T extends import('./types').Float64
    ? typeof float64
    : typeof float32 {
  if (dtype.__dtype === 'float32') {
    return float32 as any;
  }
  if (dtype.__dtype === 'float64') {
    return float64 as any;
  }
  // All other types (integers, bool) convert to float32
  return float32 as any;
}
