/**
 * DType System - Type-safe numeric types for tensor operations
 *
 * This module provides a comprehensive type system for numeric data with:
 * - Compile-time type safety through branded types
 * - NumPy-compatible type promotion rules
 * - Safe type conversion with overflow/precision loss detection
 * - Efficient TypedArray integration with zero-copy operations
 * - Runtime validation and error handling
 *
 * @example Basic Usage
 * ```typescript
 * import { getDType, createTypedArray, promoteTypes } from './dtype';
 *
 * const float32 = getDType('float32');
 * const int32 = getDType('int32');
 *
 * // Create typed arrays
 * const floatArray = createTypedArray(float32, 1000);
 * const intArray = createTypedArray(int32, 1000);
 *
 * // Type promotion
 * const promoted = promoteTypes(float32, int32); // RuntimeDType<Float64>
 * ```
 *
 * @example Type-Safe Operations
 * ```typescript
 * import type { Float32, Int32, Promote } from './dtype';
 *
 * // Compile-time type checking
 * type ResultType = Promote<Float32, Int32>; // Float64
 *
 * function addTensors<A extends AnyDType, B extends AnyDType>(
 *   a: TensorType<A, Shape>,
 *   b: TensorType<B, Shape>
 * ): TensorType<Promote<A, B>, Shape> {
 *   // Implementation with type safety
 * }
 * ```
 */

// =============================================================================
// Type Exports
// =============================================================================

export type {
  // Core branded types
  DType,
  AnyDType,
  DTypeName,
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

  // Utility types
  DTypeNameOf,
  DTypeValue,
  JSTypeOf,
  ArrayConstructorOf,
  ByteSizeOf,
  DTypeFromName,
  IsIntegerDType,
  IsFloatDType,
  IsSignedDType,

  // Type operations
  Promote,
  CanPromote,
  CanSafelyCast,
  IsLossyCast,

  // Error types (type-only)
  IncompatibleDTypesError,
  InvalidCastError,

  // Collections
  IntegerDTypes,
  FloatDTypes,
  SignedDTypes,
  UnsignedDTypes,
  NumberDTypes,
  BigIntDTypes,
  DefaultDType,
  DefaultBoolDType,
  DefaultIntDType,

  // Constructor types
  TypedArrayConstructor,
} from './types.js';

// TypedArray interface export
export type { DTypedArray } from './typedarray.js';

// =============================================================================
// Runtime Class and Registry Exports
// =============================================================================

export {
  // Core runtime class
  RuntimeDType,

  // Registry and factory functions
  DTYPES,
  getDType,
  getDTypeNames,
  isValidDTypeName,
  getDefaultDType,

  // Type guards and validation
  isRuntimeDType,
  validateTypedArrayDType,
  getTypedArrayDType,
  validateArrayData,

  // Memory utilities
  calculateByteSize,
  validateBufferSize,
  isAligned,
  getAlignedOffset,

  // Error classes
  DTypeError as DTypeErrorClass,
  DTypeValidationError,
  DTypeBufferError,
} from './runtime.js';

// Import functions needed for convenience utilities
import { getDType, getDefaultDType as _getDefaultDType, getDTypeNames } from './runtime.js';
import {
  findCommonType,
  validatePromotionMatrix,
  promoteTypes,
  getPromotionRules,
} from './promotion.js';
import { convertValue } from './conversion.js';
import { createTypedArrayFromData, type DTypedArray } from './typedarray.js';
import type {
  AnyDType,
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
} from './types.js';

// =============================================================================
// Type Promotion Exports
// =============================================================================

export {
  // Runtime promotion functions
  promoteTypes,
  canPromoteTypes,
  promoteMultipleTypes,
  findCommonType,
  toPromotedDType,

  // Result type computation
  computeResultType,
  computeUnaryResultType,

  // Validation and debugging
  validatePromotionMatrix,
  analyzePromotion,
  getPromotionRules,
} from './promotion.js';

// =============================================================================
// Type Conversion Exports
// =============================================================================

export {
  // Conversion options and policies
  type ConversionOptions,
  type ConversionResult,
  type PrecisionLossInfo,
  STRICT_CONVERSION_OPTIONS,
  PERMISSIVE_CONVERSION_OPTIONS,

  // Core conversion functions
  convertValue,
  convertArray,

  // Type-safe conversion utilities
  safeCast,
  unsafeCast,
  wouldBeLossy,

  // Error classes
  ConversionError,
  PrecisionLossError,
  OverflowError,
} from './conversion.js';

// =============================================================================
// TypedArray Integration Exports
// =============================================================================

export {
  // Factory functions
  createTypedArray,
  createTypedArrayFromBuffer,
  createTypedArrayFromData,
  createReadonlyTypedArray,

  // Conversion and casting
  convertTypedArray,
  createTypedArrayView,

  // Utility functions
  wrapTypedArray,
  sharesSameBuffer,
  hasMemoryOverlap,
  calculateMemoryUsage,
  validateTypedArray,

  // Performance utilities
  copyTypedArrayData,
  createAlignedBuffer,

  // Error classes
  TypedArrayError,
  AlignmentError,
  BoundsError,
} from './typedarray.js';

// =============================================================================
// Runtime DType Constants
// =============================================================================

export {
  bool,
  int8,
  uint8,
  int16,
  uint16,
  int32,
  uint32,
  int64,
  uint64,
  float32,
  float64,
  DTYPE_CONSTANTS_MAP,
  getDTypeConstant,
  toFloatDType,
} from './constants.js';

// =============================================================================
// Convenience Re-exports and Utilities
// =============================================================================

/**
 * Common DType instances for convenience
 *
 * @example
 * import { CommonDTypes } from './dtype';
 * const array = createTypedArray(CommonDTypes.float32, 1000);
 */
export const CommonDTypes = Object.freeze({
  bool: getDType('bool'),
  int8: getDType('int8'),
  uint8: getDType('uint8'),
  int16: getDType('int16'),
  uint16: getDType('uint16'),
  int32: getDType('int32'),
  uint32: getDType('uint32'),
  int64: getDType('int64'),
  uint64: getDType('uint64'),
  float32: getDType('float32'),
  float64: getDType('float64'),
} as const);

/**
 * Type-level DType constants for compile-time operations
 * These are phantom types that exist only at compile time
 */
export const DTypeConstants = Object.freeze({
  bool: null as unknown as Bool,
  int8: null as unknown as Int8,
  uint8: null as unknown as Uint8,
  int16: null as unknown as Int16,
  uint16: null as unknown as Uint16,
  int32: null as unknown as Int32,
  uint32: null as unknown as Uint32,
  int64: null as unknown as Int64,
  uint64: null as unknown as Uint64,
  float32: null as unknown as Float32,
  float64: null as unknown as Float64,
} as const);

/**
 * Create a typed array with automatic DType inference from data
 *
 * @example
 * const array1 = createInferredArray([1, 2, 3]); // int32
 * const array2 = createInferredArray([1.5, 2.5]); // float32
 * const array3 = createInferredArray([true, false]); // bool
 */
export function createInferredArray(data: readonly unknown[]): DTypedArray<AnyDType> {
  const inferredDType = findCommonType(data);
  // We need to cast because findCommonType returns RuntimeDType but we need the values to match
  const validatedData = data.map((value) => {
    const result = convertValue(value as never, _getDefaultDType(value), inferredDType);
    if (!result.success) {
      throw new Error(`Failed to convert value ${String(value)}: ${result.errors.join(', ')}`);
    }
    return result.value;
  });

  return createTypedArrayFromData(inferredDType as never, validatedData as never);
}

/**
 * Get comprehensive information about the DType system
 * Useful for debugging and system introspection
 */
export function getDTypeSystemInfo(): {
  availableDTypes: readonly string[];
  defaultDTypes: Record<string, string>;
  promotionRules: string;
  memoryLayout: Record<string, { byteSize: number; signed: boolean; isInteger: boolean }>;
} {
  const availableDTypes = getDTypeNames();

  const defaultDTypes = {
    boolean: 'bool',
    integer: 'int32',
    float: 'float32',
    bigint: 'int64',
  };

  const memoryLayout: Record<string, { byteSize: number; signed: boolean; isInteger: boolean }> =
    {};
  for (const name of availableDTypes) {
    const dtype = getDType(name);
    memoryLayout[name] = {
      byteSize: dtype.byteSize,
      signed: dtype.signed,
      isInteger: dtype.isInteger,
    };
  }

  return {
    availableDTypes,
    defaultDTypes,
    promotionRules: getPromotionRules(),
    memoryLayout,
  };
}

/**
 * Validate that the entire DType system is working correctly
 * Useful for system tests and validation
 */
export function validateDTypeSystem(): { success: boolean; errors: string[] } {
  const errors: string[] = [];

  try {
    // Validate promotion matrix
    validatePromotionMatrix();
  } catch (error) {
    errors.push(
      `Promotion matrix validation failed: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  try {
    // Test basic DType creation
    for (const name of getDTypeNames()) {
      const dtype = getDType(name);
      if (dtype.name !== name) {
        errors.push(`Failed to create DType: ${name}`);
      }
    }
  } catch (error) {
    errors.push(`DType creation failed: ${error instanceof Error ? error.message : String(error)}`);
  }

  try {
    // Test type promotion symmetry
    const dtypeNames = getDTypeNames();
    for (const nameA of dtypeNames) {
      for (const nameB of dtypeNames) {
        const dtypeA = getDType(nameA);
        const dtypeB = getDType(nameB);
        const promotedAB = promoteTypes(dtypeA, dtypeB);
        const promotedBA = promoteTypes(dtypeB, dtypeA);

        if (promotedAB.name !== promotedBA.name) {
          errors.push(
            `Promotion asymmetry: ${nameA} + ${nameB} = ${promotedAB.name}, but ${nameB} + ${nameA} = ${promotedBA.name}`,
          );
        }
      }
    }
  } catch (error) {
    errors.push(
      `Promotion symmetry test failed: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  return {
    success: errors.length === 0,
    errors,
  };
}
