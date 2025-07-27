/**
 * Type-level DType operations for compile-time numeric type safety
 *
 * This module provides TypeScript branded types for representing different
 * numeric types (float32, int32, bool, etc.) with compile-time type checking
 * and zero runtime overhead. It enables type-safe tensor operations with
 * NumPy-compatible type promotion rules.
 */

import type { Max } from 'ts-arithmetic';

// =============================================================================
// Core DType Branded Types
// =============================================================================

/**
 * Base branded type interface for numeric data types
 * Uses phantom type parameters to enable compile-time type checking
 * without any runtime overhead
 */
export interface DType<
  Name extends string,
  JSType extends number | boolean | bigint,
  TypedArrayType extends TypedArrayConstructor,
  ByteSize extends number = number,
  Signed extends boolean = boolean,
  IsInteger extends boolean = boolean,
> {
  readonly __dtype: Name;
  readonly __jsType: JSType;
  readonly __typedArray: TypedArrayType;
  readonly __byteSize: ByteSize;
  readonly __signed: Signed;
  readonly __isInteger: IsInteger;
}

/**
 * TypedArray constructor types for compile-time validation
 */
export type TypedArrayConstructor =
  | Int8ArrayConstructor
  | Uint8ArrayConstructor
  | Uint8ClampedArrayConstructor
  | Int16ArrayConstructor
  | Uint16ArrayConstructor
  | Int32ArrayConstructor
  | Uint32ArrayConstructor
  | Float32ArrayConstructor
  | Float64ArrayConstructor
  | BigInt64ArrayConstructor
  | BigUint64ArrayConstructor;

// =============================================================================
// Concrete DType Definitions
// =============================================================================

/**
 * Boolean type (stored as uint8)
 */
export type Bool = DType<'bool', boolean, Uint8ArrayConstructor, 1, false, true>;

/**
 * 8-bit signed integer
 */
export type Int8 = DType<'int8', number, Int8ArrayConstructor, 1, true, true>;

/**
 * 8-bit unsigned integer
 */
export type Uint8 = DType<'uint8', number, Uint8ArrayConstructor, 1, false, true>;

/**
 * 16-bit signed integer
 */
export type Int16 = DType<'int16', number, Int16ArrayConstructor, 2, true, true>;

/**
 * 16-bit unsigned integer
 */
export type Uint16 = DType<'uint16', number, Uint16ArrayConstructor, 2, false, true>;

/**
 * 32-bit signed integer
 */
export type Int32 = DType<'int32', number, Int32ArrayConstructor, 4, true, true>;

/**
 * 32-bit unsigned integer
 */
export type Uint32 = DType<'uint32', number, Uint32ArrayConstructor, 4, false, true>;

/**
 * 32-bit floating point
 */
export type Float32 = DType<'float32', number, Float32ArrayConstructor, 4, true, false>;

/**
 * 64-bit floating point
 */
export type Float64 = DType<'float64', number, Float64ArrayConstructor, 8, true, false>;

/**
 * 64-bit signed integer (BigInt)
 */
export type Int64 = DType<'int64', bigint, BigInt64ArrayConstructor, 8, true, true>;

/**
 * 64-bit unsigned integer (BigInt)
 */
export type Uint64 = DType<'uint64', bigint, BigUint64ArrayConstructor, 8, false, true>;

// =============================================================================
// DType Utility Types
// =============================================================================

/**
 * Union of all supported DType names
 */
export type DTypeName =
  | 'bool'
  | 'int8'
  | 'uint8'
  | 'int16'
  | 'uint16'
  | 'int32'
  | 'uint32'
  | 'float32'
  | 'float64'
  | 'int64'
  | 'uint64';

/**
 * Numeric dtype names (all except bool)
 */
export type NumericDTypeName =
  | 'int8'
  | 'uint8'
  | 'int16'
  | 'uint16'
  | 'int32'
  | 'uint32'
  | 'float32'
  | 'float64'
  | 'int64'
  | 'uint64';

/**
 * Integer dtype names
 */
export type IntegerDTypeName =
  | 'int8'
  | 'uint8'
  | 'int16'
  | 'uint16'
  | 'int32'
  | 'uint32'
  | 'int64'
  | 'uint64';

/**
 * Floating-point dtype names
 */
export type FloatDTypeName = 'float32' | 'float64';

/**
 * Boolean dtype name
 */
export type BooleanDTypeName = 'bool';

/**
 * Union of all DType branded types
 */
export type AnyDType =
  | Bool
  | Int8
  | Uint8
  | Int16
  | Uint16
  | Int32
  | Uint32
  | Float32
  | Float64
  | Int64
  | Uint64;

/**
 * Extract the name from a DType
 *
 * @example
 * type Name = DTypeNameOf<Float32> // 'float32'
 */
export type DTypeNameOf<T extends AnyDType> = T['__dtype'];

/**
 * Extract the value type from a DType
 *
 * @example
 * type Value = DTypeValue<Float32> // number
 * type Value2 = DTypeValue<Bool> // boolean
 */
export type DTypeValue<D extends AnyDType> = D['__jsType'];

/**
 * Extract the JavaScript type from a DType
 *
 * @example
 * type JSType = JSTypeOf<Float32> // number
 * type JSType2 = JSTypeOf<Bool> // boolean
 */
export type JSTypeOf<T extends AnyDType> = T['__jsType'];

/**
 * Extract the TypedArray constructor from a DType
 *
 * @example
 * type ArrayType = ArrayConstructorOf<Float32> // Float32ArrayConstructor
 */
export type ArrayConstructorOf<T extends AnyDType> = T['__typedArray'];

/**
 * Get DType from name
 *
 * @example
 * type Type = DTypeFromName<'float32'> // Float32
 */
export type DTypeFromName<Name extends DTypeName> = Name extends 'bool'
  ? Bool
  : Name extends 'int8'
    ? Int8
    : Name extends 'uint8'
      ? Uint8
      : Name extends 'int16'
        ? Int16
        : Name extends 'uint16'
          ? Uint16
          : Name extends 'int32'
            ? Int32
            : Name extends 'uint32'
              ? Uint32
              : Name extends 'float32'
                ? Float32
                : Name extends 'float64'
                  ? Float64
                  : Name extends 'int64'
                    ? Int64
                    : Name extends 'uint64'
                      ? Uint64
                      : never;

/**
 * Get DType union from numeric dtype names
 */
export type NumericDTypeFromName<Name extends NumericDTypeName> = DTypeFromName<Name>;

/**
 * Get DType union from integer dtype names
 */
export type IntegerDTypeFromName<Name extends IntegerDTypeName> = DTypeFromName<Name>;

/**
 * Get DType union from float dtype names
 */
export type FloatDTypeFromName<Name extends FloatDTypeName> = DTypeFromName<Name>;

/**
 * Get DType union from boolean dtype name
 */
export type BooleanDTypeFromName<Name extends BooleanDTypeName> = DTypeFromName<Name>;

/**
 * Check if a DType is an integer type
 *
 * @example
 * type IsInt = IsIntegerDType<Int32> // true
 * type IsFloat = IsIntegerDType<Float32> // false
 */
export type IsIntegerDType<T extends AnyDType> = T['__isInteger'];

/**
 * Check if a DType is a floating point type
 *
 * @example
 * type IsFloat = IsFloatDType<Float32> // true
 * type IsInt = IsFloatDType<Int32> // false
 */
export type IsFloatDType<T extends AnyDType> = T['__isInteger'] extends false ? true : false;

/**
 * Check if a DType is signed
 *
 * @example
 * type IsSigned = IsSignedDType<Int32> // true
 * type IsUnsigned = IsSignedDType<Uint32> // false
 */
export type IsSignedDType<T extends AnyDType> = T['__signed'];

/**
 * Get byte size of a DType
 *
 * @example
 * type Size = ByteSizeOf<Float32> // 4
 * type Size2 = ByteSizeOf<Float64> // 8
 */
export type ByteSizeOf<T extends AnyDType> = T['__byteSize'];

// =============================================================================
// Type Promotion System
// =============================================================================

/**
 * Get the promotion index of a DType
 */
type PromotionIndexOf<T extends AnyDType> = T extends Bool
  ? 0
  : T extends Int8
    ? 1
    : T extends Uint8
      ? 2
      : T extends Int16
        ? 3
        : T extends Uint16
          ? 4
          : T extends Int32
            ? 5
            : T extends Uint32
              ? 6
              : T extends Int64
                ? 7
                : T extends Uint64
                  ? 8
                  : T extends Float32
                    ? 9
                    : T extends Float64
                      ? 10
                      : never;

/**
 * Get DType from promotion index
 */
type DTypeFromIndex<Index extends number> = Index extends 0
  ? Bool
  : Index extends 1
    ? Int8
    : Index extends 2
      ? Uint8
      : Index extends 3
        ? Int16
        : Index extends 4
          ? Uint16
          : Index extends 5
            ? Int32
            : Index extends 6
              ? Uint32
              : Index extends 7
                ? Int64
                : Index extends 8
                  ? Uint64
                  : Index extends 9
                    ? Float32
                    : Index extends 10
                      ? Float64
                      : never;

/**
 * Promote two DTypes to their common type following NumPy rules
 * Handles special cases for mixed signedness promotion
 *
 * @example
 * type Result = Promote<Int32, Float32> // Float32
 * type Result2 = Promote<Bool, Int8> // Int8
 * type Result3 = Promote<Float32, Float64> // Float64
 * type Result4 = Promote<Int8, Uint8> // Int16 (mixed signedness)
 */
export type Promote<A extends AnyDType, B extends AnyDType> =
  // Same type returns itself
  A extends B
    ? A
    : // Bool promotes to the other type
      A extends Bool
      ? B
      : B extends Bool
        ? A
        : // Mixed signedness promotion rules
          A extends Int8
          ? B extends Uint8
            ? Int16
            : B extends Uint16
              ? Int32
              : B extends Uint32
                ? Int64
                : B extends Uint64
                  ? Float64
                  : DTypeFromIndex<Max<PromotionIndexOf<A>, PromotionIndexOf<B>>>
          : A extends Uint8
            ? B extends Int8
              ? Int16
              : DTypeFromIndex<Max<PromotionIndexOf<A>, PromotionIndexOf<B>>>
            : A extends Int16
              ? B extends Uint8
                ? Int16
                : B extends Uint16
                  ? Int32
                  : B extends Uint32
                    ? Int64
                    : B extends Uint64
                      ? Float64
                      : DTypeFromIndex<Max<PromotionIndexOf<A>, PromotionIndexOf<B>>>
              : A extends Uint16
                ? B extends Int8
                  ? Int32
                  : B extends Int16
                    ? Int32
                    : DTypeFromIndex<Max<PromotionIndexOf<A>, PromotionIndexOf<B>>>
                : A extends Int32
                  ? B extends Uint8
                    ? Int32
                    : B extends Uint16
                      ? Int32
                      : B extends Uint32
                        ? Int64
                        : B extends Uint64
                          ? Float64
                          : B extends Float32
                            ? Float64 // int32 precision requires float64
                            : DTypeFromIndex<Max<PromotionIndexOf<A>, PromotionIndexOf<B>>>
                  : A extends Uint32
                    ? B extends Int8
                      ? Int64
                      : B extends Int16
                        ? Int64
                        : B extends Int32
                          ? Int64
                          : B extends Float32
                            ? Float64 // uint32 precision requires float64
                            : DTypeFromIndex<Max<PromotionIndexOf<A>, PromotionIndexOf<B>>>
                    : A extends Int64
                      ? B extends Uint64
                        ? Float64 // Can't represent both ranges in any integer type
                        : B extends Float32
                          ? Float64 // int64 precision requires float64
                          : DTypeFromIndex<Max<PromotionIndexOf<A>, PromotionIndexOf<B>>>
                      : A extends Uint64
                        ? B extends Int8
                          ? Float64
                          : B extends Int16
                            ? Float64
                            : B extends Int32
                              ? Float64
                              : B extends Int64
                                ? Float64 // Can't represent both ranges in any integer type
                                : B extends Float32
                                  ? Float64 // uint64 precision requires float64
                                  : DTypeFromIndex<Max<PromotionIndexOf<A>, PromotionIndexOf<B>>>
                        : // Float32 special cases (must handle symmetrically)
                          A extends Float32
                          ? B extends Int32
                            ? Float64 // int32 precision requires float64
                            : B extends Uint32
                              ? Float64 // uint32 precision requires float64
                              : B extends Int64
                                ? Float64 // int64 precision requires float64
                                : B extends Uint64
                                  ? Float64 // uint64 precision requires float64
                                  : DTypeFromIndex<Max<PromotionIndexOf<A>, PromotionIndexOf<B>>>
                          : // Default to max index for remaining cases
                            DTypeFromIndex<Max<PromotionIndexOf<A>, PromotionIndexOf<B>>>;

/**
 * Check if two DTypes can be promoted together
 *
 * @example
 * type CanPromote = CanPromote<Int32, Float32> // true
 * type CanPromote2 = CanPromote<Bool, Int64> // true
 */
export type CanPromote<A extends AnyDType, B extends AnyDType> =
  Promote<A, B> extends never ? false : true;

// =============================================================================
// Type Compatibility and Validation
// =============================================================================

/**
 * Check if one DType can be safely cast to another without data loss
 *
 * @example
 * type CanCast = CanSafelyCast<Int8, Int32> // true (widening)
 * type CannotCast = CanSafelyCast<Float64, Int32> // false (potential precision loss)
 */
export type CanSafelyCast<From extends AnyDType, To extends AnyDType> =
  // Same type is always safe
  From extends To
    ? true
    : // Bool can cast to any type safely
      From extends Bool
      ? true
      : // Integer to integer: safe if target is wider and same signedness or signed
        From extends Int8
        ? To extends Int8 | Int16 | Int32 | Int64 | Float32 | Float64
          ? true
          : false
        : From extends Uint8
          ? To extends Uint8 | Int16 | Uint16 | Int32 | Uint32 | Int64 | Uint64 | Float32 | Float64
            ? true
            : false
          : From extends Int16
            ? To extends Int16 | Int32 | Int64 | Float32 | Float64
              ? true
              : false
            : From extends Uint16
              ? To extends Uint16 | Int32 | Uint32 | Int64 | Uint64 | Float32 | Float64
                ? true
                : false
              : From extends Int32
                ? To extends Int32 | Int64 | Float64
                  ? true
                  : false
                : From extends Uint32
                  ? To extends Uint32 | Int64 | Uint64 | Float64
                    ? true
                    : false
                  : From extends Int64
                    ? To extends Int64 | Float64
                      ? true
                      : false
                    : From extends Uint64
                      ? To extends Uint64 | Float64
                        ? true
                        : false
                      : From extends Float32
                        ? To extends Float32 | Float64
                          ? true
                          : false
                        : From extends Float64
                          ? To extends Float64
                            ? true
                            : false
                          : false;

/**
 * Check if a cast would result in potential data loss
 *
 * @example
 * type WillLosePrecision = IsLossyCast<Float64, Int32> // true
 * type WillNotLosePrecision = IsLossyCast<Int8, Int32> // false
 */
export type IsLossyCast<From extends AnyDType, To extends AnyDType> =
  CanSafelyCast<From, To> extends true ? false : true;

// =============================================================================
// DType Error Types
// =============================================================================

/**
 * Type-level error for invalid DType operations
 */
export interface DTypeError<Message extends string, Context = unknown> {
  readonly __error: 'DTypeError';
  readonly message: Message;
  readonly context: Context;
}

/**
 * Error for incompatible DType operations
 */
export type IncompatibleDTypesError<A extends AnyDType, B extends AnyDType> = DTypeError<
  `Incompatible DTypes: cannot operate on ${DTypeNameOf<A>} and ${DTypeNameOf<B>}`,
  { dtypeA: A; dtypeB: B }
>;

/**
 * Error for invalid DType casts
 */
export type InvalidCastError<From extends AnyDType, To extends AnyDType> = DTypeError<
  `Invalid cast: cannot safely cast ${DTypeNameOf<From>} to ${DTypeNameOf<To>} without data loss`,
  { from: From; to: To }
>;

// =============================================================================
// Common DType Collections
// =============================================================================

/**
 * All integer DTypes
 */
export type IntegerDTypes = Int8 | Uint8 | Int16 | Uint16 | Int32 | Uint32 | Int64 | Uint64;

/**
 * All floating point DTypes
 */
export type FloatDTypes = Float32 | Float64;

/**
 * All signed DTypes
 */
export type SignedDTypes = Int8 | Int16 | Int32 | Int64 | Float32 | Float64;

/**
 * All unsigned DTypes
 */
export type UnsignedDTypes = Bool | Uint8 | Uint16 | Uint32 | Uint64;

/**
 * DTypes that use JavaScript's number type
 */
export type NumberDTypes = Int8 | Uint8 | Int16 | Uint16 | Int32 | Uint32 | Float32 | Float64;

/**
 * DTypes that use JavaScript's bigint type
 */
export type BigIntDTypes = Int64 | Uint64;

/**
 * Default DType for numeric literals
 */
export type DefaultDType = Float32;

/**
 * Default DType for boolean literals
 */
export type DefaultBoolDType = Bool;

/**
 * Default DType for integer literals
 */
export type DefaultIntDType = Int32;

// =============================================================================
// DType Conversion Utilities
// =============================================================================

/**
 * Convert a DType to its floating point equivalent
 * - Float types remain unchanged (preserve precision)
 * - Integer and boolean types convert to Float32
 *
 * This is used for operations that mathematically require floating point output
 * like sin, cos, exp, log, sqrt, etc.
 *
 * @example
 * type F32 = ToFloat<Int32> // Float32
 * type F32_2 = ToFloat<Float32> // Float32 (unchanged)
 * type F64 = ToFloat<Float64> // Float64 (preserved)
 * type F32_3 = ToFloat<Bool> // Float32
 */
export type ToFloat<T extends AnyDType> = T extends Float32
  ? Float32
  : T extends Float64
    ? Float64
    : Float32; // Default for all integer and boolean types
