/**
 * Type tests for dtype/types.ts
 *
 * Tests for core DType interfaces, branded types, and type-level utilities
 */

import { expectTypeOf } from 'expect-type';
import type {
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
  JSTypeOf,
  ArrayConstructorOf,
  ByteSizeOf,
  DTypeFromName,
  IsIntegerDType,
  IsFloatDType,
  IsSignedDType,

  // Type operations
  CanSafelyCast,
  IsLossyCast,

  // Error types
  DTypeError,
  IncompatibleDTypesError,
  InvalidCastError,

  // Collections
  IntegerDTypes,
  FloatDTypes,
  SignedDTypes,
  UnsignedDTypes,
  NumberDTypes,
  BigIntDTypes,
} from './types';

// =============================================================================
// Core Branded Type Tests
// =============================================================================

// Test that DType is a branded interface
expectTypeOf<DType<'test', number, Float32ArrayConstructor>>().toMatchTypeOf<{
  readonly __dtype: 'test';
  readonly __jsType: number;
  readonly __typedArray: Float32ArrayConstructor;
}>();

// Boolean type
expectTypeOf<Bool>().toMatchTypeOf<DType<'bool', boolean, Uint8ArrayConstructor, 1, false, true>>();

// 8-bit integer types
expectTypeOf<Int8>().toMatchTypeOf<DType<'int8', number, Int8ArrayConstructor, 1, true, true>>();
expectTypeOf<Uint8>().toMatchTypeOf<
  DType<'uint8', number, Uint8ArrayConstructor, 1, false, true>
>();

// 16-bit integer types
expectTypeOf<Int16>().toMatchTypeOf<DType<'int16', number, Int16ArrayConstructor, 2, true, true>>();
expectTypeOf<Uint16>().toMatchTypeOf<
  DType<'uint16', number, Uint16ArrayConstructor, 2, false, true>
>();

// 32-bit integer types
expectTypeOf<Int32>().toMatchTypeOf<DType<'int32', number, Int32ArrayConstructor, 4, true, true>>();
expectTypeOf<Uint32>().toMatchTypeOf<
  DType<'uint32', number, Uint32ArrayConstructor, 4, false, true>
>();

// 64-bit integer types
expectTypeOf<Int64>().toMatchTypeOf<
  DType<'int64', bigint, BigInt64ArrayConstructor, 8, true, true>
>();
expectTypeOf<Uint64>().toMatchTypeOf<
  DType<'uint64', bigint, BigUint64ArrayConstructor, 8, false, true>
>();

// Floating-point types
expectTypeOf<Float32>().toMatchTypeOf<
  DType<'float32', number, Float32ArrayConstructor, 4, true, false>
>();
expectTypeOf<Float64>().toMatchTypeOf<
  DType<'float64', number, Float64ArrayConstructor, 8, true, false>
>();

// DTypeName union
expectTypeOf<DTypeName>().toEqualTypeOf<
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
  | 'uint64'
>();

// AnyDType union
expectTypeOf<AnyDType>().toEqualTypeOf<
  Bool | Int8 | Uint8 | Int16 | Uint16 | Int32 | Uint32 | Float32 | Float64 | Int64 | Uint64
>();

// =============================================================================
// Utility Type Tests
// =============================================================================

// Extract DType names
expectTypeOf<DTypeNameOf<Bool>>().toEqualTypeOf<'bool'>();
expectTypeOf<DTypeNameOf<Int32>>().toEqualTypeOf<'int32'>();
expectTypeOf<DTypeNameOf<Float64>>().toEqualTypeOf<'float64'>();

// Extract JavaScript types
expectTypeOf<JSTypeOf<Bool>>().toEqualTypeOf<boolean>();
expectTypeOf<JSTypeOf<Int32>>().toEqualTypeOf<number>();
expectTypeOf<JSTypeOf<Float32>>().toEqualTypeOf<number>();
expectTypeOf<JSTypeOf<Int64>>().toEqualTypeOf<bigint>();

// Extract TypedArray constructors
expectTypeOf<ArrayConstructorOf<Bool>>().toEqualTypeOf<Uint8ArrayConstructor>();
expectTypeOf<ArrayConstructorOf<Int32>>().toEqualTypeOf<Int32ArrayConstructor>();
expectTypeOf<ArrayConstructorOf<Float32>>().toEqualTypeOf<Float32ArrayConstructor>();
expectTypeOf<ArrayConstructorOf<Int64>>().toEqualTypeOf<BigInt64ArrayConstructor>();

// Extract byte sizes
expectTypeOf<ByteSizeOf<Bool>>().toEqualTypeOf<1>();
expectTypeOf<ByteSizeOf<Int16>>().toEqualTypeOf<2>();
expectTypeOf<ByteSizeOf<Int32>>().toEqualTypeOf<4>();
expectTypeOf<ByteSizeOf<Float64>>().toEqualTypeOf<8>();

// Map names to DTypes
expectTypeOf<DTypeFromName<'bool'>>().toEqualTypeOf<Bool>();
expectTypeOf<DTypeFromName<'int32'>>().toEqualTypeOf<Int32>();
expectTypeOf<DTypeFromName<'float64'>>().toEqualTypeOf<Float64>();

// Identify integer types
expectTypeOf<IsIntegerDType<Bool>>().toEqualTypeOf<true>();
expectTypeOf<IsIntegerDType<Int32>>().toEqualTypeOf<true>();
expectTypeOf<IsIntegerDType<Float32>>().toEqualTypeOf<false>();

// Identify floating-point types
expectTypeOf<IsFloatDType<Bool>>().toEqualTypeOf<false>();
expectTypeOf<IsFloatDType<Int32>>().toEqualTypeOf<false>();
expectTypeOf<IsFloatDType<Float32>>().toEqualTypeOf<true>();
expectTypeOf<IsFloatDType<Float64>>().toEqualTypeOf<true>();

// Identify signed types
expectTypeOf<IsSignedDType<Bool>>().toEqualTypeOf<false>();
expectTypeOf<IsSignedDType<Uint8>>().toEqualTypeOf<false>();
expectTypeOf<IsSignedDType<Int8>>().toEqualTypeOf<true>();
expectTypeOf<IsSignedDType<Float32>>().toEqualTypeOf<true>();

// =============================================================================
// Type Casting Safety Tests
// =============================================================================

// Safe casts (widening)
// Same type
expectTypeOf<CanSafelyCast<Int32, Int32>>().toEqualTypeOf<true>();

// Bool to any type
expectTypeOf<CanSafelyCast<Bool, Int8>>().toEqualTypeOf<true>();
expectTypeOf<CanSafelyCast<Bool, Float64>>().toEqualTypeOf<true>();

// Integer widening (same signedness)
expectTypeOf<CanSafelyCast<Int8, Int16>>().toEqualTypeOf<true>();
expectTypeOf<CanSafelyCast<Int16, Int32>>().toEqualTypeOf<true>();
expectTypeOf<CanSafelyCast<Uint8, Uint16>>().toEqualTypeOf<true>();

// Integer to float (if precision preserved)
expectTypeOf<CanSafelyCast<Int8, Float32>>().toEqualTypeOf<true>();
expectTypeOf<CanSafelyCast<Int16, Float32>>().toEqualTypeOf<true>();
expectTypeOf<CanSafelyCast<Int32, Float64>>().toEqualTypeOf<true>();

// Float to larger float
expectTypeOf<CanSafelyCast<Float32, Float64>>().toEqualTypeOf<true>();

// Unsafe casts (narrowing)
// Integer narrowing
expectTypeOf<CanSafelyCast<Int16, Int8>>().toEqualTypeOf<false>();
expectTypeOf<CanSafelyCast<Int32, Int16>>().toEqualTypeOf<false>();
expectTypeOf<CanSafelyCast<Uint16, Uint8>>().toEqualTypeOf<false>();

// Signed to unsigned (potential negative values)
expectTypeOf<CanSafelyCast<Int8, Uint8>>().toEqualTypeOf<false>();
expectTypeOf<CanSafelyCast<Int32, Uint32>>().toEqualTypeOf<false>();

// Float to integer (potential precision loss)
expectTypeOf<CanSafelyCast<Float32, Int32>>().toEqualTypeOf<false>();
expectTypeOf<CanSafelyCast<Float64, Int64>>().toEqualTypeOf<false>();

// Float narrowing
expectTypeOf<CanSafelyCast<Float64, Float32>>().toEqualTypeOf<false>();

// Lossy cast identification
// Safe casts are not lossy
expectTypeOf<IsLossyCast<Int8, Int32>>().toEqualTypeOf<false>();
expectTypeOf<IsLossyCast<Float32, Float64>>().toEqualTypeOf<false>();

// Unsafe casts are lossy
expectTypeOf<IsLossyCast<Float64, Int32>>().toEqualTypeOf<true>();
expectTypeOf<IsLossyCast<Int32, Int8>>().toEqualTypeOf<true>();
expectTypeOf<IsLossyCast<Float64, Float32>>().toEqualTypeOf<true>();

// =============================================================================
// Error Type Tests
// =============================================================================

// DType error structure
type TestError = DTypeError<'Test message', { context: 'test' }>;
expectTypeOf<TestError>().toMatchTypeOf<{
  readonly __error: 'DTypeError';
  readonly message: string;
  readonly context: unknown;
}>();

// Incompatible DTypes error
type IncompatibleError = IncompatibleDTypesError<Int32, Float32>;
expectTypeOf<IncompatibleError>().toMatchTypeOf<
  DTypeError<string, { dtypeA: Int32; dtypeB: Float32 }>
>();

// Invalid cast error
type CastError = InvalidCastError<Float64, Int32>;
expectTypeOf<CastError>().toMatchTypeOf<DTypeError<string, { from: Float64; to: Int32 }>>();

// =============================================================================
// DType Collection Tests
// =============================================================================

// Integer DTypes collection
expectTypeOf<IntegerDTypes>().toEqualTypeOf<
  Int8 | Uint8 | Int16 | Uint16 | Int32 | Uint32 | Int64 | Uint64
>();

// Float DTypes collection
expectTypeOf<FloatDTypes>().toEqualTypeOf<Float32 | Float64>();

// Signed DTypes collection
expectTypeOf<SignedDTypes>().toEqualTypeOf<Int8 | Int16 | Int32 | Int64 | Float32 | Float64>();

// Unsigned DTypes collection
expectTypeOf<UnsignedDTypes>().toEqualTypeOf<Bool | Uint8 | Uint16 | Uint32 | Uint64>();

// JavaScript type collections
expectTypeOf<NumberDTypes>().toEqualTypeOf<
  Int8 | Uint8 | Int16 | Uint16 | Int32 | Uint32 | Float32 | Float64
>();

expectTypeOf<BigIntDTypes>().toEqualTypeOf<Int64 | Uint64>();
