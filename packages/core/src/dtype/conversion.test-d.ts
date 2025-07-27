/**
 * Type tests for dtype/conversion.ts
 *
 * Tests for type-level conversion operations and safety
 */

import { expectTypeOf } from 'expect-type';
import type {
  Bool,
  Int8,
  Uint8,
  Int32,
  Int64,
  Float32,
  Float64,
  AnyDType,
  JSTypeOf,
  CanSafelyCast,
  IsLossyCast,
} from './types';
import {
  type ConversionResult,
  type ConversionOptions,
  type PrecisionLossInfo,
  convertValue,
  convertArray,
  safeCast,
  unsafeCast,
  wouldBeLossy,
  STRICT_CONVERSION_OPTIONS,
  PERMISSIVE_CONVERSION_OPTIONS,
} from './conversion';
import type { RuntimeDType } from './runtime';

// =============================================================================
// ConversionResult Type Tests
// =============================================================================

// Success result structure
type SuccessResult = ConversionResult<Int32> & { success: true };
expectTypeOf<SuccessResult>().toMatchTypeOf<{
  readonly success: true;
  readonly value: number;
  readonly warnings: readonly string[];
}>();

// Failure result structure
type FailureResult = ConversionResult<Int32> & { success: false };
expectTypeOf<FailureResult>().toMatchTypeOf<{
  readonly success: false;
  readonly errors: readonly string[];
}>();

// Value types based on target dtype
type BoolResult = ConversionResult<Bool> & { success: true };
type Int32Result = ConversionResult<Int32> & { success: true };
type Float64Result = ConversionResult<Float64> & { success: true };
type Int64Result = ConversionResult<Int64> & { success: true };

expectTypeOf<BoolResult['value']>().toEqualTypeOf<boolean>();
expectTypeOf<Int32Result['value']>().toEqualTypeOf<number>();
expectTypeOf<Float64Result['value']>().toEqualTypeOf<number>();
expectTypeOf<Int64Result['value']>().toEqualTypeOf<bigint>();

// =============================================================================
// ConversionOptions Type Tests
// =============================================================================

// ConversionOptions structure
expectTypeOf<ConversionOptions>().toMatchTypeOf<{
  readonly allowPrecisionLoss?: boolean;
  readonly allowOverflow?: boolean;
  readonly nanHandling?: 'error' | 'zero' | 'clamp';
  readonly infinityHandling?: 'error' | 'clamp';
  readonly overflowHandling?: 'error' | 'clamp' | 'wrap';
}>();

// Predefined options
expectTypeOf<typeof STRICT_CONVERSION_OPTIONS>().toEqualTypeOf<Required<ConversionOptions>>();
expectTypeOf<typeof PERMISSIVE_CONVERSION_OPTIONS>().toEqualTypeOf<Required<ConversionOptions>>();

// =============================================================================
// PrecisionLossInfo Type Tests
// =============================================================================

// PrecisionLossInfo structure
expectTypeOf<PrecisionLossInfo>().toMatchTypeOf<{
  readonly originalValue: number | bigint | boolean;
  readonly convertedValue: number | bigint | boolean;
  readonly lossType: 'truncation' | 'rounding' | 'overflow' | 'underflow' | 'range';
  readonly message: string;
}>();

// =============================================================================
// Type-Safe Conversion Function Tests
// =============================================================================

declare const safeCastResult: ReturnType<typeof safeCast<Int8, Int32>>;
expectTypeOf(safeCastResult).toEqualTypeOf<number>();

// Compile-time checking for safe casts
type SafeCastReturn<From extends AnyDType, To extends AnyDType> =
  CanSafelyCast<From, To> extends true ? JSTypeOf<To> : never;

expectTypeOf<SafeCastReturn<Int8, Int32>>().toEqualTypeOf<number>();
expectTypeOf<SafeCastReturn<Float64, Int32>>().toEqualTypeOf<never>();

// =============================================================================
// Conversion Function Return Types
// =============================================================================

// convertValue return type
declare const convertResult: ReturnType<typeof convertValue<Int32, Float32>>;
expectTypeOf(convertResult).toMatchTypeOf<ConversionResult<Float32>>();

// Array conversion result type
declare const arrayResult: ReturnType<typeof convertArray<Int32, Float32>>;
expectTypeOf(arrayResult).toMatchTypeOf<
  ConversionResult<Float32> & { values?: readonly number[] }
>();

// wouldBeLossy return type
declare const lossyResult: ReturnType<typeof wouldBeLossy<Float64, Int32>>;
expectTypeOf(lossyResult).toEqualTypeOf<boolean>();

// unsafeCast return type
declare const unsafeResult: ReturnType<typeof unsafeCast<Float32, Int32>>;
expectTypeOf(unsafeResult).toMatchTypeOf<ConversionResult<Int32>>();

// =============================================================================
// Generic Type Constraints
// =============================================================================

// Generic conversion function
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare function testConversion<From extends AnyDType, To extends AnyDType>(
  value: JSTypeOf<From>,
  from: RuntimeDType<From>,
  to: RuntimeDType<To>,
): ConversionResult<To>;

declare const genericResult: ReturnType<typeof testConversion<Int32, Float32>>;
expectTypeOf(genericResult).toMatchTypeOf<ConversionResult<Float32>>();

// =============================================================================
// Compile-Time Cast Safety
// =============================================================================

// Safe casts
expectTypeOf<CanSafelyCast<Bool, Int32>>().toEqualTypeOf<true>();
expectTypeOf<CanSafelyCast<Int8, Int32>>().toEqualTypeOf<true>();
expectTypeOf<CanSafelyCast<Float32, Float64>>().toEqualTypeOf<true>();

// Unsafe casts
expectTypeOf<CanSafelyCast<Float64, Int32>>().toEqualTypeOf<false>();
expectTypeOf<CanSafelyCast<Int32, Int8>>().toEqualTypeOf<false>();
expectTypeOf<CanSafelyCast<Int8, Uint8>>().toEqualTypeOf<false>();

// Non-lossy casts
expectTypeOf<IsLossyCast<Int8, Int32>>().toEqualTypeOf<false>();
expectTypeOf<IsLossyCast<Float32, Float64>>().toEqualTypeOf<false>();

// Lossy casts
expectTypeOf<IsLossyCast<Float64, Float32>>().toEqualTypeOf<true>();
expectTypeOf<IsLossyCast<Int32, Int8>>().toEqualTypeOf<true>();
expectTypeOf<IsLossyCast<Float32, Int32>>().toEqualTypeOf<true>();

// =============================================================================
// Conditional Conversion Result Types
// =============================================================================

// Conditional type based on conversion success
type ConversionValue<T extends ConversionResult<AnyDType>> = T extends {
  success: true;
  value: infer V;
}
  ? V
  : never;

type SuccessValue = ConversionValue<ConversionResult<Float32> & { success: true }>;
expectTypeOf<SuccessValue>().toEqualTypeOf<number>();

// Array conversion value extraction
type ArrayConversionValue<T extends ConversionResult<AnyDType> & { values?: readonly unknown[] }> =
  T extends { success: true; values: readonly (infer V)[] } ? readonly V[] : never;

// Test that array conversion types are properly inferred  
expectTypeOf<ArrayConversionValue<any>>().not.toBeNever();
