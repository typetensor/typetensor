/**
 * Type tests for dtype/constants.ts
 *
 * Tests for compile-time type checking of dtype constants
 */

import { expectTypeOf } from 'expect-type';
import {
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
} from './constants';
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

// =============================================================================
// Individual Constant Type Tests
// =============================================================================

// Test that constants match their expected types
expectTypeOf(bool).toMatchTypeOf<Bool>();
expectTypeOf(int8).toMatchTypeOf<Int8>();
expectTypeOf(uint8).toMatchTypeOf<Uint8>();
expectTypeOf(int16).toMatchTypeOf<Int16>();
expectTypeOf(uint16).toMatchTypeOf<Uint16>();
expectTypeOf(int32).toMatchTypeOf<Int32>();
expectTypeOf(uint32).toMatchTypeOf<Uint32>();
expectTypeOf(int64).toMatchTypeOf<Int64>();
expectTypeOf(uint64).toMatchTypeOf<Uint64>();
expectTypeOf(float32).toMatchTypeOf<Float32>();
expectTypeOf(float64).toMatchTypeOf<Float64>();

// =============================================================================
// Property Readonly Tests
// =============================================================================

// Test that all properties are readonly
expectTypeOf(bool).toMatchTypeOf<{
  readonly __dtype: 'bool';
  readonly __jsType: boolean;
  readonly __typedArray: Uint8ArrayConstructor;
  readonly __byteSize: 1;
  readonly __signed: false;
  readonly __isInteger: true;
}>();

expectTypeOf(float32).toMatchTypeOf<{
  readonly __dtype: 'float32';
  readonly __jsType: number;
  readonly __typedArray: Float32ArrayConstructor;
  readonly __byteSize: 4;
  readonly __signed: true;
  readonly __isInteger: false;
}>();

expectTypeOf(int64).toMatchTypeOf<{
  readonly __dtype: 'int64';
  readonly __jsType: bigint;
  readonly __typedArray: BigInt64ArrayConstructor;
  readonly __byteSize: 8;
  readonly __signed: true;
  readonly __isInteger: true;
}>();

// =============================================================================
// Constants Map Type Tests
// =============================================================================

// Test DTYPE_CONSTANTS_MAP structure
expectTypeOf(DTYPE_CONSTANTS_MAP).toMatchTypeOf<{
  readonly bool: Bool;
  readonly int8: Int8;
  readonly uint8: Uint8;
  readonly int16: Int16;
  readonly uint16: Uint16;
  readonly int32: Int32;
  readonly uint32: Uint32;
  readonly int64: Int64;
  readonly uint64: Uint64;
  readonly float32: Float32;
  readonly float64: Float64;
}>();

// Test getDTypeConstant return types
expectTypeOf(getDTypeConstant('bool')).toEqualTypeOf<Bool>();
expectTypeOf(getDTypeConstant('int32')).toEqualTypeOf<Int32>();
expectTypeOf(getDTypeConstant('float64')).toEqualTypeOf<Float64>();

// =============================================================================
// Literal Type Tests
// =============================================================================

// Test specific literal types for bool
expectTypeOf<typeof bool.__dtype>().toEqualTypeOf<'bool'>();
expectTypeOf<typeof bool.__byteSize>().toEqualTypeOf<1>();
expectTypeOf<typeof bool.__signed>().toEqualTypeOf<false>();
expectTypeOf<typeof bool.__isInteger>().toEqualTypeOf<true>();

// Test specific literal types for float32
expectTypeOf<typeof float32.__dtype>().toEqualTypeOf<'float32'>();
expectTypeOf<typeof float32.__byteSize>().toEqualTypeOf<4>();
expectTypeOf<typeof float32.__signed>().toEqualTypeOf<true>();
expectTypeOf<typeof float32.__isInteger>().toEqualTypeOf<false>();

// =============================================================================
// Conditional Type Context Tests
// =============================================================================

// Test that constants can be used in conditional type contexts
type TestConditional<T extends Bool | Int32 | Float32> = T extends Bool
  ? 'boolean'
  : T extends Int32
    ? 'integer'
    : T extends Float32
      ? 'float'
      : never;

expectTypeOf<TestConditional<typeof bool>>().toEqualTypeOf<'boolean'>();
expectTypeOf<TestConditional<typeof int32>>().toEqualTypeOf<'integer'>();
expectTypeOf<TestConditional<typeof float32>>().toEqualTypeOf<'float'>();

// =============================================================================
// Const Assertion Behavior Tests
// =============================================================================

// Verify that 'as const' maintains literal types
declare const testBool: typeof bool;
expectTypeOf(testBool.__dtype).toEqualTypeOf<'bool'>();
expectTypeOf(testBool.__byteSize).toEqualTypeOf<1>();

// Should not be just 'string' or 'number'
expectTypeOf<typeof testBool.__dtype>().not.toEqualTypeOf<string>();
expectTypeOf<typeof testBool.__byteSize>().not.toEqualTypeOf<number>();
