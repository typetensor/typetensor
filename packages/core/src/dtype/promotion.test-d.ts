/**
 * Type tests for dtype/promotion.ts
 *
 * Tests for type-level promotion operations
 */

import { expectTypeOf } from 'expect-type';
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
  AnyDType,
  Promote,
  CanPromote,
} from './types';
import { promoteTypes, computeResultType, computeUnaryResultType } from './promotion';

import type { RuntimeDType } from './runtime';

// =============================================================================
// Type Promotion Tests
// =============================================================================

// Same types promote to themselves
expectTypeOf<Promote<Bool, Bool>>().toEqualTypeOf<Bool>();
expectTypeOf<Promote<Int32, Int32>>().toEqualTypeOf<Int32>();
expectTypeOf<Promote<Float64, Float64>>().toEqualTypeOf<Float64>();

// Bool promotes to other types
expectTypeOf<Promote<Bool, Int8>>().toEqualTypeOf<Int8>();
expectTypeOf<Promote<Bool, Uint16>>().toEqualTypeOf<Uint16>();
expectTypeOf<Promote<Bool, Float32>>().toEqualTypeOf<Float32>();
expectTypeOf<Promote<Int32, Bool>>().toEqualTypeOf<Int32>();

// Integer promotion hierarchy
// Same signedness - promote to larger
expectTypeOf<Promote<Int8, Int16>>().toEqualTypeOf<Int16>();
expectTypeOf<Promote<Int16, Int32>>().toEqualTypeOf<Int32>();
expectTypeOf<Promote<Uint8, Uint16>>().toEqualTypeOf<Uint16>();

// Mixed signedness - promote to larger signed
expectTypeOf<Promote<Int8, Uint8>>().toEqualTypeOf<Int16>();
expectTypeOf<Promote<Int16, Uint16>>().toEqualTypeOf<Int32>();
expectTypeOf<Promote<Int32, Uint32>>().toEqualTypeOf<Int64>();

// Integer to float promotion
// Small integers to float32
expectTypeOf<Promote<Int8, Float32>>().toEqualTypeOf<Float32>();
expectTypeOf<Promote<Uint8, Float32>>().toEqualTypeOf<Float32>();
expectTypeOf<Promote<Int16, Float32>>().toEqualTypeOf<Float32>();
expectTypeOf<Promote<Uint16, Float32>>().toEqualTypeOf<Float32>();

// Large integers need float64 for precision
expectTypeOf<Promote<Int32, Float32>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Uint32, Float32>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Int64, Float32>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Uint64, Float32>>().toEqualTypeOf<Float64>();

// Float to larger float
expectTypeOf<Promote<Float32, Float64>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Float64, Float32>>().toEqualTypeOf<Float64>();

// Large integer promotions
// uint64 with signed integers often goes to float64
expectTypeOf<Promote<Int8, Uint64>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Int16, Uint64>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Int32, Uint64>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Int64, Uint64>>().toEqualTypeOf<Float64>();

// Promotion compatibility checks
expectTypeOf<CanPromote<Int32, Float32>>().toEqualTypeOf<true>();
expectTypeOf<CanPromote<Bool, Int64>>().toEqualTypeOf<true>();
expectTypeOf<CanPromote<Float32, Float64>>().toEqualTypeOf<true>();

// =============================================================================
// Runtime Function Return Type Tests
// =============================================================================

// promoteTypes return type
declare const int32: RuntimeDType<Int32>;
declare const float32: RuntimeDType<Float32>;
declare const promoted: ReturnType<typeof promoteTypes<Int32, Float32>>;
expectTypeOf(promoted).toMatchTypeOf<RuntimeDType<Float64>>();

// computeResultType return type
declare const int8: RuntimeDType<Int8>;
declare const uint8: RuntimeDType<Uint8>;
declare const result: ReturnType<typeof computeResultType<Int8, Uint8>>;
expectTypeOf(result).toMatchTypeOf<RuntimeDType<Int16>>();

// computeUnaryResultType return type
declare const unaryResult: ReturnType<typeof computeUnaryResultType<Float32>>;
expectTypeOf(unaryResult).toMatchTypeOf<RuntimeDType<AnyDType>>();

// Generic promotion function
declare function testPromotion<A extends AnyDType, B extends AnyDType>(
  a: RuntimeDType<A>,
  b: RuntimeDType<B>,
): RuntimeDType<Promote<A, B>>;

declare const testResult: ReturnType<typeof testPromotion<Int32, Float32>>;
expectTypeOf(testResult).toMatchTypeOf<RuntimeDType<Float64>>();

// =============================================================================
// Complex Type Integration Tests
// =============================================================================

// Chained promotions
type Chain1 = Promote<Bool, Int8>; // Int8
type Chain2 = Promote<Chain1, Int16>; // Int16
type Chain3 = Promote<Chain2, Float32>; // Float32
type Chain4 = Promote<Chain3, Float64>; // Float64

expectTypeOf<Chain1>().toEqualTypeOf<Int8>();
expectTypeOf<Chain2>().toEqualTypeOf<Int16>();
expectTypeOf<Chain3>().toEqualTypeOf<Float32>();
expectTypeOf<Chain4>().toEqualTypeOf<Float64>();

// Complex mixed-type scenarios
// Large unsigned + signed integers
expectTypeOf<Promote<Uint64, Int32>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Uint32, Int64>>().toEqualTypeOf<Int64>();

// Mixed precision requirements
expectTypeOf<Promote<Int32, Float32>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Uint32, Float32>>().toEqualTypeOf<Float64>();

// Promotion symmetry at type level
// Test that Promote<A, B> === Promote<B, A>
expectTypeOf<Promote<Int8, Uint8>>().toEqualTypeOf<Promote<Uint8, Int8>>();
expectTypeOf<Promote<Int32, Float32>>().toEqualTypeOf<Promote<Float32, Int32>>();
expectTypeOf<Promote<Float32, Float64>>().toEqualTypeOf<Promote<Float64, Float32>>();

// =============================================================================
// Edge Cases and Boundary Conditions
// =============================================================================

// Bool is always promoted away
expectTypeOf<Promote<Bool, Int8>>().toEqualTypeOf<Int8>();
expectTypeOf<Promote<Bool, Uint64>>().toEqualTypeOf<Uint64>();
expectTypeOf<Promote<Bool, Float32>>().toEqualTypeOf<Float32>();

// Maximum precision requirements
// When both int64 and uint64 are involved with other types
expectTypeOf<Promote<Int64, Uint64>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Int64, Float32>>().toEqualTypeOf<Float64>();
expectTypeOf<Promote<Uint64, Float32>>().toEqualTypeOf<Float64>();

// =============================================================================
// Multi-type Promotion Chains
// =============================================================================

// Testing promotion with 3 or more types
type MultiPromote<A extends AnyDType, B extends AnyDType, C extends AnyDType> = Promote<
  Promote<A, B>,
  C
>;

expectTypeOf<MultiPromote<Bool, Int8, Float32>>().toEqualTypeOf<Float32>();
expectTypeOf<MultiPromote<Int32, Uint32, Float32>>().toEqualTypeOf<Float64>();
expectTypeOf<MultiPromote<Int64, Uint64, Float64>>().toEqualTypeOf<Float64>();
