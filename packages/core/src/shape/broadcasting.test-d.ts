/**
 * Type tests for the broadcasting system
 *
 * These tests validate that our type-level broadcasting operations work correctly
 * at compile time.
 */

import { describe, it } from 'bun:test';
import { expectTypeOf } from 'expect-type';
import type { CanBroadcast, BroadcastShapes } from './types';

// =============================================================================
// Broadcasting Rules
// =============================================================================

describe('Broadcasting Rules', () => {
  it('should validate broadcastable shapes', () => {
    // Compatible shapes
    expectTypeOf<CanBroadcast<[1, 3], [2, 1]>>().toEqualTypeOf<true>();
    expectTypeOf<CanBroadcast<[5, 1, 3], [1, 3]>>().toEqualTypeOf<true>();
    expectTypeOf<CanBroadcast<[2, 3], [3]>>().toEqualTypeOf<true>();
    expectTypeOf<CanBroadcast<[], [2, 3]>>().toEqualTypeOf<true>(); // Scalar

    // Incompatible shapes
    expectTypeOf<CanBroadcast<[2, 3], [4, 5]>>().toEqualTypeOf<false>();
    expectTypeOf<CanBroadcast<[2, 3], [2, 4]>>().toEqualTypeOf<false>();
  });

  it('should compute broadcast shapes', () => {
    expectTypeOf<BroadcastShapes<[1, 3], [2, 1]>>().toEqualTypeOf<readonly [2, 3]>();
    expectTypeOf<BroadcastShapes<[5, 1, 3], [1, 3]>>().toEqualTypeOf<readonly [5, 1, 3]>();
    expectTypeOf<BroadcastShapes<[2, 3], [3]>>().toEqualTypeOf<readonly [2, 3]>();
    expectTypeOf<BroadcastShapes<[], [2, 3]>>().toEqualTypeOf<readonly [2, 3]>();

    // These should be never (incompatible)
    expectTypeOf<BroadcastShapes<[2, 3], [4, 5]>>().toEqualTypeOf<never>();
    expectTypeOf<BroadcastShapes<[2, 3], [2, 4]>>().toEqualTypeOf<never>();
  });

  it('should handle neural network broadcasting patterns', () => {
    // Batch normalization: [batch, features] + [features] -> [batch, features]
    type BatchInput = readonly [32, 256];
    type BNParams = readonly [256];

    expectTypeOf<CanBroadcast<BatchInput, BNParams>>().toEqualTypeOf<true>();
    expectTypeOf<BroadcastShapes<BatchInput, BNParams>>().toEqualTypeOf<BatchInput>();

    // Layer normalization: [batch, seq, features] + [features] -> [batch, seq, features]
    type LNInput = readonly [32, 128, 768];
    type LNParams = readonly [768];

    expectTypeOf<CanBroadcast<LNInput, LNParams>>().toEqualTypeOf<true>();
    expectTypeOf<BroadcastShapes<LNInput, LNParams>>().toEqualTypeOf<LNInput>();
  });

  it('should handle edge cases with ones', () => {
    expectTypeOf<CanBroadcast<[1, 1, 5], [3, 4, 1]>>().toEqualTypeOf<true>();
    expectTypeOf<BroadcastShapes<[1, 1, 5], [3, 4, 1]>>().toEqualTypeOf<readonly [3, 4, 5]>();
  });

  it('should handle scalar broadcasting', () => {
    // Broadcasting with scalars
    expectTypeOf<CanBroadcast<[], [2, 3]>>().toEqualTypeOf<true>();
    expectTypeOf<BroadcastShapes<[], [2, 3]>>().toEqualTypeOf<readonly [2, 3]>();
  });
});
