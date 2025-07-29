/**
 * Type tests for the runtime shape system
 *
 * These tests validate that our shape runtime types work correctly
 * at compile time with TypeScript.
 */

import { describe, it } from 'bun:test';
import { expectTypeOf } from 'expect-type';
import type { Shape, DynamicShape, SymbolicShape, SymbolicDim } from './types';

// =============================================================================
// Basic Shape Types
// =============================================================================

describe('Basic Shape Types', () => {
  it('should define Shape as readonly number array', () => {
    expectTypeOf<Shape>().toEqualTypeOf<readonly number[]>();
    expectTypeOf<readonly [2, 3, 4]>().toExtend<Shape>();
  });

  it('should define DynamicShape with -1 for dynamic dimensions', () => {
    // eslint-disable-next-line @typescript-eslint/no-redundant-type-constituents
    expectTypeOf<DynamicShape>().toEqualTypeOf<readonly (number | -1)[]>();
    expectTypeOf<[2, -1, 4]>().toExtend<DynamicShape>();
    expectTypeOf<readonly [2, -1, 4]>().toExtend<DynamicShape>();
  });

  it('should define SymbolicShape with symbolic dimensions', () => {
    type TestSymbolic = SymbolicDim<'batch'>;
    expectTypeOf<readonly [TestSymbolic, 224, 224, 3]>().toExtend<SymbolicShape>();
  });
});
