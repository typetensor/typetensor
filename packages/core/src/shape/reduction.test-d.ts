/**
 * Type-level tests for reduction shape utilities
 */

import { expectTypeOf } from 'expect-type';
import type { ValidateAxes, ReduceShape, ValidateReduction, DimensionError } from './types';

// =============================================================================
// ValidateAxes Tests
// =============================================================================

// Valid cases
{
  type CaseOne = ValidateAxes<[0, -1], [2, 3, 4]>;
  expectTypeOf<CaseOne>().toEqualTypeOf<readonly [0, 2]>(); // normalized
  type CaseTwo = ValidateAxes<[1], [2, 3, 4]>;
  expectTypeOf<CaseTwo>().toEqualTypeOf<readonly [1]>();
  type CaseThree = ValidateAxes<[], [2, 3, 4]>;
  expectTypeOf<CaseThree>().toEqualTypeOf<readonly []>(); // empty axes
}

// Invalid cases - should return DimensionError
{
  type CaseOne = ValidateAxes<[5], [2, 3, 4]>;
  expectTypeOf<CaseOne>().toEqualTypeOf<
    DimensionError<'Invalid dimension 5 for tensor with 3 dimensions (must be < 3)'>
  >(); // out of bounds
  type CaseTwo = ValidateAxes<[-5], [2, 3, 4]>;
  expectTypeOf<CaseTwo>().toEqualTypeOf<
    DimensionError<'Invalid dimension -5 for tensor with 3 dimensions (must be >= -3)'>
  >(); // out of bounds negative
}

// =============================================================================
// ReduceShape Tests
// =============================================================================

// keepdims=false (default)
expectTypeOf<ReduceShape<[2, 3, 4], [1], false>>().toEqualTypeOf<readonly [2, 4]>(); // remove middle dim
expectTypeOf<ReduceShape<[2, 3, 4], [0, 2], false>>().toEqualTypeOf<readonly [3]>(); // remove first and last
expectTypeOf<ReduceShape<[2, 3, 4], [0, 1, 2], false>>().toEqualTypeOf<readonly []>(); // remove all -> scalar

// keepdims=true
expectTypeOf<ReduceShape<[2, 3, 4], [1], true>>().toEqualTypeOf<readonly [2, 1, 4]>(); // keep middle as 1
expectTypeOf<ReduceShape<[2, 3, 4], [0, 2], true>>().toEqualTypeOf<readonly [1, 3, 1]>(); // keep first/last as 1
expectTypeOf<ReduceShape<[2, 3, 4], [0, 1, 2], true>>().toEqualTypeOf<readonly [1, 1, 1]>(); // keep all as 1

// Edge cases
expectTypeOf<ReduceShape<[2, 3, 4], [], false>>().toEqualTypeOf<readonly [2, 3, 4]>(); // no reduction
expectTypeOf<ReduceShape<[2, 3, 4], [], true>>().toEqualTypeOf<readonly [2, 3, 4]>(); // no reduction with keepdims

// =============================================================================
// ValidateReduction Tests
// =============================================================================

// Valid cases
expectTypeOf<ValidateReduction<[2, 3, 4], [1]>>().toEqualTypeOf<true>();
expectTypeOf<ValidateReduction<[2, 3, 4], undefined>>().toEqualTypeOf<true>(); // reduce all
expectTypeOf<ValidateReduction<[2, 3, 4], []>>().toEqualTypeOf<true>(); // no reduction

// Invalid cases
{
  type CaseOne = ValidateReduction<[2, 3, 4], [5]>;
  expectTypeOf<CaseOne>().toEqualTypeOf<
    DimensionError<'Invalid dimension 5 for tensor with 3 dimensions (must be < 3)'>
  >(); // out of bounds
  type CaseTwo = ValidateReduction<[2, 3, 4], [1, 1]>;
  expectTypeOf<CaseTwo>().toEqualTypeOf<
    DimensionError<'Duplicate axes found in reduction. Each axis can only appear once.'>
  >(); // duplicate axes
}
