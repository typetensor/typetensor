/**
 * Type-level tests for reduction shape utilities
 */

import { expectType, expectError } from 'tsd';
import type { 
  ValidateAxes, 
  ReduceShape, 
  ValidateReduction,
  DimensionError 
} from './types';

// =============================================================================
// ValidateAxes Tests
// =============================================================================

// Valid cases
expectType<readonly [0, 2]>(true as any as ValidateAxes<[0, -1], [2, 3, 4]>); // normalized
expectType<readonly [1]>(true as any as ValidateAxes<[1], [2, 3, 4]>);
expectType<readonly []>(true as any as ValidateAxes<[], [2, 3, 4]>); // empty axes

// Invalid cases - should return DimensionError
expectType<DimensionError<string>>(true as any as ValidateAxes<[5], [2, 3, 4]>); // out of bounds
expectType<DimensionError<string>>(true as any as ValidateAxes<[-5], [2, 3, 4]>); // out of bounds negative

// =============================================================================
// ReduceShape Tests  
// =============================================================================

// keepdims=false (default)
expectType<readonly [2, 4]>(true as any as ReduceShape<[2, 3, 4], [1], false>); // remove middle dim
expectType<readonly [3]>(true as any as ReduceShape<[2, 3, 4], [0, 2], false>); // remove first and last
expectType<readonly []>(true as any as ReduceShape<[2, 3, 4], [0, 1, 2], false>); // remove all -> scalar

// keepdims=true
expectType<readonly [2, 1, 4]>(true as any as ReduceShape<[2, 3, 4], [1], true>); // keep middle as 1
expectType<readonly [1, 3, 1]>(true as any as ReduceShape<[2, 3, 4], [0, 2], true>); // keep first/last as 1
expectType<readonly [1, 1, 1]>(true as any as ReduceShape<[2, 3, 4], [0, 1, 2], true>); // keep all as 1

// Edge cases
expectType<readonly [2, 3, 4]>(true as any as ReduceShape<[2, 3, 4], [], false>); // no reduction
expectType<readonly [2, 3, 4]>(true as any as ReduceShape<[2, 3, 4], [], true>); // no reduction with keepdims

// =============================================================================
// ValidateReduction Tests
// =============================================================================

// Valid cases
expectType<true>(true as any as ValidateReduction<[2, 3, 4], [1]>);
expectType<true>(true as any as ValidateReduction<[2, 3, 4], undefined>); // reduce all
expectType<true>(true as any as ValidateReduction<[2, 3, 4], []>); // no reduction

// Invalid cases
expectType<DimensionError<string>>(true as any as ValidateReduction<[2, 3, 4], [5]>); // out of bounds
expectType<DimensionError<string>>(true as any as ValidateReduction<[2, 3, 4], [1, 1]>); // duplicate axes