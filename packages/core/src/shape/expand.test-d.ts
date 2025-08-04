/**
 * Type tests for expand operation
 *
 * These tests validate that our type-level expand operations work correctly
 * at compile time.
 */

import { expectTypeOf } from 'expect-type';
import type { ExpandShape, CanExpand } from './types';

// =============================================================================
// ExpandShape Tests
// =============================================================================

// Basic singleton expansion
{
  expectTypeOf<ExpandShape<[2, 1, 3], [2, 5, 3]>>().toEqualTypeOf<readonly [2, 5, 3]>();
  expectTypeOf<ExpandShape<[1, 1, 3], [4, 5, 3]>>().toEqualTypeOf<readonly [4, 5, 3]>();
  expectTypeOf<ExpandShape<[1], [10]>>().toEqualTypeOf<readonly [10]>();
}

// Using -1 to keep dimensions
{
  expectTypeOf<ExpandShape<[2, 1, 3], [-1, 5, -1]>>().toEqualTypeOf<readonly [2, 5, 3]>();
  expectTypeOf<ExpandShape<[1, 3, 1], [-1, -1, 7]>>().toEqualTypeOf<readonly [1, 3, 7]>();
  expectTypeOf<ExpandShape<[2, 3, 4], [-1, -1, -1]>>().toEqualTypeOf<readonly [2, 3, 4]>();
}

// Adding new dimensions on the left
{
  expectTypeOf<ExpandShape<[3, 4], [2, 3, 4]>>().toEqualTypeOf<readonly [2, 3, 4]>();
  expectTypeOf<ExpandShape<[5], [2, 3, 4, 5]>>().toEqualTypeOf<readonly [2, 3, 4, 5]>();
  expectTypeOf<ExpandShape<[2, 3], [5, 4, 2, 3]>>().toEqualTypeOf<readonly [5, 4, 2, 3]>();
}

// Scalar expansion
{
  expectTypeOf<ExpandShape<[], [3, 4, 5]>>().toEqualTypeOf<readonly [3, 4, 5]>();
  expectTypeOf<ExpandShape<[], [10]>>().toEqualTypeOf<readonly [10]>();
  expectTypeOf<ExpandShape<[], []>>().toEqualTypeOf<readonly []>();
}

// Complex expansions
{
  // Mix of new dims, expansions, and -1
  expectTypeOf<ExpandShape<[1, 3, 1], [2, 5, 3, 4]>>().toEqualTypeOf<readonly [2, 5, 3, 4]>();
  expectTypeOf<ExpandShape<[1, 1], [10, 20, 30]>>().toEqualTypeOf<readonly [10, 20, 30]>();
}

// Neural network patterns
{
  // Batch broadcasting
  expectTypeOf<ExpandShape<[1, 768], [32, 768]>>().toEqualTypeOf<readonly [32, 768]>();
  
  // Attention mask expansion
  expectTypeOf<ExpandShape<[1, 1, 128, 128], [32, 8, 128, 128]>>().toEqualTypeOf<readonly [32, 8, 128, 128]>();
  
  // Bias expansion
  expectTypeOf<ExpandShape<[1, 1, 768], [32, 128, 768]>>().toEqualTypeOf<readonly [32, 128, 768]>();
}

// =============================================================================
// CanExpand Validation Tests
// =============================================================================

// Valid expansions
{
  expectTypeOf<CanExpand<[2, 1, 3], [2, 5, 3]>>().toEqualTypeOf<true>();
  expectTypeOf<CanExpand<[1, 3], [4, 3]>>().toEqualTypeOf<true>();
  expectTypeOf<CanExpand<[1, 1, 1], [2, 3, 4]>>().toEqualTypeOf<true>();
  expectTypeOf<CanExpand<[], [2, 3, 4]>>().toEqualTypeOf<true>(); // scalar
  expectTypeOf<CanExpand<[3], [2, 3]>>().toEqualTypeOf<true>(); // add dimension
}

// Valid with -1
{
  expectTypeOf<CanExpand<[2, 1, 3], [-1, 5, -1]>>().toEqualTypeOf<true>();
  expectTypeOf<CanExpand<[2, 3], [-1, -1]>>().toEqualTypeOf<true>();
  expectTypeOf<CanExpand<[1, 1], [-1, 10]>>().toEqualTypeOf<true>();
}

// Invalid expansions
{
  expectTypeOf<CanExpand<[2, 3], [2, 5]>>().toEqualTypeOf<false>(); // can't expand 3 to 5
  expectTypeOf<CanExpand<[3, 4], [2, 4]>>().toEqualTypeOf<false>(); // can't expand 3 to 2
  expectTypeOf<CanExpand<[2, 3, 4], [2, 3, 5]>>().toEqualTypeOf<false>(); // can't expand 4 to 5
}

// Edge cases
{
  expectTypeOf<CanExpand<[0, 1], [0, 5]>>().toEqualTypeOf<true>(); // zero dimension
  expectTypeOf<CanExpand<[1, 0], [5, 0]>>().toEqualTypeOf<true>(); // zero preserved
  expectTypeOf<CanExpand<[], []>>().toEqualTypeOf<true>(); // scalar to scalar
}