/**
 * Type-level tests for einops reduce shape resolver
 */

import type { ResolveReduceShape, ValidReducePattern } from './type-shape-resolver-reduce';
import { expectTypeOf } from 'expect-type';

// =============================================================================
// Basic Reduction Tests
// =============================================================================

// Test 1: Simple reductions
{
  // Reduce single axis
  type Result1 = ResolveReduceShape<'h w c -> h c', readonly [2, 3, 4]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 4]>();

  // Reduce multiple axes
  type Result2 = ResolveReduceShape<'h w c -> c', readonly [2, 3, 4]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [4]>();

  // Reduce all to scalar
  type Result3 = ResolveReduceShape<'h w c ->', readonly [2, 3, 4]>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly []>();
}

// Test 2: Identity patterns (no reduction)
{
  type Result1 = ResolveReduceShape<'h w -> h w', readonly [2, 3]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 3]>();

  type Result2 = ResolveReduceShape<'a b c -> a b c', readonly [2, 3, 4]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 3, 4]>();
}

// Test 3: Different reduction operations (all should give same shape)
{
  type ResultSum = ResolveReduceShape<'h w -> w', readonly [2, 3]>;
  type ResultMean = ResolveReduceShape<'h w -> w', readonly [2, 3]>;
  type ResultMax = ResolveReduceShape<'h w -> w', readonly [2, 3]>;
  type ResultMin = ResolveReduceShape<'h w -> w', readonly [2, 3]>;
  type ResultProd = ResolveReduceShape<'h w -> w', readonly [2, 3]>;

  expectTypeOf<ResultSum>().toEqualTypeOf<readonly [3]>();
  expectTypeOf<ResultMean>().toEqualTypeOf<readonly [3]>();
  expectTypeOf<ResultMax>().toEqualTypeOf<readonly [3]>();
  expectTypeOf<ResultMin>().toEqualTypeOf<readonly [3]>();
  expectTypeOf<ResultProd>().toEqualTypeOf<readonly [3]>();
}

// =============================================================================
// Singleton Pattern Tests
// =============================================================================

// Test 4: Adding singletons to output
{
  // Add singleton in middle
  type Result1 = ResolveReduceShape<'h w -> h 1', readonly [2, 3]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 1]>();

  // Add multiple singletons
  type Result2 = ResolveReduceShape<'h w -> h 1 w 1', readonly [2, 3]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 1, 3, 1]>();

  // Reduce to singletons
  type Result3 = ResolveReduceShape<'h w -> 1 1', readonly [2, 3]>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [1, 1]>();
}

// Test 5: Removing singletons from input
{
  type Result1 = ResolveReduceShape<'h 1 w -> h w', readonly [2, 1, 3]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 3]>();

  type Result2 = ResolveReduceShape<'a 1 b 1 c -> a b c', readonly [2, 1, 3, 1, 4]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 3, 4]>();
}

// Test 6: Mixed singleton operations
{
  // Reduce some, add singleton
  type Result1 = ResolveReduceShape<'h w c -> h 1 c', readonly [2, 3, 4]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 1, 4]>();

  // Keep singleton, reduce others
  type Result2 = ResolveReduceShape<'h 1 w -> h 1', readonly [2, 1, 3]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 1]>();
}

// =============================================================================
// Keep Dimensions Tests
// =============================================================================

// Test 7: KeepDims = true
{
  // Single axis reduction with keepdims
  type Result1 = ResolveReduceShape<'h w c -> h c', readonly [2, 3, 4], true>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 1, 4]>();

  // Multiple axes reduction with keepdims
  type Result2 = ResolveReduceShape<'h w c -> c', readonly [2, 3, 4], true>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [1, 1, 4]>();

  // Global reduction with keepdims
  type Result3 = ResolveReduceShape<'h w c ->', readonly [2, 3, 4], true>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [1, 1, 1]>();
}

// =============================================================================
// Composite Pattern Tests
// =============================================================================

// Test 8: Composite patterns with provided axes
{
  // Simple composite reduction
  type Result1 = ResolveReduceShape<
    '(h h2) (w w2) -> h w',
    readonly [4, 6],
    false,
    { h: 2; h2: 2; w: 3; w2: 2 }
  >;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 3]>();

  // Partial composite reduction
  type Result2 = ResolveReduceShape<'(h h2) w -> h w', readonly [4, 6], false, { h: 2; h2: 2 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 6]>();

  // Inferred dimension
  type Result3 = ResolveReduceShape<'(h h2) w -> h w', readonly [4, 6], false, { h: 2 }>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [2, 6]>();
}

// Test 9: Nested composite patterns
{
  type Result = ResolveReduceShape<
    '(h (h2 h3)) (w w2) -> h h2 w',
    readonly [8, 6],
    false,
    { h: 2; h2: 2; h3: 2; w: 3; w2: 2 }
  >;
  expectTypeOf<Result>().toEqualTypeOf<readonly [2, 2, 3]>();
}

// =============================================================================
// Ellipsis Pattern Tests
// =============================================================================

// Test 10: Basic ellipsis patterns
{
  // Reduce ellipsis dimensions
  type Result1 = ResolveReduceShape<'batch ... c -> batch c', readonly [2, 3, 4, 5]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 5]>();

  // Preserve ellipsis in output
  type Result2 = ResolveReduceShape<'batch ... c -> batch ...', readonly [2, 3, 4, 5]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 3, 4]>();

  // Ellipsis at beginning
  type Result3 = ResolveReduceShape<'... h w -> ...', readonly [2, 3, 4, 5]>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [2, 3]>();
}

// Test 11: Ellipsis with keepdims
{
  type Result = ResolveReduceShape<'batch ... c -> batch c', readonly [2, 3, 4, 5], true>;
  expectTypeOf<Result>().toEqualTypeOf<readonly [2, 1, 1, 5]>();
}

// =============================================================================
// Edge Cases
// =============================================================================

// Test 12: 1D tensors
{
  type Result1 = ResolveReduceShape<'x ->', readonly [5]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly []>();

  type Result2 = ResolveReduceShape<'x -> 1', readonly [5]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [1]>();
}

// Test 13: Scalar patterns
{
  // Scalar to scalar (identity)
  type Result1 = ResolveReduceShape<' -> ', readonly []>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly []>();

  // Scalar to singleton
  type Result2 = ResolveReduceShape<' -> 1', readonly []>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [1]>();
}

// =============================================================================
// Real-World Patterns
// =============================================================================

// Test 14: Pooling operations
{
  // 2x2 max pooling
  type Result1 = ResolveReduceShape<
    'batch channel (h h2) (w w2) -> batch channel h w',
    readonly [32, 3, 224, 224],
    false,
    { h2: 2; w2: 2 }
  >;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [32, 3, 112, 112]>();

  // Global average pooling
  type Result2 = ResolveReduceShape<
    'batch channel h w -> batch channel',
    readonly [32, 3, 7, 7],
    false,
    { h: 7; w: 7 }
  >;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [32, 3]>();
}

// Test 15: Attention patterns
{
  // Reduce over sequence dimension
  type Result = ResolveReduceShape<
    'batch seq dim -> batch dim',
    readonly [8, 128, 512],
    false,
    { seq: 128 }
  >;
  expectTypeOf<Result>().toEqualTypeOf<readonly [8, 512]>();
}

// =============================================================================
// Error Cases (should resolve to never with ResolveReduceShape)
// =============================================================================

// Test 16: Invalid patterns with ResolveReduceShape (legacy behavior)
{
  // Unknown axis in output
  type Result1 = ResolveReduceShape<'h w -> h w c', readonly [2, 3], false, { c: 1 }>;
  expectTypeOf<Result1>().toBeNever();

  // Duplicate axes
  type Result2 = ResolveReduceShape<'h h w -> h', readonly [2, 2, 3], false, { h: 2 }>;
  expectTypeOf<Result2>().toBeNever();

  // Multiple ellipsis
  type Result3 = ResolveReduceShape<'... a ... -> a', readonly [2, 3, 4], false, { a: 2 }>;
  expectTypeOf<Result3>().toBeNever();
}

// Test 17: Shape mismatch in composite patterns
{
  // Wrong decomposition
  type Result = ResolveReduceShape<'(h h2) w -> h w', readonly [4, 6], false, { h: 3; h2: 2 }>;
  expectTypeOf<Result>().toBeNever();
}

// =============================================================================
// Enhanced Error Cases with ValidReducePattern (should show specific errors)
// =============================================================================

// Test 18: Parse errors
{
  // Missing arrow operator
  type Result1 = ValidReducePattern<'h w c', readonly [2, 3, 4]>;
  expectTypeOf<Result1>().toEqualTypeOf<"[Reduce ❌] Parse Error: Missing arrow operator '->'. Pattern must be 'input -> output'">();

  // Empty input for non-scalar tensor
  type Result2 = ValidReducePattern<' -> c', readonly [2, 3, 4]>;
  expectTypeOf<Result2>().toEqualTypeOf<"[Reduce ❌] Parse Error: Empty input pattern. Specify input axes before '->'">();
}

// Test 19: Axis errors
{
  // New axis creation (invalid for reduce)
  type Result1 = ValidReducePattern<'h w -> h w c', readonly [2, 3]>;
  expectTypeOf<Result1>().toEqualTypeOf<'[Reduce ❌] Axis Error: Cannot create new axis \'c\' in reduce output. Available input axes: [\'h\', \'w\']. Reduce can only preserve or remove axes'>();

  // Duplicate axes in input
  type Result2 = ValidReducePattern<'h h w -> h', readonly [2, 2, 3]>;
  expectTypeOf<Result2>().toEqualTypeOf<'[Reduce ❌] Axis Error: Duplicate axis \'h\' in input. Each axis can appear at most once per side'>();

  // Duplicate axes in output
  type Result3 = ValidReducePattern<'h w c -> h h', readonly [2, 3, 4]>;
  expectTypeOf<Result3>().toEqualTypeOf<'[Reduce ❌] Axis Error: Duplicate axis \'h\' in output. Each axis can appear at most once per side'>();

  // Multiple ellipsis in input
  type Result4 = ValidReducePattern<'... a ... -> a', readonly [2, 3, 4]>;
  expectTypeOf<Result4>().toEqualTypeOf<'[Reduce ❌] Axis Error: Multiple ellipsis \'...\' in input. Only one ellipsis allowed per side'>();

  // Multiple ellipsis in output
  type Result5 = ValidReducePattern<'a -> ... a ...', readonly [2, 3, 4]>;
  expectTypeOf<Result5>().toEqualTypeOf<'[Reduce ❌] Axis Error: Multiple ellipsis \'...\' in output. Only one ellipsis allowed per side'>();
}

// Test 20: Shape errors
{
  // Rank mismatch
  type Result1 = ValidReducePattern<'h w c -> c', readonly [2, 3]>;
  expectTypeOf<Result1>().toEqualTypeOf<'[Reduce ❌] Shape Error: Reduce pattern expects 3 dimensions but tensor has 2'>();

  // Composite resolution error
  type Result2 = ValidReducePattern<'(h h2) w -> h w', readonly [4, 6], false, { h: 3; h2: 2 }>;
  expectTypeOf<Result2>().toEqualTypeOf<'[Reduce ❌] Shape Error: Cannot resolve \'(h h2)\' from dimension 4. Specify axis values: reduce(tensor, pattern, operation, keepDims, {axis: number})'>();
}

// Test 21: Fractional dimension errors (if we add integer validation)
{
  // This would catch fractional dimensions if our integer validation detects them
  // type Result = ValidReducePattern<'h w -> h', readonly [2.5, 3]>;
  // expectTypeOf<Result>().toEqualTypeOf<'[Reduce ❌] Shape Error: Reduce pattern \'h w -> h\' produces fractional dimensions. Composite axes must divide evenly. Use integer axis values: reduce(tensor, pattern, operation, keepDims, {axis: integer})'>();
}

// =============================================================================
// Valid Cases with ValidReducePattern (should return shapes)
// =============================================================================

// Test 22: Valid patterns should return shapes, not error messages
{
  // Simple reduction
  type Result1 = ValidReducePattern<'h w c -> c', readonly [2, 3, 4]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [4]>();

  // Identity pattern
  type Result2 = ValidReducePattern<'h w -> h w', readonly [2, 3]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 3]>();

  // Global reduction
  type Result3 = ValidReducePattern<'h w c ->', readonly [2, 3, 4]>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly []>();

  // Ellipsis patterns
  type Result4 = ValidReducePattern<'batch ... c -> batch c', readonly [2, 3, 4, 5]>;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [2, 5]>();

  // Composite patterns
  type Result5 = ValidReducePattern<'(h h2) w -> h w', readonly [4, 6], false, { h: 2 }>;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [2, 6]>();

  // KeepDims
  type Result6 = ValidReducePattern<'h w c -> c', readonly [2, 3, 4], true>;
  expectTypeOf<Result6>().toEqualTypeOf<readonly [1, 1, 4]>();
}
