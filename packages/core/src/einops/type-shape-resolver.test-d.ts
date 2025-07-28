/**
 * Type-level tests for einops shape resolver
 */

import type { ResolveEinopsShape } from './type-shape-resolver';
import { expectTypeOf } from 'expect-type';

// =============================================================================
// Simple Transpose Tests
// =============================================================================

// Test 1: Basic 2D transpose
{
  type Result = ResolveEinopsShape<'h w -> w h', readonly [2, 3]>;
  expectTypeOf<Result>().toEqualTypeOf<readonly [3, 2]>();
}

// Test 2: 3D permutation patterns
{
  // Transpose last two dimensions
  type Result1 = ResolveEinopsShape<'batch height width -> batch width height', readonly [2, 3, 4]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 4, 3]>();

  // Move batch to end
  type Result2 = ResolveEinopsShape<'batch height width -> height width batch', readonly [2, 3, 4]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [3, 4, 2]>();

  // Full permutation
  type Result3 = ResolveEinopsShape<'a b c -> c a b', readonly [2, 3, 4]>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [4, 2, 3]>();
}

// Test 3: Identity patterns
{
  type Result1 = ResolveEinopsShape<'h w -> h w', readonly [2, 3]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 3]>();

  type Result2 = ResolveEinopsShape<'a b c -> a b c', readonly [2, 3, 4]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 3, 4]>();
}

// =============================================================================
// Composite Pattern Tests
// =============================================================================

// Test 4: Merge patterns (simple to composite)
{
  type Result1 = ResolveEinopsShape<'h w -> (h w)', readonly [2, 3]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [6]>();

  type Result2 = ResolveEinopsShape<'batch h w -> batch (h w)', readonly [2, 3, 4]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 12]>();

  type Result3 = ResolveEinopsShape<'a b c -> (a b) c', readonly [2, 3, 4]>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [6, 4]>();
}

// Test 5: Split patterns (composite to simple) with axes
{
  type Result1 = ResolveEinopsShape<'(h w) -> h w', readonly [6], { h: 2 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 3]>();

  type Result2 = ResolveEinopsShape<'(h w) c -> h w c', readonly [2, 3], { h: 2 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 1, 3]>();

  type Result3 = ResolveEinopsShape<'batch (h w) -> batch h w', readonly [2, 12], { h: 3 }>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [2, 3, 4]>();
}

// =============================================================================
// Singleton Pattern Tests
// =============================================================================

// Test 6: Adding singleton dimensions
{
  type Result1 = ResolveEinopsShape<'h w -> h 1 w', readonly [2, 3]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 1, 3]>();

  type Result2 = ResolveEinopsShape<'h w -> 1 h w 1', readonly [2, 3]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [1, 2, 3, 1]>();
}

// Test 7: Removing singleton dimensions
{
  type Result = ResolveEinopsShape<'a 1 b 1 -> a b', readonly [2, 1, 3, 1]>;
  expectTypeOf<Result>().toEqualTypeOf<readonly [2, 3]>();
}

// Test 7b: Singleton with ellipsis
{
  type Result = ResolveEinopsShape<'... 1 -> 1 ...', readonly [2, 3, 4, 1]>;
  expectTypeOf<Result>().toEqualTypeOf<readonly [1, 2, 3, 4]>();
}

// =============================================================================
// Ellipsis Pattern Tests
// =============================================================================

// Test 8: Basic ellipsis patterns
{
  // Move last dimension to front
  type Result1 = ResolveEinopsShape<'... c -> c ...', readonly [2, 3, 4]>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [4, 2, 3]>();

  // Move first dimension to end
  type Result2 = ResolveEinopsShape<'a ... -> ... a', readonly [2, 3, 4]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [3, 4, 2]>();
}

// Test 9: Ellipsis with named axes
{
  type Result1 = ResolveEinopsShape<
    'batch ... height width -> height width batch ...',
    readonly [2, 3, 4, 5]
  >;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [4, 5, 2, 3]>();

  type Result2 = ResolveEinopsShape<'a ... z -> z a ...', readonly [2, 3, 4]>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [4, 2, 3]>();
}

// =============================================================================
// Complex Real-World Patterns
// =============================================================================

// Test 10: Vision transformer patterns
{
  // Image to patches
  type Result1 = ResolveEinopsShape<
    'c (h p1) (w p2) -> (h w) (p1 p2 c)',
    readonly [3, 224, 224],
    { p1: 16; p2: 16 }
  >;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [196, 768]>();

  // Multi-head attention reshape
  type Result2 = ResolveEinopsShape<
    'batch seq (heads dim) -> batch heads seq dim',
    readonly [8, 64, 512],
    { heads: 8 }
  >;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [8, 8, 64, 64]>();
}

// Test 11: Channel first/last conversion
{
  // NCHW to NHWC
  type Result1 = ResolveEinopsShape<
    'batch channel height width -> batch height width channel',
    readonly [32, 3, 224, 224]
  >;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [32, 224, 224, 3]>();

  // NHWC to NCHW
  type Result2 = ResolveEinopsShape<
    'batch height width channel -> batch channel height width',
    readonly [32, 224, 224, 3]
  >;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [32, 3, 224, 224]>();
}

// =============================================================================
// Scalar Pattern Tests  
// =============================================================================

// Test 12: Scalar patterns
{
  // Scalar identity
  type Result1 = ResolveEinopsShape<' -> ', readonly []>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly []>();
  
  // Scalar to singleton
  type Result2 = ResolveEinopsShape<' -> 1', readonly []>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [1]>();
  
  // Singleton to scalar
  type Result3 = ResolveEinopsShape<'1 -> ', readonly [1]>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly []>();
}

// =============================================================================
// Error Cases (should resolve to never)
// =============================================================================

// Test 12: Invalid patterns
{
  // Unknown axis in output
  type Result1 = ResolveEinopsShape<'h w -> h w c', readonly [2, 3]>;
  expectTypeOf<Result1>().toBeNever();

  // Duplicate axes in input
  type Result2 = ResolveEinopsShape<'h h -> h', readonly [2, 3]>;
  expectTypeOf<Result2>().toBeNever();

  // Multiple ellipsis
  type Result3 = ResolveEinopsShape<'... a ... -> a', readonly [2, 3, 4]>;
  expectTypeOf<Result3>().toBeNever();
}

// Test 13: Shape mismatch in composite patterns
{
  // Wrong decomposition - 6 != 2*4
  type Result1 = ResolveEinopsShape<'(h w) -> h w', readonly [6], { h: 2; w: 4 }>;
  expectTypeOf<Result1>().toBeNever();

  // Missing required axis
  type Result2 = ResolveEinopsShape<'(h w) c -> h w c', readonly [2, 3]>; // Missing h dimension
  expectTypeOf<Result2>().toBeNever();
}
