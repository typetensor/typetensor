/**
 * Type-level tests for einops repeat shape resolver
 */

import type { ValidRepeatPattern } from './type-shape-resolver-repeat';
import { expectTypeOf } from 'expect-type';

// =============================================================================
// Basic Repeat Operations Tests (15 tests)
// =============================================================================

// Test 1: Simple new axis addition
{
  // Add single new axis
  type Result1 = ValidRepeatPattern<'h w -> h w c', readonly [2, 3], { c: 4 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 3, 4]>();

  // Add multiple new axes
  type Result2 = ValidRepeatPattern<'h -> h w c', readonly [2], { w: 3; c: 4 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 3, 4]>();

  // Identity pattern (no repetition)
  type Result3 = ValidRepeatPattern<'h w -> h w', readonly [2, 3]>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [2, 3]>();

  // Add axis at beginning
  type Result4 = ValidRepeatPattern<'h w -> batch h w', readonly [2, 3], { batch: 8 }>;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [8, 2, 3]>();

  // Add axis in middle
  type Result5 = ValidRepeatPattern<'h w -> h c w', readonly [2, 3], { c: 4 }>;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [2, 4, 3]>();
}

// Test 2: Axis repetition patterns
{
  // Repeat existing axis
  type Result1 = ValidRepeatPattern<'w -> (w w2)', readonly [3], { w2: 2 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [6]>();

  // Repeat multiple axes
  type Result2 = ValidRepeatPattern<'h w -> (h h2) (w w2)', readonly [2, 3], { h2: 2; w2: 3 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [4, 9]>();

  // Repeat single axis multiple times
  type Result3 = ValidRepeatPattern<'x -> (x x2)', readonly [5], { x2: 4 }>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [20]>();

  // Repeat with large factors
  type Result4 = ValidRepeatPattern<'h w -> (h h2) w', readonly [2, 3], { h2: 10 }>;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [20, 3]>();

  // Chain repetition
  type Result5 = ValidRepeatPattern<'x -> (x x2)', readonly [2], { x2: 3 }>;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [6]>();
}

// =============================================================================
// Advanced Patterns Tests (20 tests)
// =============================================================================

// Test 3: Mixed repetition and new axes
{
  // Repeat existing + add new
  type Result1 = ValidRepeatPattern<'h w -> (h h2) w c', readonly [2, 3], { h2: 2; c: 4 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [4, 3, 4]>();

  // Complex upsampling pattern (like einops examples)
  type Result2 = ValidRepeatPattern<'h w -> (h h2) (w w2)', readonly [30, 40], { h2: 2; w2: 2 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [60, 80]>();

  // Add channel + upsample
  type Result3 = ValidRepeatPattern<
    'h w -> (h h2) (w w2) c',
    readonly [16, 16],
    { h2: 2; w2: 2; c: 3 }
  >;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [32, 32, 3]>();

  // Multiple new axes with repetition
  type Result4 = ValidRepeatPattern<'x -> (x x2) y z', readonly [5], { x2: 2; y: 3; z: 4 }>;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [10, 3, 4]>();

  // Interleaved pattern
  type Result5 = ValidRepeatPattern<
    'h w -> batch (h h2) c (w w2)',
    readonly [2, 3],
    { batch: 4; h2: 2; c: 3; w2: 2 }
  >;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [4, 4, 3, 6]>();
}

// Test 4: Complex composite patterns
{
  // Decompose then repeat
  type Result1 = ValidRepeatPattern<
    '(h h2) w -> h (w w3)',
    readonly [4, 6],
    { h: 2; h2: 2; w3: 3 }
  >;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 18]>();

  // Nested composite with new axes
  type Result2 = ValidRepeatPattern<
    '(h (h2 h3)) w -> h h2 w c',
    readonly [8, 6],
    { h: 2; h2: 2; h3: 2; c: 3 }
  >;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 2, 6, 3]>();

  // Multiple composite patterns
  type Result3 = ValidRepeatPattern<
    '(a a2) (b b2) -> (a a3) b c',
    readonly [4, 6],
    { a: 2; a2: 2; b: 3; b2: 2; a3: 3; c: 4 }
  >;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [6, 3, 4]>();

  // Composite with repetition factor
  type Result4 = ValidRepeatPattern<'(h h2) -> (h h2 h3)', readonly [6], { h: 2; h2: 3; h3: 2 }>;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [12]>();

  // Deep nesting with new axes
  type Result5 = ValidRepeatPattern<
    '(a (b b2)) -> a b c d',
    readonly [8],
    { a: 2; b: 2; b2: 2; c: 3; d: 4 }
  >;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [2, 2, 3, 4]>();
}

// =============================================================================
// Ellipsis Patterns Tests (15 tests)
// =============================================================================

// Test 5: Ellipsis with new axes
{
  // Add channel to image
  type Result1 = ValidRepeatPattern<'... -> ... c', readonly [32, 32], { c: 3 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [32, 32, 3]>();

  // Batch processing with new axis
  type Result2 = ValidRepeatPattern<'batch ... -> batch ... c', readonly [8, 64, 64], { c: 3 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [8, 64, 64, 3]>();

  // Add multiple axes with ellipsis
  type Result3 = ValidRepeatPattern<'... -> batch ... c', readonly [28, 28], { batch: 16; c: 3 }>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [16, 28, 28, 3]>();

  // Ellipsis in middle
  type Result4 = ValidRepeatPattern<
    'batch ... c -> batch ... c d',
    readonly [4, 32, 32, 3],
    { d: 2 }
  >;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [4, 32, 32, 3, 2]>();

  // Complex ellipsis with new axes
  type Result5 = ValidRepeatPattern<'... -> a ... b c', readonly [16, 16], { a: 2; b: 3; c: 4 }>;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [2, 16, 16, 3, 4]>();
}

// Test 6: Ellipsis with repetition
{
  // Repeat ellipsis dimensions
  type Result1 = ValidRepeatPattern<'... -> (... r)', readonly [4, 4], { r: 2 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [32]>();

  // Ellipsis with axis repetition
  type Result2 = ValidRepeatPattern<'batch ... -> (batch b2) ...', readonly [2, 32, 32], { b2: 4 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [8, 32, 32]>();

  // Mixed ellipsis operations
  type Result3 = ValidRepeatPattern<'... c -> ... (c c2)', readonly [64, 64, 3], { c2: 2 }>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [64, 64, 6]>();

  // Complex ellipsis repetition
  type Result4 = ValidRepeatPattern<'a ... -> (a a2) ... c', readonly [2, 16, 16], { a2: 3; c: 4 }>;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [6, 16, 16, 4]>();

  // Nested ellipsis patterns
  type Result5 = ValidRepeatPattern<'... -> batch (... r)', readonly [8, 8], { batch: 4; r: 2 }>;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [4, 128]>();
}

// =============================================================================
// Edge Cases Tests (15 tests)
// =============================================================================

// Test 7: 1D tensors
{
  type Result1 = ValidRepeatPattern<'x -> x c', readonly [5], { c: 3 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [5, 3]>();

  type Result2 = ValidRepeatPattern<'x -> (x x2)', readonly [5], { x2: 2 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [10]>();

  type Result3 = ValidRepeatPattern<'x -> c x', readonly [5], { c: 4 }>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [4, 5]>();

  type Result4 = ValidRepeatPattern<'x -> (x x2) c', readonly [3], { x2: 3; c: 2 }>;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [9, 2]>();

  type Result5 = ValidRepeatPattern<'x -> c (x x2) d', readonly [2], { c: 2; x2: 4; d: 3 }>;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [2, 8, 3]>();
}

// Test 8: Scalar operations
{
  // Scalar to vector
  type Result1 = ValidRepeatPattern<' -> c', readonly [], { c: 3 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [3]>();

  // Scalar to matrix
  type Result2 = ValidRepeatPattern<' -> h w', readonly [], { h: 2; w: 3 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 3]>();

  // Scalar to 3D
  type Result3 = ValidRepeatPattern<' -> h w c', readonly [], { h: 2; w: 3; c: 4 }>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [2, 3, 4]>();

  // Scalar with composite
  type Result4 = ValidRepeatPattern<' -> (h h2) w', readonly [], { h: 2; h2: 3; w: 4 }>;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [6, 4]>();

  // Scalar with complex pattern
  type Result5 = ValidRepeatPattern<' -> a (b b2) c', readonly [], { a: 2; b: 3; b2: 2; c: 4 }>;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [2, 6, 4]>();
}

// Test 9: Singleton dimensions
{
  // Singleton handling
  type Result1 = ValidRepeatPattern<'h 1 w -> h 1 w c', readonly [2, 1, 3], { c: 4 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 1, 3, 4]>();

  // Singleton to non-singleton
  type Result2 = ValidRepeatPattern<'h 1 w -> h c w', readonly [2, 1, 3], { c: 4 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [2, 4, 3]>();

  // Multiple singletons
  type Result3 = ValidRepeatPattern<
    '1 h 1 w 1 -> a h b w c',
    readonly [1, 2, 1, 3, 1],
    { a: 2; b: 4; c: 5 }
  >;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [2, 2, 4, 3, 5]>();

  // Singleton with repetition
  type Result4 = ValidRepeatPattern<'h 1 -> (h h2) c', readonly [2, 1], { h2: 3; c: 4 }>;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [6, 4]>();

  // Singleton expansion
  type Result5 = ValidRepeatPattern<'1 -> a b c', readonly [1], { a: 2; b: 3; c: 4 }>;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [2, 3, 4]>();
}

// =============================================================================
// Real-World Use Cases Tests (10 tests)
// =============================================================================

// Test 10: Computer vision patterns
{
  // Grayscale to RGB
  type Result1 = ValidRepeatPattern<'h w -> h w c', readonly [224, 224], { c: 3 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [224, 224, 3]>();

  // Upsampling 2x2
  type Result2 = ValidRepeatPattern<'h w -> (h h2) (w w2)', readonly [112, 112], { h2: 2; w2: 2 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [224, 224]>();

  // Batch expansion
  type Result3 = ValidRepeatPattern<'h w c -> batch h w c', readonly [32, 32, 3], { batch: 8 }>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [8, 32, 32, 3]>();

  // Patch extraction simulation
  type Result4 = ValidRepeatPattern<
    'h w -> (h h2) (w w2) c',
    readonly [16, 16],
    { h2: 2; w2: 2; c: 3 }
  >;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [32, 32, 3]>();

  // Channel multiplication
  type Result5 = ValidRepeatPattern<'h w c -> h w (c c2)', readonly [64, 64, 3], { c2: 4 }>;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [64, 64, 12]>();
}

// Test 11: Time series patterns
{
  // Add feature dimension
  type Result1 = ValidRepeatPattern<'time -> time features', readonly [100], { features: 64 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [100, 64]>();

  // Temporal upsampling
  type Result2 = ValidRepeatPattern<'batch time -> batch (time t2)', readonly [32, 50], { t2: 4 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [32, 200]>();

  // Multi-scale features
  type Result3 = ValidRepeatPattern<
    'time -> (time t2) features',
    readonly [25],
    { t2: 4; features: 128 }
  >;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [100, 128]>();

  // Sequence batching
  type Result4 = ValidRepeatPattern<
    'time features -> batch time features',
    readonly [50, 64],
    { batch: 16 }
  >;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [16, 50, 64]>();

  // Time series expansion
  type Result5 = ValidRepeatPattern<
    '... -> batch ... features',
    readonly [100],
    { batch: 8; features: 32 }
  >;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [8, 100, 32]>();
}

// =============================================================================
// Error Cases - Parse Errors Tests (5 tests)
// =============================================================================

// Test 12: Parse errors (following reduce pattern)
{
  // Missing arrow operator
  type Result1 = ValidRepeatPattern<'h w c', readonly [2, 3, 4], { c: 3 }>;
  expectTypeOf<Result1>().toEqualTypeOf<"[Repeat ❌] Parse Error: Missing arrow operator '->'. Pattern must be 'input -> output'">();

  // Empty input for non-scalar tensor
  type Result2 = ValidRepeatPattern<' -> c', readonly [2, 3, 4], { c: 3 }>;
  expectTypeOf<Result2>().toEqualTypeOf<"[Repeat ❌] Parse Error: Empty input pattern. Specify input axes before '->'">();

  // Empty output
  type Result3 = ValidRepeatPattern<'h w -> ', readonly [2, 3]>;
  expectTypeOf<Result3>().toEqualTypeOf<"[Repeat ❌] Parse Error: Empty output pattern. Specify output axes after '->'">();

  // Malformed arrow
  type Result4 = ValidRepeatPattern<'h w - > c', readonly [2, 3], { c: 4 }>;
  expectTypeOf<Result4>().toEqualTypeOf<"[Repeat ❌] Parse Error: Missing arrow operator '->'. Pattern must be 'input -> output'">();

  // Multiple arrows
  type Result5 = ValidRepeatPattern<'h -> w -> c', readonly [2], { w: 3; c: 4 }>;
  expectTypeOf<Result5>().toEqualTypeOf<"[Repeat ❌] Parse Error: Missing arrow operator '->'. Pattern must be 'input -> output'">();
}

// =============================================================================
// Error Cases - Axis Errors Tests (10 tests)
// =============================================================================

// Test 13: Missing axis dimensions (unique to repeat)
{
  // New axis without size specification
  type Result1 = ValidRepeatPattern<'h w -> h w c', readonly [2, 3]>;
  expectTypeOf<Result1>().toEqualTypeOf<"[Repeat ❌] Axis Error: New axis 'c' requires explicit size. Specify: repeat(tensor, pattern, {c: number})">();

  // Multiple missing axes
  type Result2 = ValidRepeatPattern<'h -> h w c', readonly [2]>;
  expectTypeOf<Result2>().toEqualTypeOf<"[Repeat ❌] Axis Error: New axes ['w', 'c'] require explicit sizes. Specify: repeat(tensor, pattern, {w: number, c: number})">();

  // Repetition axis without size
  type Result3 = ValidRepeatPattern<'h w -> (h h2) w', readonly [2, 3]>;
  expectTypeOf<Result3>().toEqualTypeOf<"[Repeat ❌] Axis Error: New axis 'h2' requires explicit size. Specify: repeat(tensor, pattern, {h2: number})">();

  // Complex missing axes
  type Result4 = ValidRepeatPattern<'h -> (h h2) w c d', readonly [2]>;
  expectTypeOf<Result4>().toEqualTypeOf<"[Repeat ❌] Axis Error: New axes ['h2', 'w', 'c', 'd'] require explicit sizes. Specify: repeat(tensor, pattern, {h2: number, w: number, c: number, d: number})">();

  // Ellipsis with missing axes
  type Result5 = ValidRepeatPattern<'... -> ... c d', readonly [32, 32]>;
  expectTypeOf<Result5>().toEqualTypeOf<"[Repeat ❌] Axis Error: New axes ['c', 'd'] require explicit sizes. Specify: repeat(tensor, pattern, {c: number, d: number})">();
}

// Test 14: Invalid axis sizes
{
  // Zero size axis
  type Result1 = ValidRepeatPattern<'h w -> h w c', readonly [2, 3], { c: 0 }>;
  expectTypeOf<Result1>().toEqualTypeOf<"[Repeat ❌] Axis Error: Invalid size 0 for axis 'c'. Repeat sizes must be positive integers">();

  // Negative size axis
  type Result2 = ValidRepeatPattern<'h w -> h w c', readonly [2, 3], { c: -1 }>;
  expectTypeOf<Result2>().toEqualTypeOf<"[Repeat ❌] Axis Error: Invalid size -1 for axis 'c'. Repeat sizes must be positive integers">();

  // Fractional size (if detectable)
  // type Result3 = ValidRepeatPattern<'h w -> h w c', readonly [2, 3], { c: 2.5 }>;
  // expectTypeOf<Result3>().toEqualTypeOf<"[Repeat ❌] Axis Error: Invalid size 2.5 for axis 'c'. Repeat sizes must be positive integers">();

  // Zero repetition factor
  type Result4 = ValidRepeatPattern<'h -> (h h2)', readonly [2], { h2: 0 }>;
  expectTypeOf<Result4>().toEqualTypeOf<"[Repeat ❌] Axis Error: Invalid size 0 for axis 'h2'. Repeat sizes must be positive integers">();

  // Negative repetition factor
  type Result5 = ValidRepeatPattern<'w -> (w w2)', readonly [3], { w2: -2 }>;
  expectTypeOf<Result5>().toEqualTypeOf<"[Repeat ❌] Axis Error: Invalid size -2 for axis 'w2'. Repeat sizes must be positive integers">();
}

// Test 15: Duplicate axes (same as reduce/rearrange)
{
  // Duplicate in input
  type Result1 = ValidRepeatPattern<'h h w -> h w c', readonly [2, 2, 3], { c: 3 }>;
  expectTypeOf<Result1>().toEqualTypeOf<"[Repeat ❌] Axis Error: Duplicate axis 'h' in input. Each axis can appear at most once per side">();

  // Duplicate in output
  type Result2 = ValidRepeatPattern<'h w -> h h c', readonly [2, 3], { c: 3 }>;
  expectTypeOf<Result2>().toEqualTypeOf<"[Repeat ❌] Axis Error: Duplicate axis 'h' in output. Each axis can appear at most once per side">();

  // Duplicate with new axis
  type Result3 = ValidRepeatPattern<'h w -> c c h', readonly [2, 3], { c: 4 }>;
  expectTypeOf<Result3>().toEqualTypeOf<"[Repeat ❌] Axis Error: Duplicate axis 'c' in output. Each axis can appear at most once per side">();

  // Duplicate in composite
  type Result4 = ValidRepeatPattern<'h w -> (h h) c', readonly [2, 3], { c: 4 }>;
  expectTypeOf<Result4>().toEqualTypeOf<"[Repeat ❌] Axis Error: Duplicate axis 'h' in output. Each axis can appear at most once per side">();

  // Multiple duplicates
  type Result5 = ValidRepeatPattern<'a b c -> a b a c', readonly [2, 3, 4]>;
  expectTypeOf<Result5>().toEqualTypeOf<"[Repeat ❌] Axis Error: Duplicate axis 'a' in output. Each axis can appear at most once per side">();
}

// =============================================================================
// Error Cases - Shape Errors Tests (5 tests)
// =============================================================================

// Test 16: Rank mismatch
{
  type Result1 = ValidRepeatPattern<'h w c -> h w c d', readonly [2, 3], { d: 4 }>;
  expectTypeOf<Result1>().toEqualTypeOf<'[Repeat ❌] Shape Error: Repeat pattern expects 3 dimensions but tensor has 2'>();

  type Result2 = ValidRepeatPattern<'a b -> a b c d', readonly [2], { c: 3; d: 4 }>;
  expectTypeOf<Result2>().toEqualTypeOf<'[Repeat ❌] Shape Error: Repeat pattern expects 2 dimensions but tensor has 1'>();

  type Result3 = ValidRepeatPattern<'h -> h w', readonly [2, 3], { w: 4 }>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [2, 4]>();

  type Result4 = ValidRepeatPattern<'... a -> ... a b', readonly [], { b: 2 }>;
  expectTypeOf<Result4>().toEqualTypeOf<'[Repeat ❌] Shape Error: Repeat pattern expects 1 dimensions but tensor has 0'>();

  type Result5 = ValidRepeatPattern<'a b c d -> a b c d e', readonly [2, 3], { e: 5 }>;    
  expectTypeOf<Result5>().toEqualTypeOf<'[Repeat ❌] Shape Error: Repeat pattern expects 4 dimensions but tensor has 2'>();
}

// Test 17: Composite resolution errors
{
  type Result1 = ValidRepeatPattern<
    '(h h2) w -> h w c',
    readonly [4, 6],
    { h: 3; h2: 2; c: 3 }
  >;
  expectTypeOf<Result1>().toEqualTypeOf<"[Repeat ❌] Shape Error: Cannot resolve '(h h2)' from dimension 4. Specify axis values: repeat(tensor, pattern, {axis: number})">();

  type Result2 = ValidRepeatPattern<
    'h (w w2) -> h w c',
    readonly [2, 9],
    { w: 2; w2: 3; c: 4 }
  >;
  expectTypeOf<Result2>().toEqualTypeOf<"[Repeat ❌] Shape Error: Cannot resolve '(w w2)' from dimension 9. Specify axis values: repeat(tensor, pattern, {axis: number})">();

  type Result3 = ValidRepeatPattern<
    '(a a2) (b b2) -> a b c',
    readonly [6, 8],
    { a: 2; a2: 2; b: 3; b2: 2; c: 4 }
  >;
  expectTypeOf<Result3>().toEqualTypeOf<"[Repeat ❌] Shape Error: Cannot resolve '(a a2)' from dimension 6. Specify axis values: repeat(tensor, pattern, {axis: number})">();

  type Result4 = ValidRepeatPattern<
    '((a a2) a3) -> a c',
    readonly [12],
    { a: 2; a2: 2; a3: 2; c: 3 }
  >;
  expectTypeOf<Result4>().toEqualTypeOf<"[Repeat ❌] Shape Error: Cannot resolve '((a a2) a3)' from dimension 12. Specify axis values: repeat(tensor, pattern, {axis: number})">();

  type Result5 = ValidRepeatPattern<
    '(h h2) -> (h h2) c',
    readonly [5],
    { h: 2; h2: 3; c: 4 }
  >;
  expectTypeOf<Result5>().toEqualTypeOf<"[Repeat ❌] Shape Error: Cannot resolve '(h h2)' from dimension 5. Specify axis values: repeat(tensor, pattern, {axis: number})">();
}

// =============================================================================
// Valid Cases with ValidRepeatPattern (should return shapes) Tests (10 tests)
// =============================================================================

// Test 18: Valid patterns should return shapes, not error messages
{
  // Simple new axis addition
  type Result1 = ValidRepeatPattern<'h w -> h w c', readonly [2, 3], { c: 4 }>;
  expectTypeOf<Result1>().toEqualTypeOf<readonly [2, 3, 4]>();

  // Axis repetition
  type Result2 = ValidRepeatPattern<'h w -> (h h2) w', readonly [2, 3], { h2: 2 }>;
  expectTypeOf<Result2>().toEqualTypeOf<readonly [4, 3]>();

  // Identity pattern
  type Result3 = ValidRepeatPattern<'h w c -> h w c', readonly [2, 3, 4]>;
  expectTypeOf<Result3>().toEqualTypeOf<readonly [2, 3, 4]>();

  // Ellipsis patterns
  type Result4 = ValidRepeatPattern<'batch ... -> batch ... c', readonly [8, 64, 64], { c: 3 }>;
  expectTypeOf<Result4>().toEqualTypeOf<readonly [8, 64, 64, 3]>();

  // Composite patterns
  type Result5 = ValidRepeatPattern<'(h h2) w -> h w c', readonly [4, 6], { h: 2; h2: 2; c: 3 }>;
  expectTypeOf<Result5>().toEqualTypeOf<readonly [2, 6, 3]>();

  // Scalar expansion
  type Result6 = ValidRepeatPattern<' -> h w c', readonly [], { h: 2; w: 3; c: 4 }>;
  expectTypeOf<Result6>().toEqualTypeOf<readonly [2, 3, 4]>();

  // Complex mixed pattern
  type Result7 = ValidRepeatPattern<
    'h w -> (h h2) (w w2) c d',
    readonly [2, 3],
    { h2: 2; w2: 3; c: 4; d: 5 }
  >;
  expectTypeOf<Result7>().toEqualTypeOf<readonly [4, 9, 4, 5]>();

  // Multiple ellipsis dims
  type Result8 = ValidRepeatPattern<
    '... -> batch ... features',
    readonly [32, 32, 3],
    { batch: 8; features: 64 }
  >;
  expectTypeOf<Result8>().toEqualTypeOf<readonly [8, 32, 32, 3, 64]>();

  // Nested composite
  type Result9 = ValidRepeatPattern<
    '(h (h2 h3)) -> h h2 c',
    readonly [8],
    { h: 2; h2: 2; h3: 2; c: 4 }
  >;
  expectTypeOf<Result9>().toEqualTypeOf<readonly [2, 2, 4]>();

  // Large dimensions
  type Result10 = ValidRepeatPattern<'h w -> (h h2) (w w2)', readonly [224, 224], { h2: 2; w2: 2 }>;
  expectTypeOf<Result10>().toEqualTypeOf<readonly [448, 448]>();
}
