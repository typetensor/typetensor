/**
 * Type tests for einops validation - Step 2: Basic Implementation
 *
 * These tests validate our simple ValidEinopsPattern implementation:
 * - Syntax error cases should return branded error strings
 * - Valid patterns should return computed shapes
 * - Existing functionality should work unchanged
 */

import { expectTypeOf } from 'expect-type';
import type { ValidEinopsPattern } from './type-validation';

// =============================================================================
// Step 2: Syntax Error Tests
// =============================================================================

{
  // Test missing arrow operator
  type NoArrow = ValidEinopsPattern<'h w', readonly [2, 3]>;
  expectTypeOf<NoArrow>().toEqualTypeOf<"[Einops ❌] Parse Error: Missing arrow operator '->'. Pattern must be 'input -> output'">();

  type NoArrowComplex = ValidEinopsPattern<'batch height width', readonly [2, 3, 4]>;
  expectTypeOf<NoArrowComplex>().toEqualTypeOf<"[Einops ❌] Parse Error: Missing arrow operator '->'. Pattern must be 'input -> output'">();
}

{
  // Test empty input patterns
  type EmptyInput1 = ValidEinopsPattern<' -> h w', readonly [2, 3]>;
  expectTypeOf<EmptyInput1>().toEqualTypeOf<"[Einops ❌] Parse Error: Empty input pattern. Specify input axes before '->'">();

  type EmptyInput2 = ValidEinopsPattern<'-> h w', readonly [2, 3]>;
  expectTypeOf<EmptyInput2>().toEqualTypeOf<"[Einops ❌] Parse Error: Empty input pattern. Specify input axes before '->'">();

  type EmptyInput3 = ValidEinopsPattern<'  -> h w', readonly [2, 3]>;
  expectTypeOf<EmptyInput3>().toEqualTypeOf<"[Einops ❌] Parse Error: Empty input pattern. Specify input axes before '->'">();
}

{
  // Test empty output patterns (these are valid - they reduce to scalar)
  type EmptyOutput1 = ValidEinopsPattern<'h w -> ', readonly [2, 3]>;
  expectTypeOf<EmptyOutput1>().toEqualTypeOf<readonly []>(); // Reduces to scalar

  type EmptyOutput2 = ValidEinopsPattern<'h w ->', readonly [2, 3]>;
  expectTypeOf<EmptyOutput2>().toEqualTypeOf<readonly []>(); // Reduces to scalar

  type EmptyOutput3 = ValidEinopsPattern<'h w ->  ', readonly [2, 3]>;
  expectTypeOf<EmptyOutput3>().toEqualTypeOf<readonly []>(); // Reduces to scalar
}

// =============================================================================
// Step 2: Success Cases (Should Work Like Before)
// =============================================================================

{
  // Test basic successful patterns
  type Transpose = ValidEinopsPattern<'h w -> w h', readonly [2, 3]>;
  expectTypeOf<Transpose>().toEqualTypeOf<readonly [3, 2]>();

  type Identity = ValidEinopsPattern<'a b c -> a b c', readonly [2, 3, 4]>;
  expectTypeOf<Identity>().toEqualTypeOf<readonly [2, 3, 4]>();

  type Permute = ValidEinopsPattern<'batch height width -> height width batch', readonly [2, 3, 4]>;
  expectTypeOf<Permute>().toEqualTypeOf<readonly [3, 4, 2]>();
}

{
  // Test composite patterns
  type Flatten = ValidEinopsPattern<'h w -> (h w)', readonly [2, 3]>;
  expectTypeOf<Flatten>().toEqualTypeOf<readonly [6]>();

  type Split = ValidEinopsPattern<'(h w) -> h w', readonly [6], { h: 2 }>;
  expectTypeOf<Split>().toEqualTypeOf<readonly [2, 3]>();
}

{
  // Test ellipsis patterns
  type EllipsisMove = ValidEinopsPattern<'... c -> c ...', readonly [2, 3, 4]>;
  expectTypeOf<EllipsisMove>().toEqualTypeOf<readonly [4, 2, 3]>();
}

{
  // Test singleton patterns
  type AddSingleton = ValidEinopsPattern<'h w -> h 1 w', readonly [2, 3]>;
  expectTypeOf<AddSingleton>().toEqualTypeOf<readonly [2, 1, 3]>();
}

// =============================================================================
// Step 2: Current Limitation Tests
// =============================================================================

{
  // Step 3: Now returns specific error messages instead of generic failures
  type UnknownAxis = ValidEinopsPattern<'h w -> h w c', readonly [2, 3]>;
  expectTypeOf<UnknownAxis>().toEqualTypeOf<"[Einops ❌] Axis Error: Unknown axis 'c' in output. Available axes: ['h', 'w']">();

  type DuplicateAxis = ValidEinopsPattern<'h h -> h', readonly [2, 3]>;
  expectTypeOf<DuplicateAxis>().toEqualTypeOf<"[Einops ❌] Axis Error: Duplicate axis 'h' in input. Each axis can appear at most once per side">();

  type BadComposite = ValidEinopsPattern<'(h w) -> h w', readonly [6]>; // Missing axis spec
  expectTypeOf<BadComposite>().toEqualTypeOf<"[Einops ❌] Shape Error: Cannot resolve '(h w)' from dimension 6. Specify axis values: rearrange(tensor, pattern, {axis: number})">();

  // Test multiple ellipsis detection
  type MultipleEllipsisInput = ValidEinopsPattern<'h ... w ... -> h w', readonly [2, 3, 4, 5]>;
  expectTypeOf<MultipleEllipsisInput>().toEqualTypeOf<"[Einops ❌] Axis Error: Multiple ellipsis '...' in input. Only one ellipsis allowed per side">();

  type MultipleEllipsisOutput = ValidEinopsPattern<'h w -> ... c ...', readonly [2, 3]>;
  expectTypeOf<MultipleEllipsisOutput>().toEqualTypeOf<"[Einops ❌] Axis Error: Multiple ellipsis '...' in output. Only one ellipsis allowed per side">();

  // Test product mismatch detection
  type ProductMismatch = ValidEinopsPattern<'(h w) -> h w', readonly [10], { h: 3, w: 4 }>; // 3*4=12, but dimension is 10
  expectTypeOf<ProductMismatch>().toEqualTypeOf<"[Einops ❌] Shape Error: Composite '(h w)' expects product 10 but axes give 12. Check axis values">();

  // Test fractional dimension detection
  type FractionalDimension = ValidEinopsPattern<'(h w) -> h w', readonly [13], { h: 4 }>; // 13/4 = 3.25 (non-integer)
  expectTypeOf<FractionalDimension>().toEqualTypeOf<"[Einops ❌] Shape Error: Pattern '(h w) -> h w' produces fractional dimensions. Composite axes must divide evenly. Use integer axis values: rearrange(tensor, pattern, {axis: integer})">();
}

// =============================================================================
// Step 2: Validation of Type Discrimination
// =============================================================================

{
  // Success cases should return Shape types
  type Success = ValidEinopsPattern<'h w -> w h', readonly [2, 3]>;
  expectTypeOf<Success>().toMatchTypeOf<readonly number[]>();

  // Error cases should return branded error strings
  type ParseError = ValidEinopsPattern<'h w', readonly [2, 3]>;
  expectTypeOf<ParseError>().toMatchTypeOf<string>();
  expectTypeOf<ParseError>().not.toMatchTypeOf<readonly number[]>();
}

{
  // Test that different error types are properly branded
  type NoArrowError = ValidEinopsPattern<'h w', readonly [2, 3]>;
  type EmptyError = ValidEinopsPattern<' -> h w', readonly [2, 3]>;

  // Both should be strings but with different content
  expectTypeOf<NoArrowError>().toMatchTypeOf<string>();
  expectTypeOf<EmptyError>().toMatchTypeOf<string>();
  expectTypeOf<NoArrowError>().not.toEqualTypeOf<EmptyError>();
}
