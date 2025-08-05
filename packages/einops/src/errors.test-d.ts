/**
 * Type tests for einops error system
 *
 * These tests validate that our error types work correctly and provide
 * helpful messages following the TypeTensor error patterns.
 */

import { expectTypeOf } from 'expect-type';
import type {
  EinopsError,
  EinopsParseError,
  EinopsAxisError,
  EinopsShapeError,
  ParseErrorMessages,
  AxisErrorMessages,
  ShapeErrorMessages,
  UnknownAxisError,
  DuplicateAxisError,
  RankMismatchError,
  CompositeResolutionError,
  ProductMismatchError,
} from './errors';

// Import helper types for testing (these are internal but we want to test them)
type FormatAxesList<T extends readonly string[]> = 
  T extends readonly [infer First extends string, ...infer Rest extends readonly string[]]
    ? Rest['length'] extends 0
      ? `'${First}'`
      : `'${First}', ${FormatAxesList<Rest>}`
    : never;

// =============================================================================
// Helper Type Tests
// =============================================================================

{
  // Test FormatAxesList with various inputs
  type SingleAxis = FormatAxesList<readonly ['h']>;
  expectTypeOf<SingleAxis>().toEqualTypeOf<"'h'">();
  
  type TwoAxes = FormatAxesList<readonly ['h', 'w']>;
  expectTypeOf<TwoAxes>().toEqualTypeOf<"'h', 'w'">();
  
  type ThreeAxes = FormatAxesList<readonly ['batch', 'height', 'width']>;
  expectTypeOf<ThreeAxes>().toEqualTypeOf<"'batch', 'height', 'width'">();
  
  type ManyAxes = FormatAxesList<readonly ['a', 'b', 'c', 'd']>;
  expectTypeOf<ManyAxes>().toEqualTypeOf<"'a', 'b', 'c', 'd'">();
}

// =============================================================================
// Core Error Interface Tests
// =============================================================================

{
  // Test EinopsError interface structure
  type TestError = EinopsError<'Test message', { pattern: 'h w -> w h'; shape: readonly [2, 3] }>;
  
  expectTypeOf<TestError['__error']>().toEqualTypeOf<'EinopsError'>();
  expectTypeOf<TestError['message']>().toEqualTypeOf<'Test message'>();
  expectTypeOf<TestError['context']>().toEqualTypeOf<{ pattern: 'h w -> w h'; shape: readonly [2, 3] }>();
}

{
  // Test error without context
  type SimpleError = EinopsError<'Simple message'>;
  expectTypeOf<SimpleError['__error']>().toEqualTypeOf<'EinopsError'>();
  expectTypeOf<SimpleError['message']>().toEqualTypeOf<'Simple message'>();
  expectTypeOf<SimpleError['context']>().toEqualTypeOf<unknown>();
}

// =============================================================================
// Branded Error String Type Tests
// =============================================================================

{
  // Test EinopsParseError formatting
  type ParseError = EinopsParseError<'Missing arrow operator'>;
  expectTypeOf<ParseError>().toEqualTypeOf<'[Einops ❌] Parse Error: Missing arrow operator'>();
  
  // Test with complex message
  type ComplexParseError = EinopsParseError<"Unmatched parenthesis '(' at position 12">;
  expectTypeOf<ComplexParseError>().toEqualTypeOf<"[Einops ❌] Parse Error: Unmatched parenthesis '(' at position 12">();
}

{
  // Test EinopsAxisError formatting
  type AxisError = EinopsAxisError<'Duplicate axis found'>;
  expectTypeOf<AxisError>().toEqualTypeOf<'[Einops ❌] Axis Error: Duplicate axis found'>();
  
  // Test with axis name
  type AxisNameError = EinopsAxisError<"Unknown axis 'z' in output">;
  expectTypeOf<AxisNameError>().toEqualTypeOf<"[Einops ❌] Axis Error: Unknown axis 'z' in output">();
}

{
  // Test EinopsShapeError formatting
  type ShapeError = EinopsShapeError<'Dimension mismatch'>;
  expectTypeOf<ShapeError>().toEqualTypeOf<'[Einops ❌] Shape Error: Dimension mismatch'>();
  
  // Test with shape information
  type ShapeInfoError = EinopsShapeError<'Cannot resolve composite from shape [24]'>;
  expectTypeOf<ShapeInfoError>().toEqualTypeOf<'[Einops ❌] Shape Error: Cannot resolve composite from shape [24]'>();
}

// =============================================================================
// Common Error Message Tests
// =============================================================================

{
  // Test ParseErrorMessages
  expectTypeOf<ParseErrorMessages['MissingArrow']>().toEqualTypeOf<
    "[Einops ❌] Parse Error: Missing arrow operator '->'. Pattern must be 'input -> output'"
  >();
  
  expectTypeOf<ParseErrorMessages['EmptyInput']>().toEqualTypeOf<
    "[Einops ❌] Parse Error: Empty input pattern. Specify input axes before '->'"
  >();
  
  expectTypeOf<ParseErrorMessages['EmptyOutput']>().toEqualTypeOf<
    "[Einops ❌] Parse Error: Empty output pattern. Specify output axes after '->'"
  >();
  
  expectTypeOf<ParseErrorMessages['InvalidSyntax']>().toEqualTypeOf<
    '[Einops ❌] Parse Error: Invalid pattern syntax. Check parentheses and axis names'
  >();
}

{
  // Test AxisErrorMessages
  expectTypeOf<AxisErrorMessages['DuplicateInput']>().toEqualTypeOf<
    '[Einops ❌] Axis Error: Duplicate axes in input. Each axis can appear at most once per side'
  >();
  
  expectTypeOf<AxisErrorMessages['DuplicateOutput']>().toEqualTypeOf<
    '[Einops ❌] Axis Error: Duplicate axes in output. Each axis can appear at most once per side'
  >();
  
  expectTypeOf<AxisErrorMessages['MultipleEllipsisInput']>().toEqualTypeOf<
    "[Einops ❌] Axis Error: Multiple ellipsis '...' in input. Only one ellipsis allowed per side"
  >();
  
  expectTypeOf<AxisErrorMessages['MultipleEllipsisOutput']>().toEqualTypeOf<
    "[Einops ❌] Axis Error: Multiple ellipsis '...' in output. Only one ellipsis allowed per side"
  >();
  
  expectTypeOf<AxisErrorMessages['UnknownAxis']>().toEqualTypeOf<
    '[Einops ❌] Axis Error: Unknown axis in output. All output axes must exist in input'
  >();
}

{
  // Test ShapeErrorMessages
  expectTypeOf<ShapeErrorMessages['RankMismatch']>().toEqualTypeOf<
    '[Einops ❌] Shape Error: Pattern rank does not match tensor dimensions'
  >();
  
  expectTypeOf<ShapeErrorMessages['CompositeResolution']>().toEqualTypeOf<
    '[Einops ❌] Shape Error: Cannot resolve composite pattern. Specify axis dimensions'
  >();
  
  expectTypeOf<ShapeErrorMessages['DimensionMismatch']>().toEqualTypeOf<
    '[Einops ❌] Shape Error: Composite pattern dimension mismatch. Check axis values'
  >();
  
  expectTypeOf<ShapeErrorMessages['InvalidDecomposition']>().toEqualTypeOf<
    '[Einops ❌] Shape Error: Cannot decompose dimension with given axes'
  >();
}

// =============================================================================
// Error Factory Function Tests
// =============================================================================

{
  // Test UnknownAxisError
  type UnknownZ = UnknownAxisError<'z', readonly ['h', 'w', 'c']>;
  expectTypeOf<UnknownZ>().toEqualTypeOf<
    "[Einops ❌] Axis Error: Unknown axis 'z' in output. Available axes: ['h', 'w', 'c']"
  >();
  
  type UnknownBatch = UnknownAxisError<'batch', readonly ['height', 'width']>;
  expectTypeOf<UnknownBatch>().toEqualTypeOf<
    "[Einops ❌] Axis Error: Unknown axis 'batch' in output. Available axes: ['height', 'width']"
  >();
}

{
  // Test DuplicateAxisError
  type DuplicateH = DuplicateAxisError<'h', 'input'>;
  expectTypeOf<DuplicateH>().toEqualTypeOf<
    "[Einops ❌] Axis Error: Duplicate axis 'h' in input. Each axis can appear at most once per side"
  >();
  
  type DuplicateW = DuplicateAxisError<'w', 'output'>;
  expectTypeOf<DuplicateW>().toEqualTypeOf<
    "[Einops ❌] Axis Error: Duplicate axis 'w' in output. Each axis can appear at most once per side"
  >();
}

{
  // Test RankMismatchError
  type Rank3vs2 = RankMismatchError<3, 2>;
  expectTypeOf<Rank3vs2>().toEqualTypeOf<
    '[Einops ❌] Shape Error: Pattern expects 3 dimensions but tensor has 2'
  >();
  
  type Rank2vs4 = RankMismatchError<2, 4>;
  expectTypeOf<Rank2vs4>().toEqualTypeOf<
    '[Einops ❌] Shape Error: Pattern expects 2 dimensions but tensor has 4'
  >();
}

{
  // Test CompositeResolutionError
  type CompositeHW = CompositeResolutionError<'(h w)', 24>;
  expectTypeOf<CompositeHW>().toEqualTypeOf<
    "[Einops ❌] Shape Error: Cannot resolve '(h w)' from dimension 24. Specify axis values: rearrange(tensor, pattern, {axis: number})"
  >();
  
  type ComplexComposite = CompositeResolutionError<'(batch head seq)', 512>;
  expectTypeOf<ComplexComposite>().toEqualTypeOf<
    "[Einops ❌] Shape Error: Cannot resolve '(batch head seq)' from dimension 512. Specify axis values: rearrange(tensor, pattern, {axis: number})"
  >();
}

{
  // Test ProductMismatchError
  type Mismatch24vs30 = ProductMismatchError<'(h w)', 24, 30>;
  expectTypeOf<Mismatch24vs30>().toEqualTypeOf<
    "[Einops ❌] Shape Error: Composite '(h w)' expects product 24 but axes give 30. Check axis values"
  >();
  
  type MismatchComplex = ProductMismatchError<'(heads dim)', 512, 480>;
  expectTypeOf<MismatchComplex>().toEqualTypeOf<
    "[Einops ❌] Shape Error: Composite '(heads dim)' expects product 512 but axes give 480. Check axis values"
  >();
}

// =============================================================================
// Error Type Discrimination Tests
// =============================================================================

{
  // Test that different error types are distinct
  type ParseErr = EinopsParseError<'parse error'>;
  type AxisErr = EinopsAxisError<'axis error'>;
  type ShapeErr = EinopsShapeError<'shape error'>;
  
  // These should be different types
  expectTypeOf<ParseErr>().not.toEqualTypeOf<AxisErr>();
  expectTypeOf<AxisErr>().not.toEqualTypeOf<ShapeErr>();
  expectTypeOf<ParseErr>().not.toEqualTypeOf<ShapeErr>();
  
  // But they should all be strings
  expectTypeOf<ParseErr>().toMatchTypeOf<string>();
  expectTypeOf<AxisErr>().toMatchTypeOf<string>();
  expectTypeOf<ShapeErr>().toMatchTypeOf<string>();
}

{
  // Test that all error strings start with the correct prefix
  type ParseTest = EinopsParseError<'test'>;
  type AxisTest = EinopsAxisError<'test'>;
  type ShapeTest = EinopsShapeError<'test'>;
  
  // TypeScript should understand these are branded strings
  expectTypeOf<ParseTest>().toEqualTypeOf<'[Einops ❌] Parse Error: test'>();
  expectTypeOf<AxisTest>().toEqualTypeOf<'[Einops ❌] Axis Error: test'>();
  expectTypeOf<ShapeTest>().toEqualTypeOf<'[Einops ❌] Shape Error: test'>();
}