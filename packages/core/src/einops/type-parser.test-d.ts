/**
 * Type tests for the einops type-level parser
 *
 * These tests validate that our type-level parser correctly parses
 * einops patterns at compile time and produces the expected AST structures.
 */

import { expectTypeOf } from 'expect-type';
import type {
  ParsePattern,
  IsValidPattern,
  GetParseError,
  TypeEinopsAST,
  TypeAxisPattern,
  TypeSimpleAxis,
  TypeCompositeAxis,
  TypeEllipsisAxis,
  TypeSingletonAxis,
  FirstChar,
  RestChars,
  SkipWhitespace,
  ExtractAxisName,
  ShiftUntil,
  IsWhitespace,
  IsAxisChar,
  TypeParseError,
} from './type-parser';

// =============================================================================
// String Utility Type Tests
// =============================================================================

{
  // Test FirstChar
  expectTypeOf<FirstChar<'hello'>>().toEqualTypeOf<'h'>();
  expectTypeOf<FirstChar<'a'>>().toEqualTypeOf<'a'>();
  expectTypeOf<FirstChar<''>>().toEqualTypeOf<''>();
  expectTypeOf<FirstChar<'   '>>().toEqualTypeOf<' '>();
}

{
  // Test RestChars
  expectTypeOf<RestChars<'hello'>>().toEqualTypeOf<'ello'>();
  expectTypeOf<RestChars<'a'>>().toEqualTypeOf<''>();
  expectTypeOf<RestChars<''>>().toEqualTypeOf<''>();
  expectTypeOf<RestChars<'abc def'>>().toEqualTypeOf<'bc def'>();
}

{
  // Test SkipWhitespace
  expectTypeOf<SkipWhitespace<'  hello'>>().toEqualTypeOf<'hello'>();
  expectTypeOf<SkipWhitespace<'\t\n  world'>>().toEqualTypeOf<'world'>();
  expectTypeOf<SkipWhitespace<'nowhitespace'>>().toEqualTypeOf<'nowhitespace'>();
  expectTypeOf<SkipWhitespace<'   '>>().toEqualTypeOf<''>();
  expectTypeOf<SkipWhitespace<''>>().toEqualTypeOf<''>();
}

{
  // Test ShiftUntil
  expectTypeOf<ShiftUntil<'hello world', ' '>>().toEqualTypeOf<['hello', ' world']>();
  expectTypeOf<ShiftUntil<'abc', 'x'>>().toEqualTypeOf<['abc', '']>();
  expectTypeOf<ShiftUntil<'a->b', '-'>>().toEqualTypeOf<['a', '->b']>();
  expectTypeOf<ShiftUntil<'', 'x'>>().toEqualTypeOf<['', '']>();
}

{
  // Test ExtractAxisName
  expectTypeOf<ExtractAxisName<'height width'>>().toEqualTypeOf<['height', ' width']>();
  expectTypeOf<ExtractAxisName<'h('>>().toEqualTypeOf<['h', '(']>();
  expectTypeOf<ExtractAxisName<'axis_name123'>>().toEqualTypeOf<['axis_name123', '']>();
  expectTypeOf<ExtractAxisName<'a->'>>().toEqualTypeOf<['a', '->']>();
}

{
  // Test IsWhitespace
  expectTypeOf<IsWhitespace<' '>>().toEqualTypeOf<true>();
  expectTypeOf<IsWhitespace<'\t'>>().toEqualTypeOf<true>();
  expectTypeOf<IsWhitespace<'\n'>>().toEqualTypeOf<true>();
  expectTypeOf<IsWhitespace<'a'>>().toEqualTypeOf<false>();
  expectTypeOf<IsWhitespace<'('>>().toEqualTypeOf<false>();
}

{
  // Test IsAxisChar
  expectTypeOf<IsAxisChar<'a'>>().toEqualTypeOf<true>();
  expectTypeOf<IsAxisChar<'Z'>>().toEqualTypeOf<true>();
  expectTypeOf<IsAxisChar<'5'>>().toEqualTypeOf<true>();
  expectTypeOf<IsAxisChar<'_'>>().toEqualTypeOf<true>();
  expectTypeOf<IsAxisChar<' '>>().toEqualTypeOf<false>();
  expectTypeOf<IsAxisChar<'('>>().toEqualTypeOf<false>();
  expectTypeOf<IsAxisChar<'-'>>().toEqualTypeOf<false>();
}

// =============================================================================
// Type AST Structure Tests
// =============================================================================

{
  // Test TypeSimpleAxis structure
  expectTypeOf<TypeSimpleAxis>().toHaveProperty('type').toEqualTypeOf<'simple'>();
  expectTypeOf<TypeSimpleAxis>().toHaveProperty('name').toEqualTypeOf<string>();

  const simpleAxis: TypeSimpleAxis = { type: 'simple', name: 'height' };
  expectTypeOf<typeof simpleAxis>().toEqualTypeOf<TypeSimpleAxis>();
}

{
  // Test TypeCompositeAxis structure
  expectTypeOf<TypeCompositeAxis>().toHaveProperty('type').toEqualTypeOf<'composite'>();
  expectTypeOf<TypeCompositeAxis>()
    .toHaveProperty('axes')
    .toEqualTypeOf<readonly TypeAxisPattern[]>();

  const compositeAxis: TypeCompositeAxis = {
    type: 'composite',
    axes: [
      { type: 'simple', name: 'h' },
      { type: 'simple', name: 'w' },
    ],
  };
  expectTypeOf<typeof compositeAxis>().toEqualTypeOf<TypeCompositeAxis>();
}

{
  // Test TypeEllipsisAxis structure
  expectTypeOf<TypeEllipsisAxis>().toHaveProperty('type').toEqualTypeOf<'ellipsis'>();

  const ellipsisAxis: TypeEllipsisAxis = { type: 'ellipsis' };
  expectTypeOf<typeof ellipsisAxis>().toEqualTypeOf<TypeEllipsisAxis>();
}

{
  // Test TypeSingletonAxis structure
  expectTypeOf<TypeSingletonAxis>().toHaveProperty('type').toEqualTypeOf<'singleton'>();

  const singletonAxis: TypeSingletonAxis = { type: 'singleton' };
  expectTypeOf<typeof singletonAxis>().toEqualTypeOf<TypeSingletonAxis>();
}

{
  // Test TypeAxisPattern union
  expectTypeOf<TypeAxisPattern>().toEqualTypeOf<
    TypeSimpleAxis | TypeCompositeAxis | TypeEllipsisAxis | TypeSingletonAxis
  >();
}

{
  // Test TypeEinopsAST structure
  expectTypeOf<TypeEinopsAST>().toHaveProperty('input').toEqualTypeOf<readonly TypeAxisPattern[]>();
  expectTypeOf<TypeEinopsAST>()
    .toHaveProperty('output')
    .toEqualTypeOf<readonly TypeAxisPattern[]>();
}

// =============================================================================
// Simple Pattern Parsing Tests
// =============================================================================

{
  // Test simple single axis pattern
  type SimplePattern = ParsePattern<'a -> b'>;
  expectTypeOf<SimplePattern>().toEqualTypeOf<{
    input: readonly [{ type: 'simple'; name: 'a' }];
    output: readonly [{ type: 'simple'; name: 'b' }];
  }>();
}

{
  // Test simple transpose pattern
  type TransposePattern = ParsePattern<'h w -> w h'>;
  expectTypeOf<TransposePattern>().toEqualTypeOf<{
    input: readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
    output: readonly [{ type: 'simple'; name: 'w' }, { type: 'simple'; name: 'h' }];
  }>();
}

{
  // Test multi-axis pattern
  type MultiAxisPattern =
    ParsePattern<'batch height width channels -> batch channels height width'>;
  expectTypeOf<MultiAxisPattern>().toEqualTypeOf<{
    input: readonly [
      { type: 'simple'; name: 'batch' },
      { type: 'simple'; name: 'height' },
      { type: 'simple'; name: 'width' },
      { type: 'simple'; name: 'channels' },
    ];
    output: readonly [
      { type: 'simple'; name: 'batch' },
      { type: 'simple'; name: 'channels' },
      { type: 'simple'; name: 'height' },
      { type: 'simple'; name: 'width' },
    ];
  }>();
}

{
  // Test pattern with extra whitespace
  type WhitespacePattern = ParsePattern<'  a   b  ->  c   d  '>;
  expectTypeOf<WhitespacePattern>().toEqualTypeOf<{
    input: readonly [{ type: 'simple'; name: 'a' }, { type: 'simple'; name: 'b' }];
    output: readonly [{ type: 'simple'; name: 'c' }, { type: 'simple'; name: 'd' }];
  }>();
}

// =============================================================================
// Composite Pattern Parsing Tests
// =============================================================================

{
  // Test simple composite pattern
  type CompositePattern = ParsePattern<'(h w) c -> h w c'>;
  expectTypeOf<CompositePattern>().toEqualTypeOf<{
    input: readonly [
      {
        type: 'composite';
        axes: readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
      },
      { type: 'simple'; name: 'c' },
    ];
    output: readonly [
      { type: 'simple'; name: 'h' },
      { type: 'simple'; name: 'w' },
      { type: 'simple'; name: 'c' },
    ];
  }>();
}

{
  // Test nested composite pattern
  type NestedCompositePattern = ParsePattern<'((h w) c) d -> h w c d'>;
  expectTypeOf<NestedCompositePattern>().toEqualTypeOf<{
    input: readonly [
      {
        type: 'composite';
        axes: readonly [
          {
            type: 'composite';
            axes: readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
          },
          { type: 'simple'; name: 'c' },
        ];
      },
      { type: 'simple'; name: 'd' },
    ];
    output: readonly [
      { type: 'simple'; name: 'h' },
      { type: 'simple'; name: 'w' },
      { type: 'simple'; name: 'c' },
      { type: 'simple'; name: 'd' },
    ];
  }>();
}

{
  // Test multiple composites
  type MultipleComposites = ParsePattern<'(h w) (a b) -> h w a b'>;
  expectTypeOf<MultipleComposites>().toEqualTypeOf<{
    input: readonly [
      {
        type: 'composite';
        axes: readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
      },
      {
        type: 'composite';
        axes: readonly [{ type: 'simple'; name: 'a' }, { type: 'simple'; name: 'b' }];
      },
    ];
    output: readonly [
      { type: 'simple'; name: 'h' },
      { type: 'simple'; name: 'w' },
      { type: 'simple'; name: 'a' },
      { type: 'simple'; name: 'b' },
    ];
  }>();
}

{
  // Test empty composite
  type EmptyComposite = ParsePattern<'() -> a'>;
  expectTypeOf<EmptyComposite>().toEqualTypeOf<{
    input: readonly [
      {
        type: 'composite';
        axes: readonly [];
      },
    ];
    output: readonly [{ type: 'simple'; name: 'a' }];
  }>();
}

// =============================================================================
// Ellipsis Pattern Parsing Tests
// =============================================================================

{
  // Test ellipsis in input and output
  type EllipsisPattern = ParsePattern<'batch ... -> ...'>;
  expectTypeOf<EllipsisPattern>().toEqualTypeOf<{
    input: readonly [{ type: 'simple'; name: 'batch' }, { type: 'ellipsis' }];
    output: readonly [{ type: 'ellipsis' }];
  }>();
}

{
  // Test ellipsis with other axes
  type EllipsisWithAxes = ParsePattern<'batch ... height -> height batch ...'>;
  expectTypeOf<EllipsisWithAxes>().toEqualTypeOf<{
    input: readonly [
      { type: 'simple'; name: 'batch' },
      { type: 'ellipsis' },
      { type: 'simple'; name: 'height' },
    ];
    output: readonly [
      { type: 'simple'; name: 'height' },
      { type: 'simple'; name: 'batch' },
      { type: 'ellipsis' },
    ];
  }>();
}

{
  // Test multiple ellipses
  type MultipleEllipses = ParsePattern<'... -> ... ...'>;
  expectTypeOf<MultipleEllipses>().toEqualTypeOf<{
    input: readonly [{ type: 'ellipsis' }];
    output: readonly [{ type: 'ellipsis' }, { type: 'ellipsis' }];
  }>();
}

// =============================================================================
// Singleton Pattern Parsing Tests
// =============================================================================

{
  // Test singleton dimensions
  type SingletonPattern = ParsePattern<'h w 1 -> h w'>;
  expectTypeOf<SingletonPattern>().toEqualTypeOf<{
    input: readonly [
      { type: 'simple'; name: 'h' },
      { type: 'simple'; name: 'w' },
      { type: 'singleton' },
    ];
    output: readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
  }>();
}

{
  // Test multiple singletons
  type MultipleSingletons = ParsePattern<'1 1 1 -> 1'>;
  expectTypeOf<MultipleSingletons>().toEqualTypeOf<{
    input: readonly [{ type: 'singleton' }, { type: 'singleton' }, { type: 'singleton' }];
    output: readonly [{ type: 'singleton' }];
  }>();
}

{
  // Test singleton placement
  type SingletonPlacement = ParsePattern<'batch height 1 -> batch 1 height'>;
  expectTypeOf<SingletonPlacement>().toEqualTypeOf<{
    input: readonly [
      { type: 'simple'; name: 'batch' },
      { type: 'simple'; name: 'height' },
      { type: 'singleton' },
    ];
    output: readonly [
      { type: 'simple'; name: 'batch' },
      { type: 'singleton' },
      { type: 'simple'; name: 'height' },
    ];
  }>();
}

// =============================================================================
// Mixed Pattern Parsing Tests
// =============================================================================

{
  // Test composite with ellipsis
  type CompositeEllipsis = ParsePattern<'(batch ...) -> batch ...'>;
  expectTypeOf<CompositeEllipsis>().toEqualTypeOf<{
    input: readonly [
      {
        type: 'composite';
        axes: readonly [{ type: 'simple'; name: 'batch' }, { type: 'ellipsis' }];
      },
    ];
    output: readonly [{ type: 'simple'; name: 'batch' }, { type: 'ellipsis' }];
  }>();
}

{
  // Test composite with singleton
  type CompositeSingleton = ParsePattern<'(h w 1) -> h w'>;
  expectTypeOf<CompositeSingleton>().toEqualTypeOf<{
    input: readonly [
      {
        type: 'composite';
        axes: readonly [
          { type: 'simple'; name: 'h' },
          { type: 'simple'; name: 'w' },
          { type: 'singleton' },
        ];
      },
    ];
    output: readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
  }>();
}

{
  // Test complex mixed pattern
  type ComplexMixed = ParsePattern<'(batch seq) embed ... 1 -> batch seq embed ... 1'>;
  expectTypeOf<ComplexMixed>().toEqualTypeOf<{
    input: readonly [
      {
        type: 'composite';
        axes: readonly [{ type: 'simple'; name: 'batch' }, { type: 'simple'; name: 'seq' }];
      },
      { type: 'simple'; name: 'embed' },
      { type: 'ellipsis' },
      { type: 'singleton' },
    ];
    output: readonly [
      { type: 'simple'; name: 'batch' },
      { type: 'simple'; name: 'seq' },
      { type: 'simple'; name: 'embed' },
      { type: 'ellipsis' },
      { type: 'singleton' },
    ];
  }>();
}

// =============================================================================
// Error Handling Tests
// =============================================================================

{
  // Test IsValidPattern utility
  expectTypeOf<IsValidPattern<'h w -> w h'>>().toEqualTypeOf<true>();
  expectTypeOf<IsValidPattern<'invalid pattern without arrow'>>().toEqualTypeOf<false>();
  expectTypeOf<IsValidPattern<'(unbalanced -> parens'>>().toEqualTypeOf<false>();
  expectTypeOf<IsValidPattern<'h w -> w h -> multiple arrows'>>().toEqualTypeOf<false>();
}

{
  // Test missing arrow error
  type MissingArrowError = ParsePattern<'a b c'>;
  expectTypeOf<MissingArrowError>().toEqualTypeOf<
    TypeParseError<"[Einops] Missing arrow operator '->': einops patterns must have input -> output format">
  >();

  type MissingArrowError2 = GetParseError<'a b c'>;
  expectTypeOf<MissingArrowError2>().toEqualTypeOf<"Missing arrow operator '->': einops patterns must have input -> output format">();
}

{
  // Test unbalanced parentheses error
  type UnbalancedParensError = ParsePattern<'(h w -> h w'>;
  expectTypeOf<UnbalancedParensError>().toEqualTypeOf<
    TypeParseError<"[Einops] Input parsing failed: Unbalanced parentheses: missing closing ')'">
  >();
}

{
  // Test invalid singleton error
  type InvalidSingletonError = ParsePattern<'1abc -> h'>;
  expectTypeOf<InvalidSingletonError>().toEqualTypeOf<
    TypeParseError<"[Einops] Input parsing failed: Singleton '1' must be followed by delimiter">
  >();
}

// =============================================================================
// Integration Tests with Complex Patterns
// =============================================================================

{
  // Test realistic einops patterns
  type ImageReshape = ParsePattern<'batch height width channels -> batch channels height width'>;
  expectTypeOf<ImageReshape>().toMatchTypeOf<{
    input: readonly TypeAxisPattern[];
    output: readonly TypeAxisPattern[];
  }>();
}

{
  // Test sequence modeling pattern
  type SequencePattern = ParsePattern<'(batch seq) embed -> batch seq embed'>;
  expectTypeOf<SequencePattern>().toMatchTypeOf<{
    input: readonly TypeAxisPattern[];
    output: readonly TypeAxisPattern[];
  }>();
}

{
  // Test attention pattern with ellipsis
  type AttentionPattern = ParsePattern<'batch heads ... -> batch ...'>;
  expectTypeOf<AttentionPattern>().toMatchTypeOf<{
    input: readonly TypeAxisPattern[];
    output: readonly TypeAxisPattern[];
  }>();
}

{
  // Test flatten pattern
  type FlattenPattern = ParsePattern<'(h w) c -> h w c'>;
  expectTypeOf<FlattenPattern>().toMatchTypeOf<{
    input: readonly TypeAxisPattern[];
    output: readonly TypeAxisPattern[];
  }>();
}

// =============================================================================
// Edge Case Tests
// =============================================================================

{
  // Test single character axes
  type SingleCharAxes = ParsePattern<'a b c -> c b a'>;
  expectTypeOf<SingleCharAxes>().toMatchTypeOf<{
    input: readonly TypeAxisPattern[];
    output: readonly TypeAxisPattern[];
  }>();
}

{
  // Test axes with numbers
  type AxesWithNumbers = ParsePattern<'axis1 axis2 -> axis2 axis1'>;
  expectTypeOf<AxesWithNumbers>().toMatchTypeOf<{
    input: readonly TypeAxisPattern[];
    output: readonly TypeAxisPattern[];
  }>();
}

{
  // Test axes with underscores
  type AxesWithUnderscores = ParsePattern<'batch_size seq_len -> seq_len batch_size'>;
  expectTypeOf<AxesWithUnderscores>().toMatchTypeOf<{
    input: readonly TypeAxisPattern[];
    output: readonly TypeAxisPattern[];
  }>();
}

{
  // Test whitespace variations
  type WhitespaceVariations = ParsePattern<'\t h \n w \r -> \t w \n h \r'>;
  expectTypeOf<WhitespaceVariations>().toMatchTypeOf<{
    input: readonly TypeAxisPattern[];
    output: readonly TypeAxisPattern[];
  }>();
}
