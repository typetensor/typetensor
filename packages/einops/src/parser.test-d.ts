/**
 * Type tests for the einops parser
 *
 * These tests validate that our parser functions work correctly
 * at compile time and maintain proper type safety.
 */

import { expectTypeOf } from 'expect-type';
import type {
  EinopsAST,
  AxisPattern,
  SimpleAxis,
  CompositeAxis,
  EllipsisAxis,
  SingletonAxis,
} from './ast';
import type { EinopsToken } from './types';
import {
  parseTokens,
  ParseError,
  UnexpectedTokenError,
  UnbalancedParenthesesError,
  MissingArrowError,
  MultipleArrowError,
} from './parser';
import { tokenize } from './scanner';

// =============================================================================
// Core Parser Function Type Tests
// =============================================================================

{
  // Test parseTokens function signature
  expectTypeOf<typeof parseTokens>().parameter(0).toEqualTypeOf<readonly EinopsToken[]>();
  expectTypeOf<typeof parseTokens>().returns.toEqualTypeOf<EinopsAST>();
}

{
  // Test parseTokens with actual token arrays
  const tokens = tokenize('h w -> w h').tokens;
  const ast = parseTokens(tokens);

  expectTypeOf<typeof ast>().toEqualTypeOf<EinopsAST>();
  expectTypeOf<typeof ast.input>().toEqualTypeOf<readonly AxisPattern[]>();
  expectTypeOf<typeof ast.output>().toEqualTypeOf<readonly AxisPattern[]>();
  expectTypeOf<typeof ast.metadata>().toMatchTypeOf<{
    readonly originalPattern: string;
    readonly arrowPosition: { readonly start: number; readonly end: number };
    readonly inputTokenCount: number;
    readonly outputTokenCount: number;
  }>();
}

// =============================================================================
// Error Type Tests
// =============================================================================

{
  // Test ParseError hierarchy
  expectTypeOf<ParseError>().toMatchTypeOf<Error>();
  expectTypeOf<UnexpectedTokenError>().toMatchTypeOf<ParseError>();
  expectTypeOf<UnbalancedParenthesesError>().toMatchTypeOf<ParseError>();
  expectTypeOf<MissingArrowError>().toMatchTypeOf<ParseError>();
  expectTypeOf<MultipleArrowError>().toMatchTypeOf<ParseError>();
}

{
  // Test error constructors
  const position = { start: 0, end: 1 } as const;
  const context = { test: 'value' } as const;

  // Test ParseError constructor works with all parameter combinations
  new ParseError('test message');
  new ParseError('test message', position);
  new ParseError('test message', position, context);
  new ParseError('test message', undefined, context);

  const error = new ParseError('test message', position, context);
  expectTypeOf<typeof error.position>().toEqualTypeOf<
    { readonly start: number; readonly end: number } | undefined
  >();
  expectTypeOf<typeof error.context>().toEqualTypeOf<Record<string, unknown> | undefined>();
}

// =============================================================================
// AST Structure Type Tests
// =============================================================================

{
  // Test that parser returns properly typed AST patterns
  const tokens = tokenize('(h w) c -> h w c').tokens;
  const ast = parseTokens(tokens);

  // Input should contain CompositeAxis and SimpleAxis
  expectTypeOf<(typeof ast.input)[0]>().toEqualTypeOf<AxisPattern>();
  expectTypeOf<(typeof ast.input)[1]>().toEqualTypeOf<AxisPattern>();

  // Output should contain SimpleAxis patterns
  expectTypeOf<(typeof ast.output)[0]>().toEqualTypeOf<AxisPattern>();
  expectTypeOf<(typeof ast.output)[1]>().toEqualTypeOf<AxisPattern>();
  expectTypeOf<(typeof ast.output)[2]>().toEqualTypeOf<AxisPattern>();
}

{
  // Test ellipsis patterns
  const tokens = tokenize('batch ... -> ...').tokens;
  const ast = parseTokens(tokens);

  expectTypeOf<typeof ast.input>().toEqualTypeOf<readonly AxisPattern[]>();
  expectTypeOf<typeof ast.output>().toEqualTypeOf<readonly AxisPattern[]>();

  // Each pattern should be in the AxisPattern union
  expectTypeOf<(typeof ast.input)[0]>().toEqualTypeOf<AxisPattern>();
  expectTypeOf<(typeof ast.input)[1]>().toEqualTypeOf<AxisPattern>();
  expectTypeOf<(typeof ast.output)[0]>().toEqualTypeOf<AxisPattern>();
}

{
  // Test singleton patterns
  const tokens = tokenize('h w 1 -> h w').tokens;
  const ast = parseTokens(tokens);

  expectTypeOf<typeof ast.input>().toEqualTypeOf<readonly AxisPattern[]>();
  expectTypeOf<typeof ast.output>().toEqualTypeOf<readonly AxisPattern[]>();

  expectTypeOf<(typeof ast.input)[0]>().toEqualTypeOf<AxisPattern>();
  expectTypeOf<(typeof ast.input)[1]>().toEqualTypeOf<AxisPattern>();
  expectTypeOf<(typeof ast.input)[2]>().toEqualTypeOf<AxisPattern>();
}

// =============================================================================
// Pattern Type Discrimination Tests
// =============================================================================

{
  // Test that AxisPattern union maintains discriminated properties
  const tokens = tokenize('(h w) ... 1 a -> h w').tokens;
  const ast = parseTokens(tokens);

  // Test that ast has the expected structure
  expectTypeOf<typeof ast>().toEqualTypeOf<EinopsAST>();
  expectTypeOf<typeof ast.input>().toEqualTypeOf<readonly AxisPattern[]>();

  // Each pattern should have the discriminated type property
  expectTypeOf<AxisPattern>()
    .toHaveProperty('type')
    .toEqualTypeOf<'simple' | 'composite' | 'ellipsis' | 'singleton'>();
  expectTypeOf<AxisPattern>()
    .toHaveProperty('position')
    .toEqualTypeOf<{ readonly start: number; readonly end: number }>();

  // Test specific pattern types
  expectTypeOf<SimpleAxis>().toHaveProperty('type').toEqualTypeOf<'simple'>();
  expectTypeOf<SimpleAxis>().toHaveProperty('name').toEqualTypeOf<string>();

  expectTypeOf<CompositeAxis>().toHaveProperty('type').toEqualTypeOf<'composite'>();
  expectTypeOf<CompositeAxis>().toHaveProperty('axes').toEqualTypeOf<readonly AxisPattern[]>();

  expectTypeOf<EllipsisAxis>().toHaveProperty('type').toEqualTypeOf<'ellipsis'>();

  expectTypeOf<SingletonAxis>().toHaveProperty('type').toEqualTypeOf<'singleton'>();
}

// =============================================================================
// Metadata Type Tests
// =============================================================================

{
  // Test metadata structure
  const tokens = tokenize('h w -> w h').tokens;
  const ast = parseTokens(tokens);

  expectTypeOf<typeof ast.metadata.originalPattern>().toEqualTypeOf<string>();
  expectTypeOf<typeof ast.metadata.arrowPosition>().toEqualTypeOf<{
    readonly start: number;
    readonly end: number;
  }>();
  expectTypeOf<typeof ast.metadata.inputTokenCount>().toEqualTypeOf<number>();
  expectTypeOf<typeof ast.metadata.outputTokenCount>().toEqualTypeOf<number>();
}

// =============================================================================
// Readonly Constraint Tests
// =============================================================================

{
  // Test that all AST properties are readonly
  const tokens = tokenize('h w -> w h').tokens;
  const ast = parseTokens(tokens);

  expectTypeOf<typeof ast.input>().toEqualTypeOf<readonly AxisPattern[]>();
  expectTypeOf<typeof ast.output>().toEqualTypeOf<readonly AxisPattern[]>();

  // Test that individual patterns maintain readonly properties
  expectTypeOf<SimpleAxis['position']>().toEqualTypeOf<{
    readonly start: number;
    readonly end: number;
  }>();
  expectTypeOf<CompositeAxis['axes']>().toEqualTypeOf<readonly AxisPattern[]>();
}

// =============================================================================
// Integration Type Tests
// =============================================================================

{
  // Test integration with scanner types
  const scanResult = tokenize('(h w) -> h w');
  expectTypeOf<typeof scanResult.tokens>().toEqualTypeOf<readonly EinopsToken[]>();

  const ast = parseTokens(scanResult.tokens);
  expectTypeOf<typeof ast>().toEqualTypeOf<EinopsAST>();
}

{
  // Test complex nested patterns maintain proper typing
  const tokens = tokenize('((h w) c) d -> h w c d').tokens;
  const ast = parseTokens(tokens);

  expectTypeOf<typeof ast.input>().toEqualTypeOf<readonly AxisPattern[]>();
  expectTypeOf<typeof ast.output>().toEqualTypeOf<readonly AxisPattern[]>();

  // Test that we can access nested patterns with proper typing
  expectTypeOf<(typeof ast.input)[0]>().toEqualTypeOf<AxisPattern>();
  expectTypeOf<(typeof ast.input)[1]>().toEqualTypeOf<AxisPattern>();
}

// =============================================================================
// Error Handling Type Tests
// =============================================================================

{
  // Test that error functions maintain proper types
  expectTypeOf<ParseError['position']>().toEqualTypeOf<
    { readonly start: number; readonly end: number } | undefined
  >();
  expectTypeOf<ParseError['context']>().toEqualTypeOf<Record<string, unknown> | undefined>();

  // Test other error constructors work with expected parameters
  const sampleToken: EinopsToken = { type: 'axis', name: 'test', position: { start: 0, end: 4 } };
  const testPosition = { start: 5, end: 10 } as const;
  
  const unexpectedError1 = new UnexpectedTokenError('axis', sampleToken);
  new UnexpectedTokenError('axis', sampleToken, 'test pattern');
  
  const unbalancedError = new UnbalancedParenthesesError(testPosition, 'test pattern');
  const missingArrowError = new MissingArrowError('test pattern');
  const multipleArrowError = new MultipleArrowError(testPosition, 'test pattern');
  
  // Test that they all extend ParseError
  expectTypeOf<typeof unexpectedError1>().toMatchTypeOf<ParseError>();
  expectTypeOf<typeof unbalancedError>().toMatchTypeOf<ParseError>();
  expectTypeOf<typeof missingArrowError>().toMatchTypeOf<ParseError>();
  expectTypeOf<typeof multipleArrowError>().toMatchTypeOf<ParseError>();
}

// =============================================================================
// Type-level Pattern Validation Tests
// =============================================================================

{
  // Test that parseTokens function works with any valid token array
  const emptyTokens: readonly EinopsToken[] = [];
  const simpleTokens = tokenize('a -> b').tokens;
  const complexTokens = tokenize('(batch seq) embed ... 1 -> batch seq embed').tokens;

  // All should have the same return type
  expectTypeOf<typeof parseTokens>().parameter(0).toEqualTypeOf<readonly EinopsToken[]>();
  expectTypeOf<typeof parseTokens>().returns.toEqualTypeOf<EinopsAST>();

  // Function should accept any readonly array of EinopsToken
  expectTypeOf<typeof parseTokens>().toBeCallableWith(emptyTokens);
  expectTypeOf<typeof parseTokens>().toBeCallableWith(simpleTokens);
  expectTypeOf<typeof parseTokens>().toBeCallableWith(complexTokens);
}
