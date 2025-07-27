/**
 * Type tests for the einops scanner
 *
 * These tests validate that our token types and scanner interfaces work
 * correctly at compile time using expect-type.
 */

import { expectTypeOf } from 'expect-type';
import type {
  EinopsToken,
  AxisToken,
  ArrowToken,
  WhitespaceToken,
  LparenToken,
  RparenToken,
  EllipsisToken,
  SingletonToken,
  Position,
  TokenizeResult,
} from './types';
import { EinopsError, InvalidCharacterError, MalformedArrowError } from './types';
import { EinopsScanner, tokenize } from './scanner';

// =============================================================================
// Token Type Tests
// =============================================================================

{
  // Test that EinopsToken is properly discriminated by 'type' field
  const axisToken: AxisToken = {
    type: 'axis',
    name: 'test',
    position: { start: 0, end: 4 },
  };
  
  const arrowToken: ArrowToken = {
    type: 'arrow',
    position: { start: 0, end: 2 },
  };
  
  const whitespaceToken: WhitespaceToken = {
    type: 'whitespace',
    position: { start: 0, end: 1 },
  };

  const lparenToken: LparenToken = {
    type: 'lparen',
    position: { start: 0, end: 1 },
  };
  
  const rparenToken: RparenToken = {
    type: 'rparen', 
    position: { start: 4, end: 5 },
  };
  
  const ellipsisToken: EllipsisToken = {
    type: 'ellipsis',
    position: { start: 0, end: 3 },
  };
  
  const singletonToken: SingletonToken = {
    type: 'singleton',
    position: { start: 0, end: 1 },
  };

  // All should be assignable to EinopsToken
  expectTypeOf<typeof axisToken>().toMatchTypeOf<EinopsToken>();
  expectTypeOf<typeof arrowToken>().toMatchTypeOf<EinopsToken>();
  expectTypeOf<typeof whitespaceToken>().toMatchTypeOf<EinopsToken>();
  expectTypeOf<typeof lparenToken>().toMatchTypeOf<EinopsToken>();
  expectTypeOf<typeof rparenToken>().toMatchTypeOf<EinopsToken>();
  expectTypeOf<typeof ellipsisToken>().toMatchTypeOf<EinopsToken>();
  expectTypeOf<typeof singletonToken>().toMatchTypeOf<EinopsToken>();
}

{
  // Test token property types
  expectTypeOf<AxisToken['type']>().toEqualTypeOf<'axis'>();
  expectTypeOf<AxisToken['name']>().toEqualTypeOf<string>();
  expectTypeOf<AxisToken['position']>().toEqualTypeOf<Position>();
  
  expectTypeOf<ArrowToken['type']>().toEqualTypeOf<'arrow'>();
  expectTypeOf<ArrowToken['position']>().toEqualTypeOf<Position>();
  
  expectTypeOf<WhitespaceToken['type']>().toEqualTypeOf<'whitespace'>();
  expectTypeOf<WhitespaceToken['position']>().toEqualTypeOf<Position>();
  
  expectTypeOf<LparenToken['type']>().toEqualTypeOf<'lparen'>();
  expectTypeOf<LparenToken['position']>().toEqualTypeOf<Position>();
  
  expectTypeOf<RparenToken['type']>().toEqualTypeOf<'rparen'>();
  expectTypeOf<RparenToken['position']>().toEqualTypeOf<Position>();
  
  expectTypeOf<EllipsisToken['type']>().toEqualTypeOf<'ellipsis'>();
  expectTypeOf<EllipsisToken['position']>().toEqualTypeOf<Position>();
  
  expectTypeOf<SingletonToken['type']>().toEqualTypeOf<'singleton'>();
  expectTypeOf<SingletonToken['position']>().toEqualTypeOf<Position>();
}

// =============================================================================
// Position Type Tests
// =============================================================================

{
  const position: Position = { start: 0, end: 5 };
  
  expectTypeOf<typeof position.start>().toEqualTypeOf<number>();
  expectTypeOf<typeof position.end>().toEqualTypeOf<number>();
  
  // Should be readonly
  expectTypeOf<Position['start']>().toEqualTypeOf<number>();
  expectTypeOf<Position['end']>().toEqualTypeOf<number>();
}

// =============================================================================
// TokenizeResult Type Tests
// =============================================================================

{
  const result: TokenizeResult = {
    tokens: [],
    pattern: 'test',
  };
  
  expectTypeOf<typeof result.tokens>().toEqualTypeOf<readonly EinopsToken[]>();
  expectTypeOf<typeof result.pattern>().toEqualTypeOf<string>();
}

{
  const result = tokenize('a b');
  
  expectTypeOf<typeof result>().toMatchTypeOf<TokenizeResult>();
  expectTypeOf<typeof result.tokens>().toMatchTypeOf<readonly EinopsToken[]>();
  expectTypeOf<typeof result.pattern>().toEqualTypeOf<string>();
}

// =============================================================================
// Scanner Type Tests
// =============================================================================

{
  const scanner = new EinopsScanner('test');
  
  expectTypeOf<typeof scanner.lookahead>().toEqualTypeOf<string>();
  expectTypeOf<typeof scanner.nextLookahead>().toEqualTypeOf<string>();
  expectTypeOf<typeof scanner.position>().toEqualTypeOf<number>();
  expectTypeOf<typeof scanner.isAtEnd>().toEqualTypeOf<boolean>();
  expectTypeOf<typeof scanner.unscanned>().toEqualTypeOf<string>();
  expectTypeOf<typeof scanner.scanned>().toEqualTypeOf<string>();
  
  expectTypeOf<typeof scanner.shift>().toMatchTypeOf<() => string>();
  expectTypeOf<typeof scanner.jumpForward>().toMatchTypeOf<(count: number) => void>();
  expectTypeOf<typeof scanner.tokenize>().toMatchTypeOf<() => TokenizeResult>();
}

// =============================================================================
// Error Type Tests
// =============================================================================

{
  const baseError = new EinopsError('test', 'pattern');
  const invalidCharError = new InvalidCharacterError('x', 'pattern', { start: 0, end: 1 });
  const malformedArrowError = new MalformedArrowError('pattern', { start: 0, end: 1 });
  
  expectTypeOf<typeof baseError>().toMatchTypeOf<Error>();
  expectTypeOf<typeof invalidCharError>().toMatchTypeOf<EinopsError>();
  expectTypeOf<typeof malformedArrowError>().toMatchTypeOf<EinopsError>();
  
  expectTypeOf<typeof baseError.pattern>().toEqualTypeOf<string>();
  expectTypeOf<typeof baseError.position>().toEqualTypeOf<Position | undefined>();
}

{
  const error = new EinopsError('test', 'pattern', { start: 0, end: 1 });
  
  expectTypeOf<typeof error.name>().toEqualTypeOf<string>();
  expectTypeOf<typeof error.message>().toEqualTypeOf<string>();
  expectTypeOf<typeof error.pattern>().toEqualTypeOf<string>();
  expectTypeOf<typeof error.position>().toEqualTypeOf<Position | undefined>();
}

// =============================================================================
// Type Inference Tests
// =============================================================================

{
  const tokens: EinopsToken[] = [
    { type: 'lparen', position: { start: 0, end: 1 } },
    { type: 'axis', name: 'h', position: { start: 1, end: 2 } },
    { type: 'whitespace', position: { start: 2, end: 3 } },
    { type: 'axis', name: 'w', position: { start: 3, end: 4 } },
    { type: 'rparen', position: { start: 4, end: 5 } },
    { type: 'whitespace', position: { start: 5, end: 6 } },
    { type: 'axis', name: 'c', position: { start: 6, end: 7 } },
  ];
  
  // Should be able to filter and narrow types
  const axisTokens = tokens.filter((t): t is AxisToken => t.type === 'axis');
  expectTypeOf<typeof axisTokens>().toEqualTypeOf<AxisToken[]>();
  
  const parenTokens = tokens.filter((t): t is LparenToken | RparenToken => 
    t.type === 'lparen' || t.type === 'rparen'
  );
  expectTypeOf<typeof parenTokens>().toEqualTypeOf<(LparenToken | RparenToken)[]>();
  
  const firstAxis = axisTokens[0];
  if (firstAxis) {
    expectTypeOf<typeof firstAxis.name>().toEqualTypeOf<string>();
  }
}

{
  const tokens: readonly EinopsToken[] = [];
  
  const axisNames = tokens
    .filter((t): t is AxisToken => t.type === 'axis')
    .map(t => t.name);
  
  expectTypeOf<typeof axisNames>().toEqualTypeOf<string[]>();
}

// =============================================================================
// New Token Type Tests
// =============================================================================

{
  // Test filtering with new tokens
  const tokens: readonly EinopsToken[] = [];
  
  const ellipsisTokens = tokens.filter((t): t is EllipsisToken => t.type === 'ellipsis');
  expectTypeOf<typeof ellipsisTokens>().toEqualTypeOf<EllipsisToken[]>();
  
  const singletonTokens = tokens.filter((t): t is SingletonToken => t.type === 'singleton');
  expectTypeOf<typeof singletonTokens>().toEqualTypeOf<SingletonToken[]>();
  
  const specialTokens = tokens.filter((t): t is EllipsisToken | SingletonToken => 
    t.type === 'ellipsis' || t.type === 'singleton'
  );
  expectTypeOf<typeof specialTokens>().toEqualTypeOf<(EllipsisToken | SingletonToken)[]>();
}