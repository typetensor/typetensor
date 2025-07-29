/**
 * Type-level parser for einops patterns using template literal types
 *
 * This module provides compile-time pattern validation using TypeScript's
 * template literal type system, heavily inspired by ArkType's approach to
 * string parsing at the type level.
 *
 * The parser converts string patterns like "h w -> w h" into type-level
 * AST structures that match our runtime AST, enabling compile-time validation
 * and error reporting.
 */

import type { Add, Subtract } from 'ts-arithmetic';

// =============================================================================
// String Parsing Utilities
// =============================================================================

/**
 * Extract the first character from a string type
 */
export type FirstChar<S extends string> = S extends `${infer Head}${string}` ? Head : '';

/**
 * Extract all characters after the first from a string type
 */
export type RestChars<S extends string> = S extends `${string}${infer Tail}` ? Tail : '';

/**
 * Extract the last character from a string type
 */
export type LastChar<S extends string> = S extends `${infer Head}${infer Tail}`
  ? Tail extends ''
    ? Head
    : LastChar<Tail>
  : '';

/**
 * Check if a string is empty
 */
export type IsEmpty<S extends string> = S extends '' ? true : false;

/**
 * Check if a character is whitespace
 */
export type IsWhitespace<C extends string> = C extends ' ' | '\t' | '\n' | '\r' ? true : false;

/**
 * Check if a character is a valid axis name character (letter, digit, underscore)
 */
export type IsAxisChar<C extends string> = C extends
  | 'a'
  | 'b'
  | 'c'
  | 'd'
  | 'e'
  | 'f'
  | 'g'
  | 'h'
  | 'i'
  | 'j'
  | 'k'
  | 'l'
  | 'm'
  | 'n'
  | 'o'
  | 'p'
  | 'q'
  | 'r'
  | 's'
  | 't'
  | 'u'
  | 'v'
  | 'w'
  | 'x'
  | 'y'
  | 'z'
  | 'A'
  | 'B'
  | 'C'
  | 'D'
  | 'E'
  | 'F'
  | 'G'
  | 'H'
  | 'I'
  | 'J'
  | 'K'
  | 'L'
  | 'M'
  | 'N'
  | 'O'
  | 'P'
  | 'Q'
  | 'R'
  | 'S'
  | 'T'
  | 'U'
  | 'V'
  | 'W'
  | 'X'
  | 'Y'
  | 'Z'
  | '0'
  | '1'
  | '2'
  | '3'
  | '4'
  | '5'
  | '6'
  | '7'
  | '8'
  | '9'
  | '_'
  ? true
  : false;

/**
 * Shift characters until a terminator is found, returns [scanned, remaining]
 */
export type ShiftUntil<
  S extends string,
  Terminator extends string,
  Scanned extends string = '',
> = S extends `${infer Head}${infer Tail}`
  ? Head extends Terminator
    ? [Scanned, S]
    : ShiftUntil<Tail, Terminator, `${Scanned}${Head}`>
  : [Scanned, ''];

/**
 * Skip whitespace characters and return the trimmed string
 */
export type SkipWhitespace<S extends string> = S extends `${infer Head}${infer Tail}`
  ? IsWhitespace<Head> extends true
    ? SkipWhitespace<Tail>
    : S
  : S;

/**
 * Extract an axis name (sequence of valid axis characters)
 */
export type ExtractAxisName<S extends string> = ShiftUntil<
  S,
  ' ' | '\t' | '\n' | '\r' | '(' | ')' | '-' | '.'
>;

/**
 * Find matching closing parenthesis for an opening parenthesis
 * Returns [content between parens, remaining after closing paren]
 */
export type FindMatchingParen<
  S extends string,
  Depth extends number = 1,
  Content extends string = '',
> = S extends `${infer Head}${infer Tail}`
  ? Head extends '('
    ? FindMatchingParen<Tail, Add<Depth, 1>, `${Content}${Head}`>
    : Head extends ')'
      ? Depth extends 1
        ? [Content, Tail] // Found matching paren
        : FindMatchingParen<Tail, Subtract<Depth, 1>, `${Content}${Head}`>
      : FindMatchingParen<Tail, Depth, `${Content}${Head}`>
  : ParseError<"Unbalanced parentheses: missing closing ')'">;

// =============================================================================
// Type-Level AST Types
// =============================================================================

/**
 * Type-level representation of axis patterns matching runtime AST
 */
export type TypeAxisPattern =
  | TypeSimpleAxis
  | TypeCompositeAxis
  | TypeEllipsisAxis
  | TypeSingletonAxis;

export interface TypeSimpleAxis {
  readonly type: 'simple';
  readonly name: string;
}

export interface TypeCompositeAxis {
  readonly type: 'composite';
  readonly axes: readonly TypeAxisPattern[];
}

export interface TypeEllipsisAxis {
  readonly type: 'ellipsis';
}

export interface TypeSingletonAxis {
  readonly type: 'singleton';
}

/**
 * Type-level AST structure matching runtime EinopsAST
 */
export interface TypeEinopsAST {
  readonly input: readonly TypeAxisPattern[];
  readonly output: readonly TypeAxisPattern[];
}

// =============================================================================
// Parse State Management
// =============================================================================

/**
 * Parser state for tracking position and context during parsing
 */
export interface ParseState {
  readonly currentSide: 'input' | 'output';
  readonly nestingDepth: number;
  readonly seenAxes: readonly string[];
  readonly errors: readonly string[];
}

/**
 * Initial parse state
 */
export interface InitialParseState {
  readonly currentSide: 'input';
  readonly nestingDepth: 0;
  readonly seenAxes: readonly [];
  readonly errors: readonly [];
}

// =============================================================================
// Error Types
// =============================================================================

/**
 * Type-level parse error with helpful messages
 */
export interface TypeParseError<Message extends string> {
  readonly __error: 'EinopsParseError';
  readonly message: Message;
}

/**
 * Create a parse error with context
 */
export type ParseError<Message extends string> = TypeParseError<`[Einops] ${Message}`>;

// =============================================================================
// Core Parser Implementation
// =============================================================================

/**
 * Parse a single axis pattern from the input string
 */
export type ParseAxisPattern<S extends string> =
  SkipWhitespace<S> extends infer Trimmed
    ? Trimmed extends string
      ? FirstChar<Trimmed> extends '('
        ? ParseCompositeAxis<Trimmed>
        : FirstChar<Trimmed> extends '.'
          ? ParseEllipsisAxis<Trimmed>
          : FirstChar<Trimmed> extends '1'
            ? ParseSingletonAxis<Trimmed>
            : ParseSimpleAxis<Trimmed>
      : ParseError<'Invalid input'>
    : ParseError<'Invalid input'>;

/**
 * Parse a simple axis name
 */
export type ParseSimpleAxis<S extends string> =
  ExtractAxisName<S> extends [infer Name, infer Remaining]
    ? Name extends string
      ? Remaining extends string
        ? Name extends ''
          ? ParseError<'Empty axis name'>
          : {
              pattern: { type: 'simple'; name: Name };
              remaining: Remaining;
            }
        : ParseError<'Invalid remaining string'>
      : ParseError<'Invalid axis name'>
    : ParseError<'Failed to extract axis name'>;

/**
 * Parse a composite axis (parenthesized group)
 */
export type ParseCompositeAxis<S extends string> = S extends `(${string}`
  ? FindMatchingParen<S extends `(${infer Rest}` ? Rest : never> extends [
      infer Inner,
      infer Remaining,
    ]
    ? Inner extends string
      ? Remaining extends string
        ? ParseAxisList<Inner> extends {
            patterns: infer Patterns;
            remaining: infer InnerRemaining;
          }
          ? InnerRemaining extends ''
            ? Patterns extends readonly TypeAxisPattern[]
              ? {
                  pattern: { type: 'composite'; axes: Patterns };
                  remaining: Remaining;
                }
              : ParseError<'Invalid composite patterns'>
            : ParseError<'Invalid composite inner patterns - unparsed content'>
          : ParseAxisList<Inner> extends ParseError<infer Message>
            ? ParseError<`Invalid composite inner patterns: ${Message}`>
            : ParseError<'Invalid composite inner patterns'>
        : ParseError<'Invalid remaining after composite'>
      : ParseError<'Invalid inner content'>
    : FindMatchingParen<S extends `(${infer Rest}` ? Rest : never> extends ParseError<infer Message>
      ? ParseError<Message>
      : ParseError<'Failed to find matching parenthesis'>
  : ParseError<"Expected opening parenthesis '('">;

/**
 * Parse an ellipsis pattern
 */
export type ParseEllipsisAxis<S extends string> = S extends `...${infer Remaining}`
  ? {
      pattern: { type: 'ellipsis' };
      remaining: Remaining;
    }
  : ParseError<"Expected ellipsis '...'">;

/**
 * Parse a singleton pattern
 */
export type ParseSingletonAxis<S extends string> = S extends `1${infer Remaining}`
  ? FirstChar<Remaining> extends '' | ' ' | '\t' | '\n' | '\r' | '(' | ')' | '-'
    ? {
        pattern: { type: 'singleton' };
        remaining: Remaining;
      }
    : ParseError<"Singleton '1' must be followed by delimiter">
  : ParseError<"Expected singleton '1'">;

/**
 * Parse a list of axis patterns separated by whitespace
 */
export type ParseAxisList<
  S extends string,
  Patterns extends readonly TypeAxisPattern[] = readonly [],
> =
  SkipWhitespace<S> extends infer Trimmed
    ? Trimmed extends string
      ? IsEmpty<Trimmed> extends true
        ? { patterns: Patterns; remaining: '' }
        : ParseAxisPattern<Trimmed> extends {
              pattern: infer Pattern;
              remaining: infer Remaining;
            }
          ? Pattern extends TypeAxisPattern
            ? Remaining extends string
              ? ParseAxisList<Remaining, readonly [...Patterns, Pattern]>
              : ParseError<'Invalid remaining string in axis list'>
            : ParseError<'Invalid pattern in axis list'>
          : ParseAxisPattern<Trimmed> extends ParseError<infer Message>
            ? ParseError<Message>
            : ParseError<'Failed to parse axis pattern'>
      : ParseError<'Invalid trimmed string'>
    : ParseError<'Failed to trim whitespace'>;

/**
 * Split pattern at arrow operator
 */
export type SplitAtArrow<S extends string> = S extends `${infer Before}->${infer After}`
  ? { input: Before; output: After }
  : ParseError<"Missing arrow operator '->': einops patterns must have input -> output format">;

/**
 * Main pattern parser - converts string to type-level AST
 */
export type ParseEinopsPattern<Pattern extends string> =
  SplitAtArrow<Pattern> extends {
    input: infer InputStr;
    output: infer OutputStr;
  }
    ? InputStr extends string
      ? OutputStr extends string
        ? ParseAxisList<InputStr> extends {
            patterns: infer InputPatterns;
            remaining: '';
          }
          ? ParseAxisList<OutputStr> extends {
              patterns: infer OutputPatterns;
              remaining: '';
            }
            ? InputPatterns extends readonly TypeAxisPattern[]
              ? OutputPatterns extends readonly TypeAxisPattern[]
                ? {
                    input: InputPatterns;
                    output: OutputPatterns;
                  }
                : ParseError<'Invalid output patterns'>
              : ParseError<'Invalid input patterns'>
            : ParseAxisList<OutputStr> extends ParseError<infer Message>
              ? ParseError<`Output parsing failed: ${Message}`>
              : ParseError<'Failed to parse output section'>
          : ParseAxisList<InputStr> extends ParseError<infer Message>
            ? ParseError<`Input parsing failed: ${Message}`>
            : ParseError<'Failed to parse input section'>
        : ParseError<'Invalid output string'>
      : ParseError<'Invalid input string'>
    : SplitAtArrow<Pattern> extends ParseError<infer Message>
      ? ParseError<Message>
      : ParseError<'Failed to split pattern at arrow'>;

// =============================================================================
// Helper Types for Pattern Analysis
// =============================================================================

/**
 * Extract all axis names from a list of patterns
 */
export type ExtractAxisNames<Patterns extends readonly TypeAxisPattern[]> = {
  [K in keyof Patterns]: Patterns[K] extends TypeSimpleAxis
    ? Patterns[K]['name']
    : Patterns[K] extends TypeCompositeAxis
      ? ExtractAxisNames<Patterns[K]['axes']>
      : never;
}[number];

/**
 * Check if patterns contain an ellipsis
 */
export type HasEllipsis<Patterns extends readonly TypeAxisPattern[]> = {
  [K in keyof Patterns]: Patterns[K] extends TypeEllipsisAxis ? true : never;
}[number] extends never
  ? false
  : true;

/**
 * Count the number of simple axes in patterns
 */
export type CountSimpleAxes<
  Patterns extends readonly TypeAxisPattern[],
  Count extends number = 0,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? CountSimpleAxes<Tail, Count> // TODO: Increment count
      : Count
    : Head extends TypeCompositeAxis
      ? Tail extends readonly TypeAxisPattern[]
        ? CountSimpleAxes<Tail, Count> // TODO: Add composite count
        : Count
      : Tail extends readonly TypeAxisPattern[]
        ? CountSimpleAxes<Tail, Count>
        : Count
  : Count;

// =============================================================================
// Public API
// =============================================================================

/**
 * Parse an einops pattern at the type level
 *
 * @example
 * type Result = ParsePattern<"h w -> w h">
 * // Result = { input: [{ type: 'simple', name: 'h' }, { type: 'simple', name: 'w' }], output: [...] }
 */
export type ParsePattern<Pattern extends string> = ParseEinopsPattern<Pattern>;

/**
 * Check if a pattern is valid (doesn't contain parse errors)
 */
export type IsValidPattern<Pattern extends string> =
  ParsePattern<Pattern> extends ParseError<string> ? false : true;

/**
 * Get parse error message if pattern is invalid
 */
export type GetParseError<Pattern extends string> =
  ParsePattern<Pattern> extends ParseError<infer Message> ? Message : never;

// Re-export validation utilities
export type { ValidatePattern as ValidateEinopsPattern } from './validation';
