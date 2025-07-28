/**
 * Type definitions for einops pattern parsing
 *
 * This module defines the basic types used for tokenizing and parsing
 * einops patterns, including token types and position tracking.
 */

// =============================================================================
// Position Tracking
// =============================================================================

/**
 * Represents a position in the source pattern string
 */
export interface Position {
  /** Start index (inclusive) */
  readonly start: number;
  /** End index (exclusive) */
  readonly end: number;
}

// =============================================================================
// Token Types
// =============================================================================

/**
 * An axis name token representing a tensor dimension
 * Examples: "a", "batch", "height_1"
 */
export interface AxisToken {
  readonly type: 'axis';
  readonly name: string;
  readonly position: Position;
}

/**
 * The arrow operator token separating input and output patterns
 * Matches: "->"
 */
export interface ArrowToken {
  readonly type: 'arrow';
  readonly position: Position;
}

/**
 * A whitespace token for pattern separation
 * Matches one or more whitespace characters
 */
export interface WhitespaceToken {
  readonly type: 'whitespace';
  readonly position: Position;
}

/**
 * Left parenthesis token for composite dimension start
 * Matches: "("
 */
export interface LparenToken {
  readonly type: 'lparen';
  readonly position: Position;
}

/**
 * Right parenthesis token for composite dimension end
 * Matches: ")"
 */
export interface RparenToken {
  readonly type: 'rparen';
  readonly position: Position;
}

/**
 * Ellipsis token for variable dimensions
 * Matches: "..."
 */
export interface EllipsisToken {
  readonly type: 'ellipsis';
  readonly position: Position;
}

/**
 * Singleton token for unit dimensions
 * Matches: "1"
 */
export interface SingletonToken {
  readonly type: 'singleton';
  readonly position: Position;
}

/**
 * Union type representing all possible einops tokens
 */
export type EinopsToken =
  | AxisToken
  | ArrowToken
  | WhitespaceToken
  | LparenToken
  | RparenToken
  | EllipsisToken
  | SingletonToken;

// =============================================================================
// Tokenization Result
// =============================================================================

/**
 * Result of tokenizing an einops pattern
 */
export interface TokenizeResult {
  /** Array of tokens in order */
  readonly tokens: readonly EinopsToken[];
  /** Original pattern string */
  readonly pattern: string;
}

// =============================================================================
// Error Types
// =============================================================================

/**
 * Base error for einops parsing issues
 */
export class EinopsError extends Error {
  constructor(
    message: string,
    public readonly pattern: string,
    public readonly position?: Position,
  ) {
    super(EinopsError.formatMessage(message, pattern, position));
    this.name = 'EinopsError';
  }

  private static formatMessage(message: string, pattern: string, position?: Position): string {
    if (position) {
      const pointer =
        ' '.repeat(position.start) + '^'.repeat(Math.max(1, position.end - position.start));
      return `${message}\n  ${pattern}\n  ${pointer}`;
    }
    return `${message} in pattern: "${pattern}"`;
  }
}

/**
 * Error thrown when encountering invalid characters during tokenization
 */
export class InvalidCharacterError extends EinopsError {
  constructor(char: string, pattern: string, position: Position) {
    super(`Invalid character '${char}'`, pattern, position);
    this.name = 'InvalidCharacterError';
  }
}

/**
 * Error thrown when encountering malformed arrow operator
 */
export class MalformedArrowError extends EinopsError {
  constructor(pattern: string, position: Position) {
    super('Malformed arrow operator (expected "->")', pattern, position);
    this.name = 'MalformedArrowError';
  }
}

/**
 * Error thrown for unmatched parentheses in patterns
 */
export class UnmatchedParenthesesError extends EinopsError {
  constructor(char: '(' | ')', pattern: string, position: Position) {
    super(`Unmatched ${char === '(' ? 'opening' : 'closing'} parenthesis`, pattern, position);
    this.name = 'UnmatchedParenthesesError';
  }
}
