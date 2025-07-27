/**
 * Scanner for einops pattern parsing
 *
 * This module provides a character-by-character scanner for parsing einops
 * patterns, adapted from ArkType's Scanner architecture but specialized
 * for einops syntax.
 */

import type {
  EinopsToken,
  AxisToken,
  ArrowToken,
  WhitespaceToken,
  LparenToken,
  RparenToken,
  EllipsisToken,
  SingletonToken,
  TokenizeResult,
} from './types';
import { EinopsError, InvalidCharacterError, MalformedArrowError } from './types';

// =============================================================================
// Character Classification
// =============================================================================

/**
 * Characters that are considered whitespace in einops patterns
 */
const WHITESPACE_CHARS = new Set([' ', '\t', '\n', '\r']);

/**
 * Characters that are valid for starting an axis name
 */
const AXIS_START_CHARS = /^[a-zA-Z_]$/;

/**
 * Characters that are valid for continuing an axis name
 */
const AXIS_CONTINUE_CHARS = /^[a-zA-Z0-9_]$/;


// =============================================================================
// Scanner Class
// =============================================================================

/**
 * Scanner for tokenizing einops patterns character by character
 *
 * Based on ArkType's Scanner but specialized for einops syntax:
 * - Axis names (letters, numbers, underscores)
 * - Arrow operator (->)
 * - Whitespace as significant delimiters
 */
export class EinopsScanner {
  private readonly chars: readonly string[];
  private readonly pattern: string;
  private index: number = 0;

  constructor(pattern: string) {
    this.pattern = pattern;
    this.chars = [...pattern];
  }

  // =============================================================================
  // Basic Scanner Interface
  // =============================================================================

  /**
   * Get the current character without advancing
   */
  get lookahead(): string {
    return this.chars[this.index] ?? '';
  }

  /**
   * Get the next character without advancing
   */
  get nextLookahead(): string {
    return this.chars[this.index + 1] ?? '';
  }

  /**
   * Get the current position in the pattern
   */
  get position(): number {
    return this.index;
  }

  /**
   * Check if we're at the end of the pattern
   */
  get isAtEnd(): boolean {
    return this.index >= this.chars.length;
  }

  /**
   * Get the remaining unscanned portion of the pattern
   */
  get unscanned(): string {
    return this.chars.slice(this.index).join('');
  }

  /**
   * Get the already scanned portion of the pattern
   */
  get scanned(): string {
    return this.chars.slice(0, this.index).join('');
  }

  /**
   * Advance scanner by one character and return the consumed character
   */
  shift(): string {
    return this.chars[this.index++] ?? '';
  }

  /**
   * Advance scanner by multiple characters
   */
  jumpForward(count: number): void {
    this.index += count;
  }

  // =============================================================================
  // Einops-Specific Scanning
  // =============================================================================

  /**
   * Check if current character is whitespace
   */
  private isWhitespace(): boolean {
    return WHITESPACE_CHARS.has(this.lookahead);
  }

  /**
   * Check if current character can start an axis name
   */
  private canStartAxis(): boolean {
    return AXIS_START_CHARS.test(this.lookahead);
  }

  /**
   * Check if current character can continue an axis name
   */
  private canContinueAxis(): boolean {
    return AXIS_CONTINUE_CHARS.test(this.lookahead);
  }

  /**
   * Scan whitespace and return a whitespace token
   */
  private scanWhitespace(): WhitespaceToken {
    const start = this.index;

    while (!this.isAtEnd && this.isWhitespace()) {
      this.shift();
    }

    return {
      type: 'whitespace',
      position: { start, end: this.index },
    };
  }

  /**
   * Scan an axis name and return an axis token
   */
  private scanAxis(): AxisToken {
    const start = this.index;
    let name = '';

    // First character must be letter or underscore
    if (!this.canStartAxis()) {
      throw new InvalidCharacterError(this.lookahead, this.pattern, {
        start: this.index,
        end: this.index + 1,
      });
    }

    name += this.shift();

    // Continue with letters, numbers, or underscores
    while (!this.isAtEnd && this.canContinueAxis()) {
      name += this.shift();
    }

    return {
      type: 'axis',
      name,
      position: { start, end: this.index },
    };
  }

  /**
   * Scan arrow operator and return an arrow token
   */
  private scanArrow(): ArrowToken {
    const start = this.index;

    // Must be exactly "->"
    if (this.lookahead === '-' && this.nextLookahead === '>') {
      this.shift(); // consume '-'
      this.shift(); // consume '>'

      return {
        type: 'arrow',
        position: { start, end: this.index },
      };
    }

    // Handle malformed arrow (just '-' without '>')
    if (this.lookahead === '-') {
      throw new MalformedArrowError(this.pattern, { start: this.index, end: this.index + 1 });
    }

    throw new InvalidCharacterError(this.lookahead, this.pattern, {
      start: this.index,
      end: this.index + 1,
    });
  }

  /**
   * Scan left parenthesis and return lparen token
   */
  private scanLparen(): LparenToken {
    const start = this.index;
    this.shift(); // consume '('

    return {
      type: 'lparen',
      position: { start, end: this.index },
    };
  }

  /**
   * Scan right parenthesis and return rparen token
   */
  private scanRparen(): RparenToken {
    const start = this.index;
    this.shift(); // consume ')'

    return {
      type: 'rparen',
      position: { start, end: this.index },
    };
  }

  /**
   * Scan ellipsis token and return ellipsis token
   */
  private scanEllipsis(): EllipsisToken {
    const start = this.index;
    
    // Must be exactly "..."
    if (this.lookahead === '.' && 
        this.nextLookahead === '.' && 
        this.chars[this.index + 2] === '.') {
      this.shift(); // consume first '.'
      this.shift(); // consume second '.'
      this.shift(); // consume third '.'
      
      return {
        type: 'ellipsis',
        position: { start, end: this.index },
      };
    }
    
    throw new InvalidCharacterError(this.lookahead, this.pattern, {
      start: this.index,
      end: this.index + 1,
    });
  }

  /**
   * Scan singleton token and return singleton token
   */
  private scanSingleton(): SingletonToken {
    const start = this.index;
    this.shift(); // consume '1'
    
    return {
      type: 'singleton', 
      position: { start, end: this.index },
    };
  }

  /**
   * Scan the next token from the current position
   */
  private scanToken(): EinopsToken {
    // Skip to next non-empty position
    if (this.isAtEnd) {
      throw new EinopsError('Unexpected end of pattern', this.pattern);
    }

    // Handle whitespace
    if (this.isWhitespace()) {
      return this.scanWhitespace();
    }

    // Handle arrow operator
    if (this.lookahead === '-') {
      return this.scanArrow();
    }

    // Handle axis names
    if (this.canStartAxis()) {
      return this.scanAxis();
    }

    // Handle left parenthesis
    if (this.lookahead === '(') {
      return this.scanLparen();
    }

    // Handle right parenthesis
    if (this.lookahead === ')') {
      return this.scanRparen();
    }

    // Handle ellipsis
    if (this.lookahead === '.') {
      return this.scanEllipsis();
    }

    // Handle singleton
    if (this.lookahead === '1') {
      return this.scanSingleton();
    }

    // Invalid character
    throw new InvalidCharacterError(this.lookahead, this.pattern, {
      start: this.index,
      end: this.index + 1,
    });
  }

  // =============================================================================
  // Public API
  // =============================================================================

  /**
   * Tokenize the entire pattern and return all tokens
   */
  tokenize(): TokenizeResult {
    const tokens: EinopsToken[] = [];

    while (!this.isAtEnd) {
      tokens.push(this.scanToken());
    }

    return {
      tokens,
      pattern: this.pattern,
    };
  }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Tokenize an einops pattern string
 */
export function tokenize(pattern: string): TokenizeResult {
  const scanner = new EinopsScanner(pattern);
  return scanner.tokenize();
}
