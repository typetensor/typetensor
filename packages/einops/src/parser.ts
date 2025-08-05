/**
 * Runtime parser for converting einops tokens to AST structures
 *
 * Takes the token arrays produced by our scanner and converts them
 * into the AST structures defined in ast.ts, with comprehensive
 * validation and error handling.
 */

import type { EinopsToken, Position } from './types';
import type {
  EinopsAST,
  AxisPattern,
  SimpleAxis,
  CompositeAxis,
  EllipsisAxis,
  SingletonAxis,
  ASTMetadata,
} from './ast';

// =============================================================================
// Error Classes
// =============================================================================

export class ParseError extends Error {
  constructor(
    message: string,
    public readonly position?: Position,
    public readonly context?: Record<string, unknown>,
  ) {
    super(ParseError.formatMessage(message, position));
  }

  static formatMessage(message: string, position?: Position): string {
    if (position) {
      return `${message} at position ${position.start}-${position.end}`;
    }
    return message;
  }
}

export class UnexpectedTokenError extends ParseError {
  constructor(expected: string, actual: EinopsToken, pattern?: string) {
    super(`Expected ${expected}, but got '${actual.type}' token`, actual.position, {
      expected,
      actual: actual.type,
      pattern,
    });
  }
}

export class UnbalancedParenthesesError extends ParseError {
  constructor(openPosition: Position, pattern: string) {
    super(`Unbalanced parentheses: missing closing ')' for opening parenthesis`, openPosition, {
      pattern,
      issue: 'unbalanced_parens',
    });
  }
}

export class MissingArrowError extends ParseError {
  constructor(pattern: string) {
    super(
      `Missing arrow operator '->': einops patterns must have input -> output format`,
      undefined,
      { pattern, issue: 'missing_arrow' },
    );
  }
}

export class MultipleArrowError extends ParseError {
  constructor(secondArrowPosition: Position, pattern: string) {
    super(
      `Multiple arrow operators found: einops patterns can only have one '->' operator`,
      secondArrowPosition,
      { pattern, issue: 'multiple_arrows' },
    );
  }
}

// =============================================================================
// Main Parser Function
// =============================================================================

/**
 * Parse a token array into an AST structure
 *
 * @param tokens - Array of tokens from the scanner
 * @returns Complete EinopsAST with input/output patterns and metadata
 */
export function parseTokens(tokens: readonly EinopsToken[], originalPattern?: string): EinopsAST {
  const parser = new TokenParser(tokens, originalPattern);
  return parser.parse();
}

// =============================================================================
// Token Parser Class
// =============================================================================

class TokenParser {
  constructor(
    private readonly tokens: readonly EinopsToken[],
    private readonly originalPattern?: string,
  ) {}

  /**
   * Parse the token array into a complete AST
   */
  parse(): EinopsAST {
    // Handle empty token array
    if (this.tokens.length === 0) {
      throw new ParseError('Cannot parse empty pattern');
    }

    // Find the arrow operator to split input/output
    const arrowIndex = this.findArrowPosition();

    // Parse input section (before arrow)
    const input = this.parseSection(0, arrowIndex);

    // Parse output section (after arrow)
    const output = this.parseSection(arrowIndex + 1, this.tokens.length);

    // Build metadata
    const metadata = this.buildMetadata(arrowIndex);

    return { input, output, metadata };
  }

  /**
   * Find the position of the arrow token
   */
  private findArrowPosition(): number {
    let arrowIndex = -1;

    for (let i = 0; i < this.tokens.length; i++) {
      if (this.tokens[i]?.type === 'arrow') {
        if (arrowIndex !== -1) {
          // Found a second arrow
          throw new MultipleArrowError(this.tokens[i]!.position, this.getOriginalPattern());
        }
        arrowIndex = i;
      }
    }

    if (arrowIndex === -1) {
      throw new MissingArrowError(this.getOriginalPattern());
    }

    return arrowIndex;
  }

  /**
   * Parse a section of tokens (input or output)
   */
  private parseSection(start: number, end: number): AxisPattern[] {
    const patterns: AxisPattern[] = [];
    let i = start;

    while (i < end) {
      const token = this.tokens[i];

      if (!token) {
        break;
      }

      // Skip whitespace tokens
      if (token.type === 'whitespace') {
        i++;
        continue;
      }

      // Parse different token types
      if (token.type === 'lparen') {
        const { pattern, endIndex } = this.parseCompositeAxis(i);
        patterns.push(pattern);
        i = endIndex + 1;
      } else if (token.type === 'axis') {
        patterns.push(this.parseSimpleAxis(token));
        i++;
      } else if (token.type === 'ellipsis') {
        patterns.push(this.parseEllipsisAxis(token));
        i++;
      } else if (token.type === 'singleton') {
        patterns.push(this.parseSingletonAxis(token));
        i++;
      } else if (token.type === 'rparen') {
        throw new UnexpectedTokenError(
          'axis, ellipsis, singleton, or opening parenthesis',
          token,
          this.getOriginalPattern(),
        );
      } else {
        throw new UnexpectedTokenError(
          'axis, ellipsis, singleton, or parenthesis',
          token,
          this.getOriginalPattern(),
        );
      }
    }

    return patterns;
  }

  /**
   * Parse a simple axis token
   */
  private parseSimpleAxis(token: EinopsToken): SimpleAxis {
    if (token.type !== 'axis') {
      throw new UnexpectedTokenError('axis', token, this.getOriginalPattern());
    }

    return {
      type: 'simple',
      name: token.name,
      position: token.position,
    };
  }

  /**
   * Parse an ellipsis token
   */
  private parseEllipsisAxis(token: EinopsToken): EllipsisAxis {
    if (token.type !== 'ellipsis') {
      throw new UnexpectedTokenError('ellipsis', token, this.getOriginalPattern());
    }

    return {
      type: 'ellipsis',
      position: token.position,
    };
  }

  /**
   * Parse a singleton token
   */
  private parseSingletonAxis(token: EinopsToken): SingletonAxis {
    if (token.type !== 'singleton') {
      throw new UnexpectedTokenError('singleton', token, this.getOriginalPattern());
    }

    return {
      type: 'singleton',
      position: token.position,
    };
  }

  /**
   * Parse a composite axis (parenthesized group)
   */
  private parseCompositeAxis(startIndex: number): { pattern: CompositeAxis; endIndex: number } {
    const lparenToken = this.tokens[startIndex];

    if (!lparenToken || lparenToken.type !== 'lparen') {
      throw new UnexpectedTokenError('lparen', lparenToken!, this.getOriginalPattern());
    }

    // Find matching closing parenthesis
    const endIndex = this.findMatchingRparen(startIndex);

    // Parse the contents between parentheses
    const innerPatterns = this.parseSection(startIndex + 1, endIndex);

    const rparenToken = this.tokens[endIndex]!;

    return {
      pattern: {
        type: 'composite',
        axes: innerPatterns,
        position: {
          start: lparenToken.position.start,
          end: rparenToken.position.end,
        },
      },
      endIndex,
    };
  }

  /**
   * Find the matching closing parenthesis for an opening one
   */
  private findMatchingRparen(lparenIndex: number): number {
    let depth = 0;

    for (let i = lparenIndex; i < this.tokens.length; i++) {
      const token = this.tokens[i];

      if (!token) {
        break;
      }

      if (token.type === 'lparen') {
        depth++;
      } else if (token.type === 'rparen') {
        depth--;
        if (depth === 0) {
          return i;
        }
      }
    }

    // No matching closing parenthesis found
    const openToken = this.tokens[lparenIndex]!;
    throw new UnbalancedParenthesesError(openToken.position, this.getOriginalPattern());
  }

  /**
   * Build metadata for the AST
   */
  private buildMetadata(arrowIndex: number): ASTMetadata {
    const arrowToken = this.tokens[arrowIndex]!;

    // Count non-whitespace tokens in each section
    const inputTokenCount = this.countNonWhitespaceTokens(0, arrowIndex);
    const outputTokenCount = this.countNonWhitespaceTokens(arrowIndex + 1, this.tokens.length);

    return {
      originalPattern: this.getOriginalPattern(),
      arrowPosition: arrowToken.position,
      inputTokenCount,
      outputTokenCount,
    };
  }

  /**
   * Count non-whitespace tokens in a range
   */
  private countNonWhitespaceTokens(start: number, end: number): number {
    let count = 0;

    for (let i = start; i < end; i++) {
      const token = this.tokens[i];
      if (token && token.type !== 'whitespace') {
        count++;
      }
    }

    return count;
  }

  /**
   * Reconstruct the original pattern string from tokens
   */
  private getOriginalPattern(): string {
    if (this.originalPattern !== undefined) {
      return this.originalPattern;
    }

    if (this.tokens.length === 0) {
      return '';
    }

    const firstToken = this.tokens[0]!;
    const lastToken = this.tokens[this.tokens.length - 1]!;

    // Fallback if pattern wasn't provided
    return `<pattern from position ${firstToken.position.start} to ${lastToken.position.end}>`;
  }
}
