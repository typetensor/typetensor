/**
 * Runtime tests for the einops scanner
 *
 * These tests validate the Scanner's ability to tokenize einops patterns
 * correctly, including edge cases and error conditions.
 */

import { describe, it, expect } from 'bun:test';
import { EinopsScanner, tokenize } from './scanner';
import { InvalidCharacterError, MalformedArrowError } from './types';
import type { AxisToken, ArrowToken, WhitespaceToken, LparenToken, RparenToken } from './types';

// =============================================================================
// Basic Tokenization Tests
// =============================================================================

describe('EinopsScanner', () => {
  describe('Basic Tokenization', () => {
    it('should tokenize a single axis', () => {
      const result = tokenize('a');

      expect(result.pattern).toBe('a');
      expect(result.tokens).toHaveLength(1);

      const token = result.tokens[0] as AxisToken;
      expect(token.type).toBe('axis');
      expect(token.name).toBe('a');
      expect(token.position).toEqual({ start: 0, end: 1 });
    });

    it('should tokenize multiple axes with whitespace', () => {
      const result = tokenize('a b');

      expect(result.tokens).toHaveLength(3);

      const [axis1, whitespace, axis2] = result.tokens;

      expect(axis1?.type).toBe('axis');
      expect((axis1 as AxisToken)?.name).toBe('a');
      expect(axis1?.position).toEqual({ start: 0, end: 1 });

      expect(whitespace?.type).toBe('whitespace');
      expect(whitespace?.position).toEqual({ start: 1, end: 2 });

      expect(axis2?.type).toBe('axis');
      expect((axis2 as AxisToken)?.name).toBe('b');
      expect(axis2?.position).toEqual({ start: 2, end: 3 });
    });

    it('should tokenize simple transformation with arrow', () => {
      const result = tokenize('a -> b');

      expect(result.tokens).toHaveLength(5);

      const [axis1, ws1, arrow, ws2, axis2] = result.tokens;

      expect(axis1?.type).toBe('axis');
      expect((axis1 as AxisToken)?.name).toBe('a');

      expect(ws1?.type).toBe('whitespace');

      expect(arrow?.type).toBe('arrow');
      expect(arrow?.position).toEqual({ start: 2, end: 4 });

      expect(ws2?.type).toBe('whitespace');

      expect(axis2?.type).toBe('axis');
      expect((axis2 as AxisToken)?.name).toBe('b');
    });
  });

  describe('Axis Name Handling', () => {
    it('should handle complex axis names', () => {
      const testCases = ['batch', 'height_1', 'conv_output', '_private', 'CAPS', 'mixedCase123'];

      for (const name of testCases) {
        const result = tokenize(name);
        expect(result.tokens).toHaveLength(1);

        const token = result.tokens[0] as AxisToken;
        expect(token.type).toBe('axis');
        expect(token.name).toBe(name);
      }
    });

    it('should handle numeric suffixes in axis names', () => {
      const result = tokenize('conv2d_layer_3');

      expect(result.tokens).toHaveLength(1);
      const token = result.tokens[0] as AxisToken;
      expect(token.name).toBe('conv2d_layer_3');
    });

    it('should not allow numbers at start of axis names', () => {
      expect(() => tokenize('3axis')).toThrow(InvalidCharacterError);
    });
  });

  describe('Whitespace Handling', () => {
    it('should handle multiple consecutive spaces', () => {
      const result = tokenize('a   b');

      expect(result.tokens).toHaveLength(3);

      const whitespace = result.tokens[1] as WhitespaceToken;
      expect(whitespace?.type).toBe('whitespace');
      expect(whitespace?.position).toEqual({ start: 1, end: 4 });
    });

    it('should handle different whitespace types', () => {
      const result = tokenize('a\t\nb');

      expect(result.tokens).toHaveLength(3);
      expect(result.tokens[1]?.type).toBe('whitespace');
    });

    it('should handle leading and trailing whitespace', () => {
      const result = tokenize(' a b ');

      expect(result.tokens).toHaveLength(5);
      expect(result.tokens[0]?.type).toBe('whitespace');
      expect(result.tokens[4]?.type).toBe('whitespace');
    });
  });

  describe('Arrow Operator', () => {
    it('should handle arrow without surrounding spaces', () => {
      const result = tokenize('a->b');

      expect(result.tokens).toHaveLength(3);

      const [axis1, arrow, axis2] = result.tokens;
      expect(axis1?.type).toBe('axis');
      expect(arrow?.type).toBe('arrow');
      expect(axis2?.type).toBe('axis');
    });

    it('should track arrow position correctly', () => {
      const result = tokenize('abc->def');

      const arrow = result.tokens[1] as ArrowToken;
      expect(arrow?.position).toEqual({ start: 3, end: 5 });
    });
  });

  describe('Realistic Patterns', () => {
    it('should handle transpose pattern', () => {
      const result = tokenize('h w -> w h');

      expect(result.tokens).toHaveLength(9);

      const tokenTypes = result.tokens.map((t) => t.type);
      expect(tokenTypes).toEqual([
        'axis',
        'whitespace',
        'axis',
        'whitespace',
        'arrow',
        'whitespace',
        'axis',
        'whitespace',
        'axis',
      ]);

      const axisNames = result.tokens
        .filter((t) => t.type === 'axis')
        .map((t) => (t as AxisToken).name);
      expect(axisNames).toEqual(['h', 'w', 'w', 'h']);
    });

    it('should handle batch dimension pattern', () => {
      const result = tokenize('batch height width channels -> batch channels height width');

      const axisTokens = result.tokens.filter((t) => t.type === 'axis') as AxisToken[];
      const inputAxes = axisTokens.slice(0, 4).map((t) => t.name);
      const outputAxes = axisTokens.slice(4).map((t) => t.name);

      expect(inputAxes).toEqual(['batch', 'height', 'width', 'channels']);
      expect(outputAxes).toEqual(['batch', 'channels', 'height', 'width']);
    });
  });

  describe('Composite Patterns', () => {
    it('should tokenize simple composite pattern', () => {
      const result = tokenize('(h w) c');

      expect(result.tokens).toHaveLength(7);

      const tokenTypes = result.tokens.map((t) => t.type);
      expect(tokenTypes).toEqual([
        'lparen',
        'axis',
        'whitespace',
        'axis',
        'rparen',
        'whitespace',
        'axis',
      ]);

      // Verify axis names
      const axisTokens = result.tokens.filter((t) => t.type === 'axis') as AxisToken[];
      expect(axisTokens.map((t) => t.name)).toEqual(['h', 'w', 'c']);
    });

    it('should handle composite transformation', () => {
      const result = tokenize('(h w) c -> h w c');

      const tokenTypes = result.tokens.map((t) => t.type);
      expect(tokenTypes).toEqual([
        'lparen',
        'axis',
        'whitespace',
        'axis',
        'rparen',
        'whitespace',
        'axis',
        'whitespace',
        'arrow',
        'whitespace',
        'axis',
        'whitespace',
        'axis',
        'whitespace',
        'axis',
      ]);
    });

    it('should handle multiple composites', () => {
      const result = tokenize('(h w) (a b) -> h w a b');

      const parenTokens = result.tokens.filter((t) => t.type === 'lparen' || t.type === 'rparen');
      expect(parenTokens).toHaveLength(4); // 2 lparen + 2 rparen
    });

    it('should track parentheses positions correctly', () => {
      const result = tokenize('(abc)');

      const lparen = result.tokens[0] as LparenToken;
      const rparen = result.tokens[2] as RparenToken;

      expect(lparen?.position).toEqual({ start: 0, end: 1 });
      expect(rparen?.position).toEqual({ start: 4, end: 5 });
    });

    it('should handle nested composites', () => {
      const result = tokenize('((h w) c)');

      const tokenTypes = result.tokens.map((t) => t.type);
      expect(tokenTypes).toEqual([
        'lparen',
        'lparen',
        'axis',
        'whitespace',
        'axis',
        'rparen',
        'whitespace',
        'axis',
        'rparen',
      ]);
    });

    it('should handle realistic composite patterns', () => {
      const patterns = [
        '(batch seq) embed -> batch seq embed',
        'batch (height width) channels -> batch height width channels',
        '(h1 h2) (w1 w2) c -> h1 h2 w1 w2 c',
      ];

      for (const pattern of patterns) {
        const result = tokenize(pattern);
        expect(result.pattern).toBe(pattern);
        expect(result.tokens.length).toBeGreaterThan(0);

        // Should contain both lparen and rparen tokens
        const lparens = result.tokens.filter((t) => t.type === 'lparen');
        const rparens = result.tokens.filter((t) => t.type === 'rparen');
        expect(lparens.length).toEqual(rparens.length);
        expect(lparens.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Composite Edge Cases', () => {
    it('should handle unmatched opening parenthesis', () => {
      // Note: We'll implement validation in a later phase
      // For now, just ensure tokens are created correctly
      const result = tokenize('(h w c');

      const lparens = result.tokens.filter((t) => t.type === 'lparen');
      const rparens = result.tokens.filter((t) => t.type === 'rparen');
      expect(lparens.length).toBe(1);
      expect(rparens.length).toBe(0);
    });

    it('should handle unmatched closing parenthesis', () => {
      const result = tokenize('h w c)');

      const lparens = result.tokens.filter((t) => t.type === 'lparen');
      const rparens = result.tokens.filter((t) => t.type === 'rparen');
      expect(lparens.length).toBe(0);
      expect(rparens.length).toBe(1);
    });

    it('should handle empty parentheses', () => {
      const result = tokenize('()');

      expect(result.tokens).toHaveLength(2);
      expect(result.tokens[0]?.type).toBe('lparen');
      expect(result.tokens[1]?.type).toBe('rparen');
    });

    it('should handle parentheses without spaces', () => {
      const result = tokenize('(hw)c');

      const tokenTypes = result.tokens.map((t) => t.type);
      expect(tokenTypes).toEqual(['lparen', 'axis', 'rparen', 'axis']);

      const axisTokens = result.tokens.filter((t) => t.type === 'axis') as AxisToken[];
      expect(axisTokens.map((t) => t.name)).toEqual(['hw', 'c']);
    });

    it('should handle complex nesting', () => {
      const result = tokenize('(((a b) c) d)');

      const lparens = result.tokens.filter((t) => t.type === 'lparen');
      const rparens = result.tokens.filter((t) => t.type === 'rparen');
      expect(lparens.length).toBe(3);
      expect(rparens.length).toBe(3);
    });
  });

  describe('Ellipsis Patterns', () => {
    it('should tokenize simple ellipsis', () => {
      const result = tokenize('...');

      expect(result.tokens).toHaveLength(1);
      expect(result.tokens[0]?.type).toBe('ellipsis');
      expect(result.tokens[0]?.position).toEqual({ start: 0, end: 3 });
    });

    it('should tokenize ellipsis in transformation', () => {
      const result = tokenize('batch ... -> ...');

      const tokenTypes = result.tokens.map((t) => t.type);
      expect(tokenTypes).toEqual([
        'axis',
        'whitespace',
        'ellipsis',
        'whitespace',
        'arrow',
        'whitespace',
        'ellipsis',
      ]);
    });

    it('should handle ellipsis with other axes', () => {
      const result = tokenize('batch ... height -> height batch ...');

      const axisTokens = result.tokens.filter((t) => t.type === 'axis') as AxisToken[];
      expect(axisTokens.map((t) => t.name)).toEqual(['batch', 'height', 'height', 'batch']);

      const ellipsisTokens = result.tokens.filter((t) => t.type === 'ellipsis');
      expect(ellipsisTokens).toHaveLength(2);
    });

    it('should track ellipsis position correctly', () => {
      const result = tokenize('a ... b');

      const ellipsis = result.tokens[2]; // Should be the ellipsis token
      expect(ellipsis?.type).toBe('ellipsis');
      expect(ellipsis?.position).toEqual({ start: 2, end: 5 });
    });

    it('should handle multiple ellipses', () => {
      const result = tokenize('... -> ... ...');

      const ellipsisTokens = result.tokens.filter((t) => t.type === 'ellipsis');
      expect(ellipsisTokens).toHaveLength(3);
    });

    it('should reject single or double dots', () => {
      expect(() => tokenize('.')).toThrow(InvalidCharacterError);
      expect(() => tokenize('..')).toThrow(InvalidCharacterError);
    });

    it('should handle ellipsis in composite patterns', () => {
      const result = tokenize('(batch ...) -> batch ...');

      const tokenTypes = result.tokens.map((t) => t.type);
      expect(tokenTypes).toContain('ellipsis');
      expect(tokenTypes).toContain('lparen');
      expect(tokenTypes).toContain('rparen');
    });

    it('should handle realistic ellipsis patterns', () => {
      const patterns = [
        'batch ... -> ...',
        'batch height ... width -> height width batch ...',
        '... channels -> channels ...',
      ];

      for (const pattern of patterns) {
        const result = tokenize(pattern);
        expect(result.pattern).toBe(pattern);
        const ellipsisCount = result.tokens.filter((t) => t.type === 'ellipsis').length;
        expect(ellipsisCount).toBeGreaterThan(0);
      }
    });
  });

  describe('Singleton Patterns', () => {
    it('should tokenize simple singleton', () => {
      const result = tokenize('1');

      expect(result.tokens).toHaveLength(1);
      expect(result.tokens[0]?.type).toBe('singleton');
      expect(result.tokens[0]?.position).toEqual({ start: 0, end: 1 });
    });

    it('should tokenize singleton in transformation', () => {
      const result = tokenize('h w -> h w 1');

      const tokenTypes = result.tokens.map((t) => t.type);
      expect(tokenTypes).toEqual([
        'axis',
        'whitespace',
        'axis',
        'whitespace',
        'arrow',
        'whitespace',
        'axis',
        'whitespace',
        'axis',
        'whitespace',
        'singleton',
      ]);
    });

    it('should handle singleton with other axes', () => {
      const result = tokenize('batch height 1 -> batch 1 height');

      const singletonTokens = result.tokens.filter((t) => t.type === 'singleton');
      expect(singletonTokens).toHaveLength(2);
    });

    it('should track singleton position correctly', () => {
      const result = tokenize('a 1 b');

      const singleton = result.tokens[2]; // Should be the singleton token
      expect(singleton?.type).toBe('singleton');
      expect(singleton?.position).toEqual({ start: 2, end: 3 });
    });

    it('should handle multiple singletons', () => {
      const result = tokenize('1 1 1 -> 1');

      const singletonTokens = result.tokens.filter((t) => t.type === 'singleton');
      expect(singletonTokens).toHaveLength(4);
    });

    it('should handle realistic singleton patterns', () => {
      const patterns = [
        'h w 1 -> h w',
        'batch height width 1 -> batch 1 height width',
        '1 -> h w 1',
      ];

      for (const pattern of patterns) {
        const result = tokenize(pattern);
        expect(result.pattern).toBe(pattern);
        const singletonCount = result.tokens.filter((t) => t.type === 'singleton').length;
        expect(singletonCount).toBeGreaterThan(0);
      }
    });
  });
});

// =============================================================================
// Error Handling Tests
// =============================================================================

describe('Scanner Error Handling', () => {
  describe('Invalid Characters', () => {
    it('should reject special characters', () => {
      const invalidChars = ['@', '#', '$', '%', '!', '?', '.', ',', ';'];

      for (const char of invalidChars) {
        expect(() => tokenize(char), `Should reject '${char}'`).toThrow(InvalidCharacterError);
      }
    });

    it('should provide helpful error messages with position', () => {
      try {
        tokenize('a @ b');
      } catch (error) {
        expect(error).toBeInstanceOf(InvalidCharacterError);
        expect((error as InvalidCharacterError).message).toContain("Invalid character '@'");
        expect((error as InvalidCharacterError).message).toContain('a @ b');
        expect((error as InvalidCharacterError).message).toContain('  ^');
      }
    });
  });

  describe('Malformed Arrow', () => {
    it('should reject single dash', () => {
      expect(() => tokenize('a - b')).toThrow(MalformedArrowError);
    });

    it('should reject dash at end', () => {
      expect(() => tokenize('a b -')).toThrow(MalformedArrowError);
    });

    it('should provide helpful arrow error message', () => {
      try {
        tokenize('a - b');
      } catch (error) {
        expect(error).toBeInstanceOf(MalformedArrowError);
        expect((error as MalformedArrowError).message).toContain('Malformed arrow operator');
        expect((error as MalformedArrowError).message).toContain('expected "->"');
      }
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty pattern', () => {
      const result = tokenize('');
      expect(result.tokens).toHaveLength(0);
      expect(result.pattern).toBe('');
    });

    it('should handle only whitespace', () => {
      const result = tokenize('   ');
      expect(result.tokens).toHaveLength(1);
      expect(result.tokens[0]?.type).toBe('whitespace');
    });

    it('should handle very long axis names', () => {
      const longName =
        'very_long_axis_name_that_goes_on_and_on_with_many_underscores_and_numbers_123';
      const result = tokenize(longName);

      expect(result.tokens).toHaveLength(1);
      const token = result.tokens[0] as AxisToken;
      expect(token.name).toBe(longName);
    });
  });
});

// =============================================================================
// Scanner Instance Tests
// =============================================================================

describe('EinopsScanner Instance', () => {
  describe('Scanner State', () => {
    it('should track position correctly', () => {
      const scanner = new EinopsScanner('abc');

      expect(scanner.position).toBe(0);
      expect(scanner.lookahead).toBe('a');
      expect(scanner.nextLookahead).toBe('b');
      expect(scanner.isAtEnd).toBe(false);

      scanner.shift();
      expect(scanner.position).toBe(1);
      expect(scanner.lookahead).toBe('b');

      scanner.jumpForward(2);
      expect(scanner.position).toBe(3);
      expect(scanner.isAtEnd).toBe(true);
    });

    it('should handle scanned/unscanned correctly', () => {
      const scanner = new EinopsScanner('hello');

      expect(scanner.scanned).toBe('');
      expect(scanner.unscanned).toBe('hello');

      scanner.shift();
      scanner.shift();

      expect(scanner.scanned).toBe('he');
      expect(scanner.unscanned).toBe('llo');
    });
  });
});
