/**
 * Runtime tests for the einops parser
 * 
 * These tests validate the parser's ability to convert token arrays
 * into valid AST structures, including error cases and edge conditions.
 */

import { describe, it, expect } from 'bun:test';
import { 
  parseTokens, 
  ParseError, 
  UnexpectedTokenError, 
  UnbalancedParenthesesError, 
  MissingArrowError,
  MultipleArrowError 
} from './parser';
import { tokenize } from './scanner';
import type { EinopsAST, SimpleAxis, CompositeAxis, EllipsisAxis, SingletonAxis } from './ast';
import { isSimpleAxis, isCompositeAxis, isEllipsisAxis, isSingletonAxis } from './ast';

// =============================================================================
// Test Helpers
// =============================================================================

function parsePattern(pattern: string): EinopsAST {
  const { tokens } = tokenize(pattern);
  return parseTokens(tokens);
}

// =============================================================================
// Simple Pattern Tests
// =============================================================================

describe('Token Parser - Simple Patterns', () => {
  it('should parse single axis pattern', () => {
    const ast = parsePattern('a -> b');
    
    expect(ast.input).toHaveLength(1);
    expect(ast.output).toHaveLength(1);
    
    const inputAxis = ast.input[0] as SimpleAxis;
    const outputAxis = ast.output[0] as SimpleAxis;
    
    expect(isSimpleAxis(inputAxis)).toBe(true);
    expect(inputAxis.name).toBe('a');
    expect(inputAxis.position).toEqual({ start: 0, end: 1 });
    
    expect(isSimpleAxis(outputAxis)).toBe(true);
    expect(outputAxis.name).toBe('b');
    expect(outputAxis.position).toEqual({ start: 5, end: 6 });
  });

  it('should parse simple transpose pattern', () => {
    const ast = parsePattern('h w -> w h');
    
    expect(ast.input).toHaveLength(2);
    expect(ast.output).toHaveLength(2);
    
    const h1 = ast.input[0]!;
    const w1 = ast.input[1]!;
    const w2 = ast.output[0]!;
    const h2 = ast.output[1]!;
    
    expect(isSimpleAxis(h1)).toBe(true);
    expect(isSimpleAxis(w1)).toBe(true);
    expect(isSimpleAxis(w2)).toBe(true);
    expect(isSimpleAxis(h2)).toBe(true);
    
    expect((h1 as SimpleAxis).name).toBe('h');
    expect((w1 as SimpleAxis).name).toBe('w');
    expect((w2 as SimpleAxis).name).toBe('w');
    expect((h2 as SimpleAxis).name).toBe('h');
  });

  it('should parse multi-axis pattern', () => {
    const ast = parsePattern('batch height width channels -> batch channels height width');
    
    expect(ast.input).toHaveLength(4);
    expect(ast.output).toHaveLength(4);
    
    const inputNames = (ast.input as SimpleAxis[]).map(axis => axis.name);
    const outputNames = (ast.output as SimpleAxis[]).map(axis => axis.name);
    
    expect(inputNames).toEqual(['batch', 'height', 'width', 'channels']);
    expect(outputNames).toEqual(['batch', 'channels', 'height', 'width']);
  });

  it('should handle whitespace correctly', () => {
    const ast = parsePattern('  a   b  ->  c   d  ');
    
    expect(ast.input).toHaveLength(2);
    expect(ast.output).toHaveLength(2);
    
    const inputNames = (ast.input as SimpleAxis[]).map(axis => axis.name);
    const outputNames = (ast.output as SimpleAxis[]).map(axis => axis.name);
    
    expect(inputNames).toEqual(['a', 'b']);
    expect(outputNames).toEqual(['c', 'd']);
  });

  it('should track positions accurately', () => {
    const ast = parsePattern('abc def -> ghi');
    
    const abc = ast.input[0]!;
    const def = ast.input[1]!;
    const ghi = ast.output[0]!;
    
    expect(isSimpleAxis(abc)).toBe(true);
    expect(isSimpleAxis(def)).toBe(true);
    expect(isSimpleAxis(ghi)).toBe(true);
    
    expect((abc as SimpleAxis).position).toEqual({ start: 0, end: 3 });
    expect((def as SimpleAxis).position).toEqual({ start: 4, end: 7 });
    expect((ghi as SimpleAxis).position).toEqual({ start: 11, end: 14 });
  });
});

// =============================================================================
// Composite Pattern Tests
// =============================================================================

describe('Token Parser - Composite Patterns', () => {
  it('should parse simple composite pattern', () => {
    const ast = parsePattern('(h w) c -> h w c');
    
    expect(ast.input).toHaveLength(2);
    expect(ast.output).toHaveLength(3);
    
    const composite = ast.input[0]!;
    const c = ast.input[1]!;
    
    expect(isCompositeAxis(composite)).toBe(true);
    expect(isSimpleAxis(c)).toBe(true);
    
    const compositeAxis = composite as CompositeAxis;
    expect(compositeAxis.axes).toHaveLength(2);
    
    const h = compositeAxis.axes[0]!;
    const w = compositeAxis.axes[1]!;
    expect(isSimpleAxis(h)).toBe(true);
    expect(isSimpleAxis(w)).toBe(true);
    expect((h as SimpleAxis).name).toBe('h');
    expect((w as SimpleAxis).name).toBe('w');
  });

  it('should parse nested composite patterns', () => {
    const ast = parsePattern('((h w) c) d -> h w c d');
    
    expect(ast.input).toHaveLength(2);
    
    const outerComposite = ast.input[0]!;
    const d = ast.input[1]!;
    
    expect(isCompositeAxis(outerComposite)).toBe(true);
    expect(isSimpleAxis(d)).toBe(true);
    
    const outer = outerComposite as CompositeAxis;
    expect(outer.axes).toHaveLength(2);
    
    const innerComposite = outer.axes[0]!;
    const c = outer.axes[1]!;
    expect(isCompositeAxis(innerComposite)).toBe(true);
    expect(isSimpleAxis(c)).toBe(true);
    
    const inner = innerComposite as CompositeAxis;
    expect(inner.axes).toHaveLength(2);
    
    const h = inner.axes[0]!;
    const w = inner.axes[1]!;
    expect(isSimpleAxis(h)).toBe(true);
    expect(isSimpleAxis(w)).toBe(true);
    expect((h as SimpleAxis).name).toBe('h');
    expect((w as SimpleAxis).name).toBe('w');
  });

  it('should handle multiple composites', () => {
    const ast = parsePattern('(h w) (a b) -> h w a b');
    
    expect(ast.input).toHaveLength(2);
    expect(ast.output).toHaveLength(4);
    
    const comp1 = ast.input[0]!;
    const comp2 = ast.input[1]!;
    
    expect(isCompositeAxis(comp1)).toBe(true);
    expect(isCompositeAxis(comp2)).toBe(true);
    
    const composite1 = comp1 as CompositeAxis;
    const composite2 = comp2 as CompositeAxis;
    
    expect(composite1.axes).toHaveLength(2);
    expect(composite2.axes).toHaveLength(2);
    
    const h = composite1.axes[0]!;
    const w = composite1.axes[1]!;
    const a = composite2.axes[0]!;
    const b = composite2.axes[1]!;
    
    expect(isSimpleAxis(h)).toBe(true);
    expect(isSimpleAxis(w)).toBe(true);
    expect(isSimpleAxis(a)).toBe(true);
    expect(isSimpleAxis(b)).toBe(true);
    
    expect((h as SimpleAxis).name).toBe('h');
    expect((w as SimpleAxis).name).toBe('w');
    expect((a as SimpleAxis).name).toBe('a');
    expect((b as SimpleAxis).name).toBe('b');
  });

  it('should track composite positions', () => {
    // Test that patterns without arrows throw errors
    expect(() => parsePattern('(abc def)')).toThrow(MissingArrowError);
    
    // Test with arrow
    const validAst = parsePattern('(abc def) -> ghi');
    const composite = validAst.input[0]!;
    
    expect(isCompositeAxis(composite)).toBe(true);
    const compositeAxis = composite as CompositeAxis;
    
    expect(compositeAxis.position.start).toBe(0);
    expect(compositeAxis.position.end).toBe(9); // Position of closing paren + 1
  });

  it('should handle empty composites', () => {
    const ast = parsePattern('() -> a');
    
    expect(ast.input).toHaveLength(1);
    
    const composite = ast.input[0]!;
    expect(isCompositeAxis(composite)).toBe(true);
    const compositeAxis = composite as CompositeAxis;
    expect(compositeAxis.axes).toHaveLength(0);
  });

  it('should validate balanced parentheses', () => {
    expect(() => parsePattern('(h w -> h w')).toThrow(UnbalancedParenthesesError);
    expect(() => parsePattern('h w) -> h w')).toThrow(UnexpectedTokenError);
    expect(() => parsePattern('((h w) -> h w')).toThrow(UnbalancedParenthesesError);
  });
});

// =============================================================================
// Ellipsis Pattern Tests
// =============================================================================

describe('Token Parser - Ellipsis Patterns', () => {
  it('should parse ellipsis in input and output', () => {
    const ast = parsePattern('batch ... -> ...');
    
    expect(ast.input).toHaveLength(2);
    expect(ast.output).toHaveLength(1);
    
    const batch = ast.input[0]!;
    const ellipsis1 = ast.input[1]!;
    const ellipsis2 = ast.output[0]!;
    
    expect(isSimpleAxis(batch)).toBe(true);
    expect(isEllipsisAxis(ellipsis1)).toBe(true);
    expect(isEllipsisAxis(ellipsis2)).toBe(true);
    
    expect((batch as SimpleAxis).name).toBe('batch');
  });

  it('should handle ellipsis with other axes', () => {
    const ast = parsePattern('batch ... height -> height batch ...');
    
    expect(ast.input).toHaveLength(3);
    expect(ast.output).toHaveLength(3);
    
    const batch1 = ast.input[0]!;
    const ellipsis1 = ast.input[1]!;
    const height1 = ast.input[2]!;
    const height2 = ast.output[0]!;
    const batch2 = ast.output[1]!;
    const ellipsis2 = ast.output[2]!;
    
    expect(isSimpleAxis(batch1)).toBe(true);
    expect(isEllipsisAxis(ellipsis1)).toBe(true);
    expect(isSimpleAxis(height1)).toBe(true);
    expect(isSimpleAxis(height2)).toBe(true);
    expect(isSimpleAxis(batch2)).toBe(true);
    expect(isEllipsisAxis(ellipsis2)).toBe(true);
  });

  it('should validate ellipsis positions', () => {
    // Test that patterns without arrows throw errors
    expect(() => parsePattern('a ... b')).toThrow(MissingArrowError);
    
    // Test with arrow
    const validAst = parsePattern('a ... -> ...');
    const a = validAst.input[0]!;
    const ellipsis = validAst.input[1]!;
    
    expect(isSimpleAxis(a)).toBe(true);
    expect(isEllipsisAxis(ellipsis)).toBe(true);
    
    expect((a as SimpleAxis).position).toEqual({ start: 0, end: 1 });
    expect((ellipsis as EllipsisAxis).position).toEqual({ start: 2, end: 5 });
  });

  it('should handle multiple ellipses', () => {
    const ast = parsePattern('... -> ... ...');
    
    expect(ast.input).toHaveLength(1);
    expect(ast.output).toHaveLength(2);
    
    const inputEllipsis = ast.input[0]!;
    const outputEllipsis1 = ast.output[0]!;
    const outputEllipsis2 = ast.output[1]!;
    
    expect(isEllipsisAxis(inputEllipsis)).toBe(true);
    expect(isEllipsisAxis(outputEllipsis1)).toBe(true);
    expect(isEllipsisAxis(outputEllipsis2)).toBe(true);
  });
});

// =============================================================================
// Singleton Pattern Tests
// =============================================================================

describe('Token Parser - Singleton Patterns', () => {
  it('should parse singleton dimensions', () => {
    const ast = parsePattern('h w 1 -> h w');
    
    expect(ast.input).toHaveLength(3);
    expect(ast.output).toHaveLength(2);
    
    const h = ast.input[0]!;
    const w = ast.input[1]!;
    const singleton = ast.input[2]!;
    
    expect(isSimpleAxis(h)).toBe(true);
    expect(isSimpleAxis(w)).toBe(true);
    expect(isSingletonAxis(singleton)).toBe(true);
    
    expect((h as SimpleAxis).name).toBe('h');
    expect((w as SimpleAxis).name).toBe('w');
  });

  it('should handle multiple singletons', () => {
    const ast = parsePattern('1 1 1 -> 1');
    
    expect(ast.input).toHaveLength(3);
    expect(ast.output).toHaveLength(1);
    
    const s1 = ast.input[0]!;
    const s2 = ast.input[1]!;
    const s3 = ast.input[2]!;
    const s4 = ast.output[0]!;
    
    expect(isSingletonAxis(s1)).toBe(true);
    expect(isSingletonAxis(s2)).toBe(true);
    expect(isSingletonAxis(s3)).toBe(true);
    expect(isSingletonAxis(s4)).toBe(true);
  });

  it('should track singleton positions', () => {
    const ast = parsePattern('a 1 b -> c');
    
    const a = ast.input[0]!;
    const singleton = ast.input[1]!;
    const b = ast.input[2]!;
    
    expect(isSimpleAxis(a)).toBe(true);
    expect(isSingletonAxis(singleton)).toBe(true);
    expect(isSimpleAxis(b)).toBe(true);
    
    expect((a as SimpleAxis).position).toEqual({ start: 0, end: 1 });
    expect((singleton as SingletonAxis).position).toEqual({ start: 2, end: 3 });
    expect((b as SimpleAxis).position).toEqual({ start: 4, end: 5 });
  });

  it('should validate singleton placement', () => {
    const ast = parsePattern('batch height 1 -> batch 1 height');
    
    expect(ast.input).toHaveLength(3);
    expect(ast.output).toHaveLength(3);
    
    const inputTypes = ast.input.map(p => p.type);
    const outputTypes = ast.output.map(p => p.type);
    
    expect(inputTypes).toEqual(['simple', 'simple', 'singleton']);
    expect(outputTypes).toEqual(['simple', 'singleton', 'simple']);
  });
});

// =============================================================================
// Mixed Pattern Tests
// =============================================================================

describe('Token Parser - Mixed Patterns', () => {
  it('should handle composite with ellipsis', () => {
    const ast = parsePattern('(batch ...) -> batch ...');
    
    expect(ast.input).toHaveLength(1);
    expect(ast.output).toHaveLength(2);
    
    const composite = ast.input[0]!;
    expect(isCompositeAxis(composite)).toBe(true);
    const compositeAxis = composite as CompositeAxis;
    expect(compositeAxis.axes).toHaveLength(2);
    
    const batch = compositeAxis.axes[0]!;
    const ellipsis = compositeAxis.axes[1]!;
    expect(isSimpleAxis(batch)).toBe(true);
    expect(isEllipsisAxis(ellipsis)).toBe(true);
  });

  it('should handle composite with singleton', () => {
    const ast = parsePattern('(h w 1) -> h w');
    
    expect(ast.input).toHaveLength(1);
    expect(ast.output).toHaveLength(2);
    
    const composite = ast.input[0]!;
    expect(isCompositeAxis(composite)).toBe(true);
    const compositeAxis = composite as CompositeAxis;
    expect(compositeAxis.axes).toHaveLength(3);
    
    const h = compositeAxis.axes[0]!;
    const w = compositeAxis.axes[1]!;
    const singleton = compositeAxis.axes[2]!;
    expect(isSimpleAxis(h)).toBe(true);
    expect(isSimpleAxis(w)).toBe(true);
    expect(isSingletonAxis(singleton)).toBe(true);
  });

  it('should handle complex mixed patterns', () => {
    const ast = parsePattern('(batch seq) embed ... 1 -> batch seq embed ... 1');
    
    expect(ast.input).toHaveLength(4);
    expect(ast.output).toHaveLength(5);
    
    const composite = ast.input[0]!;
    const embed1 = ast.input[1]!;
    const ellipsis1 = ast.input[2]!;
    const singleton1 = ast.input[3]!;
    
    expect(isCompositeAxis(composite)).toBe(true);
    expect(isSimpleAxis(embed1)).toBe(true);
    expect(isEllipsisAxis(ellipsis1)).toBe(true);
    expect(isSingletonAxis(singleton1)).toBe(true);
    
    const compositeAxis = composite as CompositeAxis;
    expect(compositeAxis.axes).toHaveLength(2);
    
    const batch1 = compositeAxis.axes[0]!;
    const seq1 = compositeAxis.axes[1]!;
    expect(isSimpleAxis(batch1)).toBe(true);
    expect(isSimpleAxis(seq1)).toBe(true);
    expect((batch1 as SimpleAxis).name).toBe('batch');
    expect((seq1 as SimpleAxis).name).toBe('seq');
  });
});

// =============================================================================
// Error Handling Tests
// =============================================================================

describe('Token Parser - Error Handling', () => {
  it('should reject patterns without arrows', () => {
    expect(() => parsePattern('a b c')).toThrow(MissingArrowError);
    expect(() => parsePattern('(h w) c')).toThrow(MissingArrowError);
    expect(() => parsePattern('...')).toThrow(MissingArrowError);
  });

  it('should reject unbalanced parentheses', () => {
    expect(() => parsePattern('(h w -> h w')).toThrow(UnbalancedParenthesesError);
    expect(() => parsePattern('((h w) -> h w')).toThrow(UnbalancedParenthesesError);
    expect(() => parsePattern('(((h w)) -> h w')).toThrow(UnbalancedParenthesesError);
  });

  it('should reject multiple arrows', () => {
    expect(() => parsePattern('a -> b -> c')).toThrow(MultipleArrowError);
    expect(() => parsePattern('h w -> x y -> z')).toThrow(MultipleArrowError);
  });

  it('should reject unexpected closing parentheses', () => {
    expect(() => parsePattern('h w) -> h w')).toThrow(UnexpectedTokenError);
    expect(() => parsePattern('h w) c -> h w c')).toThrow(UnexpectedTokenError);
  });

  it('should provide helpful error messages', () => {
    try {
      parsePattern('(h w -> h w');
    } catch (error) {
      expect(error).toBeInstanceOf(UnbalancedParenthesesError);
      expect((error as UnbalancedParenthesesError).message).toContain('missing closing');
      expect((error as UnbalancedParenthesesError).position).toBeDefined();
    }
  });

  it('should handle unexpected token sequences', () => {
    // These would be caught by the scanner, but test parser robustness
    const tokens = tokenize('a -> b').tokens;
    
    // Valid case
    expect(() => parseTokens(tokens)).not.toThrow();
    
    // Empty tokens
    expect(() => parseTokens([])).toThrow(ParseError);
  });

  it('should validate token positions in errors', () => {
    try {
      parsePattern('a -> b -> c');
    } catch (error) {
      expect(error).toBeInstanceOf(MultipleArrowError);
      expect((error as MultipleArrowError).position).toBeDefined();
      expect((error as MultipleArrowError).position?.start).toBeGreaterThan(5);
    }
  });
});

// =============================================================================
// Metadata Tests
// =============================================================================

describe('Token Parser - Metadata', () => {
  it('should build correct metadata', () => {
    const ast = parsePattern('h w -> w h');
    
    expect(ast.metadata).toBeDefined();
    expect(ast.metadata.arrowPosition).toEqual({ start: 4, end: 6 });
    expect(ast.metadata.inputTokenCount).toBe(2); // h, w (no whitespace)
    expect(ast.metadata.outputTokenCount).toBe(2); // w, h (no whitespace)
  });

  it('should count tokens correctly with composites', () => {
    const ast = parsePattern('(h w) c -> h w c');
    
    expect(ast.metadata.inputTokenCount).toBe(5); // lparen, h, w, rparen, c
    expect(ast.metadata.outputTokenCount).toBe(3); // h, w, c
  });

  it('should handle complex patterns in metadata', () => {
    const ast = parsePattern('(batch seq) embed ... 1 -> batch seq embed');
    
    expect(ast.metadata.inputTokenCount).toBe(7); // lparen, batch, seq, rparen, embed, ellipsis, singleton
    expect(ast.metadata.outputTokenCount).toBe(3); // batch, seq, embed
  });
});

// =============================================================================
// Integration Tests
// =============================================================================

describe('Token Parser - Integration', () => {
  it('should handle realistic einops patterns', () => {
    const patterns = [
      'batch height width channels -> batch channels height width',
      '(h w) c -> h w c',
      'batch (h w) c -> batch h w c',
      'batch ... -> ...',
      'h w 1 -> h w',
      '(batch seq) embed -> batch seq embed',
      '((h1 h2) (w1 w2)) c -> h1 h2 w1 w2 c',
    ];

    for (const pattern of patterns) {
      const ast = parsePattern(pattern);
      
      expect(ast.input).toBeDefined();
      expect(ast.output).toBeDefined();
      expect(ast.metadata).toBeDefined();
      expect(ast.input.length).toBeGreaterThan(0);
      expect(ast.output.length).toBeGreaterThan(0);
    }
  });

  it('should maintain position information through parsing', () => {
    const ast = parsePattern('(h w) c -> h w c');
    
    // All patterns should have valid positions
    const allPatterns = [...ast.input, ...ast.output];
    
    for (const pattern of allPatterns) {
      expect(pattern.position).toBeDefined();
      expect(pattern.position.start).toBeGreaterThanOrEqual(0);
      expect(pattern.position.end).toBeGreaterThan(pattern.position.start);
    }
  });

  it('should parse and validate consistent axis names', () => {
    const ast = parsePattern('batch height width -> width height batch');
    
    const inputNames = ast.input.map(p => (p as SimpleAxis).name);
    const outputNames = ast.output.map(p => (p as SimpleAxis).name);
    
    expect(inputNames.sort()).toEqual(outputNames.sort());
  });
});