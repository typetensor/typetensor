/**
 * Tests for axis resolution system
 */

import { describe, it, expect } from 'bun:test';
import { AxisResolver, resolvePattern, AxisResolutionError } from './axis-resolver';
import { parse } from './scanner';

// =============================================================================
// Test Helpers
// =============================================================================

function resolvePatternString(
  pattern: string,
  inputShape: readonly number[],
  providedAxes?: Record<string, number>
) {
  const ast = parse(pattern);
  return resolvePattern(ast, inputShape, providedAxes);
}

// =============================================================================
// Simple Axis Resolution Tests
// =============================================================================

describe('Simple Axis Resolution', () => {
  it('should resolve basic transpose pattern', () => {
    const result = resolvePatternString('h w -> w h', [32, 64]);
    
    expect(result.axisDimensions.get('h')).toBe(32);
    expect(result.axisDimensions.get('w')).toBe(64);
    expect(result.outputShape).toEqual([64, 32]);
  });

  it('should resolve multi-axis pattern', () => {
    const result = resolvePatternString(
      'batch height width channels -> batch channels height width',
      [2, 32, 64, 3]
    );
    
    expect(result.axisDimensions.get('batch')).toBe(2);
    expect(result.axisDimensions.get('height')).toBe(32);
    expect(result.axisDimensions.get('width')).toBe(64);
    expect(result.axisDimensions.get('channels')).toBe(3);
    expect(result.outputShape).toEqual([2, 3, 32, 64]);
  });

  it('should resolve pattern with provided axes', () => {
    const result = resolvePatternString(
      'batch seq embed -> seq batch embed',
      [2, 10, 512],
      { batch: 2 }
    );
    
    expect(result.axisDimensions.get('batch')).toBe(2);
    expect(result.axisDimensions.get('seq')).toBe(10);
    expect(result.axisDimensions.get('embed')).toBe(512);
    expect(result.outputShape).toEqual([10, 2, 512]);
  });

  it('should handle single axis pattern', () => {
    const result = resolvePatternString('a -> a', [42]);
    
    expect(result.axisDimensions.get('a')).toBe(42);
    expect(result.outputShape).toEqual([42]);
  });

  it('should handle identity pattern', () => {
    const result = resolvePatternString('a b c -> a b c', [1, 2, 3]);
    
    expect(result.axisDimensions.get('a')).toBe(1);
    expect(result.axisDimensions.get('b')).toBe(2);
    expect(result.axisDimensions.get('c')).toBe(3);
    expect(result.outputShape).toEqual([1, 2, 3]);
  });

  it('should handle axis reordering', () => {
    const result = resolvePatternString('a b c d -> d c b a', [1, 2, 3, 4]);
    
    expect(result.outputShape).toEqual([4, 3, 2, 1]);
  });

  it('should handle partial axis selection', () => {
    const result = resolvePatternString('a b c -> a c', [10, 20, 30]);
    
    expect(result.axisDimensions.get('a')).toBe(10);
    expect(result.axisDimensions.get('b')).toBe(20);
    expect(result.axisDimensions.get('c')).toBe(30);
    expect(result.outputShape).toEqual([10, 30]);
  });

  it('should handle axis duplication', () => {
    const result = resolvePatternString('a b -> a b a', [5, 7]);
    
    expect(result.outputShape).toEqual([5, 7, 5]);
  });

  it('should error on unknown output axis', () => {
    expect(() => {
      resolvePatternString('h w -> h w c', [32, 64]);
    }).toThrow(AxisResolutionError);
    
    expect(() => {
      resolvePatternString('h w -> h w c', [32, 64]);
    }).toThrow("Unknown axis 'c' in output pattern");
  });

  it('should error on dimension mismatch', () => {
    expect(() => {
      resolvePatternString('a b c -> a b', [10, 20]);
    }).toThrow(AxisResolutionError);
    
    expect(() => {
      resolvePatternString('a b c -> a b', [10, 20]);
    }).toThrow('Pattern has more axes than tensor dimensions');
  });

  it('should error on unconsumed dimensions', () => {
    expect(() => {
      resolvePatternString('a b -> a b', [10, 20, 30]);
    }).toThrow(AxisResolutionError);
    
    expect(() => {
      resolvePatternString('a b -> a b', [10, 20, 30]);
    }).toThrow('Pattern does not consume all tensor dimensions');
  });

  it('should handle long axis names', () => {
    const result = resolvePatternString(
      'batch_size sequence_length hidden_dim -> sequence_length batch_size hidden_dim',
      [16, 128, 768]
    );
    
    expect(result.axisDimensions.get('batch_size')).toBe(16);
    expect(result.axisDimensions.get('sequence_length')).toBe(128);
    expect(result.axisDimensions.get('hidden_dim')).toBe(768);
    expect(result.outputShape).toEqual([128, 16, 768]);
  });

  it('should handle single character axis names', () => {
    const result = resolvePatternString('i j k l m n -> n m l k j i', [1, 2, 3, 4, 5, 6]);
    
    expect(result.outputShape).toEqual([6, 5, 4, 3, 2, 1]);
  });

  it('should preserve provided axis values', () => {
    const result = resolvePatternString(
      'a b c -> c b a',
      [10, 20, 30],
      { b: 20, c: 30 }
    );
    
    expect(result.axisDimensions.get('a')).toBe(10);
    expect(result.axisDimensions.get('b')).toBe(20);
    expect(result.axisDimensions.get('c')).toBe(30);
  });

  it('should handle numeric axis names', () => {
    const result = resolvePatternString('x1 x2 x3 -> x3 x1 x2', [10, 20, 30]);
    
    expect(result.outputShape).toEqual([30, 10, 20]);
  });
});

// =============================================================================
// Composite Pattern Resolution Tests
// =============================================================================

describe('Composite Pattern Resolution', () => {
  it('should resolve basic composite pattern', () => {
    const result = resolvePatternString('(h w) c -> h w c', [2048, 3], { h: 32 });
    
    expect(result.axisDimensions.get('h')).toBe(32);
    expect(result.axisDimensions.get('w')).toBe(64);
    expect(result.axisDimensions.get('c')).toBe(3);
    expect(result.outputShape).toEqual([32, 64, 3]);
  });

  it('should resolve composite pattern without provided axes', () => {
    expect(() => {
      resolvePatternString('(h w) c -> h w c', [2048, 3]);
    }).toThrow(AxisResolutionError);
    
    expect(() => {
      resolvePatternString('(h w) c -> h w c', [2048, 3]);
    }).toThrow('Cannot infer multiple unknown dimensions');
  });

  it('should validate composite dimension product', () => {
    expect(() => {
      resolvePatternString('(h w) c -> h w c', [100, 3], { h: 32 });
    }).toThrow(AxisResolutionError);
    
    expect(() => {
      resolvePatternString('(h w) c -> h w c', [100, 3], { h: 32 });
    }).toThrow('Cannot evenly split dimension');
  });

  it('should handle composite in output', () => {
    const result = resolvePatternString('h w c -> (h w) c', [32, 64, 3]);
    
    expect(result.axisDimensions.get('h')).toBe(32);
    expect(result.axisDimensions.get('w')).toBe(64);
    expect(result.axisDimensions.get('c')).toBe(3);
    expect(result.outputShape).toEqual([2048, 3]);
  });

  it('should handle multiple composites', () => {
    const result = resolvePatternString(
      '(a b) (c d) -> a b c d',
      [6, 20],
      { a: 2, c: 4 }
    );
    
    expect(result.axisDimensions.get('a')).toBe(2);
    expect(result.axisDimensions.get('b')).toBe(3);
    expect(result.axisDimensions.get('c')).toBe(4);
    expect(result.axisDimensions.get('d')).toBe(5);
    expect(result.outputShape).toEqual([2, 3, 4, 5]);
  });

  it('should handle nested composites', () => {
    const result = resolvePatternString(
      '((a b) c) d -> a b c d',
      [60, 7],
      { a: 3, b: 4 }
    );
    
    expect(result.axisDimensions.get('a')).toBe(3);
    expect(result.axisDimensions.get('b')).toBe(4);
    expect(result.axisDimensions.get('c')).toBe(5);
    expect(result.axisDimensions.get('d')).toBe(7);
    expect(result.outputShape).toEqual([3, 4, 5, 7]);
  });

  it('should handle composite with all axes known', () => {
    const result = resolvePatternString(
      '(h w) c -> h w c',
      [2048, 3],
      { h: 32, w: 64 }
    );
    
    expect(result.axisDimensions.get('h')).toBe(32);
    expect(result.axisDimensions.get('w')).toBe(64);
    expect(result.outputShape).toEqual([32, 64, 3]);
  });

  it('should error on composite dimension mismatch', () => {
    expect(() => {
      resolvePatternString('(h w) c -> h w c', [100, 3], { h: 32, w: 64 });
    }).toThrow(AxisResolutionError);
    
    expect(() => {
      resolvePatternString('(h w) c -> h w c', [100, 3], { h: 32, w: 64 });
    }).toThrow('Product of axes 2048 does not equal dimension 100');
  });

  it('should handle composite pattern rearrangement', () => {
    const result = resolvePatternString(
      '(batch seq) hidden -> batch seq hidden',
      [20, 768],
      { batch: 4 }
    );
    
    expect(result.axisDimensions.get('batch')).toBe(4);
    expect(result.axisDimensions.get('seq')).toBe(5);
    expect(result.axisDimensions.get('hidden')).toBe(768);
    expect(result.outputShape).toEqual([4, 5, 768]);
  });

  it('should handle reversing composite pattern', () => {
    const result = resolvePatternString('a b c -> c (b a)', [2, 3, 5]);
    
    expect(result.outputShape).toEqual([5, 6]);
  });
});

// =============================================================================
// Ellipsis Handling Tests
// =============================================================================

describe('Ellipsis Handling', () => {
  it('should handle basic ellipsis pattern', () => {
    const result = resolvePatternString('batch ... -> ...', [2, 3, 4, 5]);
    
    expect(result.axisDimensions.get('batch')).toBe(2);
    expect(result.ellipsisDimensions).toEqual([3, 4, 5]);
    expect(result.outputShape).toEqual([3, 4, 5]);
  });

  it('should handle ellipsis in middle', () => {
    const result = resolvePatternString('batch ... channels -> batch channels ...', [2, 3, 4, 5, 6]);
    
    expect(result.axisDimensions.get('batch')).toBe(2);
    expect(result.axisDimensions.get('channels')).toBe(6);
    expect(result.ellipsisDimensions).toEqual([3, 4, 5]);
    expect(result.outputShape).toEqual([2, 6, 3, 4, 5]);
  });

  it('should handle ellipsis with no dimensions', () => {
    const result = resolvePatternString('a ... b -> b ... a', [10, 20]);
    
    expect(result.axisDimensions.get('a')).toBe(10);
    expect(result.axisDimensions.get('b')).toBe(20);
    expect(result.ellipsisDimensions).toEqual([]);
    expect(result.outputShape).toEqual([20, 10]);
  });

  it('should handle ellipsis consuming multiple dimensions', () => {
    const result = resolvePatternString('... last -> last ...', [1, 2, 3, 4, 5]);
    
    expect(result.axisDimensions.get('last')).toBe(5);
    expect(result.ellipsisDimensions).toEqual([1, 2, 3, 4]);
    expect(result.outputShape).toEqual([5, 1, 2, 3, 4]);
  });

  it('should handle ellipsis only pattern', () => {
    const result = resolvePatternString('... -> ...', [2, 3, 4]);
    
    expect(result.ellipsisDimensions).toEqual([2, 3, 4]);
    expect(result.outputShape).toEqual([2, 3, 4]);
  });

  it('should handle ellipsis with composite', () => {
    const result = resolvePatternString('(a b) ... c -> a b c ...', [6, 7, 8, 9], { a: 2 });
    
    expect(result.axisDimensions.get('a')).toBe(2);
    expect(result.axisDimensions.get('b')).toBe(3);
    expect(result.axisDimensions.get('c')).toBe(9);
    expect(result.ellipsisDimensions).toEqual([7, 8]);
    expect(result.outputShape).toEqual([2, 3, 9, 7, 8]);
  });

  it('should error on insufficient dimensions for ellipsis', () => {
    expect(() => {
      resolvePatternString('a ... b c -> a b c', [10, 20]);
    }).toThrow(AxisResolutionError);
    
    expect(() => {
      resolvePatternString('a ... b c -> a b c', [10, 20]);
    }).toThrow('Not enough dimensions for pattern after ellipsis');
  });

  it('should handle duplicate ellipsis in output', () => {
    const result = resolvePatternString('a ... b -> ... ... a b', [1, 2, 3, 4]);
    
    expect(result.outputShape).toEqual([2, 3, 2, 3, 1, 4]);
  });
});

// =============================================================================
// Singleton Handling Tests
// =============================================================================

describe('Singleton Handling', () => {
  it('should handle singleton in input', () => {
    const result = resolvePatternString('h w 1 -> h w', [32, 64, 1]);
    
    expect(result.axisDimensions.get('h')).toBe(32);
    expect(result.axisDimensions.get('w')).toBe(64);
    expect(result.outputShape).toEqual([32, 64]);
  });

  it('should handle singleton in output', () => {
    const result = resolvePatternString('h w -> h w 1', [32, 64]);
    
    expect(result.outputShape).toEqual([32, 64, 1]);
  });

  it('should handle multiple singletons', () => {
    const result = resolvePatternString('1 h 1 w 1 -> h w 1 1', [1, 32, 1, 64, 1]);
    
    expect(result.outputShape).toEqual([32, 64, 1, 1]);
  });

  it('should error on non-singleton dimension', () => {
    expect(() => {
      resolvePatternString('h w 1 -> h w', [32, 64, 2]);
    }).toThrow(AxisResolutionError);
    
    expect(() => {
      resolvePatternString('h w 1 -> h w', [32, 64, 2]);
    }).toThrow('Expected singleton dimension but got 2');
  });

  it('should handle singleton with ellipsis', () => {
    const result = resolvePatternString('... 1 -> 1 ...', [2, 3, 4, 1]);
    
    expect(result.ellipsisDimensions).toEqual([2, 3, 4]);
    expect(result.outputShape).toEqual([1, 2, 3, 4]);
  });
});

// =============================================================================
// Error Handling Tests
// =============================================================================

describe('Error Handling', () => {
  it('should provide helpful error for unknown axis', () => {
    try {
      resolvePatternString('a b -> a b c', [10, 20]);
      expect(true).toBe(false); // Should not reach here
    } catch (error) {
      expect(error).toBeInstanceOf(AxisResolutionError);
      expect(error.message).toContain("Unknown axis 'c'");
      expect(error.pattern).toBe('a b -> a b c');
      expect(error.context?.axis).toBe('c');
    }
  });

  it('should provide context for dimension mismatch', () => {
    try {
      resolvePatternString('a b c d -> a b', [10, 20]);
      expect(true).toBe(false);
    } catch (error) {
      expect(error).toBeInstanceOf(AxisResolutionError);
      expect(error.context?.inputShape).toEqual([10, 20]);
    }
  });

  it('should error on composite with non-simple axes', () => {
    expect(() => {
      resolvePatternString('(a ...) b -> a b', [10, 20]);
    }).toThrow('Composite patterns can only contain simple axes');
  });

  it('should error on unresolvable composite', () => {
    expect(() => {
      resolvePatternString('(a b c) d -> a b c d', [60, 5]);
    }).toThrow('Cannot infer multiple unknown dimensions');
  });

  it('should provide helpful error for composite mismatch', () => {
    try {
      resolvePatternString('(h w) c -> h w c', [100, 3], { h: 32, w: 64 });
      expect(true).toBe(false);
    } catch (error) {
      expect(error).toBeInstanceOf(AxisResolutionError);
      expect(error.message).toContain('2048 does not equal dimension 100');
    }
  });

  it('should handle empty pattern', () => {
    expect(() => {
      const ast = parse(' -> ');
      new AxisResolver().resolvePattern(ast, []);
    }).toThrow();
  });

  it('should error on pattern with only whitespace', () => {
    expect(() => {
      parse('   ->   ');
    }).toThrow();
  });

  it('should provide position context in errors', () => {
    try {
      resolvePatternString('a b -> a b unknown', [10, 20]);
      expect(true).toBe(false);
    } catch (error) {
      expect(error).toBeInstanceOf(AxisResolutionError);
      expect(error.pattern).toBe('a b -> a b unknown');
    }
  });

  it('should handle very long patterns gracefully', () => {
    const longPattern = 'a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 -> a10 a9 a8 a7 a6 a5 a4 a3 a2 a1';
    const shape = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    const result = resolvePatternString(longPattern, shape);
    expect(result.outputShape).toEqual([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
  });

  it('should validate all output axes before computing shape', () => {
    expect(() => {
      resolvePatternString('a b -> a c d', [10, 20]);
    }).toThrow("Unknown axis 'c'");
  });
});

// =============================================================================
// Edge Cases Tests
// =============================================================================

describe('Edge Cases', () => {
  it('should handle scalar tensor', () => {
    const result = resolvePatternString(' -> ', []);
    
    expect(result.axisDimensions.size).toBe(0);
    expect(result.outputShape).toEqual([]);
  });

  it('should handle scalar to singleton', () => {
    const result = resolvePatternString(' -> 1', []);
    
    expect(result.outputShape).toEqual([1]);
  });

  it('should handle empty tensor with dimensions', () => {
    const result = resolvePatternString('a b c -> c b a', [0, 5, 3]);
    
    expect(result.axisDimensions.get('a')).toBe(0);
    expect(result.axisDimensions.get('b')).toBe(5);
    expect(result.axisDimensions.get('c')).toBe(3);
    expect(result.outputShape).toEqual([3, 5, 0]);
  });

  it('should handle very large dimensions', () => {
    const result = resolvePatternString('a b -> b a', [1000000, 2000000]);
    
    expect(result.axisDimensions.get('a')).toBe(1000000);
    expect(result.axisDimensions.get('b')).toBe(2000000);
    expect(result.outputShape).toEqual([2000000, 1000000]);
  });

  it('should handle axis names that look like numbers', () => {
    const result = resolvePatternString('a1 b2 c3 -> c3 a1 b2', [10, 20, 30]);
    
    expect(result.outputShape).toEqual([30, 10, 20]);
  });
});

// =============================================================================
// Integration Tests
// =============================================================================

describe('Integration with Parser', () => {
  it('should work with complex real-world patterns', () => {
    const result = resolvePatternString(
      'batch (seq head) dim -> batch seq head dim',
      [2, 60, 768],
      { seq: 10 }
    );
    
    expect(result.axisDimensions.get('batch')).toBe(2);
    expect(result.axisDimensions.get('seq')).toBe(10);
    expect(result.axisDimensions.get('head')).toBe(6);
    expect(result.axisDimensions.get('dim')).toBe(768);
    expect(result.outputShape).toEqual([2, 10, 6, 768]);
  });

  it('should handle attention pattern', () => {
    const result = resolvePatternString(
      'batch heads seq dim -> batch seq (heads dim)',
      [4, 8, 128, 64]
    );
    
    expect(result.outputShape).toEqual([4, 128, 512]);
  });

  it('should handle convolution pattern', () => {
    const result = resolvePatternString(
      'batch channel (height width) -> batch channel height width',
      [32, 3, 1024],
      { height: 32 }
    );
    
    expect(result.outputShape).toEqual([32, 3, 32, 32]);
  });

  it('should handle mixed pattern types', () => {
    const result = resolvePatternString(
      '(batch seq) ... dim 1 -> batch seq 1 dim ...',
      [20, 10, 20, 30, 768, 1],
      { batch: 4 }
    );
    
    expect(result.axisDimensions.get('batch')).toBe(4);
    expect(result.axisDimensions.get('seq')).toBe(5);
    expect(result.axisDimensions.get('dim')).toBe(768);
    expect(result.ellipsisDimensions).toEqual([10, 20, 30]);
    expect(result.outputShape).toEqual([4, 5, 1, 768, 10, 20, 30]);
  });
});