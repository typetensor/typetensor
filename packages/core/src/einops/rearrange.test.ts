/**
 * Integration tests for rearrange function
 */

import { describe, it, expect } from 'bun:test';
import { rearrange, RearrangeError, type RearrangeTensor } from './rearrange';

// =============================================================================
// Mock Tensor Implementation
// =============================================================================

/**
 * Mock tensor implementation for testing
 * This is a simple implementation that demonstrates the API
 */
class MockTensor implements RearrangeTensor {
  constructor(
    public readonly shape: readonly number[],
    public readonly data?: number[],
  ) {}

  reshape(newShape: readonly number[]): MockTensor {
    // For testing purposes, we'll allow any reshape
    // In a real implementation, this would need proper tensor operations
    return new MockTensor(newShape, this.data);
  }

  permute(axes: readonly number[]): MockTensor {
    // Validate axes length matches tensor dimensions for proper permutations
    if (axes.length !== this.shape.length) {
      throw new Error(`Permutation axes length ${axes.length} does not match tensor dimensions ${this.shape.length}`);
    }
    
    const newShape = axes.map(axis => {
      const dim = this.shape[axis];
      if (dim === undefined) {
        throw new Error(`Invalid axis ${axis} for shape ${this.shape}`);
      }
      return dim;
    });
    return new MockTensor(newShape, this.data);
  }

  transpose(): MockTensor {
    if (this.shape.length !== 2) {
      throw new Error('Transpose only supported for 2D tensors');
    }
    
    const dim0 = this.shape[0];
    const dim1 = this.shape[1];
    if (dim0 === undefined || dim1 === undefined) {
      throw new Error('Invalid shape for transpose');
    }
    
    const newShape = [dim1, dim0];
    return new MockTensor(newShape, this.data);
  }
}

// Helper function to create test tensors
function createTensor(shape: number[]): MockTensor {
  return new MockTensor(shape);
}

// =============================================================================
// Simple Transpose Tests
// =============================================================================

describe('Simple Transpose Operations', () => {
  it('should handle basic 2D transpose', () => {
    const tensor = createTensor([3, 4]);
    const result = rearrange(tensor, 'h w -> w h');
    
    expect(result.shape).toEqual([4, 3]);
  });

  it('should handle identity pattern', () => {
    const tensor = createTensor([2, 3, 4]);
    const result = rearrange(tensor, 'a b c -> a b c');
    
    expect(result.shape).toEqual([2, 3, 4]);
  });

  it('should handle 3D transpose', () => {
    const tensor = createTensor([2, 3, 4]);
    const result = rearrange(tensor, 'a b c -> c a b');
    
    expect(result.shape).toEqual([4, 2, 3]);
  });

  it('should handle partial axis selection', () => {
    const tensor = createTensor([2, 3, 4]);
    const result = rearrange(tensor, 'a b c -> a c');
    
    expect(result.shape).toEqual([2, 4]);
  });

  it('should handle axis duplication', () => {
    const tensor = createTensor([2, 3]);
    const result = rearrange(tensor, 'a b -> a b a');
    
    expect(result.shape).toEqual([2, 3, 2]);
  });
});

// =============================================================================
// Axis Reordering Tests
// =============================================================================

describe('Axis Reordering Operations', () => {
  it('should handle 4D reordering (channels first to channels last)', () => {
    const tensor = createTensor([2, 3, 32, 32]); // [batch, channels, height, width]
    const result = rearrange(tensor, 'b c h w -> b h w c');
    
    expect(result.shape).toEqual([2, 32, 32, 3]);
  });

  it('should handle reverse order', () => {
    const tensor = createTensor([1, 2, 3, 4]);
    const result = rearrange(tensor, 'a b c d -> d c b a');
    
    expect(result.shape).toEqual([4, 3, 2, 1]);
  });

  it('should handle complex reordering', () => {
    const tensor = createTensor([2, 3, 4, 5, 6]);
    const result = rearrange(tensor, 'a b c d e -> c e a d b');
    
    expect(result.shape).toEqual([4, 6, 2, 5, 3]);
  });

  it('should handle single axis pattern', () => {
    const tensor = createTensor([42]);
    const result = rearrange(tensor, 'a -> a');
    
    expect(result.shape).toEqual([42]);
  });
});

// =============================================================================
// Composite Pattern Tests
// =============================================================================

describe('Composite Pattern Operations', () => {
  it('should handle basic composite splitting', () => {
    const tensor = createTensor([2048, 3]); // [h*w, channels]
    const result = rearrange(tensor, '(h w) c -> h w c', { axes: { h: 32 } });
    
    expect(result.shape).toEqual([32, 64, 3]);
  });

  it('should handle composite merging', () => {
    const tensor = createTensor([32, 64, 3]); // [height, width, channels]
    const result = rearrange(tensor, 'h w c -> (h w) c');
    
    expect(result.shape).toEqual([2048, 3]);
  });

  it('should handle multiple composites', () => {
    const tensor = createTensor([6, 20]); // [a*b, c*d]
    const result = rearrange(tensor, '(a b) (c d) -> a b c d', { 
      axes: { a: 2, c: 4 } 
    });
    
    expect(result.shape).toEqual([2, 3, 4, 5]);
  });

  it('should handle composite with reordering', () => {
    const tensor = createTensor([20, 768]); // [batch*seq, hidden]
    const result = rearrange(tensor, '(batch seq) hidden -> batch seq hidden', {
      axes: { batch: 4 }
    });
    
    expect(result.shape).toEqual([4, 5, 768]);
  });

  it('should infer unknown dimension in composite', () => {
    const tensor = createTensor([100, 3]); // [h*w, channels]
    const result = rearrange(tensor, '(h w) c -> h w c', { axes: { h: 10 } });
    
    expect(result.shape).toEqual([10, 10, 3]);
  });
});

// =============================================================================
// Ellipsis Pattern Tests
// =============================================================================

describe('Ellipsis Pattern Operations', () => {
  it('should handle basic ellipsis identity', () => {
    const tensor = createTensor([2, 3, 4, 5]);
    const result = rearrange(tensor, '... -> ...');
    
    expect(result.shape).toEqual([2, 3, 4, 5]);
  });

  it('should handle ellipsis with named axes', () => {
    const tensor = createTensor([2, 3, 4, 5]);
    const result = rearrange(tensor, 'batch ... -> ...');
    
    expect(result.shape).toEqual([3, 4, 5]);
  });

  it('should handle ellipsis in middle', () => {
    const tensor = createTensor([2, 3, 4, 5, 6]);
    const result = rearrange(tensor, 'batch ... channels -> batch channels ...');
    
    expect(result.shape).toEqual([2, 6, 3, 4, 5]);
  });

  it('should handle ellipsis consuming multiple dimensions', () => {
    const tensor = createTensor([1, 2, 3, 4, 5]);
    const result = rearrange(tensor, '... last -> last ...');
    
    expect(result.shape).toEqual([5, 1, 2, 3, 4]);
  });

  it('should handle ellipsis with no dimensions', () => {
    const tensor = createTensor([10, 20]);
    const result = rearrange(tensor, 'a ... b -> b ... a');
    
    expect(result.shape).toEqual([20, 10]);
  });
});

// =============================================================================
// Singleton Pattern Tests
// =============================================================================

describe('Singleton Pattern Operations', () => {
  it('should handle singleton removal', () => {
    const tensor = createTensor([32, 64, 1]);
    const result = rearrange(tensor, 'h w 1 -> h w');
    
    expect(result.shape).toEqual([32, 64]);
  });

  it('should handle singleton addition', () => {
    const tensor = createTensor([32, 64]);
    const result = rearrange(tensor, 'h w -> h w 1');
    
    expect(result.shape).toEqual([32, 64, 1]);
  });

  it('should handle multiple singletons', () => {
    const tensor = createTensor([1, 32, 1, 64, 1]);
    const result = rearrange(tensor, '1 h 1 w 1 -> h w 1 1');
    
    expect(result.shape).toEqual([32, 64, 1, 1]);
  });

  it('should handle singleton with ellipsis', () => {
    const tensor = createTensor([2, 3, 4, 1]);
    const result = rearrange(tensor, '... 1 -> 1 ...');
    
    expect(result.shape).toEqual([1, 2, 3, 4]);
  });
});

// =============================================================================
// Mixed Pattern Tests
// =============================================================================

describe('Mixed Pattern Operations', () => {
  it('should handle composite with ellipsis', () => {
    const tensor = createTensor([6, 7, 8, 9]); // [(a*b), ..., c]
    const result = rearrange(tensor, '(a b) ... c -> a b c ...', { axes: { a: 2 } });
    
    expect(result.shape).toEqual([2, 3, 9, 7, 8]);
  });

  it('should handle complex real-world pattern', () => {
    const tensor = createTensor([2, 60, 768]); // [batch, seq*head, dim]
    const result = rearrange(tensor, 'batch (seq head) dim -> batch seq head dim', {
      axes: { seq: 10 }
    });
    
    expect(result.shape).toEqual([2, 10, 6, 768]);
  });

  it('should handle attention pattern', () => {
    const tensor = createTensor([4, 8, 128, 64]); // [batch, heads, seq, dim]
    const result = rearrange(tensor, 'batch heads seq dim -> batch seq (heads dim)');
    
    expect(result.shape).toEqual([4, 128, 512]);
  });

  it('should handle convolution pattern', () => {
    const tensor = createTensor([32, 3, 1024]); // [batch, channels, h*w]
    const result = rearrange(tensor, 'batch channel (height width) -> batch channel height width', {
      axes: { height: 32 }
    });
    
    expect(result.shape).toEqual([32, 3, 32, 32]);
  });
});

// =============================================================================
// Edge Cases Tests
// =============================================================================

describe('Edge Cases', () => {
  it('should handle scalar tensor', () => {
    const tensor = createTensor([]);
    const result = rearrange(tensor, ' -> ');
    
    expect(result.shape).toEqual([]);
  });

  it('should handle scalar to singleton', () => {
    const tensor = createTensor([]);
    const result = rearrange(tensor, ' -> 1');
    
    expect(result.shape).toEqual([1]);
  });

  it('should handle empty tensor with dimensions', () => {
    const tensor = createTensor([0, 5, 3]);
    const result = rearrange(tensor, 'a b c -> c b a');
    
    expect(result.shape).toEqual([3, 5, 0]);
  });

  it('should handle very large dimensions', () => {
    const tensor = createTensor([1000000, 2000000]);
    const result = rearrange(tensor, 'a b -> b a');
    
    expect(result.shape).toEqual([2000000, 1000000]);
  });

  it('should handle axis names with numbers', () => {
    const tensor = createTensor([10, 20, 30]);
    const result = rearrange(tensor, 'x1 x2 x3 -> x3 x1 x2');
    
    expect(result.shape).toEqual([30, 10, 20]);
  });

  it('should handle long axis names', () => {
    const tensor = createTensor([16, 128, 768]);
    const result = rearrange(tensor, 'batch_size sequence_length hidden_dim -> sequence_length batch_size hidden_dim');
    
    expect(result.shape).toEqual([128, 16, 768]);
  });
});

// =============================================================================
// Error Handling Tests
// =============================================================================

describe('Error Handling', () => {
  it('should error on unknown output axis', () => {
    const tensor = createTensor([32, 64]);
    
    expect(() => {
      rearrange(tensor, 'h w -> h w c');
    }).toThrow(RearrangeError);
  });

  it('should error on dimension mismatch', () => {
    const tensor = createTensor([10, 20]);
    
    expect(() => {
      rearrange(tensor, 'a b c -> a b');
    }).toThrow(RearrangeError);
  });

  it('should error on provided axis mismatch', () => {
    const tensor = createTensor([10, 20, 30]);
    
    expect(() => {
      rearrange(tensor, 'a b c -> a b c', { axes: { b: 25 } });
    }).toThrow(RearrangeError);
  });

  it('should error on invalid composite split', () => {
    const tensor = createTensor([100, 3]);
    
    expect(() => {
      rearrange(tensor, '(h w) c -> h w c', { axes: { h: 30 } }); // 100/30 is not integer
    }).toThrow(RearrangeError);
  });

  it('should error on multiple unknowns in composite', () => {
    const tensor = createTensor([60, 5]);
    
    expect(() => {
      rearrange(tensor, '(a b c) d -> a b c d'); // Cannot infer a, b, c
    }).toThrow(RearrangeError);
  });

  it('should error on non-singleton dimension', () => {
    const tensor = createTensor([32, 64, 2]);
    
    expect(() => {
      rearrange(tensor, 'h w 1 -> h w'); // Expected 1, got 2
    }).toThrow(RearrangeError);
  });

  it('should error on malformed pattern', () => {
    const tensor = createTensor([10, 20]);
    
    expect(() => {
      rearrange(tensor, 'a b -> a b)'); // Unbalanced parentheses
    }).toThrow(RearrangeError);
  });

  it('should provide helpful error context', () => {
    const tensor = createTensor([10, 20]);
    
    try {
      rearrange(tensor, 'a b -> a b c');
      expect(true).toBe(false); // Should not reach here
    } catch (error) {
      expect(error).toBeInstanceOf(RearrangeError);
      if (error instanceof RearrangeError) {
        expect(error.pattern).toBe('a b -> a b c');
        expect(error.context?.inputShape).toEqual([10, 20]);
      }
    }
  });
});

// =============================================================================
// Performance Tests
// =============================================================================

describe('Performance', () => {
  it('should handle large tensors efficiently', () => {
    const tensor = createTensor([1000, 1000, 10]);
    
    const start = performance.now();
    const result = rearrange(tensor, 'h w c -> c h w');
    const end = performance.now();
    
    expect(result.shape).toEqual([10, 1000, 1000]);
    expect(end - start).toBeLessThan(100); // Should complete in under 100ms
  });

  it('should handle many small operations efficiently', () => {
    const tensor = createTensor([10, 10]);
    
    const start = performance.now();
    for (let i = 0; i < 1000; i++) {
      rearrange(tensor, 'h w -> w h');
    }
    const end = performance.now();
    
    expect(end - start).toBeLessThan(1000); // Should complete 1000 ops in under 1s
  });
});

// =============================================================================
// Integration Tests
// =============================================================================

describe('Integration with Real Patterns', () => {
  it('should work with transformer attention patterns', () => {
    // Multi-head attention reshape
    const qkv = createTensor([32, 512, 1536]); // [batch, seq, 3*heads*dim]
    const reshaped = rearrange(qkv, 'batch seq (three heads dim) -> three batch heads seq dim', {
      axes: { three: 3, heads: 8 }
    });
    
    expect(reshaped.shape).toEqual([3, 32, 8, 512, 64]);
  });

  it('should work with CNN channel reordering', () => {
    // Convert from channels-first to channels-last
    const image = createTensor([1, 3, 224, 224]); // [batch, channels, height, width]
    const result = rearrange(image, 'batch channels height width -> batch height width channels');
    
    expect(result.shape).toEqual([1, 224, 224, 3]);
  });

  it('should work with patch embedding patterns', () => {
    // Convert image patches to embedding
    const patches = createTensor([1, 196, 768]); // [batch, num_patches, embed_dim]
    const result = rearrange(patches, 'batch (h w) embed -> batch h w embed', {
      axes: { h: 14 }
    });
    
    expect(result.shape).toEqual([1, 14, 14, 768]);
  });

  it('should work with batch matrix operations', () => {
    // Batch matrix multiplication setup
    const tensor = createTensor([32, 10, 512]); // [batch, seq, hidden]
    const result = rearrange(tensor, 'batch seq hidden -> (batch seq) hidden');
    
    expect(result.shape).toEqual([320, 512]);
  });
});