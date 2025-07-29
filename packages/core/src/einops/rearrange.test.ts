/**
 * Integration tests for rearrange function
 */

import { describe, it, expect } from 'bun:test';
import { rearrange, RearrangeError } from './rearrange';
import { tensor, float32, ones } from '..';
import { cpu } from '@typetensor/backend-cpu';

describe('Simple Transpose Operations', () => {
  it('should handle basic 2D transpose', async () => {
    const testTensor = await ones([3, 4] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'h w -> w h');
    expect(result.shape).toEqual([4, 3]);
  });

  it('should handle identity pattern', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'a b c -> a b c');

    expect(result.shape).toEqual([2, 3, 4]);
  });

  it('should handle 3D transpose', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'a b c -> c a b');

    expect(result.shape).toEqual([4, 2, 3]);
  });

  // These tests are moved to invalid patterns section
});

// =============================================================================
// Axis Reordering Tests
// =============================================================================

describe('Axis Reordering Operations', () => {
  it('should handle 4D reordering (channels first to channels last)', async () => {
    const testTensor = await ones([2, 3, 32, 32] as const, { device: cpu, dtype: float32 }); // [batch, channels, height, width]
    const result = await rearrange(testTensor, 'b c h w -> b h w c');

    expect(result.shape).toEqual([2, 32, 32, 3]);
  });

  it('should handle reverse order', async () => {
    const testTensor = await ones([1, 2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'a b c d -> d c b a');

    expect(result.shape).toEqual([4, 3, 2, 1]);
  });

  it('should handle complex reordering', async () => {
    const testTensor = await ones([2, 3, 4, 5, 6] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'a b c d e -> c e a d b');

    expect(result.shape).toEqual([4, 6, 2, 5, 3]);
  });

  it('should handle single axis pattern', async () => {
    const testTensor = await ones([42] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'a -> a');

    expect(result.shape).toEqual([42]);
  });
});

// =============================================================================
// Composite Pattern Tests
// =============================================================================

describe('Composite Pattern Operations', () => {
  it('should handle basic composite splitting', async () => {
    const testTensor = await ones([2048, 3] as const, { device: cpu, dtype: float32 }); // [h*w, channels]
    const result = await rearrange(testTensor, '(h w) c -> h w c', { h: 32 }); // Need to provide h dimension

    expect(result.shape).toEqual([32, 64, 3]);
  });

  it('should handle composite merging', async () => {
    const testTensor = await ones([32, 64, 3] as const, { device: cpu, dtype: float32 }); // [height, width, channels]
    const result = await rearrange(testTensor, 'h w c -> (h w) c');

    expect(result.shape).toEqual([2048, 3]);
  });

  it('should handle multiple composites', async () => {
    const testTensor = await ones([6, 20] as const, { device: cpu, dtype: float32 }); // [a*b, c*d]
    const result = await rearrange(testTensor, '(a b) (c d) -> a b c d', { a: 2, c: 4 }); // Need to provide both a and c dimensions

    expect(result.shape).toEqual([2, 3, 4, 5]);
  });

  it('should handle composite with reordering', async () => {
    const testTensor = await ones([20, 768] as const, { device: cpu, dtype: float32 }); // [batch*seq, hidden]
    const result = await rearrange(testTensor, '(batch seq) hidden -> batch seq hidden', {
      batch: 4,
    });

    expect(result.shape).toEqual([4, 5, 768]);
  });

  it('should infer unknown dimension in composite', async () => {
    const testTensor = await ones([100, 3] as const, { device: cpu, dtype: float32 }); // [h*w, channels]
    const result = await rearrange(testTensor, '(h w) c -> h w c', { h: 10 });

    expect(result.shape).toEqual([10, 10, 3]);
  });
});

// =============================================================================
// Ellipsis Pattern Tests
// =============================================================================

describe('Ellipsis Pattern Operations', () => {
  it('should handle basic ellipsis identity', async () => {
    const testTensor = await ones([2, 3, 4, 5] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, '... -> ...');

    expect(result.shape).toEqual([2, 3, 4, 5]);
  });

  // This test is moved to invalid patterns section

  it('should handle ellipsis in middle', async () => {
    const testTensor = await ones([2, 3, 4, 5, 6] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'batch ... channels -> batch channels ...');

    expect(result.shape).toEqual([2, 6, 3, 4, 5]);
  });

  it('should handle ellipsis consuming multiple dimensions', async () => {
    const testTensor = await ones([1, 2, 3, 4, 5] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, '... last -> last ...');

    expect(result.shape).toEqual([5, 1, 2, 3, 4]);
  });

  it('should handle ellipsis with no dimensions', async () => {
    const testTensor = await ones([10, 20] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'a ... b -> b ... a');

    expect(result.shape).toEqual([20, 10]);
  });
});

// =============================================================================
// Singleton Pattern Tests
// =============================================================================

describe('Singleton Pattern Operations', () => {
  it('should handle singleton removal', async () => {
    const testTensor = await ones([32, 64, 1] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'h w 1 -> h w');

    expect(result.shape).toEqual([32, 64]);
  });

  it('should handle singleton addition', async () => {
    const testTensor = await ones([32, 64] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'h w -> h w 1');

    expect(result.shape).toEqual([32, 64, 1]);
  });

  it('should handle multiple singletons', async () => {
    const testTensor = await ones([1, 32, 1, 64, 1] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, '1 h 1 w 1 -> h w 1 1');

    expect(result.shape).toEqual([32, 64, 1, 1]);
  });

  it('should handle singleton with ellipsis', async () => {
    const testTensor = await ones([2, 3, 4, 1] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, '... 1 -> 1 ...');

    expect(result.shape).toEqual([1, 2, 3, 4]);
  });
});

// =============================================================================
// Mixed Pattern Tests
// =============================================================================

describe('Mixed Pattern Operations', () => {
  it('should handle composite with ellipsis', async () => {
    const testTensor = await ones([6, 7, 8, 9] as const, { device: cpu, dtype: float32 }); // [(a*b), ..., c]
    const result = await rearrange(testTensor, '(a b) ... c -> a b c ...', { a: 2 });

    expect(result.shape).toEqual([2, 3, 9, 7, 8]);
  });

  it('should handle complex real-world pattern', async () => {
    const testTensor = await ones([2, 60, 768] as const, { device: cpu, dtype: float32 }); // [batch, seq*head, dim]
    const result = await rearrange(testTensor, 'batch (seq head) dim -> batch seq head dim', {
      seq: 10,
    });

    expect(result.shape).toEqual([2, 10, 6, 768]);
  });

  it('should handle attention pattern', async () => {
    const testTensor = await ones([4, 8, 128, 64] as const, { device: cpu, dtype: float32 }); // [batch, heads, seq, dim]
    const result = await rearrange(testTensor, 'batch heads seq dim -> batch seq (heads dim)');

    expect(result.shape).toEqual([4, 128, 512]);
  });

  it('should handle convolution pattern', async () => {
    const testTensor = await ones([32, 3, 1024] as const, { device: cpu, dtype: float32 }); // [batch, channels, h*w]
    const result = await rearrange(
      testTensor,
      'batch channel (height width) -> batch channel height width',
      { height: 32 },
    );

    expect(result.shape).toEqual([32, 3, 32, 32]);
  });
});

// =============================================================================
// Edge Cases Tests
// =============================================================================

describe('Edge Cases', () => {
  it('should handle scalar tensor', async () => {
    const testTensor = await tensor(42, { device: cpu, dtype: float32 }); // Create scalar with value 42
    const result = await rearrange(testTensor, ' -> ');

    expect(result.shape).toEqual([]);
  });

  it('should handle scalar to singleton', async () => {
    const testTensor = await tensor(42, { device: cpu, dtype: float32 }); // Create scalar with value 42
    const result = await rearrange(testTensor, ' -> 1');

    expect(result.shape).toEqual([1]);
  });

  it('should handle empty tensor with dimensions', async () => {
    const testTensor = await ones([0, 5, 3] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'a b c -> c b a');

    expect(result.shape).toEqual([3, 5, 0]);
  });

  it('should handle moderately large dimensions', async () => {
    const testTensor = await ones([1000, 2000] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'a b -> b a');

    expect(result.shape).toEqual([2000, 1000]);
  });

  it('should handle axis names with numbers', async () => {
    const testTensor = await ones([10, 20, 30] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(testTensor, 'x1 x2 x3 -> x3 x1 x2');

    expect(result.shape).toEqual([30, 10, 20]);
  });

  it('should handle long axis names', async () => {
    const testTensor = await ones([16, 128, 768] as const, { device: cpu, dtype: float32 });
    const result = await rearrange(
      testTensor,
      'batch_size sequence_length hidden_dim -> sequence_length batch_size hidden_dim',
    );

    expect(result.shape).toEqual([128, 16, 768]);
  });
});

// =============================================================================
// Invalid Patterns (PyTorch einops compatibility)
// =============================================================================

describe('Invalid Patterns - PyTorch Compatibility', () => {
  it('should error on partial axis selection (axes only on one side)', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });

    await expect(rearrange(testTensor, 'a b c -> a c')).rejects.toThrow(
      /Identifiers only on one side of expression.*\{b\}/
    );
  });

  it('should error on axis duplication', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });

    await expect(rearrange(testTensor, 'a b -> a b a')).rejects.toThrow(
      /Indexing expression contains duplicate dimension "a"/
    );
  });

  it('should error on ellipsis with named axes dropped', async () => {
    const testTensor = await ones([2, 3, 4, 5] as const, { device: cpu, dtype: float32 });

    await expect(rearrange(testTensor, 'batch ... -> ...')).rejects.toThrow(
      /Identifiers only on one side of expression.*\{batch\}/
    );
  });
});

// =============================================================================
// Error Handling Tests
// =============================================================================

describe('Error Handling', () => {
  it('should error on unknown output axis', async () => {
    const testTensor = await ones([32, 64] as const, { device: cpu, dtype: float32 });

    await expect(rearrange(testTensor, 'h w -> h w c')).rejects.toThrow(RearrangeError);
  });

  it('should error on dimension mismatch', async () => {
    const testTensor = await ones([10, 20] as const, { device: cpu, dtype: float32 });

    await expect(rearrange(testTensor, 'a b c -> a b')).rejects.toThrow(RearrangeError);
  });

  it('should error on provided axis mismatch', async () => {
    const testTensor = await ones([10, 20, 30] as const, { device: cpu, dtype: float32 });

    await expect(rearrange(testTensor, 'a b c -> a b c', { b: 25 })).rejects.toThrow(
      RearrangeError,
    );
  });

  it('should error on invalid composite split', async () => {
    const testTensor = await ones([100, 3] as const, { device: cpu, dtype: float32 });

    await expect(rearrange(testTensor, '(h w) c -> h w c', { h: 30 })).rejects.toThrow(
      RearrangeError,
    ); // 100/30 is not integer
  });

  it('should error on multiple unknowns in composite', async () => {
    const testTensor = await ones([60, 5] as const, { device: cpu, dtype: float32 });

    await expect(rearrange(testTensor, '(a b c) d -> a b c d')).rejects.toThrow(RearrangeError); // Cannot infer a, b, c
  });

  it('should error on non-singleton dimension', async () => {
    const testTensor = await ones([32, 64, 2] as const, { device: cpu, dtype: float32 });

    await expect(rearrange(testTensor, 'h w 1 -> h w')).rejects.toThrow(RearrangeError); // Expected 1, got 2
  });

  it('should error on malformed pattern', async () => {
    const testTensor = await ones([10, 20] as const, { device: cpu, dtype: float32 });

    await expect(rearrange(testTensor, 'a b -> a b)')).rejects.toThrow(RearrangeError); // Unbalanced parentheses
  });

  it('should provide helpful error context', async () => {
    const testTensor = await ones([10, 20] as const, { device: cpu, dtype: float32 });

    try {
      await rearrange(testTensor, 'a b -> a b c');
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
  it('should handle large tensors efficiently', async () => {
    const testTensor = await ones([1000, 1000, 10] as const, { device: cpu, dtype: float32 });

    const start = performance.now();
    const result = await rearrange(testTensor, 'h w c -> c h w');
    const end = performance.now();

    expect(result.shape).toEqual([10, 1000, 1000]);
    expect(end - start).toBeLessThan(100); // Should complete in under 100ms
  });

  it('should handle many small operations efficiently', async () => {
    const testTensor = await ones([10, 10] as const, { device: cpu, dtype: float32 });

    const start = performance.now();
    for (let i = 0; i < 1000; i++) {
      await rearrange(testTensor, 'h w -> w h');
    }
    const end = performance.now();

    expect(end - start).toBeLessThan(1000); // Should complete 1000 ops in under 1s
  });
});

// =============================================================================
// Integration Tests
// =============================================================================

describe('Integration with Real Patterns', () => {
  it('should work with transformer attention patterns', async () => {
    // Multi-head attention reshape
    const qkv = await ones([32, 512, 1536] as const, { device: cpu, dtype: float32 }); // [batch, seq, 3*heads*dim]
    const reshaped = await rearrange(
      qkv,
      'batch seq (three heads dim) -> three batch heads seq dim',
      {
        three: 3,
        heads: 8,
      },
    );

    expect(reshaped.shape).toEqual([3, 32, 8, 512, 64]);
  });

  it('should work with CNN channel reordering', async () => {
    // Convert from channels-first to channels-last
    const image = await ones([1, 3, 224, 224] as const, { device: cpu, dtype: float32 }); // [batch, channels, height, width]
    const result = await rearrange(
      image,
      'batch channels height width -> batch height width channels',
    );

    expect(result.shape).toEqual([1, 224, 224, 3]);
  });

  it('should work with patch embedding patterns', async () => {
    // Convert image patches to embedding
    const patches = await ones([1, 196, 768] as const, { device: cpu, dtype: float32 }); // [batch, num_patches, embed_dim]
    const result = await rearrange(patches, 'batch (h w) embed -> batch h w embed', { h: 14 });

    expect(result.shape).toEqual([1, 14, 14, 768]);
  });

  it('should work with batch matrix operations', async () => {
    // Batch matrix multiplication setup
    const testTensor = await ones([32, 10, 512] as const, { device: cpu, dtype: float32 }); // [batch, seq, hidden]
    const result = await rearrange(testTensor, 'batch seq hidden -> (batch seq) hidden');

    expect(result.shape).toEqual([320, 512]);
  });
});
