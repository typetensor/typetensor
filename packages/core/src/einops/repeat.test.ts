/**
 * Tests for repeat function
 */

import { describe, it, expect } from 'bun:test';
import { repeat, RepeatError } from './repeat';
import { tensor, float32, ones, zeros } from '..';
import { cpu } from '@typetensor/backend-cpu';

// =============================================================================
// Basic Repeat Operations Tests (25 tests)
// =============================================================================

describe('Basic Repeat Operations', () => {
  it('should add new axis with repetition', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> h w c', { c: 4 });
    expect(result.shape).toEqual([2, 3, 4]);

    // Check values - each element repeated 4 times
    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1, 1, 1]);
    expect(data[1][2]).toEqual([1, 1, 1, 1]);
  });

  it('should repeat along existing axis', async () => {
    const testTensor = await tensor([1, 2, 3], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'w -> (w w2)', { w2: 2 });
    expect(result.shape).toEqual([6]);

    const data = await result.toArray();
    expect(data).toEqual([1, 1, 2, 2, 3, 3]);
  });

  it('should handle identity pattern', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> h w');
    expect(result.shape).toEqual([2, 3]);

    const data = await result.toArray();
    expect(data).toEqual(await testTensor.toArray());
  });

  it('should add axis at beginning', async () => {
    const testTensor = await tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await repeat(testTensor, 'h w -> batch h w', { batch: 2 });
    expect(result.shape).toEqual([2, 2, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [1, 2],
        [3, 4],
      ],
    ]);
  });

  it('should add axis in middle', async () => {
    const testTensor = await tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await repeat(testTensor, 'h w -> h c w', { c: 3 });
    expect(result.shape).toEqual([2, 3, 2]);

    const data = await result.toArray();
    expect(data[0]).toEqual([
      [1, 2],
      [1, 2],
      [1, 2],
    ]);
    expect(data[1]).toEqual([
      [3, 4],
      [3, 4],
      [3, 4],
    ]);
  });

  it('should add multiple new axes', async () => {
    const testTensor = await tensor([5], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'x -> x h w', { h: 2, w: 3 });
    expect(result.shape).toEqual([1, 2, 3]);

    const data = await result.toArray();
    expect(data).toEqual([
      [
        [5, 5, 5],
        [5, 5, 5],
      ],
    ]);
  });

  it('should handle large repetition factors', async () => {
    const testTensor = await tensor([1, 2], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'w -> (w w2)', { w2: 5 });
    expect(result.shape).toEqual([10]);

    const data = await result.toArray();
    expect(data).toEqual([1, 1, 1, 1, 1, 2, 2, 2, 2, 2]);
  });

  it('should repeat multiple axes independently', async () => {
    const testTensor = await tensor([[1, 2]], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> (h h2) (w w2)', { h2: 2, w2: 3 });
    expect(result.shape).toEqual([2, 6]);

    const data = await result.toArray();
    expect(data).toEqual([
      [1, 1, 1, 2, 2, 2],
      [1, 1, 1, 2, 2, 2],
    ]);
  });

  it('should handle single element tensor', async () => {
    const testTensor = await tensor([[42]], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> (h h2) (w w2)', { h2: 3, w2: 2 });
    expect(result.shape).toEqual([3, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [42, 42],
      [42, 42],
      [42, 42],
    ]);
  });

  it('should preserve data types', async () => {
    const testTensor = await tensor([1.5, 2.5, 3.5], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'w -> w c', { c: 2 });
    expect(result.shape).toEqual([3, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [1.5, 1.5],
      [2.5, 2.5],
      [3.5, 3.5],
    ]);
  });

  it('should handle zero tensor', async () => {
    const testTensor = await zeros([2, 2] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> h w c', { c: 3 });
    expect(result.shape).toEqual([2, 2, 3]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([0, 0, 0]);
  });

  it('should handle negative values', async () => {
    const testTensor = await tensor([-1, -2, -3], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'w -> (w w2)', { w2: 2 });
    expect(result.shape).toEqual([6]);

    const data = await result.toArray();
    expect(data).toEqual([-1, -1, -2, -2, -3, -3]);
  });

  it('should create new axis with size 1', async () => {
    const testTensor = await tensor([1, 2, 3], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'w -> w c', { c: 1 });
    expect(result.shape).toEqual([3, 1]);

    const data = await result.toArray();
    expect(data).toEqual([[1], [2], [3]]);
  });

  it('should repeat with factor 1', async () => {
    const testTensor = await tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await repeat(testTensor, 'h w -> (h h2) w', { h2: 1 });
    expect(result.shape).toEqual([2, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it('should handle mixed data values', async () => {
    const testTensor = await tensor([0, 1, -1, 2.5, -3.7], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'w -> w c', { c: 2 });
    expect(result.shape).toEqual([5, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [0, 0],
      [1, 1],
      [-1, -1],
      [2.5, 2.5],
      [-3.7, -3.7],
    ]);
  });
});

// =============================================================================
// Advanced Patterns Tests (20 tests)
// =============================================================================

describe('Advanced Repeat Patterns', () => {
  it('should handle mixed repetition and new axes', async () => {
    const testTensor = await tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await repeat(testTensor, 'h w -> (h h2) w c', { h2: 2, c: 3 });
    expect(result.shape).toEqual([4, 2, 3]);

    // Check pattern: each row repeated twice, each element gets 3 channels
    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1, 1]); // First element repeated in channels
    expect(data[1][0]).toEqual([1, 1, 1]); // Same row repeated
    expect(data[2][1]).toEqual([4, 4, 4]); // Second row, second element
  });

  it('should handle upsampling pattern', async () => {
    const testTensor = await tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await repeat(testTensor, 'h w -> (h h2) (w w2)', { h2: 2, w2: 2 });
    expect(result.shape).toEqual([4, 4]);

    // Classic 2x2 upsampling
    const data = await result.toArray();
    expect(data).toEqual([
      [1, 1, 2, 2],
      [1, 1, 2, 2],
      [3, 3, 4, 4],
      [3, 3, 4, 4],
    ]);
  });

  it('should handle complex multi-axis repetition', async () => {
    const testTensor = await tensor([[[1]], [[2]]], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'a b c -> (a a2) (b b2) (c c2) d', {
      a2: 2,
      b2: 3,
      c2: 2,
      d: 2,
    });
    expect(result.shape).toEqual([4, 3, 2, 2]);

    const data = await result.toArray();
    expect(data[0][0][0]).toEqual([1, 1]);
    expect(data[2][0][0]).toEqual([2, 2]); // Second original element repeated
  });

  it('should handle interleaved new axes and repetition', async () => {
    const testTensor = await tensor([[5, 6]], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> batch (h h2) c (w w2)', {
      batch: 2,
      h2: 2,
      c: 3,
      w2: 2,
    });
    expect(result.shape).toEqual([2, 2, 3, 4]);

    const data = await result.toArray();
    expect(data[0][0][0]).toEqual([5, 5, 6, 6]);
    expect(data[1][1][2]).toEqual([5, 5, 6, 6]); // Same pattern repeated across batch and channel
  });

  it('should handle asymmetric repetition factors', async () => {
    const testTensor = await tensor([[1, 2, 3]], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> (h h2) (w w2)', { h2: 5, w2: 2 });
    expect(result.shape).toEqual([5, 6]);

    const data = await result.toArray();
    expect(data[0]).toEqual([1, 1, 2, 2, 3, 3]);
    expect(data[4]).toEqual([1, 1, 2, 2, 3, 3]); // 5th repetition of row
  });

  it('should handle new axes with different sizes', async () => {
    const testTensor = await tensor([42], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'x -> a x b c', { a: 2, b: 3, c: 4 });
    expect(result.shape).toEqual([2, 1, 3, 4]);

    const data = await result.toArray();
    expect(data[0][0][0]).toEqual([42, 42, 42, 42]);
    expect(data[1][0][2]).toEqual([42, 42, 42, 42]);
  });

  it('should handle repetition with prime factors', async () => {
    const testTensor = await tensor([[1, 2, 3]], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> (h h2) (w w2)', { h2: 7, w2: 11 });
    expect(result.shape).toEqual([7, 33]);

    const data = await result.toArray();
    expect(data[0].slice(0, 11)).toEqual(Array(11).fill(1));
    expect(data[6].slice(22, 33)).toEqual(Array(11).fill(3));
  });

  it('should handle deep nesting with new axes', async () => {
    const testTensor = await tensor([1, 2], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'x -> a (x x2) b c d', {
      a: 2,
      x2: 3,
      b: 2,
      c: 2,
      d: 2,
    });
    expect(result.shape).toEqual([2, 6, 2, 2, 2]);

    const data = await result.toArray();
    expect(data[0][0][0][0]).toEqual([1, 1]);
    expect(data[1][3][1][1]).toEqual([2, 2]);
  });

  it('should handle large scale repetition', async () => {
    const testTensor = await tensor([[1]], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> (h h2) (w w2)', { h2: 100, w2: 100 });
    expect(result.shape).toEqual([100, 100]);

    // Check that all elements are 1
    const data = await result.toArray();
    expect(data[0]![0]).toBe(1);
    expect(data[50]![50]).toBe(1);
    expect(data[99]![99]).toBe(1);
  });

  it('should handle repetition preserving spatial relationships', async () => {
    const testTensor = await tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await repeat(testTensor, 'h w -> (h h2) (w w2)', { h2: 2, w2: 3 });
    expect(result.shape).toEqual([4, 6]);

    const data = await result.toArray();
    // Verify spatial structure is preserved
    expect(data[0].slice(0, 3)).toEqual([1, 1, 1]); // Top-left repeated
    expect(data[0].slice(3, 6)).toEqual([2, 2, 2]); // Top-right repeated
    expect(data[2].slice(0, 3)).toEqual([3, 3, 3]); // Bottom-left repeated
    expect(data[3].slice(3, 6)).toEqual([4, 4, 4]); // Bottom-right repeated
  });
});

// =============================================================================
// Composite Patterns Tests (15 tests)
// =============================================================================

describe('Composite Pattern Repeat', () => {
  it('should handle composite with provided axes', async () => {
    const testTensor = await ones([4, 6] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '(h h2) w -> h w c', { h: 2, h2: 2, c: 3 });
    expect(result.shape).toEqual([2, 6, 3]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1, 1]);
    expect(data[1][5]).toEqual([1, 1, 1]);
  });

  it('should infer missing composite dimensions', async () => {
    const testTensor = await ones([4, 6] as const, { device: cpu, dtype: float32 });
    // h2 should be inferred as 4/2 = 2
    const result = await repeat(testTensor, '(h h2) w -> h w c', { h: 2, c: 3 });
    expect(result.shape).toEqual([2, 6, 3]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1, 1]);
  });

  it('should handle nested composite patterns', async () => {
    const testTensor = await ones([8, 6] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '(h (h2 h3)) w -> h h2 w c', {
      h: 2,
      h2: 2,
      h3: 2,
      c: 3,
    });
    expect(result.shape).toEqual([2, 2, 6, 3]);

    const data = await result.toArray();
    expect(data[0][0][0]).toEqual([1, 1, 1]);
    expect(data[1][1][5]).toEqual([1, 1, 1]);
  });

  it('should handle composite with repetition', async () => {
    const testTensor = await tensor([1, 2, 3, 4, 5, 6], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '(h h2) -> h (h2 h3) c', { h: 2, h2: 3, h3: 2, c: 2 });
    expect(result.shape).toEqual([2, 6, 2]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1]); // First element of first group
    expect(data[1][0]).toEqual([4, 4]); // First element of second group
  });

  it('should handle multiple composite axes', async () => {
    const testTensor = await ones([4, 6, 8] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '(a a2) (b b2) (c c2) -> a b c d', {
      a: 2,
      a2: 2,
      b: 3,
      b2: 2,
      c: 4,
      c2: 2,
      d: 5,
    });
    expect(result.shape).toEqual([2, 3, 4, 5]);

    const data = await result.toArray();
    expect(data[0][0][0]).toEqual([1, 1, 1, 1, 1]);
  });

  it('should handle composite decomposition with new axes', async () => {
    const testTensor = await tensor([1, 2, 3, 4, 5, 6, 7, 8], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '(h w) -> h w c', { h: 2, w: 4, c: 3 });
    expect(result.shape).toEqual([2, 4, 3]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1, 1]);
    expect(data[0][3]).toEqual([4, 4, 4]);
    expect(data[1][0]).toEqual([5, 5, 5]);
    expect(data[1][3]).toEqual([8, 8, 8]);
  });

  it('should handle composite with partial specification', async () => {
    const testTensor = await ones([12] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '(h w c) -> h w c d', { h: 3, w: 4, d: 2 });
    expect(result.shape).toEqual([3, 4, 1, 2]); // c inferred as 1

    const data = await result.toArray();
    expect(data[0][0][0]).toEqual([1, 1]);
  });

  it('should handle complex composite nesting', async () => {
    const testTensor = await ones([24] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '(a (b c) d) -> a b c d e', {
      a: 2,
      b: 3,
      c: 2,
      d: 2,
      e: 4,
    });
    expect(result.shape).toEqual([2, 3, 2, 2, 4]);

    const data = await result.toArray();
    expect(data[0][0][0][0]).toEqual([1, 1, 1, 1]);
    expect(data[1][2][1][1]).toEqual([1, 1, 1, 1]);
  });

  it('should handle composite with large factors', async () => {
    const testTensor = await ones([6] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '(h w) -> (h h2) (w w2)', { h: 2, w: 3, h2: 5, w2: 7 });
    expect(result.shape).toEqual([10, 21]);

    const data = await result.toArray();
    expect(data[0][0]).toBe(1);
    expect(data[9][20]).toBe(1);
  });

  it('should handle composite reshaping with repetition', async () => {
    const testTensor = await tensor(
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await repeat(testTensor, 'h (w w2) -> (h h2) w c', { w: 2, w2: 2, h2: 3, c: 2 });
    expect(result.shape).toEqual([6, 2, 2]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1]); // First element
    expect(data[3][0]).toEqual([5, 5]); // First element of second original row
  });
});

// =============================================================================
// Ellipsis Patterns Tests (15 tests)
// =============================================================================

describe('Ellipsis Patterns', () => {
  it('should add channels to image', async () => {
    const testTensor = await ones([32, 32] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '... -> ... c', { c: 3 });
    expect(result.shape).toEqual([32, 32, 3]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1, 1]);
    expect(data[31][31]).toEqual([1, 1, 1]);
  });

  it('should handle batch ellipsis', async () => {
    const testTensor = await ones([8, 64, 64] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'batch ... -> batch ... c', { c: 3 });
    expect(result.shape).toEqual([8, 64, 64, 3]);

    const data = await result.toArray();
    expect(data[0]![0]![0]).toEqual([1, 1, 1]);
    expect(data[7]![63]![63]).toEqual([1, 1, 1]);
  });

  it('should repeat ellipsis dimensions', async () => {
    const testTensor = await ones([4, 4] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '... -> (... r)', { r: 2 });
    expect(result.shape).toEqual([32]);

    const data = await result.toArray();
    expect(data[0]).toBe(1);
    expect(data[31]).toBe(1);
  });

  it('should handle ellipsis at beginning', async () => {
    const testTensor = await ones([3, 4, 5] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '... -> batch ...', { batch: 2 });
    expect(result.shape).toEqual([2, 3, 4, 5]);

    const data = await result.toArray();
    expect(data[0][0][0][0]).toBe(1);
    expect(data[1][2][3][4]).toBe(1);
  });

  it('should handle ellipsis in middle', async () => {
    const testTensor = await ones([2, 3, 4, 5] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'batch ... c -> batch ... c d', { d: 6 });
    expect(result.shape).toEqual([2, 3, 4, 5, 6]);

    const data = await result.toArray();
    expect(data[0][0][0][0]).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it('should handle ellipsis with repetition factor', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'batch ... -> (batch b2) ...', { b2: 3 });
    expect(result.shape).toEqual([6, 3, 4]);

    const data = await result.toArray();
    expect(data[0][0][0]).toBe(1);
    expect(data[3][2][3]).toBe(1); // Second original batch, repeated
  });

  it('should handle multiple ellipsis operations', async () => {
    const testTensor = await ones([16, 16] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '... -> batch ... features', { batch: 4, features: 8 });
    expect(result.shape).toEqual([4, 16, 16, 8]);

    const data = await result.toArray();
    expect(data[0]?.[0]?.[0] as number[]).toEqual(Array(8).fill(1));
    expect(data[3]?.[15]?.[15] as number[]).toEqual(Array(8).fill(1));
  });

  it('should handle ellipsis with complex patterns', async () => {
    const testTensor = await ones([2, 8, 8, 3] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'batch ... c -> batch (... r) (c c2)', { r: 2, c2: 2 });
    expect(result.shape).toEqual([2, 128, 6]);

    const data = await result.toArray();
    expect(data[0]?.[0] as number[]).toEqual(Array(6).fill(1));
  });

  it('should preserve ellipsis content', async () => {
    const testTensor = await tensor(
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await repeat(testTensor, '... -> ... c', { c: 2 });
    expect(result.shape).toEqual([2, 2, 2, 2]);

    const data = await result.toArray();
    expect(data[0][0][0]).toEqual([1, 1]);
    expect(data[0][0][1]).toEqual([2, 2]);
    expect(data[1][1][1]).toEqual([8, 8]);
  });

  it('should handle 1D ellipsis', async () => {
    const testTensor = await tensor([1, 2, 3, 4, 5], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '... -> batch ...', { batch: 3 });
    expect(result.shape).toEqual([3, 5]);

    const data = await result.toArray();
    expect(data[0]).toEqual([1, 2, 3, 4, 5]);
    expect(data[2]).toEqual([1, 2, 3, 4, 5]);
  });

  it('should handle 3D ellipsis', async () => {
    const testTensor = await ones([4, 4, 4] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '... -> ... c', { c: 2 });
    expect(result.shape).toEqual([4, 4, 4, 2]);

    const data = await result.toArray();
    expect(data[0][0][0]).toEqual([1, 1]);
    expect(data[3][3][3]).toEqual([1, 1]);
  });

  it('should handle ellipsis with large dimensions', async () => {
    const testTensor = await ones([100, 100] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '... -> ... c', { c: 1 });
    expect(result.shape).toEqual([100, 100, 1]);

    const data = await result.toArray();
    expect(data[0]![0]).toEqual([1]);
    expect(data[99]![99]).toEqual([1]);
  });
});

// =============================================================================
// Edge Cases Tests (15 tests)
// =============================================================================

describe('Edge Cases', () => {
  it('should handle 1D tensors', async () => {
    const testTensor = await tensor([1, 2, 3, 4, 5], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'x -> x c', { c: 3 });
    expect(result.shape).toEqual([5, 3]);

    const data = await result.toArray();
    expect(data[0]).toEqual([1, 1, 1]);
    expect(data[4]).toEqual([5, 5, 5]);
  });

  it('should handle scalar to vector', async () => {
    const testTensor = await tensor(42, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, ' -> c', { c: 5 });
    expect(result.shape).toEqual([5]);

    const data = await result.toArray();
    expect(data).toEqual([42, 42, 42, 42, 42]);
  });

  it('should handle scalar to matrix', async () => {
    const testTensor = await tensor(1, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, ' -> h w', { h: 2, w: 3 });
    expect(result.shape).toEqual([2, 3]);

    const data = await result.toArray();
    expect(data).toEqual([
      [1, 1, 1],
      [1, 1, 1],
    ]);
  });

  it('should handle scalar to 3D', async () => {
    const testTensor = await tensor(7, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, ' -> h w c', { h: 2, w: 2, c: 2 });
    expect(result.shape).toEqual([2, 2, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [
        [7, 7],
        [7, 7],
      ],
      [
        [7, 7],
        [7, 7],
      ],
    ]);
  });

  it('should handle singleton dimensions', async () => {
    const testTensor = await ones([2, 1, 3] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h 1 w -> h c w', { c: 4 });
    expect(result.shape).toEqual([2, 4, 3]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1, 1]);
    expect(data[1][3]).toEqual([1, 1, 1]);
  });

  it('should handle multiple singletons', async () => {
    const testTensor = await ones([1, 2, 1, 3, 1] as const, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, '1 h 1 w 1 -> a h b w c', { a: 2, b: 3, c: 4 });
    expect(result.shape).toEqual([2, 2, 3, 3, 4]);

    const data = await result.toArray();
    expect(data[0][0][0][0]).toEqual([1, 1, 1, 1]);
  });

  it('should handle very small tensors', async () => {
    const testTensor = await tensor([[1]], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> (h h2) (w w2) c', { h2: 10, w2: 10, c: 5 });
    expect(result.shape).toEqual([10, 10, 5]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1, 1, 1, 1]);
    expect(data[9][9]).toEqual([1, 1, 1, 1, 1]);
  });

  it('should handle empty-like patterns', async () => {
    const testTensor = await tensor(42, { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, ' -> h', { h: 3 });
    expect(result.shape).toEqual([3]);
  });

  it('should handle extreme repetition factors', async () => {
    const testTensor = await tensor([1], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'x -> (x r)', { r: 1000 });
    expect(result.shape).toEqual([1000]);

    const data = await result.toArray() as number[];
    expect(data[0] as number).toBe(1);
    expect(data[999] as number).toBe(1);
    expect(data.every((x: number) => x === 1)).toBe(true);
  });

  it('should handle fractional-like data', async () => {
    const testTensor = await tensor([0.1, 0.2, 0.3], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'x -> x r', { r: 2 });
    expect(result.shape).toEqual([3, 2]);

    const data = await result.toArray();
    expect(data[0]).toEqual([0.1, 0.1]);
    expect(data[1]).toEqual([0.2, 0.2]);
    expect(data[2]).toEqual([0.3, 0.3]);
  });

  it('should handle mixed positive/negative values', async () => {
    const testTensor = await tensor([1, -2, 3, -4], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'x -> x r', { r: 3 });
    expect(result.shape).toEqual([4, 3]);

    const data = await result.toArray();
    expect(data[0]).toEqual([1, 1, 1]);
    expect(data[1]).toEqual([-2, -2, -2]);
    expect(data[2]).toEqual([3, 3, 3]);
    expect(data[3]).toEqual([-4, -4, -4]);
  });

  it('should handle single dimension expansion', async () => {
    const testTensor = await tensor([5], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'x -> x y z', { y: 1, z: 1 });
    expect(result.shape).toEqual([1, 1, 1]);

    const data = await result.toArray();
    expect(data).toEqual([[[5]]]);
  });

  it('should handle dimension reordering with repetition', async () => {
    const testTensor = await tensor([[1, 2, 3]], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'h w -> w (h h2) c', { h2: 4, c: 2 });
    expect(result.shape).toEqual([3, 4, 2]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1]);
    expect(data[1][3]).toEqual([2, 2]);
    expect(data[2][2]).toEqual([3, 3]);
  });

  it('should preserve precision with small values', async () => {
    const testTensor = await tensor([1e-10, 2e-10], { device: cpu, dtype: float32 });
    const result = await repeat(testTensor, 'x -> x r', { r: 2 });
    expect(result.shape).toEqual([2, 2]);

    const data = await result.toArray();
    expect(data[0][0]).toBeCloseTo(1e-10);
    expect(data[1][1]).toBeCloseTo(2e-10);
  });
});

// =============================================================================
// Real-World Use Cases Tests (15 tests)
// =============================================================================

describe('Computer Vision Patterns', () => {
  it('should convert grayscale to RGB', async () => {
    const grayscale = await tensor(
      Array(224)
        .fill(null)
        .map(() => Array(224).fill(128)),
      { device: cpu, dtype: float32 },
    );
    const rgb = await repeat(grayscale, 'h w -> h w c', { c: 3 });
    expect(rgb.shape).toEqual([224, 224, 3]);

    // Each pixel should have same value in all 3 channels
    const data = await rgb.toArray();
    expect(data[0]?.[0]).toEqual([128, 128, 128]);
    expect(data[100]?.[100]).toEqual([128, 128, 128]);
  });

  it('should perform 2x upsampling', async () => {
    // Small 2x2 image
    const image = await tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: cpu, dtype: float32 },
    );
    const upsampled = await repeat(image, 'h w -> (h h2) (w w2)', { h2: 2, w2: 2 });
    expect(upsampled.shape).toEqual([4, 4]);

    // Check upsampling pattern
    const data = await upsampled.toArray();
    expect(data[0]).toEqual([1, 1, 2, 2]);
    expect(data[1]).toEqual([1, 1, 2, 2]);
    expect(data[2]).toEqual([3, 3, 4, 4]);
    expect(data[3]).toEqual([3, 3, 4, 4]);
  });

  it('should add batch dimension', async () => {
    const image = await ones([32, 32, 3] as const, { device: cpu, dtype: float32 });
    const batch = await repeat(image, 'h w c -> batch h w c', { batch: 8 });
    expect(batch.shape).toEqual([8, 32, 32, 3]);

    const data = await batch.toArray();
    expect(data[0][0][0]).toEqual([1, 1, 1]);
    expect(data[7][31][31]).toEqual([1, 1, 1]);
  });

  it('should create image patches', async () => {
    const patch = await tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: cpu, dtype: float32 },
    );
    const tiled = await repeat(patch, 'h w -> (h h2) (w w2) c', { h2: 3, w2: 3, c: 3 });
    expect(tiled.shape).toEqual([6, 6, 3]);

    const data = await tiled.toArray();
    expect(data[0][0]).toEqual([1, 1, 1]);
    expect(data[2][2]).toEqual([3, 3, 3]); // Center patch element
  });

  it('should simulate data augmentation', async () => {
    const image = await tensor([[[1, 2, 3]]], { device: cpu, dtype: float32 });
    const augmented = await repeat(image, 'h w c -> (h h2) (w w2) c', { h2: 4, w2: 4 });
    expect(augmented.shape).toEqual([4, 4, 3]);

    const data = await augmented.toArray();
    expect(data[0][0]).toEqual([1, 2, 3]);
    expect(data[3][3]).toEqual([1, 2, 3]);
  });
});

describe('Time Series Patterns', () => {
  it('should add feature dimension', async () => {
    const timeSeries = await ones([100] as const, { device: cpu, dtype: float32 });
    const features = await repeat(timeSeries, 'time -> time features', { features: 64 });
    expect(features.shape).toEqual([100, 64]);

    const data = await features.toArray();
    expect(data[0]).toEqual(Array(64).fill(1));
    expect(data[99]).toEqual(Array(64).fill(1));
  });

  it('should perform temporal upsampling', async () => {
    const batch = await ones([32, 50] as const, { device: cpu, dtype: float32 });
    const upsampled = await repeat(batch, 'batch time -> batch (time t2)', { t2: 4 });
    expect(upsampled.shape).toEqual([32, 200]);

    const data = await upsampled.toArray();
    expect(data[0].slice(0, 4)).toEqual([1, 1, 1, 1]);
    expect(data[31].slice(196, 200)).toEqual([1, 1, 1, 1]);
  });

  it('should create multi-scale features', async () => {
    const sequence = await tensor(
      Array(25)
        .fill(null)
        .map((_, i) => i + 1),
      { device: cpu, dtype: float32 },
    );
    const multiScale = await repeat(sequence, 'time -> (time t2) features', {
      t2: 4,
      features: 16,
    });
    expect(multiScale.shape).toEqual([100, 16]);

    const data = await multiScale.toArray();
    expect(data[0] as number[]).toEqual(Array(16).fill(1));
    expect(data[4] as number[]).toEqual(Array(16).fill(2));
  });

  it('should handle sequence batching', async () => {
    const sequence = await ones([50, 64] as const, { device: cpu, dtype: float32 });
    const batched = await repeat(sequence, 'time features -> batch time features', { batch: 16 });
    expect(batched.shape).toEqual([16, 50, 64]);

    const data = await batched.toArray();
    expect(data[0]?.[0] as number[]).toEqual(Array(64).fill(1));
    expect(data[15]?.[49] as number[]).toEqual(Array(64).fill(1));
  });

  it('should expand time series dimensions', async () => {
    const signal = await tensor(
      Array(100)
        .fill(null)
        .map((_, i) => Math.sin(i / 10)),
      { device: cpu, dtype: float32 },
    );
    const expanded = await repeat(signal, 'time -> batch time features', {
      batch: 8,
      features: 32,
    });
    expect(expanded.shape).toEqual([8, 100, 32]);

    const data = await expanded.toArray();
    expect(data[0]?.[0]?.[0]).toBeCloseTo(0);
    expect(data[7]?.[50]?.[31]).toBeCloseTo(Math.sin(50 / 10));
  });
});

// =============================================================================
// Error Cases Tests (15 tests)
// =============================================================================

describe('Error Cases', () => {
  it('should throw error for missing axis dimensions', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    // @ts-expect-error - TypeScript should show: [Repeat ❌] Axis Error: New axis 'c' requires explicit size
    await expect(repeat(testTensor, 'h w -> h w c')).rejects.toThrow(RepeatError);
  });

  it('should throw error for multiple missing axes', async () => {
    const testTensor = await ones([2] as const, { device: cpu, dtype: float32 });
    // @ts-expect-error - TypeScript should show: [Repeat ❌] Axis Error: New axes require explicit sizes
    await expect(repeat(testTensor, 'h -> h w c')).rejects.toThrow(RepeatError);
  });

  it('should throw error for invalid axis sizes', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    // @ts-expect-error - TypeScript should show: [Repeat ❌] Axis Error: Invalid size 0 for axis 'c'
    await expect(repeat(testTensor, 'h w -> h w c', { c: 0 })).rejects.toThrow(RepeatError);

    // @ts-expect-error - TypeScript should show: [Repeat ❌] Axis Error: Invalid size 0 for axis 'c'
    await expect(repeat(testTensor, 'h w -> h w c', { c: 0 })).rejects.toThrow(
      "Invalid size for axis 'c': 0",
    );

    // @ts-expect-error - TypeScript should show: [Repeat ❌] Axis Error: Invalid size -1 for axis 'c'
    await expect(repeat(testTensor, 'h w -> h w c', { c: -1 })).rejects.toThrow(RepeatError);
  });

  it('should throw error for duplicate axes', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    // @ts-expect-error - TypeScript should show: [Repeat ❌] Axis Error: Duplicate axis 'h' in output
    await expect(repeat(testTensor, 'h w c -> h h c')).rejects.toThrow('Duplicate axes');

    // @ts-expect-error - TypeScript should show: [Repeat ❌] Axis Error: Duplicate axis 'h' in input
    await expect(repeat(testTensor, 'h h c -> h c d', { d: 2 })).rejects.toThrow('Duplicate axes');
  });

  it('should throw error for multiple ellipsis', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    // @ts-expect-error - TypeScript should show: [Repeat ❌] Axis Error: Multiple ellipsis '...' in input
    await expect(repeat(testTensor, '... ... c -> c d', { d: 2 })).rejects.toThrow(
      'Multiple ellipsis',
    );

    // @ts-expect-error - TypeScript should show: [Repeat ❌] Axis Error: Multiple ellipsis '...' in output
    await expect(repeat(testTensor, 'h w c -> ... c ...', { c: 2 })).rejects.toThrow(
      'Multiple ellipsis',
    );
  });

  it('should throw error for rank mismatch', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    // @ts-expect-error - TypeScript should show: [Repeat ❌] Shape Error: Pattern expects 3 dimensions
    await expect(repeat(testTensor, 'h w c -> h w c d', { d: 4 })).rejects.toThrow(RepeatError);
  });

  it('should throw error for composite resolution failure', async () => {
    const testTensor = await ones([4, 6] as const, { device: cpu, dtype: float32 });
    // This should fail: repeat cannot decompose input axes, only create new output axes
    // @ts-expect-error - TypeScript should show: [Repeat ❌] Shape Error: Cannot resolve '(h h2)' from dimension 6. Specify axis values: repeat(tensor, pattern, {axis: number})
    await expect(repeat(testTensor, '(h h2) w -> h w c', { h: 3, h2: 2, c: 3 })).rejects.toThrow(
      'Cannot resolve',
    );
  });

  it('should throw error for invalid axis names', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    await expect(repeat(testTensor, '2h w -> w c', { c: 2 })).rejects.toThrow();
  });

  it('should throw error for non-integer repetition factors', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    // Note: This might be caught at TypeScript level or runtime
    // @ts-expect-error - TypeScript should show: [Repeat ❌] Axis Error: Invalid size 2.5 for axis 'c'
    await expect(repeat(testTensor, 'h w -> h w c', { c: 2.5 } as any)).rejects.toThrow(
      RepeatError,
    );
  });

  it('should handle partial axis specification gracefully', async () => {
    const testTensor = await ones([4, 6] as const, { device: cpu, dtype: float32 });
    // Should infer h2 = 2 from 4/2
    const result = await repeat(testTensor, '(h h2) w -> h w c', { h: 2, c: 3 });
    expect(result.shape).toEqual([2, 6, 3]);
  });

  it('should throw error for conflicting specifications', async () => {
    const testTensor = await ones([6] as const, { device: cpu, dtype: float32 });
    // @ts-expect-error - TypeScript should show: [Repeat ❌] Shape Error: Cannot resolve '(h h2)' from dimension 6. Specify axis values: repeat(tensor, pattern, {axis: number})
    await expect(repeat(testTensor, '(h h2) -> h c', { h: 2, h2: 4, c: 3 })).rejects.toThrow();
  });

  it('should handle empty pattern gracefully', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    // @ts-expect-error - TypeScript should show parse error
    await expect(repeat(testTensor, '', { c: 2 })).rejects.toThrow(RepeatError);
  });

  it('should handle malformed patterns', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    // @ts-expect-error - TypeScript should show parse error
    await expect(repeat(testTensor, 'h w ->', { c: 2 })).rejects.toThrow(RepeatError);
    // @ts-expect-error - TypeScript should show parse error
    await expect(repeat(testTensor, '-> h w c', { c: 2 })).rejects.toThrow(RepeatError);
  });

  it('should handle tensor type mismatches', async () => {
    const notATensor = { shape: [2, 3], data: [1, 2, 3, 4, 5, 6] };
    // Should fail at runtime (using as any bypasses TypeScript checks)
    await expect(repeat(notATensor as any, 'h w -> h w c', { c: 2 })).rejects.toThrow();
  });

  it('should provide helpful error messages', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    try {
      // @ts-expect-error - TypeScript should show error
      await repeat(testTensor, 'h w -> h w c d e');
    } catch (error) {
      expect(error).toBeInstanceOf(RepeatError);
      expect((error as RepeatError).message).toContain('explicit sizes');
      expect((error as RepeatError).pattern).toBe('h w -> h w c d e');
    }
  });
});
