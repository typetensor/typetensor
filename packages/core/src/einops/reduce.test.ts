/**
 * Tests for reduce function
 */

import { describe, it, expect } from 'bun:test';
import { reduce, ReduceError } from './reduce';
import { tensor, float32, ones, zeros } from '..';
import { cpu } from '@typetensor/backend-cpu';

describe('Basic Reduce Operations', () => {
  it('should reduce single axis', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w c -> h c', 'sum');
    expect(result.shape).toEqual([2, 4]);

    // Check values - sum of 3 ones = 3
    const data = await result.toArray();
    expect(data).toEqual([
      [3, 3, 3, 3],
      [3, 3, 3, 3],
    ]);
  });

  it('should reduce multiple axes', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w c -> c', 'sum');
    expect(result.shape).toEqual([4]);

    // Check values - sum of 2*3=6 ones = 6
    const data = await result.toArray();
    expect(data).toEqual([6, 6, 6, 6]);
  });

  it('should handle mean reduction', async () => {
    const testTensor = await tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'h w -> w', 'mean');
    expect(result.shape).toEqual([3]);

    // Check values - mean of columns
    const data = await result.toArray();
    expect(data).toEqual([2.5, 3.5, 4.5]);
  });

  it('should handle max reduction', async () => {
    const testTensor = await tensor(
      [
        [1, 5, 3],
        [4, 2, 6],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'h w -> w', 'max');
    expect(result.shape).toEqual([3]);

    // Check values - max of columns
    const data = await result.toArray();
    expect(data).toEqual([4, 5, 6]);
  });

  it('should handle min reduction', async () => {
    const testTensor = await tensor(
      [
        [1, 5, 3],
        [4, 2, 6],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'h w -> w', 'min');
    expect(result.shape).toEqual([3]);

    // Check values - min of columns
    const data = await result.toArray();
    expect(data).toEqual([1, 2, 3]);
  });

  it('should handle prod reduction', async () => {
    const testTensor = await tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'h w -> w', 'prod');
    expect(result.shape).toEqual([3]);

    // Check values - product of columns
    const data = await result.toArray();
    expect(data).toEqual([4, 10, 18]);
  });
});

describe('Reduction with Different Patterns', () => {
  it('should reduce first dimension', async () => {
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
    const result = await reduce(testTensor, 'batch h w -> h w', 'sum');
    expect(result.shape).toEqual([2, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [6, 8],
      [10, 12],
    ]);
  });

  it('should reduce middle dimension', async () => {
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
    const result = await reduce(testTensor, 'batch h w -> batch w', 'sum');
    expect(result.shape).toEqual([2, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [4, 6],
      [12, 14],
    ]);
  });

  it('should reduce last dimension', async () => {
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
    const result = await reduce(testTensor, 'batch h w -> batch h', 'sum');
    expect(result.shape).toEqual([2, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [3, 7],
      [11, 15],
    ]);
  });

  it('should handle non-contiguous reduction', async () => {
    const testTensor = await tensor(
      [
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        [
          [7, 8, 9],
          [10, 11, 12],
        ],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'batch h w -> h', 'mean');
    expect(result.shape).toEqual([2]);

    // Mean across batch and width dimensions
    const data = await result.toArray();
    expect(data).toEqual([5, 8]); // [(1+2+3+7+8+9)/6, (4+5+6+10+11+12)/6]
  });
});

describe('Global Reduction', () => {
  it('should reduce all dimensions to scalar', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w c ->', 'sum');
    expect(result.shape).toEqual([]);

    // Check value - sum of 2*3*4=24 ones = 24
    const data = await result.toArray();
    expect(data).toBe(24);
  });

  it('should handle empty output pattern', async () => {
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
    const result = await reduce(testTensor, 'a b c ->', 'mean');
    expect(result.shape).toEqual([]);

    // Check value - mean of all elements
    const data = await result.toArray();
    expect(data).toBe(4.5);
  });

  it('should handle global max', async () => {
    const testTensor = await tensor(
      [
        [
          [1, 9],
          [3, 4],
        ],
        [
          [5, 2],
          [7, 8],
        ],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'a b c ->', 'max');
    expect(result.shape).toEqual([]);

    const data = await result.toArray();
    expect(data).toBe(9);
  });

  it('should handle global min', async () => {
    const testTensor = await tensor(
      [
        [
          [1, 9],
          [3, 4],
        ],
        [
          [5, 2],
          [7, 8],
        ],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'a b c ->', 'min');
    expect(result.shape).toEqual([]);

    const data = await result.toArray();
    expect(data).toBe(1);
  });
});

describe('Keep Dimensions', () => {
  it('should keep reduced dimensions as size 1', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w c -> h c', 'sum', true);
    expect(result.shape).toEqual([2, 1, 4]);

    // Values should be the same as without keepdims
    const data = await result.toArray();
    expect(data).toEqual([[[3, 3, 3, 3]], [[3, 3, 3, 3]]]);
  });

  it('should keep all reduced dimensions', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w c -> c', 'mean', true);
    expect(result.shape).toEqual([1, 1, 4]);

    const data = await result.toArray();
    expect(data).toEqual([[[1, 1, 1, 1]]]);
  });

  it('should handle keepdims with global reduction', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w c ->', 'sum', true);
    expect(result.shape).toEqual([1, 1, 1]);

    const data = await result.toArray();
    expect(data).toEqual([[[24]]]);
  });
});

describe('Composite Pattern Reduction', () => {
  it('should handle composite pattern with provided axes', async () => {
    const testTensor = await ones([4, 6] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, '(h h2) (w w2) -> h w', 'sum', false, {
      h: 2,
      h2: 2,
      w: 3,
      w2: 2,
    });
    expect(result.shape).toEqual([2, 3]);

    // Each output element is sum of 2*2=4 ones
    const data = await result.toArray();
    expect(data).toEqual([
      [4, 4, 4],
      [4, 4, 4],
    ]);
  });

  it('should handle partial composite reduction', async () => {
    const testTensor = await ones([4, 6] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, '(h h2) w -> h w', 'mean', false, { h: 2, h2: 2 });
    expect(result.shape).toEqual([2, 6]);

    // Each output element is mean of 2 ones = 1
    const data = await result.toArray();
    expect(data).toEqual([
      [1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1],
    ]);
  });

  it('should handle nested composite patterns', async () => {
    const testTensor = await ones([8, 6] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, '(h (h2 h3)) (w w2) -> h h2 w', 'sum', false, {
      h: 2,
      h2: 2,
      h3: 2,
      w: 3,
      w2: 2,
    });
    expect(result.shape).toEqual([2, 2, 3]);

    // Each output element is sum of h3*w2 = 2*2 = 4 ones
    const data = await result.toArray();
    expect(data).toEqual([
      [
        [4, 4, 4],
        [4, 4, 4],
      ],
      [
        [4, 4, 4],
        [4, 4, 4],
      ],
    ]);
  });

  it('should handle composite with different operations', async () => {
    const testTensor = await tensor(
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'h (w w2) -> h w', 'max', false, { w: 2, w2: 2 });
    expect(result.shape).toEqual([2, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [2, 4],
      [6, 8],
    ]);
  });
});

describe('Ellipsis Patterns', () => {
  it('should reduce ellipsis dimensions', async () => {
    const testTensor = await ones([2, 3, 4, 5] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'batch ... c -> batch c', 'sum');
    expect(result.shape).toEqual([2, 5]);

    // Sum over middle dimensions: 3*4=12
    const data = await result.toArray();
    expect(data[0]).toEqual([12, 12, 12, 12, 12]);
    expect(data[1]).toEqual([12, 12, 12, 12, 12]);
  });

  it('should preserve ellipsis in output', async () => {
    const testTensor = await ones([2, 3, 4, 5] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'batch ... c -> batch ...', 'sum');
    expect(result.shape).toEqual([2, 3, 4]);

    // Sum over last dimension: 5
    const data = await result.toArray();
    expect(data[0][0]).toEqual([5, 5, 5, 5]);
  });

  it('should handle ellipsis at beginning', async () => {
    const testTensor = await ones([2, 3, 4, 5] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, '... h w -> ...', 'sum');
    expect(result.shape).toEqual([2, 3]);

    // Sum over last two dimensions: 4*5=20
    const data = await result.toArray();
    expect(data).toEqual([
      [20, 20, 20],
      [20, 20, 20],
    ]);
  });

  it('should handle ellipsis at end', async () => {
    const testTensor = await ones([2, 3, 4, 5] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'a b ... -> a b', 'sum');
    expect(result.shape).toEqual([2, 3]);

    // Sum over last two dimensions: 4*5=20
    const data = await result.toArray();
    expect(data).toEqual([
      [20, 20, 20],
      [20, 20, 20],
    ]);
  });

  it('should handle ellipsis with keepdims', async () => {
    const testTensor = await ones([2, 3, 4, 5] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'batch ... c -> batch c', 'sum', true);
    expect(result.shape).toEqual([2, 1, 1, 5]);

    const data = await result.toArray();
    expect(data[0][0][0]).toEqual([12, 12, 12, 12, 12]);
  });
});

describe('Edge Cases', () => {
  it('should handle 1D tensor', async () => {
    const testTensor = await tensor([1, 2, 3, 4, 5], { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'x ->', 'sum');
    expect(result.shape).toEqual([]);

    const data = await result.toArray();
    expect(data).toBe(15);
  });

  it('should handle 2D tensor with single row', async () => {
    const testTensor = await tensor([[1, 2, 3, 4]], { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w -> w', 'mean');
    expect(result.shape).toEqual([4]);

    const data = await result.toArray();
    expect(data).toEqual([1, 2, 3, 4]);
  });

  it('should handle 2D tensor with single column', async () => {
    const testTensor = await tensor([[1], [2], [3], [4]], { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w -> h', 'sum');
    expect(result.shape).toEqual([4]);

    const data = await result.toArray();
    expect(data).toEqual([1, 2, 3, 4]);
  });

  it('should handle tensor with zeros', async () => {
    const testTensor = await zeros([2, 3, 4] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w c -> c', 'sum');
    expect(result.shape).toEqual([4]);

    const data = await result.toArray();
    expect(data).toEqual([0, 0, 0, 0]);
  });

  it('should handle negative values', async () => {
    const testTensor = await tensor(
      [
        [-1, -2, -3],
        [-4, -5, -6],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'h w -> w', 'sum');
    expect(result.shape).toEqual([3]);

    const data = await result.toArray();
    expect(data).toEqual([-5, -7, -9]);
  });

  it('should handle mixed positive and negative values', async () => {
    const testTensor = await tensor(
      [
        [1, -2, 3],
        [-4, 5, -6],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'h w -> w', 'mean');
    expect(result.shape).toEqual([3]);

    const data = await result.toArray();
    expect(data).toEqual([-1.5, 1.5, -1.5]);
  });
});

describe('Singleton Dimensions', () => {
  it('should handle singleton in input', async () => {
    const testTensor = await ones([2, 1, 3] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h 1 w -> h w', 'sum');
    expect(result.shape).toEqual([2, 3]);

    const data = await result.toArray();
    expect(data).toEqual([
      [1, 1, 1],
      [1, 1, 1],
    ]);
  });

  it('should handle singleton in output', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w -> h 1', 'sum');
    expect(result.shape).toEqual([2, 1]);

    const data = await result.toArray();
    expect(data).toEqual([[3], [3]]);
  });

  it('should handle multiple singletons', async () => {
    const testTensor = await ones([2, 1, 3, 1, 4] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'a 1 b 1 c -> a b c', 'sum');
    expect(result.shape).toEqual([2, 3, 4]);

    const data = await result.toArray();
    expect(data[0][0]).toEqual([1, 1, 1, 1]);
  });
});

describe('Error Cases', () => {
  it('should throw error for unknown axes in output', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    await expect(reduce(testTensor, 'h w -> h w c', 'sum')).rejects.toThrow(ReduceError);
    await expect(reduce(testTensor, 'h w -> h w c', 'sum')).rejects.toThrow(
      'Unknown axes in output',
    );
  });

  it('should handle no reduction (identity)', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, 'h w -> h w', 'sum');
    expect(result.shape).toEqual([2, 3]);

    // Should be identical to input
    const data = await result.toArray();
    expect(data).toEqual([
      [1, 1, 1],
      [1, 1, 1],
    ]);
  });

  it('should throw error for duplicate axes in output', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    await expect(reduce(testTensor, 'h w c -> h h', 'sum')).rejects.toThrow(
      'Duplicate axes in output pattern',
    );
  });

  it('should throw error for multiple ellipsis', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    await expect(reduce(testTensor, '... ... c -> c', 'sum')).rejects.toThrow('Multiple ellipsis');
  });

  it('should throw error for invalid axis names', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    await expect(reduce(testTensor, '2h w -> w', 'sum')).rejects.toThrow();
  });

  it('should infer missing axes in composite pattern', async () => {
    const testTensor = await ones([4, 6] as const, { device: cpu, dtype: float32 });
    // h2 should be inferred as 4/2 = 2
    const result = await reduce(testTensor, '(h h2) w -> h w', 'sum', false, { h: 2 });
    expect(result.shape).toEqual([2, 6]);

    // Each element sums h2=2 values
    const data = await result.toArray();
    expect(data).toEqual([
      [2, 2, 2, 2, 2, 2],
      [2, 2, 2, 2, 2, 2],
    ]);
  });

  it('should throw error for mismatched composite dimensions', async () => {
    const testTensor = await ones([4, 6] as const, { device: cpu, dtype: float32 });
    // h*h2 = 3*2 = 6 ≠ 4
    await expect(
      reduce(testTensor, '(h h2) w -> h w', 'sum', false, { h: 3, h2: 2 }),
    ).rejects.toThrow();
  });
});

describe('Integration with Rearrange', () => {
  it('should work with chained operations', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    // First transpose: [2, 3, 4] -> [2, 4, 3]
    const transposed = await testTensor.transpose();
    const result = await reduce(transposed, 'batch h w -> batch', 'sum');
    expect(result.shape).toEqual([2]);

    // Sum of 4*3=12 ones per batch
    const data = await result.toArray();
    expect(data).toEqual([12, 12]);
  });

  it('should work with permute before reduce', async () => {
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
    // Permute: [2, 2, 2] -> [2, 2, 2] (swap first and last)
    const permuted = await testTensor.permute([2, 1, 0] as const);
    const result = await reduce(permuted, 'w h batch -> w', 'mean');
    expect(result.shape).toEqual([2]);

    const data = await result.toArray();
    expect(data).toEqual([4, 5]); // [(1+3+5+7)/4, (2+4+6+8)/4]
  });
});

describe('Different Data Types', () => {
  it('should handle different numeric values', async () => {
    const testTensor = await tensor(
      [
        [
          [0.5, 1.5],
          [2.5, 3.5],
        ],
        [
          [4.5, 5.5],
          [6.5, 7.5],
        ],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'batch h w -> h w', 'mean');
    expect(result.shape).toEqual([2, 2]);

    const data = await result.toArray();
    expect(data).toEqual([
      [2.5, 3.5],
      [4.5, 5.5],
    ]);
  });

  it('should handle large values', async () => {
    const testTensor = await tensor(
      [
        [1000, 2000],
        [3000, 4000],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'h w -> w', 'sum');
    expect(result.shape).toEqual([2]);

    const data = await result.toArray();
    expect(data).toEqual([4000, 6000]);
  });

  it('should handle very small values', async () => {
    const testTensor = await tensor(
      [
        [0.001, 0.002],
        [0.003, 0.004],
      ],
      { device: cpu, dtype: float32 },
    );
    const result = await reduce(testTensor, 'h w -> w', 'mean');
    expect(result.shape).toEqual([2]);

    const data = await result.toArray();
    expect(data[0]).toBeCloseTo(0.002, 6);
    expect(data[1]).toBeCloseTo(0.003, 6);
  });
});

describe('Complex Reduction Patterns', () => {
  it('should handle alternating reductions', async () => {
    const testTensor = await ones([2, 3, 4, 5] as const, { device: cpu, dtype: float32 });
    // Reduce 1st and 3rd dimensions
    const result = await reduce(testTensor, 'a b c d -> b d', 'sum');
    expect(result.shape).toEqual([3, 5]);

    // Sum over a=2 and c=4, so 2*4=8
    const data = await result.toArray();
    expect(data[0]).toEqual([8, 8, 8, 8, 8]);
  });

  it('should handle multiple composite groups', async () => {
    const testTensor = await ones([4, 6, 8] as const, { device: cpu, dtype: float32 });
    const result = await reduce(testTensor, '(a a2) (b b2) (c c2) -> a b c', 'sum', false, {
      a: 2,
      a2: 2,
      b: 3,
      b2: 2,
      c: 4,
      c2: 2,
    });
    expect(result.shape).toEqual([2, 3, 4]);

    // Each element sums a2*b2*c2 = 2*2*2 = 8 ones
    const data = await result.toArray();
    expect(data[0]?.[0]).toEqual([8, 8, 8, 8]);
  });

  it('should handle reduction with rearrangement-like pattern', async () => {
    const testTensor = await ones([2, 12] as const, { device: cpu, dtype: float32 });
    // Split second dimension and reduce part of it
    const result = await reduce(testTensor, 'batch (h w c) -> batch h c', 'sum', false, {
      h: 3,
      w: 2,
      c: 2,
    });
    expect(result.shape).toEqual([2, 3, 2]);

    // Each element sums over w=2
    const data = await result.toArray();
    expect(data[0]).toEqual([
      [2, 2],
      [2, 2],
      [2, 2],
    ]);
  });
});
