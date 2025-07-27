/**
 * Tests for matrix multiplication operations
 */

import { describe, it, expect } from 'bun:test';
import { tensor, zeros, float32, int32 } from '@typetensor/core';
import { cpu } from '../index';

describe('Matrix Multiplication', () => {
  describe('2D × 2D (standard matrix multiplication)', () => {
    it('should multiply square matrices correctly', async () => {
      // Create test matrices
      const a = await tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: float32, device: cpu },
      );
      const b = await tensor(
        [
          [5, 6],
          [7, 8],
        ],
        { dtype: float32, device: cpu },
      );

      // Compute matmul
      const c = await a.matmul(b);

      // Expected: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
      //         = [[19, 22], [43, 50]]
      const result = await c.toArray();
      expect(result).toEqual([
        [19, 22],
        [43, 50],
      ]);
      expect(c.shape).toEqual([2, 2]);

      // Clean up
      a.dispose();
      b.dispose();
      c.dispose();
    });

    it('should multiply non-square matrices correctly', async () => {
      // A: 2x3, B: 3x4 -> C: 2x4
      const a = await tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: float32, device: cpu },
      );
      const b = await tensor(
        [
          [7, 8, 9, 10],
          [11, 12, 13, 14],
          [15, 16, 17, 18],
        ],
        { dtype: float32, device: cpu },
      );

      const c = await a.matmul(b);

      // Expected computation:
      // c[0,0] = 1*7 + 2*11 + 3*15 = 7 + 22 + 45 = 74
      // c[0,1] = 1*8 + 2*12 + 3*16 = 8 + 24 + 48 = 80
      // c[0,2] = 1*9 + 2*13 + 3*17 = 9 + 26 + 51 = 86
      // c[0,3] = 1*10 + 2*14 + 3*18 = 10 + 28 + 54 = 92
      // c[1,0] = 4*7 + 5*11 + 6*15 = 28 + 55 + 90 = 173
      // c[1,1] = 4*8 + 5*12 + 6*16 = 32 + 60 + 96 = 188
      // c[1,2] = 4*9 + 5*13 + 6*17 = 36 + 65 + 102 = 203
      // c[1,3] = 4*10 + 5*14 + 6*18 = 40 + 70 + 108 = 218
      const result = await c.toArray();
      expect(result).toEqual([
        [74, 80, 86, 92],
        [173, 188, 203, 218],
      ]);
      expect(c.shape).toEqual([2, 4]);

      a.dispose();
      b.dispose();
      c.dispose();
    });

    it('should handle identity matrix', async () => {
      const a = await tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: float32, device: cpu },
      );
      const identity = await tensor(
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        { dtype: float32, device: cpu },
      );

      const c = await a.matmul(identity);

      const result = await c.toArray();
      expect(result).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);

      a.dispose();
      identity.dispose();
      c.dispose();
    });
  });

  describe('1D × 1D (dot product)', () => {
    it('should compute dot product to scalar', async () => {
      const a = await tensor([1, 2, 3], { dtype: float32, device: cpu });
      const b = await tensor([4, 5, 6], { dtype: float32, device: cpu });

      const c = await a.matmul(b);

      // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
      const result = await c.item();
      expect(result).toBe(32);
      expect(c.shape).toEqual([]);

      a.dispose();
      b.dispose();
      c.dispose();
    });

    it('should handle zero vectors', async () => {
      const a = await tensor([1, 2, 3], { dtype: float32, device: cpu });
      const b = await tensor([0, 0, 0], { dtype: float32, device: cpu });

      const c = await a.matmul(b);

      const result = await c.item();
      expect(result).toBe(0);

      a.dispose();
      b.dispose();
      c.dispose();
    });
  });

  describe('1D × 2D (vector-matrix multiply)', () => {
    it('should multiply vector with matrix', async () => {
      const a = await tensor([1, 2, 3], { dtype: float32, device: cpu });
      const b = await tensor(
        [
          [4, 5],
          [6, 7],
          [8, 9],
        ],
        { dtype: float32, device: cpu },
      );

      const c = await a.matmul(b);

      // Expected: [1*4 + 2*6 + 3*8, 1*5 + 2*7 + 3*9]
      //         = [4 + 12 + 24, 5 + 14 + 27]
      //         = [40, 46]
      const result = await c.toArray();
      expect(result).toEqual([40, 46]);
      expect(c.shape).toEqual([2]);

      a.dispose();
      b.dispose();
      c.dispose();
    });
  });

  describe('2D × 1D (matrix-vector multiply)', () => {
    it('should multiply matrix with vector', async () => {
      const a = await tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: float32, device: cpu },
      );
      const b = await tensor([7, 8, 9], { dtype: float32, device: cpu });

      const c = await a.matmul(b);

      // Expected: [1*7 + 2*8 + 3*9, 4*7 + 5*8 + 6*9]
      //         = [7 + 16 + 27, 28 + 40 + 54]
      //         = [50, 122]
      const result = await c.toArray();
      expect(result).toEqual([50, 122]);
      expect(c.shape).toEqual([2]);

      a.dispose();
      b.dispose();
      c.dispose();
    });
  });

  describe('Batch matrix multiplication', () => {
    it('should handle 3D batch multiplication', async () => {
      // Shape: [2, 2, 3]
      const a = await tensor(
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
        { dtype: float32, device: cpu },
      );

      // Shape: [2, 3, 2]
      const b = await tensor(
        [
          [
            [1, 2],
            [3, 4],
            [5, 6],
          ],
          [
            [7, 8],
            [9, 10],
            [11, 12],
          ],
        ],
        { dtype: float32, device: cpu },
      );

      const c = await a.matmul(b);

      // Expected shape: [2, 2, 2]
      // Batch 0: [[1,2,3], [4,5,6]] × [[1,2], [3,4], [5,6]]
      // Result 0: [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
      //         = [[22, 28], [49, 64]]
      // Batch 1: [[7,8,9], [10,11,12]] × [[7,8], [9,10], [11,12]]
      // Result 1: [[7*7+8*9+9*11, 7*8+8*10+9*12], [10*7+11*9+12*11, 10*8+11*10+12*12]]
      //         = [[220, 244], [301, 334]]
      const result = await c.toArray();
      expect(result).toEqual([
        [
          [22, 28],
          [49, 64],
        ],
        [
          [220, 244],
          [301, 334],
        ],
      ]);
      expect(c.shape).toEqual([2, 2, 2]);

      a.dispose();
      b.dispose();
      c.dispose();
    });

    it('should handle broadcasting in batch dimensions', async () => {
      // A has batch size 1, B has batch size 2
      const a = await tensor(
        [
          [
            [1, 2],
            [3, 4],
          ],
        ],
        { dtype: float32, device: cpu },
      ); // [1, 2, 2]
      const b = await tensor(
        [
          [
            [5, 6],
            [7, 8],
          ],
          [
            [9, 10],
            [11, 12],
          ],
        ],
        { dtype: float32, device: cpu },
      ); // [2, 2, 2]

      const c = await a.matmul(b);

      // Expected: broadcast A to match B's batch size
      // Batch 0: [[1,2], [3,4]] × [[5,6], [7,8]]
      // Result 0: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22], [43,50]]
      // Batch 1: [[1,2], [3,4]] × [[9,10], [11,12]]
      // Result 1: [[1*9+2*11, 1*10+2*12], [3*9+4*11, 3*10+4*12]] = [[31,34], [71,78]]
      const result = await c.toArray();
      expect(result).toEqual([
        [
          [19, 22],
          [43, 50],
        ],
        [
          [31, 34],
          [71, 78],
        ],
      ]);
      expect(c.shape).toEqual([2, 2, 2]);

      a.dispose();
      b.dispose();
      c.dispose();
    });
  });

  describe('Different data types', () => {
    it('should handle integer types', async () => {
      const a = await tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: int32, device: cpu },
      );
      const b = await tensor(
        [
          [5, 6],
          [7, 8],
        ],
        { dtype: int32, device: cpu },
      );

      const c = await a.matmul(b);

      const result = await c.toArray();
      expect(result).toEqual([
        [19, 22],
        [43, 50],
      ]);
      expect(c.dtype.__dtype).toBe('int32'); // int32 × int32 → int32 (same type)

      a.dispose();
      b.dispose();
      c.dispose();
    });

    it('should handle mixed precision', async () => {
      const a = await tensor(
        [
          [1.5, 2.5],
          [3.5, 4.5],
        ],
        { dtype: float32, device: cpu },
      );
      const b = await tensor(
        [
          [1, 0],
          [0, 1],
        ],
        { dtype: int32, device: cpu },
      );

      const c = await a.matmul(b);

      const result = await c.toArray();
      expect(result).toEqual([
        [1.5, 2.5],
        [3.5, 4.5],
      ]);
      expect(c.dtype.__dtype).toBe('float64'); // float32 × int32 → float64

      a.dispose();
      b.dispose();
      c.dispose();
    });
  });

  describe('Edge cases', () => {
    it('should handle single element matrices', async () => {
      const a = await tensor([[5]], { dtype: float32, device: cpu });
      const b = await tensor([[3]], { dtype: float32, device: cpu });

      const c = await a.matmul(b);

      const result = await c.toArray();
      expect(result).toEqual([[15]]);

      a.dispose();
      b.dispose();
      c.dispose();
    });

    it('should handle zero matrices', async () => {
      const a = await tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: float32, device: cpu },
      );
      const b = await zeros([2, 2] as const, { dtype: float32, device: cpu });

      const c = await a.matmul(b);

      const result = await c.toArray();
      expect(result).toEqual([
        [0, 0],
        [0, 0],
      ]);

      a.dispose();
      b.dispose();
      c.dispose();
    });

    it('should error on incompatible shapes', async () => {
      const a = await tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: float32, device: cpu },
      ); // 2x3
      const b = await tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: float32, device: cpu },
      ); // 2x2

      // Should throw because inner dimensions don't match (3 != 2)
      // @ts-expect-error - this is expected to throw at compile time but just testing that runtime also guards
      await expect(a.matmul(b)).rejects.toThrow();

      a.dispose();
      b.dispose();
    });

    it('should error on scalar inputs', async () => {
      const a = await tensor(5, { dtype: float32, device: cpu });
      const b = await tensor(3, { dtype: float32, device: cpu });

      // Scalars cannot be matrix multiplied
      // @ts-expect-error - this is expected to throw at compile time but just testing that runtime also guards
      await expect(a.matmul(b)).rejects.toThrow();

      a.dispose();
      b.dispose();
    });
  });
});
