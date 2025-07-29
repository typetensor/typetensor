/**
 * Matrix multiplication operation test generator
 *
 * Tests matrix multiplication (matmul/@) operations including:
 * - 2D x 2D matrix multiplication
 * - 1D x 1D dot products
 * - 1D x 2D and 2D x 1D operations
 * - Batched matrix multiplication
 * - Error cases for incompatible shapes
 */

import type { Device } from '@typetensor/core';
import { tensor, float32 } from '@typetensor/core';

export function generateMatmulOperationTests(
  device: Device,
  testFramework: {
    describe: (name: string, fn: () => void) => void;
    it: (name: string, fn: () => void | Promise<void>) => void;
    expect: (actual: unknown) => {
      toBe: (expected: unknown) => void;
      toEqual: (expected: unknown) => void;
      toBeCloseTo: (expected: number, precision?: number) => void;
      toThrow: (error?: string | RegExp) => void;
      rejects: {
        toThrow: (error?: string | RegExp) => Promise<void>;
      };
    };
  },
): void {
  const { describe, it, expect } = testFramework;

  describe('Matrix Multiplication Operations', () => {
    describe('2D × 2D matrix multiplication', () => {
      it('should multiply square matrices', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]) @ torch.tensor([[5, 6], [7, 8]])
        // = tensor([[19, 22], [43, 50]])
        const a = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [5, 6],
            [7, 8],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await a.matmul(b);

        const data = await result.toArray();
        expect(data).toEqual([
          [19, 22],
          [43, 50],
        ]);
      });

      it('should multiply non-square matrices', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5, 6]]) @ torch.tensor([[7, 8], [9, 10], [11, 12]])
        // = tensor([[ 58,  64], [139, 154]])
        const a = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [7, 8],
            [9, 10],
            [11, 12],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await a.matmul(b);

        const data = await result.toArray();
        expect(data).toEqual([
          [58, 64],
          [139, 154],
        ]);
      });

      it('should handle identity matrix multiplication', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]) @ torch.tensor([[1, 0], [0, 1]])
        // Output: tensor([[1., 2.], [3., 4.]]) - unchanged
        const a = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const identity = await tensor(
          [
            [1, 0],
            [0, 1],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await a.matmul(identity);
        const data = await result.toArray();
        expect(data).toEqual([
          [1, 2],
          [3, 4],
        ]);
      });

      it('should handle zero matrix multiplication', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]) @ torch.tensor([[0, 0], [0, 0]])
        // Output: tensor([[0., 0.], [0., 0.]])
        const a = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const zeros = await tensor(
          [
            [0, 0],
            [0, 0],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await a.matmul(zeros);
        const data = await result.toArray();
        expect(data).toEqual([
          [0, 0],
          [0, 0],
        ]);
      });
    });

    describe('1D × 1D dot product', () => {
      it('should compute dot product as scalar', async () => {
        // PyTorch: torch.tensor([1, 2, 3]) @ torch.tensor([4, 5, 6])
        // = tensor(32)
        const a = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const b = await tensor([4, 5, 6] as const, { device, dtype: float32 });
        const result = await a.matmul(b);

        const value = await result.item();
        expect(value).toBe(32);
      });

      it('should handle orthogonal vectors', async () => {
        // PyTorch: torch.tensor([1, 0]) @ torch.tensor([0, 1])
        // Output: tensor(0.0)
        const a = await tensor([1, 0] as const, { device, dtype: float32 });
        const b = await tensor([0, 1] as const, { device, dtype: float32 });
        const result = await a.matmul(b);

        const value = await result.item();
        expect(value).toBe(0);
      });

      it('should handle negative values', async () => {
        // PyTorch: torch.tensor([1, -2, 3]) @ torch.tensor([-1, 2, -3])
        // Output: tensor(-14.0)
        const a = await tensor([1, -2, 3] as const, { device, dtype: float32 });
        const b = await tensor([-1, 2, -3] as const, { device, dtype: float32 });
        const result = await a.matmul(b);

        const value = await result.item();
        expect(value).toBe(-14); // 1*(-1) + (-2)*2 + 3*(-3) = -1 - 4 - 9 = -14
      });
    });

    describe('1D × 2D operations', () => {
      it('should multiply vector with matrix', async () => {
        // PyTorch: torch.tensor([1, 2]) @ torch.tensor([[3, 4, 5], [6, 7, 8]])
        // = tensor([15, 18, 21])
        const vector = await tensor([1, 2] as const, { device, dtype: float32 });
        const matrix = await tensor(
          [
            [3, 4, 5],
            [6, 7, 8],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await vector.matmul(matrix);

        const data = await result.toArray();
        expect(data).toEqual([15, 18, 21]);
      });

      it('should handle single row matrix', async () => {
        // PyTorch: torch.tensor([2, 3, 4]) @ torch.tensor([[1], [2], [3]])
        // Output: tensor([20.])
        const vector = await tensor([2, 3, 4] as const, { device, dtype: float32 });
        const matrix = await tensor([[1], [2], [3]] as const, { device, dtype: float32 });
        const result = await vector.matmul(matrix);

        const value = await result.item();
        expect(value).toBe(20); // 2*1 + 3*2 + 4*3 = 2 + 6 + 12 = 20
      });
    });

    describe('2D × 1D operations', () => {
      it('should multiply matrix with vector', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5, 6]]) @ torch.tensor([7, 8, 9])
        // = tensor([50, 122])
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const vector = await tensor([7, 8, 9] as const, { device, dtype: float32 });
        const result = await matrix.matmul(vector);

        const data = await result.toArray();
        expect(data).toEqual([50, 122]);
      });

      it('should handle single column result', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4], [5, 6]]) @ torch.tensor([2, 3])
        // Output: tensor([8, 18, 28])
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
            [5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const vector = await tensor([2, 3] as const, { device, dtype: float32 });
        const result = await matrix.matmul(vector);

        const data = await result.toArray();
        expect(data).toEqual([8, 18, 28]); // [1*2+2*3, 3*2+4*3, 5*2+6*3]
      });
    });

    describe('batched matrix multiplication', () => {
      it('should multiply 3D tensors batch-wise', async () => {
        // PyTorch: a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        //          b = torch.tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
        //          a @ b = tensor([[[ 31,  34], [ 85,  94]], [[155, 166], [211, 226]]])
        const a = await tensor(
          [
            [
              [1, 2],
              [3, 4],
            ],
            [
              [5, 6],
              [7, 8],
            ],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [
              [9, 10],
              [11, 12],
            ],
            [
              [13, 14],
              [15, 16],
            ],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await a.matmul(b);

        const data = await result.toArray();
        expect(data).toEqual([
          [
            [31, 34],
            [71, 78],
          ],
          [
            [155, 166],
            [211, 226],
          ],
        ]);
      });

      it('should handle batch × 2D broadcasting', async () => {
        // PyTorch: torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]) @ torch.tensor([[1, 0], [0, 1]])
        // = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        const batch = await tensor(
          [
            [
              [1, 2],
              [3, 4],
            ],
            [
              [5, 6],
              [7, 8],
            ],
          ] as const,
          { device, dtype: float32 },
        );
        const identity = await tensor(
          [
            [1, 0],
            [0, 1],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await batch.matmul(identity);

        const data = await result.toArray();
        expect(data).toEqual([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]);
      });
    });

    describe('special cases', () => {
      it('should handle scalar-like 1x1 matrices', async () => {
        // PyTorch: torch.tensor([[5]]) @ torch.tensor([[3]])
        // Output: tensor([[15]])
        const a = await tensor([[5]] as const, { device, dtype: float32 });
        const b = await tensor([[3]] as const, { device, dtype: float32 });
        const result = await a.matmul(b);

        const data = await result.toArray();
        expect(data).toEqual([[15]]);
      });

      it('should handle chain multiplication', async () => {
        const a = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [2, 0],
            [0, 2],
          ] as const,
          { device, dtype: float32 },
        );
        const c = await tensor(
          [
            [1, 1],
            [1, 1],
          ] as const,
          { device, dtype: float32 },
        );

        const temp = await a.matmul(b);
        const result = await temp.matmul(c);
        const data = await result.toArray();
        expect(data).toEqual([
          [6, 6],
          [14, 14],
        ]); // [[2,4],[6,8]] @ [[1,1],[1,1]]
      });

      it('should preserve numerical precision', async () => {
        // PyTorch: torch.tensor([[1.234567]]) @ torch.tensor([[2.345678]])
        // Output: tensor([[2.8959]]) - float32 precision
        const a = await tensor([[1.234567]] as const, { device, dtype: float32 });
        const b = await tensor([[2.345678]] as const, { device, dtype: float32 });
        const result = await a.matmul(b);

        const value = await result.item();
        expect(value).toBeCloseTo(2.89544, 3);
      });
    });

    describe('error handling', () => {
      it('should throw on incompatible inner dimensions', async () => {
        const a = await tensor([[1, 2]] as const, { device, dtype: float32 });
        const b = await tensor([[3], [4], [5]] as const, { device, dtype: float32 });

        // @ts-expect-error - a.matmul(b) is not a valid operation
        await expect(a.matmul(b)).rejects.toThrow(/Matrix multiplication requires/);
      });

      it('should throw on scalar inputs', async () => {
        const scalar = await tensor(5, { device, dtype: float32 });
        const vector = await tensor([1, 2, 3] as const, { device, dtype: float32 });

        // @ts-expect-error - scalar.matmul(vector) is not a valid operation
        await expect(scalar.matmul(vector)).rejects.toThrow(/scalar tensors/);
      });

      it('should throw on mismatched batch dimensions', async () => {
        const a = await tensor([[[1, 2]], [[3, 4]], [[5, 6]]] as const, { device, dtype: float32 });
        const b = await tensor(
          [
            [[1], [2]],
            [[3], [4]],
          ] as const,
          { device, dtype: float32 },
        );

        await expect(a.matmul(b)).rejects.toThrow(/[Bb]atch dimension/);
      });
    });

    describe('property preservation', () => {
      it('should preserve device', async () => {
        const a = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [5, 6],
            [7, 8],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await a.matmul(b);

        expect(result.device).toBe(device);
      });

      it('should preserve or promote dtype', async () => {
        const a = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [5, 6],
            [7, 8],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await a.matmul(b);

        expect(result.dtype).toBe(float32);
      });
    });
  });
}
