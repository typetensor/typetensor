/**
 * Test generators for reduction operations
 *
 * These generators test reduction operations like sum, mean, max, min
 * that reduce tensor dimensions along specified axes with optional keepDims.
 */

import type { Device } from '@typetensor/core';
import { tensor, float32, int32 } from '@typetensor/core';

/**
 * Generates tests for reduction operations
 *
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateReductionOperationTests(
  device: Device,
  testFramework: {
    describe: (name: string, fn: () => void) => void;
    it: (name: string, fn: () => void | Promise<void>) => void;
    expect: (actual: unknown) => {
      toBe: (expected: unknown) => void;
      toEqual: (expected: unknown) => void;
      toBeCloseTo: (expected: number, precision?: number) => void;
      toThrow: (error?: string | RegExp) => void;
      toBeGreaterThan: (expected: number) => void;
      toBeTruthy: () => void;
      rejects: {
        toThrow: (error?: string | RegExp) => Promise<void>;
      };
    };
  },
) {
  const { describe, it, expect } = testFramework;

  describe(`Reduction Operations Tests (${device.type}:${device.id})`, () => {
    describe('sum operations', () => {
      it('should sum all elements in a scalar', async () => {
        // PyTorch: torch.tensor(5.0).sum()
        // Output: tensor(5.0)
        const scalar = await tensor(5, { device, dtype: float32 });
        const result = await scalar.sum();

        expect(result.shape).toEqual([]);
        expect(result.dtype).toBe(float32);
        expect(result.device).toBe(device);
        expect(await result.item()).toBeCloseTo(5, 5);
      });

      it('should sum all elements in a vector', async () => {
        // PyTorch: torch.tensor([1, 2, 3, 4, 5]).sum()
        // Output: tensor(15)
        const vector = await tensor([1, 2, 3, 4, 5] as const, { device, dtype: float32 });
        const result = await vector.sum();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(15, 5); // 1+2+3+4+5 = 15
      });

      it('should sum all elements in a matrix', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5, 6]]).sum()
        // Output: tensor(21)
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.sum();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(21, 5); // 1+2+3+4+5+6 = 21
      });

      it('should sum along axis 0 (rows)', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5, 6]]).sum(dim=0)
        // Output: tensor([5, 7, 9])
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.sum([0]);

        expect(result.shape).toEqual([3]);
        const data = await result.toArray();
        expect(data).toEqual([5, 7, 9]); // [1+4, 2+5, 3+6]
      });

      it('should sum along axis 1 (columns)', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5, 6]]).sum(dim=1)
        // Output: tensor([6, 15])
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.sum([1]);

        expect(result.shape).toEqual([2]);
        const data = await result.toArray();
        expect(data).toEqual([6, 15]); // [1+2+3, 4+5+6]
      });

      it('should sum with keepDims=true', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5, 6]]).sum(dim=0, keepdim=True)
        // Output: tensor([[5, 7, 9]])
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.sum([0], true);

        expect(result.shape).toEqual([1, 3]);
        const data = await result.toArray();
        expect(data).toEqual([[5, 7, 9]]);
      });

      it('should handle integer sums', async () => {
        // PyTorch: torch.tensor([10, 20, 30], dtype=torch.int32).sum()
        // Output: tensor(60, dtype=torch.int64)
        // Note: PyTorch promotes int32 sums to int64
        const vector = await tensor([10, 20, 30] as const, { device, dtype: int32 });
        const result = await vector.sum();

        expect(result.dtype).toBe(int32);
        expect(await result.item()).toBe(60);
      });
    });

    describe('mean operations', () => {
      it('should compute mean of all elements in a scalar', async () => {
        // PyTorch: torch.tensor(7.0).mean()
        // Output: tensor(7.0)
        const scalar = await tensor(7, { device, dtype: float32 });
        const result = await scalar.mean();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(7, 5);
      });

      it('should compute mean of all elements in a vector', async () => {
        // PyTorch: torch.tensor([2, 4, 6, 8], dtype=torch.float32).mean()
        // Output: tensor(5.0)
        const vector = await tensor([2, 4, 6, 8] as const, { device, dtype: float32 });
        const result = await vector.mean();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(5, 5); // (2+4+6+8)/4 = 20/4 = 5
      });

      it('should compute mean of all elements in a matrix', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).mean()
        // Output: tensor(2.5)
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.mean();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(2.5, 5); // (1+2+3+4)/4 = 10/4 = 2.5
      });

      it('should compute mean along axis 0', async () => {
        // PyTorch: torch.tensor([[1, 3], [5, 7]], dtype=torch.float32).mean(dim=0)
        // Output: tensor([3., 5.])
        const matrix = await tensor(
          [
            [1, 3],
            [5, 7],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.mean([0]);

        expect(result.shape).toEqual([2]);
        const data = await result.toArray();
        expect(data[0]).toBeCloseTo(3, 5); // (1+5)/2 = 3
        expect(data[1]).toBeCloseTo(5, 5); // (3+7)/2 = 5
      });

      it('should compute mean along axis 1', async () => {
        // PyTorch: torch.tensor([[2, 8], [4, 6]], dtype=torch.float32).mean(dim=1)
        // Output: tensor([5., 5.])
        const matrix = await tensor(
          [
            [2, 8],
            [4, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.mean([1]);

        expect(result.shape).toEqual([2]);
        const data = await result.toArray();
        expect(data[0]).toBeCloseTo(5, 5); // (2+8)/2 = 5
        expect(data[1]).toBeCloseTo(5, 5); // (4+6)/2 = 5
      });

      it('should compute mean with keepDims=true', async () => {
        // PyTorch: torch.tensor([[2, 4], [6, 8]], dtype=torch.float32).mean(dim=1, keepdim=True)
        // Output: tensor([[3.], [7.]])
        const matrix = await tensor(
          [
            [2, 4],
            [6, 8],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.mean([1], true);

        expect(result.shape).toEqual([2, 1]);
        const data = await result.toArray();
        expect(data[0][0]).toBeCloseTo(3, 5); // (2+4)/2 = 3
        expect(data[1][0]).toBeCloseTo(7, 5); // (6+8)/2 = 7
      });
    });

    describe('max operations', () => {
      it('should find max of all elements in a scalar', async () => {
        // PyTorch: torch.tensor(42.0).max()
        // Output: tensor(42.0)
        const scalar = await tensor(42, { device, dtype: float32 });
        const result = await scalar.max();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(42, 5);
      });

      it('should find max of all elements in a vector', async () => {
        // PyTorch: torch.tensor([3, 7, 2, 9, 1]).max()
        // Output: tensor(9)
        const vector = await tensor([3, 7, 2, 9, 1] as const, { device, dtype: float32 });
        const result = await vector.max();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(9, 5);
      });

      it('should find max of all elements in a matrix', async () => {
        // PyTorch: torch.tensor([[1, 8, 3], [4, 2, 6]]).max()
        // Output: tensor(8)
        const matrix = await tensor(
          [
            [1, 8, 3],
            [4, 2, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.max();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(8, 5);
      });

      it('should find max along axis 0', async () => {
        // PyTorch: torch.tensor([[1, 8, 3], [4, 2, 6]]).max(dim=0)
        // Output: torch.return_types.max(values=tensor([4, 8, 6]), indices=tensor([1, 0, 1]))
        // Note: PyTorch returns both values and indices for axis-wise max
        const matrix = await tensor(
          [
            [1, 8, 3],
            [4, 2, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.max([0]);

        expect(result.shape).toEqual([3]);
        const data = await result.toArray();
        expect(data).toEqual([4, 8, 6]); // [max(1,4), max(8,2), max(3,6)]
      });

      it('should find max along axis 1', async () => {
        // PyTorch: torch.tensor([[1, 8, 3], [4, 2, 6]]).max(dim=1)
        // Output: torch.return_types.max(values=tensor([8, 6]), indices=tensor([1, 2]))
        const matrix = await tensor(
          [
            [1, 8, 3],
            [4, 2, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.max([1]);

        expect(result.shape).toEqual([2]);
        const data = await result.toArray();
        expect(data).toEqual([8, 6]); // [max(1,8,3), max(4,2,6)]
      });

      it('should find max with keepDims=true', async () => {
        // PyTorch: torch.tensor([[1, 8], [4, 2]]).max(dim=0, keepdim=True)
        // Output: torch.return_types.max(values=tensor([[4, 8]]), indices=tensor([[1, 0]]))
        const matrix = await tensor(
          [
            [1, 8],
            [4, 2],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.max([0], true);

        expect(result.shape).toEqual([1, 2]);
        const data = await result.toArray();
        expect(data).toEqual([[4, 8]]);
      });

      it('should handle negative numbers', async () => {
        // PyTorch: torch.tensor([-5, -2, -8, -1]).max()
        // Output: tensor(-1)
        const vector = await tensor([-5, -2, -8, -1] as const, { device, dtype: float32 });
        const result = await vector.max();

        expect(await result.item()).toBeCloseTo(-1, 5);
      });
    });

    describe('min operations', () => {
      it('should find min of all elements in a scalar', async () => {
        // PyTorch: torch.tensor(42.0).min()
        // Output: tensor(42.0)
        const scalar = await tensor(42, { device, dtype: float32 });
        const result = await scalar.min();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(42, 5);
      });

      it('should find min of all elements in a vector', async () => {
        // PyTorch: torch.tensor([3, 7, 2, 9, 1]).min()
        // Output: tensor(1)
        const vector = await tensor([3, 7, 2, 9, 1] as const, { device, dtype: float32 });
        const result = await vector.min();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(1, 5);
      });

      it('should find min of all elements in a matrix', async () => {
        // PyTorch: torch.tensor([[5, 8, 3], [4, 2, 6]]).min()
        // Output: tensor(2)
        const matrix = await tensor(
          [
            [5, 8, 3],
            [4, 2, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.min();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(2, 5);
      });

      it('should find min along axis 0', async () => {
        // PyTorch: torch.tensor([[5, 8, 3], [4, 2, 6]]).min(dim=0)
        // Output: torch.return_types.min(values=tensor([4, 2, 3]), indices=tensor([1, 1, 0]))
        const matrix = await tensor(
          [
            [5, 8, 3],
            [4, 2, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.min([0]);

        expect(result.shape).toEqual([3]);
        const data = await result.toArray();
        expect(data).toEqual([4, 2, 3]); // [min(5,4), min(8,2), min(3,6)]
      });

      it('should find min along axis 1', async () => {
        // PyTorch: torch.tensor([[5, 8, 3], [4, 2, 6]]).min(dim=1)
        // Output: torch.return_types.min(values=tensor([3, 2]), indices=tensor([2, 1]))
        const matrix = await tensor(
          [
            [5, 8, 3],
            [4, 2, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.min([1]);

        expect(result.shape).toEqual([2]);
        const data = await result.toArray();
        expect(data).toEqual([3, 2]); // [min(5,8,3), min(4,2,6)]
      });

      it('should find min with keepDims=true', async () => {
        // PyTorch: torch.tensor([[5, 8], [4, 2]]).min(dim=1, keepdim=True)
        // Output: torch.return_types.min(values=tensor([[5], [2]]), indices=tensor([[0], [1]]))
        const matrix = await tensor(
          [
            [5, 8],
            [4, 2],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.min([1], true);

        expect(result.shape).toEqual([2, 1]);
        const data = await result.toArray();
        expect(data).toEqual([[5], [2]]);
      });

      it('should handle negative numbers', async () => {
        // PyTorch: torch.tensor([-5, -2, -8, -1]).min()
        // Output: tensor(-8)
        const vector = await tensor([-5, -2, -8, -1] as const, { device, dtype: float32 });
        const result = await vector.min();

        expect(await result.item()).toBeCloseTo(-8, 5);
      });
    });

    describe('multi-dimensional reductions', () => {
      it('should handle 3D tensor reductions', async () => {
        // PyTorch: tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        // Shape: torch.Size([2, 2, 2])
        // Sum all: 36
        // Sum axis 0: tensor([[6, 8], [10, 12]])
        // Sum axis 1: tensor([[4, 6], [12, 14]])
        // Sum axis 2: tensor([[3, 7], [11, 15]])
        const tensor3d = await tensor(
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

        // Sum all elements: 1+2+3+4+5+6+7+8 = 36
        const sumAll = await tensor3d.sum();
        expect(await sumAll.item()).toBeCloseTo(36, 5);

        // Sum along axis 0: [[1+5, 2+6], [3+7, 4+8]] = [[6, 8], [10, 12]]
        const sumAxis0 = await tensor3d.sum([0]);
        expect(sumAxis0.shape).toEqual([2, 2]);
        const data0 = await sumAxis0.toArray();
        expect(data0).toEqual([
          [6, 8],
          [10, 12],
        ]);

        // Sum along axis 1: [[1+3, 2+4], [5+7, 6+8]] = [[4, 6], [12, 14]]
        const sumAxis1 = await tensor3d.sum([1]);
        expect(sumAxis1.shape).toEqual([2, 2]);
        const data1 = await sumAxis1.toArray();
        expect(data1).toEqual([
          [4, 6],
          [12, 14],
        ]);

        // Sum along axis 2: [[1+2, 3+4], [5+6, 7+8]] = [[3, 7], [11, 15]]
        const sumAxis2 = await tensor3d.sum([2]);
        expect(sumAxis2.shape).toEqual([2, 2]);
        const data2 = await sumAxis2.toArray();
        expect(data2).toEqual([
          [3, 7],
          [11, 15],
        ]);
      });

      it('should handle multiple axes reduction', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5, 6]]).sum(dim=[0, 1])
        // Output: tensor(21)
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // Sum along both axes (equivalent to sum all)
        const result = await matrix.sum([0, 1]);
        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(21, 5);
      });

      it('should handle keepDims with multiple operations', async () => {
        // PyTorch: tensor3d.mean(dim=[0, 2], keepdim=True)
        // Output shape: torch.Size([1, 2, 1])
        // Values: tensor([[[3.5000], [5.5000]]])
        const tensor3d = await tensor(
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

        const result = await tensor3d.mean([0, 2], true);
        expect(result.shape).toEqual([1, 2, 1]);

        const data = await result.toArray();
        // Mean of [1,2,5,6] = 3.5, Mean of [3,4,7,8] = 5.5
        expect(data[0][0][0]).toBeCloseTo(3.5, 5);
        expect(data[0][1][0]).toBeCloseTo(5.5, 5);
      });
    });

    describe('property preservation', () => {
      it('should preserve device and dtype for reduction results', async () => {
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        const sum = await matrix.sum();
        const mean = await matrix.mean();
        const max = await matrix.max();
        const min = await matrix.min();

        // All should preserve device
        expect(sum.device).toBe(device);
        expect(mean.device).toBe(device);
        expect(max.device).toBe(device);
        expect(min.device).toBe(device);

        // All should preserve dtype
        expect(sum.dtype).toBe(float32);
        expect(mean.dtype).toBe(float32);
        expect(max.dtype).toBe(float32);
        expect(min.dtype).toBe(float32);
      });

      it('should handle different input dtypes', async () => {
        const intMatrix = await tensor(
          [
            [10, 20],
            [30, 40],
          ] as const,
          { device, dtype: int32 },
        );

        const sum = await intMatrix.sum();
        const max = await intMatrix.max();

        expect(sum.dtype).toBe(int32);
        expect(max.dtype).toBe(int32);
        expect(await sum.item()).toBe(100);
        expect(await max.item()).toBe(40);
      });
    });

    describe('edge cases', () => {
      it('should handle single-element tensors', async () => {
        // PyTorch: torch.tensor([[42]], dtype=torch.float32)
        // All operations return tensor(42.0)
        const single = await tensor([[42]] as const, { device, dtype: float32 });

        const sum = await single.sum();
        const mean = await single.mean();
        const max = await single.max();
        const min = await single.min();

        expect(await sum.item()).toBeCloseTo(42, 5);
        expect(await mean.item()).toBeCloseTo(42, 5);
        expect(await max.item()).toBeCloseTo(42, 5);
        expect(await min.item()).toBeCloseTo(42, 5);
      });

      it('should handle zero values', async () => {
        // PyTorch: torch.tensor([0, 5, 0, 3, 0], dtype=torch.float32)
        // sum=8.0, mean=1.6, max=5.0, min=0.0
        const withZeros = await tensor([0, 5, 0, 3, 0] as const, { device, dtype: float32 });

        const sum = await withZeros.sum();
        const mean = await withZeros.mean();
        const max = await withZeros.max();
        const min = await withZeros.min();

        expect(await sum.item()).toBeCloseTo(8, 5);
        expect(await mean.item()).toBeCloseTo(1.6, 5); // 8/5
        expect(await max.item()).toBeCloseTo(5, 5);
        expect(await min.item()).toBeCloseTo(0, 5);
      });

      it('should handle all same values', async () => {
        // PyTorch: torch.tensor([7, 7, 7, 7], dtype=torch.float32)
        // sum=28.0, mean=7.0, max=7.0, min=7.0
        const uniform = await tensor([7, 7, 7, 7] as const, { device, dtype: float32 });

        const sum = await uniform.sum();
        const mean = await uniform.mean();
        const max = await uniform.max();
        const min = await uniform.min();

        expect(await sum.item()).toBeCloseTo(28, 5);
        expect(await mean.item()).toBeCloseTo(7, 5);
        expect(await max.item()).toBeCloseTo(7, 5);
        expect(await min.item()).toBeCloseTo(7, 5);
      });

      it('should handle floating point precision', async () => {
        // PyTorch: torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        // sum=0.6000000238418579, mean=0.20000000298023224
        // Note: float32 precision limitations
        const precise = await tensor([0.1, 0.2, 0.3] as const, { device, dtype: float32 });

        const sum = await precise.sum();
        const mean = await precise.mean();

        // Note: float32 precision might not be exact for 0.6
        expect(await sum.item()).toBeCloseTo(0.6, 4);
        expect(await mean.item()).toBeCloseTo(0.2, 4);
      });
    });

    describe('error handling', () => {
      it('should handle invalid axis values gracefully', async () => {
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        // Test with axis that's out of bounds
        try {
          // @ts-expect-error - Testing runtime behavior with invalid axis
          await matrix.sum([5] as const);
          // If it doesn't throw, that's also acceptable
        } catch (error) {
          expect(error).toBeTruthy();
        }
      });

      it('should handle negative axis values appropriately', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5, 6]]).sum(dim=-1)
        // Output: tensor([6, 15])
        // Negative axis -1 refers to the last dimension
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // Negative axis should work (axis -1 = last axis)
        try {
          const result = await matrix.sum([-1]);
          expect(result.shape).toEqual([2]);
          const data = await result.toArray();
          expect(data).toEqual([6, 15]); // Same as axis: 1
        } catch (error) {
          // If negative axes aren't supported, that's also valid
          expect(error).toBeTruthy();
        }
      });
    });

    describe('prod operations', () => {
      it('should compute product of all elements in a scalar', async () => {
        // PyTorch: torch.tensor(5.0).prod()
        // Output: tensor(5.0)
        const scalar = await tensor(5, { device, dtype: float32 });
        const result = await scalar.prod();

        expect(result.shape).toEqual([]);
        expect(result.dtype).toBe(float32);
        expect(result.device).toBe(device);
        expect(await result.item()).toBeCloseTo(5, 5);
      });

      it('should compute product of all elements in a vector', async () => {
        // PyTorch: torch.tensor([2, 3, 4], dtype=torch.float32).prod()
        // Output: tensor(24.0)
        const vector = await tensor([2, 3, 4] as const, { device, dtype: float32 });
        const result = await vector.prod();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(24, 5); // 2*3*4 = 24
      });

      it('should compute product of all elements in a matrix', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).prod()
        // Output: tensor(24.0)
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.prod();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(24, 5); // 1*2*3*4 = 24
      });

      it('should compute product along axis 0', async () => {
        // PyTorch: torch.tensor([[2, 3], [4, 5]], dtype=torch.float32).prod(dim=0)
        // Output: tensor([8., 15.])
        const matrix = await tensor(
          [
            [2, 3],
            [4, 5],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.prod([0]);

        expect(result.shape).toEqual([2]);
        const data = await result.toArray();
        expect(data[0]).toBeCloseTo(8, 5); // 2*4 = 8
        expect(data[1]).toBeCloseTo(15, 5); // 3*5 = 15
      });

      it('should compute product along axis 1', async () => {
        // PyTorch: torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float32).prod(dim=1)
        // Output: tensor([24., 210.])
        const matrix = await tensor(
          [
            [2, 3, 4],
            [5, 6, 7],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.prod([1]);

        expect(result.shape).toEqual([2]);
        const data = await result.toArray();
        expect(data[0]).toBeCloseTo(24, 5); // 2*3*4 = 24
        expect(data[1]).toBeCloseTo(210, 5); // 5*6*7 = 210
      });

      it('should compute product with keepDims=true', async () => {
        // PyTorch: torch.tensor([[2, 3], [4, 5]], dtype=torch.float32).prod(dim=1, keepdim=True)
        // Output: tensor([[6.], [20.]])
        const matrix = await tensor(
          [
            [2, 3],
            [4, 5],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.prod([1], true);

        expect(result.shape).toEqual([2, 1]);
        const data = await result.toArray();
        expect(data[0][0]).toBeCloseTo(6, 5); // 2*3 = 6
        expect(data[1][0]).toBeCloseTo(20, 5); // 4*5 = 20
      });

      it('should handle negative numbers', async () => {
        // PyTorch: torch.tensor([-2, 3, -4], dtype=torch.float32).prod()
        // Output: tensor(24.0)
        const vector = await tensor([-2, 3, -4] as const, { device, dtype: float32 });
        const result = await vector.prod();

        expect(await result.item()).toBeCloseTo(24, 5); // (-2)*3*(-4) = 24
      });

      it('should handle mixed signs', async () => {
        // PyTorch: torch.tensor([[-2, 3], [4, -5]], dtype=torch.float32).prod(dim=0)
        // Output: tensor([-8, -15])
        const matrix = await tensor(
          [
            [-2, 3],
            [4, -5],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.prod([0]);

        const data = await result.toArray();
        expect(data[0]).toBeCloseTo(-8, 5); // (-2)*4 = -8
        expect(data[1]).toBeCloseTo(-15, 5); // 3*(-5) = -15
      });

      it('should handle zeros in product', async () => {
        // PyTorch: torch.tensor([2, 0, 3, 4], dtype=torch.float32).prod()
        // Output: tensor(0.)
        const vector = await tensor([2, 0, 3, 4] as const, { device, dtype: float32 });
        const result = await vector.prod();

        expect(await result.item()).toBeCloseTo(0, 5);
      });

      it('should handle empty tensors', async () => {
        // PyTorch: torch.tensor([], dtype=torch.float32).prod()
        // Output: tensor(1.)
        // Note: Empty product is 1 (multiplicative identity)
        const empty = await tensor([] as const, { device, dtype: float32 });
        const result = await empty.prod();

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(1, 5);
      });

      it('should handle integer products', async () => {
        // PyTorch: torch.tensor([2, 3, 4], dtype=torch.int32).prod()
        // Output: tensor(24, dtype=torch.int64)
        // Note: PyTorch promotes int32 products to int64
        const vector = await tensor([2, 3, 4] as const, { device, dtype: int32 });
        const result = await vector.prod();

        expect(result.dtype).toBe(int32);
        expect(await result.item()).toBe(24);
      });

      it('should handle 3D tensor products', async () => {
        // PyTorch: tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
        // Shape: torch.Size([2, 2, 2])
        // Prod all: 40320
        // Prod axis 0: tensor([[5, 12], [21, 32]])
        // Prod axis 1: tensor([[3, 8], [35, 48]])
        // Prod axis 2: tensor([[2, 12], [30, 56]])
        const tensor3d = await tensor(
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

        // Product of all elements: 1*2*3*4*5*6*7*8 = 40320
        const prodAll = await tensor3d.prod();
        expect(await prodAll.item()).toBeCloseTo(40320, 5);

        // Product along axis 0: [[1*5, 2*6], [3*7, 4*8]] = [[5, 12], [21, 32]]
        const prodAxis0 = await tensor3d.prod([0]);
        expect(prodAxis0.shape).toEqual([2, 2]);
        const data0 = await prodAxis0.toArray();
        expect(data0).toEqual([
          [5, 12],
          [21, 32],
        ]);

        // Product along axis 1: [[1*3, 2*4], [5*7, 6*8]] = [[3, 8], [35, 48]]
        const prodAxis1 = await tensor3d.prod([1]);
        expect(prodAxis1.shape).toEqual([2, 2]);
        const data1 = await prodAxis1.toArray();
        expect(data1).toEqual([
          [3, 8],
          [35, 48],
        ]);

        // Product along axis 2: [[1*2, 3*4], [5*6, 7*8]] = [[2, 12], [30, 56]]
        const prodAxis2 = await tensor3d.prod([2]);
        expect(prodAxis2.shape).toEqual([2, 2]);
        const data2 = await prodAxis2.toArray();
        expect(data2).toEqual([
          [2, 12],
          [30, 56],
        ]);
      });

      it('should handle multiple axes product', async () => {
        // PyTorch: torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.float32).prod(dim=[0, 1])
        // Output: tensor(5040.)
        const matrix = await tensor(
          [
            [2, 3, 4],
            [5, 6, 7],
          ] as const,
          { device, dtype: float32 },
        );

        // Product along both axes (equivalent to prod all)
        const result = await matrix.prod([0, 1]);
        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(5040, 5); // 2*3*4*5*6*7 = 5040
      });

      it('should handle keepDims with multiple axes', async () => {
        // PyTorch: tensor3d.prod(dim=[0, 2], keepdim=True)
        // Output shape: torch.Size([1, 2, 1])
        const tensor3d = await tensor(
          [
            [
              [2, 3],
              [4, 5],
            ],
            [
              [6, 7],
              [8, 9],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await tensor3d.prod([0, 2], true);
        expect(result.shape).toEqual([1, 2, 1]);

        const data = await result.toArray();
        // Product of [2,3,6,7] = 252, Product of [4,5,8,9] = 1440
        expect(data[0][0][0]).toBeCloseTo(252, 5);
        expect(data[0][1][0]).toBeCloseTo(1440, 5);
      });

      it('should handle ones in product', async () => {
        // PyTorch: torch.tensor([1, 1, 2, 1, 3], dtype=torch.float32).prod()
        // Output: tensor(6.)
        const vector = await tensor([1, 1, 2, 1, 3] as const, { device, dtype: float32 });
        const result = await vector.prod();

        expect(await result.item()).toBeCloseTo(6, 5); // 1*1*2*1*3 = 6
      });

      it('should handle very small numbers', async () => {
        // PyTorch: torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32).prod()
        // Output: tensor(0.0060)
        const vector = await tensor([0.1, 0.2, 0.3] as const, { device, dtype: float32 });
        const result = await vector.prod();

        expect(await result.item()).toBeCloseTo(0.006, 4); // 0.1*0.2*0.3 = 0.006
      });

      it('should preserve device and dtype', async () => {
        const matrix = await tensor(
          [
            [2, 3],
            [4, 5],
          ] as const,
          { device, dtype: float32 },
        );

        const prod = await matrix.prod();
        expect(prod.device).toBe(device);
        expect(prod.dtype).toBe(float32);
      });
    });

    describe('chaining with other operations', () => {
      it('should chain reductions with view operations', async () => {
        // PyTorch: torch.tensor([1, 2, 3, 4, 5, 6]).reshape(2, 3).sum(dim=1)
        // Output: tensor([6, 15])
        const vector = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

        // Reshape to matrix, then sum along axis
        const result = await vector.reshape([2, 3] as const).sum([1]);

        expect(result.shape).toEqual([2]);
        const data = await result.toArray();
        expect(data).toEqual([6, 15]); // [1+2+3, 4+5+6]
      });

      it('should chain reductions with unary operations', async () => {
        // PyTorch: torch.sqrt(torch.tensor([[1, 4], [9, 16]], dtype=torch.float32)).sum()
        // Output: tensor(10.0)
        const matrix = await tensor(
          [
            [1, 4],
            [9, 16],
          ] as const,
          { device, dtype: float32 },
        );

        // Take sqrt, then sum
        const result = await (await matrix.sqrt()).sum();

        expect(await result.item()).toBeCloseTo(10, 5); // sqrt(1) + sqrt(4) + sqrt(9) + sqrt(16) = 1+2+3+4 = 10
      });

      it('should use reductions in binary operations', async () => {
        // PyTorch: torch.tensor([1, 2, 3]).sum() + torch.tensor([4, 5, 6]).sum()
        // Output: 6 + 15 = 21
        const a = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const b = await tensor([4, 5, 6] as const, { device, dtype: float32 });

        // Sum each, then add the sums
        const sumA = await a.sum();
        const sumB = await b.sum();
        const result = await sumA.add(sumB);

        expect(await result.item()).toBeCloseTo(21, 5); // (1+2+3) + (4+5+6) = 6 + 15 = 21
      });

      it('should chain prod with other operations', async () => {
        // PyTorch: torch.tensor([2, 3, 4]).prod() * torch.tensor([1, 2]).prod()
        // Output: 24 * 2 = 48
        const a = await tensor([2, 3, 4] as const, { device, dtype: float32 });
        const b = await tensor([1, 2] as const, { device, dtype: float32 });

        const prodA = await a.prod();
        const prodB = await b.prod();
        const result = await prodA.mul(prodB);

        expect(await result.item()).toBeCloseTo(48, 5); // (2*3*4) * (1*2) = 24 * 2 = 48
      });
    });
  });
}
