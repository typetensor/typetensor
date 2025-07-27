/**
 * Test generators for view operations
 *
 * These generators test tensor view operations like reshape, transpose, slice,
 * permute, and flatten that create new views of tensor data without copying.
 */

import type { Device } from '@typetensor/core';
import { tensor, float32, int32 } from '@typetensor/core';

/**
 * Generates tests for view operations
 *
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateViewOperationTests(
  device: Device,
  testFramework: {
    describe: (name: string, fn: () => void) => void;
    it: (name: string, fn: () => void | Promise<void>) => void;
    expect: (actual: unknown) => {
      toBe: (expected: unknown) => void;
      toEqual: (expected: unknown) => void;
      toThrow: (error?: string | RegExp) => void;
      toBeCloseTo: (expected: number, precision?: number) => void;
      rejects: {
        toThrow: (error?: string | RegExp) => Promise<void>;
      };
    };
  },
) {
  const { describe, it, expect } = testFramework;

  describe(`View Operations Tests (${device.type}:${device.id})`, () => {
    describe('reshape operations', () => {
      it('should reshape vectors to matrices', async () => {
        // PyTorch: torch.tensor([1,2,3,4,5,6]).reshape(2, 3)
        // NumPy: np.array([1,2,3,4,5,6]).reshape(2, 3)
        const vector = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });
        const matrix = vector.reshape([2, 3] as const);

        // Verify shape transformation
        expect(vector.shape).toEqual([6]);
        expect(matrix.shape).toEqual([2, 3]);
        expect(matrix.ndim).toBe(2);
        expect(matrix.size).toBe(6);
        expect(matrix.dtype).toBe(float32);
        expect(matrix.device).toBe(device);

        // Verify data integrity
        const matrixData = await matrix.toArray();
        expect(matrixData).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);
      });

      it('should reshape matrices to vectors', async () => {
        // PyTorch: torch.tensor([[1,2,3],[4,5,6]]).reshape(6)
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const vector = matrix.reshape([6] as const);

        expect(matrix.shape).toEqual([2, 3]);
        expect(vector.shape).toEqual([6]);
        expect(vector.ndim).toBe(1);
        expect(vector.size).toBe(6);

        const vectorData = await vector.toArray();
        expect(vectorData).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('should reshape to higher dimensions', async () => {
        // PyTorch: torch.arange(24).reshape(2, 3, 4)
        const vector = await tensor(
          [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          ] as const,
          { device, dtype: int32 },
        );
        const tensor3d = vector.reshape([2, 3, 4] as const);

        expect(vector.shape).toEqual([24]);
        expect(tensor3d.shape).toEqual([2, 3, 4]);
        expect(tensor3d.ndim).toBe(3);
        expect(tensor3d.size).toBe(24);

        const data3d = await tensor3d.toArray();
        expect(Array.isArray(data3d)).toBe(true);
        expect(data3d.length).toBe(2);
        expect(Array.isArray(data3d[0])).toBe(true);
        expect(data3d[0].length).toBe(3);
        expect(Array.isArray(data3d[0][0])).toBe(true);
        expect(data3d[0][0].length).toBe(4);
      });

      it('should preserve data in different reshape combinations', async () => {
        const original = await tensor(
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
          ] as const,
          { device, dtype: float32 },
        );

        // Multiple valid reshapes of 12 elements
        const as_1d = original.reshape([12] as const);
        const as_2x6 = original.reshape([2, 6] as const);
        const as_4x3 = original.reshape([4, 3] as const);

        expect(as_1d.shape).toEqual([12]);
        expect(as_2x6.shape).toEqual([2, 6]);
        expect(as_4x3.shape).toEqual([4, 3]);

        // All should contain the same data in row-major order
        const originalFlat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        expect(await as_1d.toArray()).toEqual(originalFlat);
        expect(await as_2x6.toArray()).toEqual([
          [1, 2, 3, 4, 5, 6],
          [7, 8, 9, 10, 11, 12],
        ]);
        expect(await as_4x3.toArray()).toEqual([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
          [10, 11, 12],
        ]);
      });

      it('should throw on invalid reshape dimensions', async () => {
        const tensor12 = await tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as const, {
          device,
          dtype: float32,
        });

        // TESTED BEHAVIOR:
        // PyTorch: torch.tensor(range(12)).reshape(3, 5)
        //   -> RuntimeError: shape '[3, 5]' is invalid for input of size 12
        // NumPy: np.arange(12).reshape(3, 5)
        //   -> ValueError: cannot reshape array of size 12 into shape (3,5)

        expect(() => {
          // @ts-expect-error - this is expected to throw at compile time but just testing that runtime also guards
          tensor12.reshape([3, 5] as const); // 15 ≠ 12 elements
        }).toThrow(/different number of elements/);

        expect(() => {
          // @ts-expect-error - this is expected to throw at compile time but just testing that runtime also guards
          tensor12.reshape([2, 2] as const); // 4 ≠ 12 elements
        }).toThrow(/different number of elements/);
      });
    });

    describe('flatten operations', () => {
      it('should flatten matrices to vectors', async () => {
        // PyTorch: torch.tensor([[1,2,3],[4,5,6]]).flatten()
        // NumPy: np.array([[1,2,3],[4,5,6]]).flatten()
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const flattened = await matrix.flatten();

        expect(matrix.shape).toEqual([2, 3]);
        expect(flattened.shape).toEqual([6]);
        expect(flattened.ndim).toBe(1);
        expect(flattened.size).toBe(6);
        expect(flattened.dtype).toBe(float32);
        expect(flattened.device).toBe(device);

        const flatData = await flattened.toArray();
        expect(flatData).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('should flatten higher dimensional tensors', async () => {
        // PyTorch: torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]]).flatten()
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
          { device, dtype: int32 },
        );
        const flattened = await tensor3d.flatten();

        expect(tensor3d.shape).toEqual([2, 2, 2]);
        expect(flattened.shape).toEqual([8]);
        expect(flattened.size).toBe(8);

        const flatData = await flattened.toArray();
        expect(flatData).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
      });

      it('should handle scalar flattening', async () => {
        const scalar = await tensor(42, { device, dtype: float32 });
        const flattened = await scalar.flatten();

        expect(scalar.shape).toEqual([]);
        expect(flattened.shape).toEqual([1]);
        expect(flattened.size).toBe(1);

        const flatData = await flattened.toArray();
        expect(flatData).toEqual([42]);
      });

      it('should handle vector flattening (no-op)', async () => {
        const vector = await tensor([1, 2, 3, 4, 5] as const, { device, dtype: float32 });
        const flattened = await vector.flatten();

        expect(vector.shape).toEqual([5]);
        expect(flattened.shape).toEqual([5]);
        expect(flattened.size).toBe(5);

        const originalData = await vector.toArray();
        const flatData = await flattened.toArray();
        expect(flatData).toEqual(originalData);
      });
    });

    describe('transpose operations', () => {
      it('should transpose 2D matrices', async () => {
        // PyTorch: torch.tensor([[1,2,3],[4,5,6]]).T
        // NumPy: np.array([[1,2,3],[4,5,6]]).T
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const transposed = matrix.transpose();

        expect(matrix.shape).toEqual([2, 3]);
        expect(transposed.shape).toEqual([3, 2]);
        expect(transposed.ndim).toBe(2);
        expect(transposed.size).toBe(6);
        expect(transposed.dtype).toBe(float32);
        expect(transposed.device).toBe(device);

        const transposedData = await transposed.toArray();
        expect(transposedData).toEqual([
          [1, 4],
          [2, 5],
          [3, 6],
        ]);
      });

      it('should handle square matrix transpose', async () => {
        const square = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ] as const,
          { device, dtype: int32 },
        );
        const transposed = square.transpose();

        expect(square.shape).toEqual([3, 3]);
        expect(transposed.shape).toEqual([3, 3]);

        const transposedData = await transposed.toArray();
        expect(transposedData).toEqual([
          [1, 4, 7],
          [2, 5, 8],
          [3, 6, 9],
        ]);
      });

      it('should handle vector transpose (no change for 1D)', async () => {
        const vector = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });
        const transposed = vector.transpose();

        expect(vector.shape).toEqual([4]);
        expect(transposed.shape).toEqual([4]);

        const originalData = await vector.toArray();
        const transposedData = await transposed.toArray();
        expect(transposedData).toEqual(originalData);
      });

      it('should handle scalar transpose (no-op)', async () => {
        const scalar = await tensor(42, { device, dtype: float32 });
        const transposed = scalar.transpose();

        expect(scalar.shape).toEqual([]);
        expect(transposed.shape).toEqual([]);

        expect(await scalar.item()).toBe(42);
        expect(await transposed.item()).toBe(42);
      });
    });

    describe('view consistency', () => {
      it('should maintain device consistency across views', async () => {
        const original = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const reshaped = original.reshape([4] as const);
        const flattened = original.flatten();
        const transposed = original.transpose();

        expect(reshaped.device).toBe(device);
        expect(flattened.device).toBe(device);
        expect(transposed.device).toBe(device);
      });

      it('should maintain dtype consistency across views', async () => {
        const original = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: int32 },
        );
        const reshaped = original.reshape([4] as const);
        const flattened = original.flatten();
        const transposed = original.transpose();

        expect(reshaped.dtype).toBe(int32);
        expect(flattened.dtype).toBe(int32);
        expect(transposed.dtype).toBe(int32);
      });

      it('should preserve total element count across views', async () => {
        const original = await tensor(
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

        const reshaped = original.reshape([4, 2] as const);
        const flattened = await original.flatten();
        const transposed = original.transpose();

        expect(original.size).toBe(8);
        expect(reshaped.size).toBe(8);
        expect(flattened.size).toBe(8);
        expect(transposed.size).toBe(8);
      });

      it('should chain view operations correctly', async () => {
        // TESTED BEHAVIOR:
        // PyTorch: torch.tensor([1,2,3,4,5,6]).reshape(2,3).T.flatten()
        //   Original: [1, 2, 3, 4, 5, 6]
        //   After reshape(2,3): [[1, 2, 3], [4, 5, 6]]
        //   After transpose: [[1, 4], [2, 5], [3, 6]]
        //   After flatten: [1, 4, 2, 5, 3, 6]
        // NumPy: np.array([1,2,3,4,5,6]).reshape(2,3).T.flatten() -> same result

        const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });
        const chained = await original
          .reshape([2, 3] as const)
          .transpose()
          .flatten();

        expect(original.shape).toEqual([6]);
        expect(chained.shape).toEqual([6]);

        const chainedData = await chained.toArray();
        expect(chainedData).toEqual([1, 4, 2, 5, 3, 6]);
      });
    });
  });
}
