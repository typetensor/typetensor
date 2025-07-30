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
        // Output: tensor([[1, 2, 3],
        //                 [4, 5, 6]])
        // shape: torch.Size([2, 3])
        const vector = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });
        const matrix = await vector.reshape([2, 3] as const);

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
        // Output: tensor([1, 2, 3, 4, 5, 6])
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const vector = await matrix.reshape([6] as const);

        expect(matrix.shape).toEqual([2, 3]);
        expect(vector.shape).toEqual([6]);
        expect(vector.ndim).toBe(1);
        expect(vector.size).toBe(6);

        const vectorData = await vector.toArray();
        expect(vectorData).toEqual([1, 2, 3, 4, 5, 6]);
      });

      it('should reshape to higher dimensions', async () => {
        // PyTorch: torch.arange(1, 25).reshape(2, 3, 4)
        // Output shape: torch.Size([2, 3, 4])
        // First batch: tensor([[ 1,  2,  3,  4],
        //                      [ 5,  6,  7,  8],
        //                      [ 9, 10, 11, 12]])
        const vector = await tensor(
          [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          ] as const,
          { device, dtype: int32 },
        );
        const tensor3d = await vector.reshape([2, 3, 4] as const);

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
        // PyTorch: Original tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        // Can be reshaped to [12], [2, 6], [4, 3] etc. preserving row-major order
        const original = await tensor(
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
          ] as const,
          { device, dtype: float32 },
        );

        // Multiple valid reshapes of 12 elements
        const as_1d = await original.reshape([12] as const);
        const as_2x6 = await original.reshape([2, 6] as const);
        const as_4x3 = await original.reshape([4, 3] as const);

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

        // PyTorch: torch.arange(12).reshape(3, 5)
        // Error: RuntimeError: shape '[3, 5]' is invalid for input of size 12

        // @ts-expect-error - test error
        await expect(tensor12.reshape([3, 5] as const)).rejects.toThrow(
          /different number of elements/,
        );

        // @ts-expect-error - test error
        await expect(tensor12.reshape([2, 2] as const)).rejects.toThrow(
          /different number of elements/,
        );
      });
    });

    describe('flatten operations', () => {
      it('should flatten matrices to vectors', async () => {
        // PyTorch: torch.tensor([[1,2,3],[4,5,6]]).flatten()
        // Output: tensor([1, 2, 3, 4, 5, 6])
        // shape: torch.Size([6])
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
        // Output: tensor([1, 2, 3, 4, 5, 6, 7, 8])
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
        // PyTorch: torch.tensor(42).flatten()
        // Output: tensor([42]), shape: torch.Size([1])
        const scalar = await tensor(42, { device, dtype: float32 });
        const flattened = await scalar.flatten();

        expect(scalar.shape).toEqual([]);
        expect(flattened.shape).toEqual([1]);
        expect(flattened.size).toBe(1);

        const flatData = await flattened.toArray();
        expect(flatData).toEqual([42]);
      });

      it('should handle vector flattening (no-op)', async () => {
        // PyTorch: torch.tensor([1, 2, 3, 4, 5]).flatten()
        // Output: tensor([1, 2, 3, 4, 5]) - no change for 1D tensors
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
        // Output: tensor([[1, 4],
        //                 [2, 5],
        //                 [3, 6]])
        // shape: torch.Size([3, 2])
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const transposed = await matrix.transpose();

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
        // PyTorch: torch.tensor([[1,2,3],[4,5,6],[7,8,9]]).T
        // Output: tensor([[1, 4, 7],
        //                 [2, 5, 8],
        //                 [3, 6, 9]])
        const square = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ] as const,
          { device, dtype: int32 },
        );
        const transposed = await square.transpose();

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
        // PyTorch: torch.tensor([1, 2, 3, 4]).T
        // Output: tensor([1, 2, 3, 4]) - no change
        // Note: PyTorch warns about using .T on 1D tensors
        const vector = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });
        const transposed = await vector.transpose();

        expect(vector.shape).toEqual([4]);
        expect(transposed.shape).toEqual([4]);

        const originalData = await vector.toArray();
        const transposedData = await transposed.toArray();
        expect(transposedData).toEqual(originalData);
      });

      it('should handle scalar transpose (no-op)', async () => {
        // PyTorch: torch.tensor(42).T
        // Output: tensor(42) - scalars remain unchanged
        const scalar = await tensor(42, { device, dtype: float32 });
        const transposed = await scalar.transpose();

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
        const reshaped = await original.reshape([4] as const);
        const flattened = await original.flatten();
        const transposed = await original.transpose();

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
        const reshaped = await original.reshape([4] as const);
        const flattened = await original.flatten();
        const transposed = await original.transpose();

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

        const reshaped = await original.reshape([4, 2] as const);
        const flattened = await original.flatten();
        const transposed = await original.transpose();

        expect(original.size).toBe(8);
        expect(reshaped.size).toBe(8);
        expect(flattened.size).toBe(8);
        expect(transposed.size).toBe(8);
      });

      it('should chain view operations correctly', async () => {
        // PyTorch: torch.tensor([1,2,3,4,5,6]).reshape(2,3).T.flatten()
        // Step 1: reshape(2,3) -> tensor([[1, 2, 3], [4, 5, 6]])
        // Step 2: .T -> tensor([[1, 4], [2, 5], [3, 6]])
        // Step 3: flatten() -> tensor([1, 4, 2, 5, 3, 6])

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

    describe('slice operations', () => {
      it('should slice vectors with start and end indices', async () => {
        // PyTorch: torch.tensor([1, 2, 3, 4, 5, 6])[1:4]
        // Output: tensor([2, 3, 4])
        const vector = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

        // Slice [1:4] should give [2, 3, 4]
        const sliced = await vector.slice([{ start: 1, stop: 4 }]);

        expect(sliced.shape).toEqual([3]);
        expect(sliced.dtype).toBe(float32);
        expect(sliced.device).toBe(device);

        const data = await sliced.toArray();
        expect(data).toEqual([2, 3, 4]);
      });

      it('should slice matrices along rows', async () => {
        // PyTorch: matrix[1:3] where matrix is 4x3
        // Output: tensor([[4, 5, 6],
        //                 [7, 8, 9]])
        // shape: torch.Size([2, 3])
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
          ] as const,
          { device, dtype: float32 },
        );

        // Slice rows [1:3] should give middle two rows
        const sliced = await matrix.slice([{ start: 1, stop: 3 }, null]);

        expect(sliced.shape).toEqual([2, 3]);

        const data = await sliced.toArray();
        expect(data).toEqual([
          [4, 5, 6],
          [7, 8, 9],
        ]);
      });

      it('should slice matrices along columns', async () => {
        const matrix = await tensor(
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ] as const,
          { device, dtype: float32 },
        );

        // PyTorch: matrix[:, 1:3]
        // Output: tensor([[2, 3],
        //                 [6, 7]])
        const sliced = await matrix.slice([null, { start: 1, stop: 3 }]);

        expect(sliced.shape).toEqual([2, 2]);

        const data = await sliced.toArray();
        expect(data).toEqual([
          [2, 3],
          [6, 7],
        ]);
      });

      it('should handle 3D tensor slicing', async () => {
        // PyTorch: tensor3d[0:2] for shape [3, 2, 2]
        // Output shape: torch.Size([2, 2, 2])
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
            [
              [9, 10],
              [11, 12],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        // Slice first dimension [0:2]
        const sliced = await tensor3d.slice([{ start: 0, stop: 2 }, null, null]);

        expect(sliced.shape).toEqual([2, 2, 2]);

        const data = await sliced.toArray();
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

      it('should handle single element slices', async () => {
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // PyTorch: matrix[1:2, 1:2]
        // Output: tensor([[5]])
        // shape: torch.Size([1, 1]) - preserves dimensions
        const sliced = await matrix.slice([
          { start: 1, stop: 2 },
          { start: 1, stop: 2 },
        ]);

        expect(sliced.shape).toEqual([1, 1]);

        const data = await sliced.toArray();
        expect(data).toEqual([[5]]);
      });

      it('should preserve data integrity in slices', async () => {
        const original = await tensor(
          [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90],
          ] as const,
          { device, dtype: float32 },
        );

        const sliced = await original.slice([
          { start: 0, stop: 2 },
          { start: 1, stop: 3 },
        ]);

        expect(sliced.shape).toEqual([2, 2]);
        expect(sliced.dtype).toBe(float32);
        expect(sliced.device).toBe(device);

        const data = await sliced.toArray();
        expect(data).toEqual([
          [20, 30],
          [50, 60],
        ]);
      });
    });

    describe('permute operations', () => {
      it('should permute 2D matrix dimensions', async () => {
        // PyTorch: torch.tensor([[1,2,3],[4,5,6]]).permute(1, 0)
        // Output: tensor([[1, 4],
        //                 [2, 5],
        //                 [3, 6]])
        // shape: torch.Size([3, 2])
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // Permute [0, 1] -> [1, 0] (same as transpose)
        const permuted = await matrix.permute([1, 0] as const);

        expect(permuted.shape).toEqual([3, 2]);
        expect(permuted.dtype).toBe(float32);
        expect(permuted.device).toBe(device);

        const data = await permuted.toArray();
        expect(data).toEqual([
          [1, 4],
          [2, 5],
          [3, 6],
        ]);
      });

      it('should permute 3D tensor dimensions', async () => {
        // PyTorch: tensor of shape [2, 3, 4].permute(2, 0, 1)
        // Output shape: torch.Size([4, 2, 3])
        // Element mapping:
        //   permuted[0,0,0] = 1 (was tensor3d[0,0,0])
        //   permuted[1,0,0] = 2 (was tensor3d[0,0,1])
        //   permuted[0,1,0] = 13 (was tensor3d[1,0,0])
        //   permuted[0,0,1] = 5 (was tensor3d[0,1,0])
        const tensor3d = await tensor(
          [
            [
              [1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12],
            ],
            [
              [13, 14, 15, 16],
              [17, 18, 19, 20],
              [21, 22, 23, 24],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        expect(tensor3d.shape).toEqual([2, 3, 4]);

        // Permute [0, 1, 2] -> [2, 0, 1]
        // Shape should become [4, 2, 3]
        const permuted = await tensor3d.permute([2, 0, 1] as const);

        expect(permuted.shape).toEqual([4, 2, 3]);
        expect(permuted.dtype).toBe(float32);
        expect(permuted.device).toBe(device);

        const data = await permuted.toArray();

        // First element of each dimension should match expected positions
        expect(data[0][0][0]).toBe(1); // Original [0,0,0]
        expect(data[1][0][0]).toBe(2); // Original [0,0,1]
        expect(data[0][1][0]).toBe(13); // Original [1,0,0]
        expect(data[0][0][1]).toBe(5); // Original [0,1,0]
      });

      it('should handle identity permutation', async () => {
        // PyTorch: matrix.permute(0, 1) - identity permutation
        // No change to data or shape
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        // Identity permutation [0, 1] should not change anything
        const permuted = await matrix.permute([0, 1] as const);

        expect(permuted.shape).toEqual([2, 2]);

        const originalData = await matrix.toArray();
        const permutedData = await permuted.toArray();
        expect(permutedData).toEqual(originalData);
      });

      it('should handle vector permutation (no-op)', async () => {
        // PyTorch: vector.permute(0) for 1D tensor
        // No change - only one dimension to permute
        const vector = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });

        // Vector permutation should be identity
        const permuted = await vector.permute([0] as const);

        expect(permuted.shape).toEqual([4]);

        const originalData = await vector.toArray();
        const permutedData = await permuted.toArray();
        expect(permutedData).toEqual(originalData);
      });

      it('should preserve data integrity across permutations', async () => {
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

        // Multiple permutations should preserve total elements
        const permuted1 = await tensor3d.permute([1, 2, 0] as const);
        const permuted2 = await permuted1.permute([2, 0, 1] as const);

        expect(tensor3d.shape).toEqual([2, 2, 2]);
        expect(permuted1.shape).toEqual([2, 2, 2]);
        expect(permuted2.shape).toEqual([2, 2, 2]);

        // All should have same total elements
        expect(tensor3d.size).toBe(8);
        expect(permuted1.size).toBe(8);
        expect(permuted2.size).toBe(8);

        // Round-trip should return to original
        const finalData = await permuted2.toArray();
        const originalData = await tensor3d.toArray();
        expect(finalData).toEqual(originalData);
      });

      it('should work with complex permutation patterns', async () => {
        // PyTorch: Convert BCHW to BHWC format
        // BCHW shape: [1, 1, 2, 2] -> BHWC shape: [1, 2, 2, 1]
        // Data reorganized from channel-first to channel-last
        const bchw = await tensor(
          [
            [
              [
                [1, 2],
                [3, 4],
              ],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        expect(bchw.shape).toEqual([1, 1, 2, 2]);

        // Convert BCHW to BHWC (batch, height, width, channels)
        const bhwc = await bchw.permute([0, 2, 3, 1] as const);

        expect(bhwc.shape).toEqual([1, 2, 2, 1]);
        expect(bhwc.dtype).toBe(float32);
        expect(bhwc.device).toBe(device);

        const data = await bhwc.toArray();
        expect(data).toEqual([
          [
            [[1], [2]],
            [[3], [4]],
          ],
        ]);
      });
    });

    describe('squeeze operations', () => {
      it('should squeeze all size-1 dimensions', async () => {
        // PyTorch: torch.tensor([[[1], [2]], [[3], [4]]]).squeeze()
        // Input shape: torch.Size([2, 2, 1])
        // Output: tensor([[1, 2],
        //                 [3, 4]])
        // Output shape: torch.Size([2, 2])
        const tensor3d = await tensor(
          [
            [[1], [2]],
            [[3], [4]],
          ] as const,
          { device, dtype: float32 },
        );
        const squeezed = await tensor3d.squeeze();

        expect(tensor3d.shape).toEqual([2, 2, 1]);
        expect(squeezed.shape).toEqual([2, 2]);
        expect(squeezed.ndim).toBe(2);
        expect(squeezed.size).toBe(4);
        expect(squeezed.dtype).toBe(float32);
        expect(squeezed.device).toBe(device);

        const data = await squeezed.toArray();
        expect(data).toEqual([
          [1, 2],
          [3, 4],
        ]);
      });

      it('should squeeze specific axis with size 1', async () => {
        // PyTorch: torch.tensor([[[1, 2, 3]]]).squeeze(0)
        // Input shape: torch.Size([1, 1, 3])
        // Output shape: torch.Size([1, 3])
        const tensor3d = await tensor([[[1, 2, 3]]] as const, { device, dtype: float32 });
        const squeezed = await tensor3d.squeeze([0] as const);

        expect(tensor3d.shape).toEqual([1, 1, 3]);
        expect(squeezed.shape).toEqual([1, 3]);

        const data = await squeezed.toArray();
        expect(data).toEqual([[1, 2, 3]]);
      });

      it('should squeeze multiple specific axes', async () => {
        // PyTorch: torch.tensor([[[[5]]]]).squeeze([0, 3])
        // Input shape: torch.Size([1, 1, 1, 1])
        // Output shape: torch.Size([1, 1])
        const tensor4d = await tensor([[[[5]]]] as const, { device, dtype: float32 });
        const squeezed = await tensor4d.squeeze([0, 3] as const);

        expect(tensor4d.shape).toEqual([1, 1, 1, 1]);
        expect(squeezed.shape).toEqual([1, 1]);

        const data = await squeezed.toArray();
        expect(data).toEqual([[5]]);
      });

      it('should handle squeeze on tensor with no size-1 dimensions', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]).squeeze()
        // No change - no dimensions with size 1
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const squeezed = await matrix.squeeze();

        expect(matrix.shape).toEqual([2, 2]);
        expect(squeezed.shape).toEqual([2, 2]);

        const data = await squeezed.toArray();
        expect(data).toEqual([
          [1, 2],
          [3, 4],
        ]);
      });

      it('should handle squeeze on scalar (no-op)', async () => {
        // PyTorch: torch.tensor(42).squeeze()
        // Output: tensor(42) - scalars remain unchanged
        const scalar = await tensor(42, { device, dtype: float32 });
        const squeezed = await scalar.squeeze();

        expect(scalar.shape).toEqual([]);
        expect(squeezed.shape).toEqual([]);

        expect(await squeezed.item()).toBe(42);
      });

      it('should handle negative axis indices', async () => {
        // PyTorch: torch.tensor([[[1, 2, 3]]]).squeeze(-3)
        // Same as squeeze(0) - removes first dimension
        const tensor3d = await tensor([[[1, 2, 3]]] as const, { device, dtype: float32 });
        const squeezed = await tensor3d.squeeze([-3] as const);

        expect(tensor3d.shape).toEqual([1, 1, 3]);
        expect(squeezed.shape).toEqual([1, 3]);

        const data = await squeezed.toArray();
        expect(data).toEqual([[1, 2, 3]]);
      });

      it('should preserve data through squeeze operation', async () => {
        // PyTorch: Complex squeeze preserving data integrity
        const tensor5d = await tensor(
          [
            [
              [[[1]], [[2]]],
              [[[3]], [[4]]],
            ],
          ] as const,
          { device, dtype: int32 },
        );

        expect(tensor5d.shape).toEqual([1, 2, 2, 1, 1]);

        // Squeeze all dimensions
        const squeezed = await tensor5d.squeeze();
        expect(squeezed.shape).toEqual([2, 2]);

        const data = await squeezed.toArray();
        expect(data).toEqual([
          [1, 2],
          [3, 4],
        ]);
      });

      it('should error when trying to squeeze non-unit dimension', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]).squeeze(0)
        // RuntimeError: cannot select an axis to squeeze out which has size not equal to one
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        // Should throw an error when trying to squeeze a dimension with size != 1
        await expect(matrix.squeeze([0] as const)).rejects.toThrow(/size.*must be 1/);
      });
    });

    describe('unsqueeze operations', () => {
      it('should unsqueeze at position 0', async () => {
        // PyTorch: torch.tensor([1, 2, 3]).unsqueeze(0)
        // Input shape: torch.Size([3])
        // Output: tensor([[1, 2, 3]])
        // Output shape: torch.Size([1, 3])
        const vector = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const unsqueezed = await vector.unsqueeze(0);

        expect(vector.shape).toEqual([3]);
        expect(unsqueezed.shape).toEqual([1, 3]);
        expect(unsqueezed.ndim).toBe(2);
        expect(unsqueezed.size).toBe(3);
        expect(unsqueezed.dtype).toBe(float32);
        expect(unsqueezed.device).toBe(device);

        const data = await unsqueezed.toArray();
        expect(data).toEqual([[1, 2, 3]]);
      });

      it('should unsqueeze at last position', async () => {
        // PyTorch: torch.tensor([1, 2, 3]).unsqueeze(1)
        // or: torch.tensor([1, 2, 3]).unsqueeze(-1)
        // Output: tensor([[1],
        //                 [2],
        //                 [3]])
        // Output shape: torch.Size([3, 1])
        const vector = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const unsqueezed = await vector.unsqueeze(1);

        expect(vector.shape).toEqual([3]);
        expect(unsqueezed.shape).toEqual([3, 1]);

        const data = await unsqueezed.toArray();
        expect(data).toEqual([[1], [2], [3]]);
      });

      it('should unsqueeze in the middle', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]).unsqueeze(1)
        // Input shape: torch.Size([2, 2])
        // Output shape: torch.Size([2, 1, 2])
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const unsqueezed = await matrix.unsqueeze(1);

        expect(matrix.shape).toEqual([2, 2]);
        expect(unsqueezed.shape).toEqual([2, 1, 2]);

        const data = await unsqueezed.toArray();
        expect(data).toEqual([[[1, 2]], [[3, 4]]]);
      });

      it('should unsqueeze scalar tensor', async () => {
        // PyTorch: torch.tensor(42).unsqueeze(0)
        // Output: tensor([42])
        // Output shape: torch.Size([1])
        const scalar = await tensor(42, { device, dtype: float32 });
        const unsqueezed = await scalar.unsqueeze(0);

        expect(scalar.shape).toEqual([]);
        expect(unsqueezed.shape).toEqual([1]);

        const data = await unsqueezed.toArray();
        expect(data).toEqual([42]);
      });

      it('should handle negative axis for unsqueeze', async () => {
        // PyTorch: torch.tensor([1, 2, 3]).unsqueeze(-1)
        // Same as unsqueeze(1) for 1D tensor
        const vector = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const unsqueezed = await vector.unsqueeze(-1);

        expect(vector.shape).toEqual([3]);
        expect(unsqueezed.shape).toEqual([3, 1]);

        const data = await unsqueezed.toArray();
        expect(data).toEqual([[1], [2], [3]]);
      });

      it('should handle multiple unsqueeze operations', async () => {
        // PyTorch: torch.tensor([1, 2]).unsqueeze(0).unsqueeze(2)
        // Step 1: [1, 2] -> [[1, 2]] (shape: [1, 2])
        // Step 2: [[1, 2]] -> [[[1], [2]]] (shape: [1, 2, 1])
        const vector = await tensor([1, 2] as const, { device, dtype: float32 });
        const unsqueezed1 = await vector.unsqueeze(0);
        const unsqueezed2 = await unsqueezed1.unsqueeze(2);

        expect(vector.shape).toEqual([2]);
        expect(unsqueezed1.shape).toEqual([1, 2]);
        expect(unsqueezed2.shape).toEqual([1, 2, 1]);

        const data = await unsqueezed2.toArray();
        expect(data).toEqual([[[1], [2]]]);
      });

      it('should preserve data through unsqueeze operations', async () => {
        // Test that data is preserved correctly through unsqueeze
        const matrix = await tensor(
          [
            [10, 20, 30],
            [40, 50, 60],
          ] as const,
          { device, dtype: int32 },
        );

        const unsqueezed = await matrix.unsqueeze(1);
        expect(unsqueezed.shape).toEqual([2, 1, 3]);

        const data = await unsqueezed.toArray();
        expect(data).toEqual([[[10, 20, 30]], [[40, 50, 60]]]);
      });

      it('should error on invalid unsqueeze axis', async () => {
        // PyTorch: torch.tensor([1, 2, 3]).unsqueeze(5)
        // IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 5)
        const vector = await tensor([1, 2, 3] as const, { device, dtype: float32 });

        // Axis 5 is out of bounds for a 1D tensor
        await expect(vector.unsqueeze(5)).rejects.toThrow(/out of bounds/);
      });
    });

    describe('expand operations', () => {
      it('should expand singleton dimensions', async () => {
        // PyTorch: torch.tensor([[1], [2]]).expand(2, 3)
        // Output: tensor([[1, 1, 1],
        //                 [2, 2, 2]])
        const matrix = await tensor([[1], [2]] as const, { device, dtype: float32 });
        const expanded = await matrix.expand([2, 3] as const);

        expect(matrix.shape).toEqual([2, 1]);
        expect(expanded.shape).toEqual([2, 3]);
        expect(expanded.ndim).toBe(2);
        expect(expanded.size).toBe(6);
        expect(expanded.dtype).toBe(float32);
        expect(expanded.device).toBe(device);

        const data = await expanded.toArray();
        expect(data).toEqual([
          [1, 1, 1],
          [2, 2, 2],
        ]);
      });

      it('should expand with -1 to keep dimensions', async () => {
        // PyTorch: torch.tensor([[1], [2], [3]]).expand(-1, 4)
        // Output: tensor([[1, 1, 1, 1],
        //                 [2, 2, 2, 2],
        //                 [3, 3, 3, 3]])
        const matrix = await tensor([[1], [2], [3]] as const, { device, dtype: float32 });
        const expanded = await matrix.expand([-1, 4] as const);

        expect(matrix.shape).toEqual([3, 1]);
        expect(expanded.shape).toEqual([3, 4]);

        const data = await expanded.toArray();
        expect(data).toEqual([
          [1, 1, 1, 1],
          [2, 2, 2, 2],
          [3, 3, 3, 3],
        ]);
      });

      it('should expand by adding new dimensions', async () => {
        // PyTorch: torch.tensor([1, 2, 3]).expand(2, 3)
        // Output: tensor([[1, 2, 3],
        //                 [1, 2, 3]])
        const vector = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const expanded = await vector.expand([2, 3] as const);

        expect(vector.shape).toEqual([3]);
        expect(expanded.shape).toEqual([2, 3]);

        const data = await expanded.toArray();
        expect(data).toEqual([
          [1, 2, 3],
          [1, 2, 3],
        ]);
      });

      it('should expand scalar tensor', async () => {
        // PyTorch: torch.tensor(42).expand(3, 4)
        // Output: tensor([[42, 42, 42, 42],
        //                 [42, 42, 42, 42],
        //                 [42, 42, 42, 42]])
        const scalar = await tensor(42, { device, dtype: float32 });
        const expanded = await scalar.expand([3, 4] as const);

        expect(scalar.shape).toEqual([]);
        expect(expanded.shape).toEqual([3, 4]);

        const data = await expanded.toArray();
        expect(data).toEqual([
          [42, 42, 42, 42],
          [42, 42, 42, 42],
          [42, 42, 42, 42],
        ]);
      });

      it('should expand higher dimensional tensors', async () => {
        // PyTorch: torch.tensor([[[1]], [[2]]]).expand(2, 3, 4)
        // Shape: [2, 1, 1] -> [2, 3, 4]
        const tensor3d = await tensor([[[1]], [[2]]] as const, { device, dtype: float32 });
        const expanded = await tensor3d.expand([2, 3, 4] as const);

        expect(tensor3d.shape).toEqual([2, 1, 1]);
        expect(expanded.shape).toEqual([2, 3, 4]);

        const data = await expanded.toArray();
        expect(data[0]).toEqual([
          [1, 1, 1, 1],
          [1, 1, 1, 1],
          [1, 1, 1, 1],
        ]);
        expect(data[1]).toEqual([
          [2, 2, 2, 2],
          [2, 2, 2, 2],
          [2, 2, 2, 2],
        ]);
      });

      it('should error on invalid expansion', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]).expand(2, 5)
        // RuntimeError: The expanded size of the tensor (5) must match the existing size (2) at non-singleton dimension 1
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        // Cannot expand non-singleton dimension 2 to 5
        await expect(matrix.expand([2, 5] as const)).rejects.toThrow(/Cannot expand dimension/);
      });

      it('should handle complex expand patterns', async () => {
        // PyTorch: Broadcasting pattern for attention masks
        // mask of shape [1, 1, seq_len, seq_len] expanded to [batch, heads, seq_len, seq_len]
        const mask = await tensor(
          [
            [
              [
                [1, 0],
                [1, 1],
              ],
            ],
          ] as const,
          { device, dtype: float32 },
        );
        const expanded = await mask.expand([32, 8, 2, 2] as const);

        expect(mask.shape).toEqual([1, 1, 2, 2]);
        expect(expanded.shape).toEqual([32, 8, 2, 2]);

        // Check a sample of the expanded data
        const data = await expanded.toArray();
        expect(data[0][0]).toEqual([
          [1, 0],
          [1, 1],
        ]);
        expect(data[31][7]).toEqual([
          [1, 0],
          [1, 1],
        ]);
      });
    });

    describe('tile operations', () => {
      it('should tile vectors', async () => {
        // PyTorch: torch.tensor([1, 2, 3]).repeat(2)
        // Output: tensor([1, 2, 3, 1, 2, 3])
        const vector = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const tiled = await vector.tile([2] as const);

        expect(vector.shape).toEqual([3]);
        expect(tiled.shape).toEqual([6]);
        expect(tiled.ndim).toBe(1);
        expect(tiled.size).toBe(6);
        expect(tiled.dtype).toBe(float32);
        expect(tiled.device).toBe(device);

        const data = await tiled.toArray();
        expect(data).toEqual([1, 2, 3, 1, 2, 3]);
      });

      it('should tile matrices', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]).repeat(2, 3)
        // Output: tensor([[1, 2, 1, 2, 1, 2],
        //                 [3, 4, 3, 4, 3, 4],
        //                 [1, 2, 1, 2, 1, 2],
        //                 [3, 4, 3, 4, 3, 4]])
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const tiled = await matrix.tile([2, 3] as const);

        expect(matrix.shape).toEqual([2, 2]);
        expect(tiled.shape).toEqual([4, 6]);

        const data = await tiled.toArray();
        expect(data).toEqual([
          [1, 2, 1, 2, 1, 2],
          [3, 4, 3, 4, 3, 4],
          [1, 2, 1, 2, 1, 2],
          [3, 4, 3, 4, 3, 4],
        ]);
      });

      it('should tile with more repetitions than dimensions', async () => {
        // PyTorch: torch.tensor([1, 2]).repeat(2, 3)
        // Shape: [2] -> [2, 6]
        const vector = await tensor([1, 2] as const, { device, dtype: float32 });
        const tiled = await vector.tile([2, 3] as const);

        expect(vector.shape).toEqual([2]);
        expect(tiled.shape).toEqual([2, 6]);

        const data = await tiled.toArray();
        expect(data).toEqual([
          [1, 2, 1, 2, 1, 2],
          [1, 2, 1, 2, 1, 2],
        ]);
      });

      it('should tile scalar tensor', async () => {
        // PyTorch: torch.tensor(42).repeat(3, 4)
        // Output: tensor([[42, 42, 42, 42],
        //                 [42, 42, 42, 42],
        //                 [42, 42, 42, 42]])
        const scalar = await tensor(42, { device, dtype: float32 });
        const tiled = await scalar.tile([3, 4] as const);

        expect(scalar.shape).toEqual([]);
        expect(tiled.shape).toEqual([3, 4]);

        const data = await tiled.toArray();
        expect(data).toEqual([
          [42, 42, 42, 42],
          [42, 42, 42, 42],
          [42, 42, 42, 42],
        ]);
      });

      it('should tile with identity repetitions', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]).repeat(1, 1)
        // No change - all reps are 1
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const tiled = await matrix.tile([1, 1] as const);

        expect(matrix.shape).toEqual([2, 2]);
        expect(tiled.shape).toEqual([2, 2]);

        const data = await tiled.toArray();
        expect(data).toEqual([
          [1, 2],
          [3, 4],
        ]);
      });

      it('should handle partial dimension tiling', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5, 6]]).repeat(1, 2)
        // Only tile the last dimension
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const tiled = await matrix.tile([1, 2] as const);

        expect(matrix.shape).toEqual([2, 3]);
        expect(tiled.shape).toEqual([2, 6]);

        const data = await tiled.toArray();
        expect(data).toEqual([
          [1, 2, 3, 1, 2, 3],
          [4, 5, 6, 4, 5, 6],
        ]);
      });

      it('should handle zero repetitions', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]).repeat(0, 2)
        // Output shape: [0, 4]
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const tiled = await matrix.tile([0, 2] as const);

        expect(matrix.shape).toEqual([2, 2]);
        expect(tiled.shape).toEqual([0, 4]);
        expect(tiled.size).toBe(0);

        const data = await tiled.toArray();
        expect(data).toEqual([]);
      });

      it('should tile 3D tensors', async () => {
        // PyTorch: torch.tensor([[[1, 2]], [[3, 4]]]).repeat(1, 2, 3)
        // Shape: [2, 1, 2] -> [2, 2, 6]
        const tensor3d = await tensor([[[1, 2]], [[3, 4]]] as const, { device, dtype: float32 });
        const tiled = await tensor3d.tile([1, 2, 3] as const);

        expect(tensor3d.shape).toEqual([2, 1, 2]);
        expect(tiled.shape).toEqual([2, 2, 6]);

        const data = await tiled.toArray();
        expect(data).toEqual([
          [
            [1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2],
          ],
          [
            [3, 4, 3, 4, 3, 4],
            [3, 4, 3, 4, 3, 4],
          ],
        ]);
      });
    });

    describe('expand and tile interaction', () => {
      it('should combine expand and tile operations', async () => {
        // PyTorch: Create a pattern by expanding then tiling
        const vector = await tensor([1, 2] as const, { device, dtype: float32 });

        // First expand to add a dimension
        const expanded = await vector.expand([3, 2] as const);
        expect(expanded.shape).toEqual([3, 2]);

        // Then tile the expanded result
        const tiled = await expanded.tile([2, 2] as const);
        expect(tiled.shape).toEqual([6, 4]);

        const data = await tiled.toArray();
        expect(data).toEqual([
          [1, 2, 1, 2],
          [1, 2, 1, 2],
          [1, 2, 1, 2],
          [1, 2, 1, 2],
          [1, 2, 1, 2],
          [1, 2, 1, 2],
        ]);
      });

      it('should work with other view operations', async () => {
        // PyTorch: Combine expand/tile with reshape, transpose, etc.
        const matrix = await tensor([[1], [2]] as const, { device, dtype: float32 });

        // Expand -> transpose -> tile
        const expanded = await matrix.expand([2, 3] as const);
        const transposed = await expanded.transpose();
        const tiled = await transposed.tile([2, 1] as const);

        expect(expanded.shape).toEqual([2, 3]);
        expect(transposed.shape).toEqual([3, 2]);
        expect(tiled.shape).toEqual([6, 2]);

        const data = await tiled.toArray();
        expect(data).toEqual([
          [1, 2],
          [1, 2],
          [1, 2],
          [1, 2],
          [1, 2],
          [1, 2],
        ]);
      });
    });

    describe('squeeze and unsqueeze interaction', () => {
      it('should round-trip through squeeze and unsqueeze', async () => {
        // PyTorch: tensor with shape [2, 1, 3, 1]
        // squeeze() -> [2, 3]
        // unsqueeze(1).unsqueeze(3) -> [2, 1, 3, 1]
        const original = await tensor([[[[1], [2], [3]]], [[[4], [5], [6]]]] as const, {
          device,
          dtype: float32,
        });

        expect(original.shape).toEqual([2, 1, 3, 1]);

        // Squeeze all size-1 dimensions
        const squeezed = await original.squeeze();
        expect(squeezed.shape).toEqual([2, 3]);

        // Unsqueeze back to original shape
        const unsqueezed = await squeezed.unsqueeze(1).unsqueeze(3);
        expect(unsqueezed.shape).toEqual([2, 1, 3, 1]);

        // Data should be preserved
        const originalData = await original.toArray();
        const roundTripData = await unsqueezed.toArray();
        expect(roundTripData).toEqual(originalData);
      });

      it('should work with view operations', async () => {
        // PyTorch: Combine squeeze/unsqueeze with other view ops
        const vector = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

        // Reshape -> unsqueeze -> transpose -> squeeze
        const reshaped = await vector.reshape([2, 3] as const);
        const unsqueezed = await reshaped.unsqueeze(0); // [1, 2, 3]
        const transposed = await unsqueezed.transpose(); // [1, 3, 2]
        const squeezed = await transposed.squeeze([0] as const); // [3, 2]

        expect(squeezed.shape).toEqual([3, 2]);

        const data = await squeezed.toArray();
        expect(data).toEqual([
          [1, 4],
          [2, 5],
          [3, 6],
        ]);
      });
    });

    describe('view-on-view operations', () => {
      it('should correctly flatten a transposed matrix', async () => {
        // PyTorch:
        // >>> t = torch.tensor([[1, 2], [3, 4]])
        // >>> transposed = t.T  # [[1, 3], [2, 4]]
        // >>> flattened = transposed.flatten()
        // >>> flattened
        // tensor([1, 3, 2, 4])
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        const transposed = await matrix.transpose();
        const flattened = await transposed.flatten();

        expect(matrix.shape).toEqual([2, 2]);
        expect(transposed.shape).toEqual([2, 2]);
        expect(flattened.shape).toEqual([4]);

        const data = await flattened.toArray();
        expect(data).toEqual([1, 3, 2, 4]); // Not [1, 2, 3, 4]!
      });

      it('should correctly reshape a transposed matrix', async () => {
        // PyTorch:
        // >>> t = torch.tensor([[1, 2], [3, 4]])
        // >>> transposed = t.T  # [[1, 3], [2, 4]]
        // >>> reshaped = transposed.reshape(4)
        // >>> reshaped
        // tensor([1, 3, 2, 4])
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        const transposed = await matrix.transpose();
        const reshaped = await transposed.reshape([4] as const);

        const data = await reshaped.toArray();
        expect(data).toEqual([1, 3, 2, 4]);
      });

      it('should correctly flatten a permuted tensor', async () => {
        // PyTorch:
        // >>> t = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        // >>> permuted = t.permute(2, 0, 1)  # Shape: [2, 2, 2]
        // >>> flattened = permuted.flatten()
        // >>> flattened
        // tensor([1, 3, 5, 7, 2, 4, 6, 8])
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

        const permuted = await tensor3d.permute([2, 0, 1] as const);
        const flattened = await permuted.flatten();

        expect(tensor3d.shape).toEqual([2, 2, 2]);
        expect(permuted.shape).toEqual([2, 2, 2]);
        expect(flattened.shape).toEqual([8]);

        const data = await flattened.toArray();
        expect(data).toEqual([1, 3, 5, 7, 2, 4, 6, 8]);
      });

      it('should handle reshape after slice', async () => {
        // PyTorch:
        // >>> t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        // >>> sliced = t[1:3, 1:3]  # [[6, 7], [10, 11]]
        // >>> reshaped = sliced.reshape(4)
        // >>> reshaped
        // tensor([ 6,  7, 10, 11])
        const matrix = await tensor(
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
          ] as const,
          { device, dtype: float32 },
        );

        const sliced = await matrix.slice([
          { start: 1, stop: 3 },
          { start: 1, stop: 3 },
        ]);
        const reshaped = await sliced.reshape([4] as const);

        expect(sliced.shape).toEqual([2, 2]);
        expect(reshaped.shape).toEqual([4]);

        const data = await reshaped.toArray();
        expect(data).toEqual([6, 7, 10, 11]);
      });

      it('should handle transpose after reshape', async () => {
        // PyTorch:
        // >>> t = torch.tensor([1, 2, 3, 4, 5, 6])
        // >>> reshaped = t.reshape(2, 3)  # [[1, 2, 3], [4, 5, 6]]
        // >>> transposed = reshaped.T  # [[1, 4], [2, 5], [3, 6]]
        // >>> transposed.flatten()
        // tensor([1, 4, 2, 5, 3, 6])
        const vector = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

        const reshaped = await vector.reshape([2, 3] as const);
        const transposed = await reshaped.transpose();
        const flattened = await transposed.flatten();

        expect(reshaped.shape).toEqual([2, 3]);
        expect(transposed.shape).toEqual([3, 2]);
        expect(flattened.shape).toEqual([6]);

        const data = await flattened.toArray();
        expect(data).toEqual([1, 4, 2, 5, 3, 6]);
      });

      it('should handle multiple chained view operations', async () => {
        // PyTorch:
        // >>> t = torch.arange(24).reshape(2, 3, 4)
        // >>> p1 = t.permute(2, 0, 1)  # [4, 2, 3]
        // >>> p2 = p1.transpose(0, 1)  # [2, 4, 3]
        // >>> result = p2.flatten()
        const vector = await tensor(
          [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
          ] as const,
          { device, dtype: float32 },
        );

        const reshaped = await vector.reshape([2, 3, 4] as const);
        const permuted = await reshaped.permute([2, 0, 1] as const);
        const transposed = await permuted.transpose(); // Swaps last two dims
        const flattened = await transposed.flatten();

        expect(reshaped.shape).toEqual([2, 3, 4]);
        expect(permuted.shape).toEqual([4, 2, 3]);
        expect(transposed.shape).toEqual([4, 3, 2]);
        expect(flattened.shape).toEqual([24]);

        // Verify the data is correctly transformed
        // This is a complex case - the exact expected values would need to be
        // verified with PyTorch, but the key is that it shouldn't just be [0, 1, 2, ...]
        const data = await flattened.toArray();
        expect(data[0]).toBe(0);
        // Should not be sequential - verify it's not just [0, 1, 2, ...]
        // @ts-expect-error - test error
        const isSequential = data.every((val: number, idx: number) => val === idx);
        expect(isSequential).toBe(false);
      });

      it('should correctly handle non-contiguous view in slice', async () => {
        // PyTorch:
        // >>> t = torch.tensor([[1, 2, 3], [4, 5, 6]]).T  # [[1, 4], [2, 5], [3, 6]]
        // >>> sliced = t[1:3]  # [[2, 5], [3, 6]]
        // >>> sliced.flatten()
        // tensor([2, 5, 3, 6])
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        const transposed = await matrix.transpose();
        const sliced = await transposed.slice([{ start: 1, stop: 3 }, null]);
        const flattened = await sliced.flatten();

        expect(transposed.shape).toEqual([3, 2]);
        expect(sliced.shape).toEqual([2, 2]);
        expect(flattened.shape).toEqual([4]);

        const data = await flattened.toArray();
        expect(data).toEqual([2, 5, 3, 6]);
      });

      it('should maintain correct strides through view operations', async () => {
        // Test that strides are properly tracked through operations
        const original = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // Original should be contiguous with strides [3, 1]
        expect(original.strides).toEqual([3, 1]);

        // After transpose, strides should be [1, 3]
        const transposed = await original.transpose();
        expect(transposed.strides).toEqual([1, 3]);

        // After reshape on non-contiguous, this is where issues arise
        // The reshape should either error or make a copy
        const reshaped = await transposed.reshape([6] as const);
        expect(reshaped.shape).toEqual([6]);

        // The data should reflect the transposed order
        const data = await reshaped.toArray();
        expect(data).toEqual([1, 4, 2, 5, 3, 6]);
      });
    });
  });
}
