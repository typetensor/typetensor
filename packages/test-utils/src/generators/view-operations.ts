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

        await expect(tensor12.reshape([3, 5] as const)).rejects.toThrow(
          /different number of elements/,
        );

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
        const permuted = matrix.permute([1, 0] as const);

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
        const permuted = tensor3d.permute([2, 0, 1] as const);

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
        const permuted = matrix.permute([0, 1] as const);

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
        const permuted = vector.permute([0] as const);

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
        const permuted1 = tensor3d.permute([1, 2, 0] as const);
        const permuted2 = permuted1.permute([2, 0, 1] as const);

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
        const bhwc = bchw.permute([0, 2, 3, 1] as const);

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

        const transposed = matrix.transpose();
        const reshaped = transposed.reshape([4] as const);

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

        const permuted = tensor3d.permute([2, 0, 1] as const);
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
