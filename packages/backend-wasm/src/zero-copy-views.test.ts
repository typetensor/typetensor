import { describe, it, expect, beforeEach } from 'vitest';
import { WASMDevice } from './device';
import { float32 } from '@typetensor/core';
import * as tt from '@typetensor/core';

describe('Zero-Copy View Implementation Tests', () => {
  let device: WASMDevice;

  beforeEach(async () => {
    device = await WASMDevice.create();
  });

  describe('zero-copy memory sharing', () => {
    it('should share underlying buffer between original and reshaped tensors', async () => {
      const tensor = await tt.tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });
      const reshaped = await tensor.reshape([2, 3] as const);

      // Check they share the same DeviceData
      expect(reshaped.data).toBe(tensor.data);

      // Modify through original view
      const originalView = device.readDataView(tensor.data, float32);
      originalView[0] = 999;

      // Check change is visible in reshaped view
      const reshapedView = device.readDataView(reshaped.data, float32);
      expect(reshapedView[0]).toBe(999);

      // Verify through toArray as well
      const reshapedArray = await reshaped.toArray();
      expect(reshapedArray[0][0]).toBe(999);
    });

    it('should not allocate new memory for view operations', async () => {
      const initialStats = device.getMemoryStats();

      // Create initial tensor
      const data = Array.from({ length: 1000 }, (_, i) => i);
      const tensor = await tt.tensor(data, {
        shape: [1000] as const,
        device,
        dtype: float32,
      });
      const afterTensorStats = device.getMemoryStats();

      // Perform multiple view operations
      const reshaped = await tensor.reshape([10, 100] as const);
      const flattened = await reshaped.flatten();
      const squeezed = await flattened.unsqueeze(0).squeeze();
      const view = await squeezed.view([20, 50] as const);

      const finalStats = device.getMemoryStats();
      const viewAllocation = finalStats.totalAllocated - afterTensorStats.totalAllocated;

      // No additional allocation should occur for views
      expect(viewAllocation).toBe(0);

      // All views should share the same buffer
      expect(reshaped.data).toBe(tensor.data);
      expect(flattened.data).toBe(tensor.data);
      expect(squeezed.data).toBe(tensor.data);
      expect(view.data).toBe(tensor.data);
    });

    it('should handle view lifetime correctly with buffer disposal', async () => {
      const tensor = await tt.tensor([1, 2, 3, 4], { device, dtype: float32 });
      const reshaped = await tensor.reshape([2, 2] as const);

      // Get raw views
      const tensorView = device.readDataView(tensor.data, float32);
      const reshapedView = device.readDataView(reshaped.data, float32);

      // Verify initial access works
      expect(tensorView[0]).toBe(1);
      expect(reshapedView[0]).toBe(1);

      // Dispose the original tensor's data
      device.disposeData(tensor.data);

      // Both views should now be invalid
      expect(() => tensorView[0]).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => reshapedView[0]).toThrow('View is no longer valid: buffer has been disposed');
    });
  });

  describe('non-contiguous view handling', () => {
    it('should handle transpose creating non-contiguous views', async () => {
      const matrix = await tt.tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { device, dtype: float32 },
      );
      const transposed = await matrix.transpose();

      // Original strides: [3, 1] (row-major)
      expect(matrix.strides).toEqual([3, 1]);

      // Transposed strides: [1, 3] (column-major)
      expect(transposed.strides).toEqual([1, 3]);

      // They should still share the same buffer
      expect(transposed.data).toBe(matrix.data);

      // Data should be logically transposed
      const transposedArray = await transposed.toArray();
      expect(transposedArray).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('should handle permute creating complex stride patterns', async () => {
      const tensor3d = await tt.tensor(
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
        { device, dtype: float32 },
      );
      const permuted = await tensor3d.permute([2, 0, 1] as const);

      // Should share buffer
      expect(permuted.data).toBe(tensor3d.data);

      // Verify correct permutation
      const permutedArray = await permuted.toArray();
      expect(permutedArray[0][0][0]).toBe(1); // Original [0,0,0]
      expect(permutedArray[1][0][0]).toBe(2); // Original [0,0,1]
      expect(permutedArray[0][1][0]).toBe(5); // Original [1,0,0]
    });
  });

  describe('expand operation with broadcasting', () => {
    it('should expand singleton dimensions without copying', async () => {
      const tensor = await tt.tensor([[1], [2], [3]], { device, dtype: float32 });
      const expanded = await tensor.expand([3, 4] as const);

      // Should share buffer for broadcast
      expect(expanded.data).toBe(tensor.data);

      // Verify expansion
      const expandedArray = await expanded.toArray();
      expect(expandedArray).toEqual([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
      ]);
    });

    it('should use stride tricks for broadcasting', async () => {
      const scalar = await tt.tensor(42, { device, dtype: float32 });
      const expanded = await scalar.expand([3, 4] as const);

      // Scalar expansion should use zero strides
      expect(expanded.strides).toEqual([0, 0]);

      // Should still share buffer
      expect(expanded.data).toBe(scalar.data);
    });
  });

  describe('error cases', () => {
    it('should throw when reshape has incompatible dimensions', async () => {
      const tensor = await tt.tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

      // 6 elements cannot be reshaped to 2x4 (8 elements)
      // @ts-expect-error - type error expected here since we validate reshapes at compile and runtime
      await expect(tensor.reshape([2, 4] as const)).rejects.toThrow(/different number of elements/);
    });

    it('should throw when expand tries to change non-singleton dimension', async () => {
      const matrix = await tt.tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { device, dtype: float32 },
      );

      // Cannot expand dimension of size 2 to size 3
      await expect(matrix.expand([3, 2] as const)).rejects.toThrow(/Cannot expand dimension/);
    });

    it('should throw when squeeze targets non-unit dimension', async () => {
      const matrix = await tt.tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { device, dtype: float32 },
      );

      // Cannot squeeze dimension with size 2
      await expect(matrix.squeeze([0] as const)).rejects.toThrow(/size.*must be 1/);
    });
  });

  describe('performance characteristics', () => {
    it('should have O(1) time complexity for view creation', async () => {
      // Create tensors of different sizes
      const sizes = [100, 1000, 10000];
      const times: number[] = [];

      for (const size of sizes) {
        const data = Array.from({ length: size }, (_, i) => i);
        const tensor = await tt.tensor(data, {
          shape: [size] as const,
          device,
          dtype: float32,
        });

        const start = performance.now();
        // Perform multiple view operations
        for (let i = 0; i < 100; i++) {
          await tensor.view([size] as const);
        }
        const end = performance.now();

        times.push(end - start);
      }

      // Time should not scale with tensor size (within reasonable variance)
      // If views are copying, larger tensors would take proportionally longer
      const ratio = times[2]! / times[0]!;
      expect(ratio).toBeLessThan(2); // Should be roughly constant, not 100x
    });
  });
});
