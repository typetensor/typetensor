/**
 * Test generators for utility operations
 *
 * These generators test utility operations including:
 * - clone: Deep copying tensors
 * - contiguous: Creating contiguous memory layouts
 * - view: Advanced view operations with -1 inference
 * - Memory layout preservation and optimization
 */

import type { Device } from '@typetensor/core';
import { tensor, zeros, float32, int32, bool } from '@typetensor/core';

/**
 * Generates tests for utility operations
 *
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateUtilityOperationTests(
  device: Device,
  testFramework: {
    describe: (name: string, fn: () => void) => void;
    it: (name: string, fn: () => void | Promise<void>) => void;
    expect: (actual: unknown) => {
      toBe: (expected: unknown) => void;
      toEqual: (expected: unknown) => void;
      toBeCloseTo: (expected: number, precision?: number) => void;
      toThrow: (error?: string | RegExp) => void;
      toBeTruthy: () => void;
      toBeFalsy: () => void;
      not: {
        toBe: (expected: unknown) => void;
      };
      rejects: {
        toThrow: (error?: string | RegExp) => Promise<void>;
      };
    };
  },
) {
  const { describe, it, expect } = testFramework;

  describe(`Utility Operations Tests (${device.type}:${device.id})`, () => {
    describe('clone operations', () => {
      it('should create independent copy of scalar tensors', async () => {
        // PyTorch: tensor.clone() creates a new tensor with copied data
        const original = await tensor(42.5, { device, dtype: float32 });
        const cloned = await original.clone();

        // Should be different objects
        expect(cloned).not.toBe(original);

        // But with same properties
        expect(cloned.shape).toEqual(original.shape);
        expect(cloned.dtype).toBe(original.dtype);
        expect(cloned.device).toBe(original.device);
        expect(cloned.ndim).toBe(original.ndim);
        expect(cloned.size).toBe(original.size);

        // And same data
        expect(await cloned.item()).toBeCloseTo(42.5, 5);
      });

      it('should create independent copy of vector tensors', async () => {
        const original = await tensor([1, 2, 3, 4, 5] as const, { device, dtype: float32 });
        const cloned = await original.clone();

        expect(cloned).not.toBe(original);
        expect(cloned.shape).toEqual([5]);

        const originalData = await original.toArray();
        const clonedData = await cloned.toArray();
        expect(clonedData).toEqual(originalData);

        // Verify independence by modifying clone (if tensor supports in-place ops)
        // This would require additional in-place operations to fully test
      });

      it('should clone matrices correctly', async () => {
        const original = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
          ] as const,
          { device, dtype: float32 },
        );
        const cloned = await original.clone();

        expect(cloned).not.toBe(original);
        expect(cloned.shape).toEqual([3, 3]);

        const originalData = await original.toArray();
        const clonedData = await cloned.toArray();
        expect(clonedData).toEqual(originalData);
      });

      it('should clone empty tensors', async () => {
        // Empty 1D
        const empty1d = await tensor([] as const, { device, dtype: float32 });
        const cloned1d = await empty1d.clone();
        expect(cloned1d).not.toBe(empty1d);
        expect(cloned1d.shape).toEqual([0]);
        expect(await cloned1d.toArray()).toEqual([]);

        // Empty 2D
        const empty2d = await zeros([0, 5] as const, { device, dtype: float32 });
        const cloned2d = await empty2d.clone();
        expect(cloned2d).not.toBe(empty2d);
        expect(cloned2d.shape).toEqual([0, 5]);
        expect(cloned2d.size).toBe(0);
      });

      it('should preserve dtypes when cloning', async () => {
        // float32
        const floatTensor = await tensor([1.5, 2.5] as const, { device, dtype: float32 });
        const floatCloned = await floatTensor.clone();
        expect(floatCloned.dtype).toBe(float32);

        // int32
        const intTensor = await tensor([1, 2, 3] as const, { device, dtype: int32 });
        const intCloned = await intTensor.clone();
        expect(intCloned.dtype).toBe(int32);

        // bool
        const boolTensor = await tensor([true, false] as const, { device, dtype: bool });
        const boolCloned = await boolTensor.clone();
        expect(boolCloned.dtype).toBe(bool);
      });

      it('should clone non-contiguous tensors', async () => {
        // Create non-contiguous via transpose
        const original = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const transposed = await original.transpose();
        const cloned = await transposed.clone();

        expect(cloned).not.toBe(transposed);
        expect(cloned.shape).toEqual([3, 2]);

        // Clone in PyTorch preserves non-contiguous layout
        expect(cloned.layout.c_contiguous).toBeFalsy();

        const data = await cloned.toArray();
        expect(data).toEqual([
          [1, 4],
          [2, 5],
          [3, 6],
        ]);
      });

      it('should clone sliced tensors', async () => {
        const original = await tensor([1, 2, 3, 4, 5] as const, { device, dtype: float32 });
        const sliced = await original.slice([{ start: 1, stop: 4 }]);
        const cloned = await sliced.clone();

        expect(cloned).not.toBe(sliced);
        expect(cloned.shape).toEqual([3]);
        expect(await cloned.toArray()).toEqual([2, 3, 4]);

        // PyTorch: 1D slices are contiguous
        expect(cloned.layout.c_contiguous).toBeTruthy();
      });

      it('should handle special values when cloning', async () => {
        const special = await tensor([Infinity, -Infinity, NaN, 0, -0] as const, {
          device,
          dtype: float32,
        });
        const cloned = await special.clone();

        const data = await cloned.toArray();
        expect(data[0]).toBe(Infinity);
        expect(data[1]).toBe(-Infinity);
        expect(Number.isNaN(data[2])).toBeTruthy();
        expect(data[3]).toBe(0);
        expect(Object.is(data[4], -0)).toBeTruthy();
      });
    });

    describe('contiguous operations', () => {
      it('should return same tensor if already contiguous', async () => {
        const original = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });

        // Freshly created tensors should be contiguous
        expect(original.layout.c_contiguous).toBeTruthy();

        const result = await original.contiguous();
        expect(result).toBe(original); // Should be same object
      });

      it('should create contiguous copy of transposed tensors', async () => {
        const original = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        const transposed = await original.transpose();
        // Transposed tensor may not be contiguous
        const contiguous = await transposed.contiguous();

        expect(contiguous).not.toBe(transposed);
        expect(contiguous.shape).toEqual([3, 2]);
        expect(contiguous.layout.c_contiguous).toBeTruthy();

        const data = await contiguous.toArray();
        expect(data).toEqual([
          [1, 4],
          [2, 5],
          [3, 6],
        ]);
      });

      it('should handle permuted tensors', async () => {
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

        const permuted = await original.permute([2, 0, 1] as const);
        const contiguous = await permuted.contiguous();

        expect(contiguous.shape).toEqual([2, 2, 2]);
        expect(contiguous.layout.c_contiguous).toBeTruthy();

        // Verify data is correctly laid out
        const data = await contiguous.toArray();
        expect(data).toEqual([
          [
            [1, 3],
            [5, 7],
          ],
          [
            [2, 4],
            [6, 8],
          ],
        ]);
      });

      it('should handle sliced tensors', async () => {
        const original = await tensor(
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
          ] as const,
          { device, dtype: float32 },
        );

        // Slice that creates non-contiguous view
        const sliced = await original.slice([
          { start: 0, stop: 2 },
          { start: 1, stop: 3 },
        ]);
        const contiguous = await sliced.contiguous();

        expect(contiguous.shape).toEqual([2, 2]);
        expect(contiguous.layout.c_contiguous).toBeTruthy();

        const data = await contiguous.toArray();
        expect(data).toEqual([
          [2, 3],
          [6, 7],
        ]);
      });

      it('should preserve empty tensors', async () => {
        const empty = await tensor([] as const, { device, dtype: float32 });
        const contiguous = await empty.contiguous();

        expect(contiguous).toBe(empty); // Already contiguous
        expect(contiguous.shape).toEqual([0]);
      });

      it('should handle scalar tensors', async () => {
        const scalar = await tensor(3.14, { device, dtype: float32 });
        const contiguous = await scalar.contiguous();

        expect(contiguous).toBe(scalar); // Scalars are always contiguous
        expect(await contiguous.item()).toBeCloseTo(3.14, 5);
      });

      it('should work with chained operations', async () => {
        const original = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // Chain of operations that might create non-contiguous tensors
        const result = await original
          .transpose()
          .contiguous()
          .reshape([6] as const);

        expect(result.shape).toEqual([6]);
        expect(await result.toArray()).toEqual([1, 4, 2, 5, 3, 6]);
      });
    });

    describe('view operations', () => {
      it('should create basic views with explicit dimensions', async () => {
        const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });
        const viewed = await original.view([2, 3] as const);

        expect(viewed.shape).toEqual([2, 3]);
        expect(viewed.size).toBe(6);

        const data = await viewed.toArray();
        expect(data).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);
      });

      it('should infer dimension with -1', async () => {
        // PyTorch: tensor.view(-1, 2) infers first dimension
        const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

        // Single -1 dimension
        const viewed1 = await original.view([-1, 2] as const);
        expect(viewed1.shape).toEqual([3, 2]);

        const viewed2 = await original.view([2, -1] as const);
        expect(viewed2.shape).toEqual([2, 3]);

        const viewed3 = await original.view([-1] as const);
        expect(viewed3.shape).toEqual([6]);
      });

      it('should handle scalar views', async () => {
        const scalar = await tensor(42, { device, dtype: float32 });

        // Scalar to 1D
        const as1d = await scalar.view([1] as const);
        expect(as1d.shape).toEqual([1]);
        expect(await as1d.toArray()).toEqual([42]);

        // 1D back to scalar
        const backToScalar = await as1d.view([] as const);
        expect(backToScalar.shape).toEqual([]);
        expect(await backToScalar.item()).toBe(42);
      });

      it('should require contiguous tensors for view', async () => {
        const original = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // Non-contiguous tensor
        const transposed = await original.transpose();

        // View should work (may make contiguous internally)
        const viewed = await transposed.view([6] as const);
        expect(viewed.shape).toEqual([6]);

        // Data should be in transposed order
        const data = await viewed.toArray();
        expect(data).toEqual([1, 4, 2, 5, 3, 6]);
      });

      it('should handle empty tensor views', async () => {
        const empty = await tensor([] as const, { device, dtype: float32 });

        // Empty tensor can be viewed to any shape with 0 elements
        // @ts-expect-error - test error
        const viewed1 = await empty.view([0] as const);
        expect(viewed1.shape).toEqual([0]);

        // @ts-expect-error - test error
        const viewed2 = await empty.view([0, 5] as const);
        expect(viewed2.shape).toEqual([0, 5]);

        // @ts-expect-error - test error
        const viewed3 = await empty.view([2, 0, 3] as const);
        expect(viewed3.shape).toEqual([2, 0, 3]);
      });

      it('should error on invalid view dimensions', async () => {
        const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

        // Total elements don't match
        // @ts-expect-error - test error
        await expect(original.view([2, 4] as const)).rejects.toThrow();

        // Multiple -1 dimensions
        // @ts-expect-error - test error
        await expect(original.view([-1, -1] as const)).rejects.toThrow();

        // Invalid inference (not divisible)
        // @ts-expect-error - test error
        await expect(original.view([-1, 4] as const)).rejects.toThrow();
      });

      it('should handle high-dimensional views', async () => {
        const original = await tensor([1, 2, 3, 4, 5, 6, 7, 8] as const, {
          device,
          dtype: float32,
        });

        // 1D to 4D
        const viewed4d = await original.view([2, 2, 1, 2] as const);
        expect(viewed4d.shape).toEqual([2, 2, 1, 2]);
        expect(viewed4d.ndim).toBe(4);

        // 4D to 2D with inference
        const viewed2d = await viewed4d.view([-1, 4] as const);
        expect(viewed2d.shape).toEqual([2, 4]);
      });

      it('should preserve data type in views', async () => {
        // int32
        const intTensor = await tensor([1, 2, 3, 4] as const, { device, dtype: int32 });
        const intView = await intTensor.view([2, 2] as const);
        expect(intView.dtype).toBe(int32);

        // bool
        const boolTensor = await tensor([true, false, true, false] as const, {
          device,
          dtype: bool,
        });
        const boolView = await boolTensor.view([2, 2] as const);
        expect(boolView.dtype).toBe(bool);
      });

      it('should chain view operations', async () => {
        const original = await zeros([24] as const, { device, dtype: float32 });

        const result = await original
          .view([2, 12] as const)
          .view([2, 3, 4] as const)
          .view([6, -1] as const);

        expect(result.shape).toEqual([6, 4]);
      });

      it('should handle view after other operations', async () => {
        const original = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        // Operations that maintain contiguity
        const added = await original.add(original);
        const viewed = await added.view([4] as const);
        expect(await viewed.toArray()).toEqual([2, 4, 6, 8]);

        // PyTorch allows view on certain non-contiguous tensors
        // Slicing columns creates non-contiguous but viewable tensor
        const sliced = await original.slice([
          { start: 0, stop: 2 },
          { start: 0, stop: 1 },
        ]);
        const sliceView = await sliced.view([2] as const);
        expect(await sliceView.toArray()).toEqual([1, 3]);
      });
    });

    describe('chainable operations', () => {
      it('should support chaining utility operations', async () => {
        const original = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // Chain: transpose -> contiguous -> clone -> view
        const result = await original
          .transpose()
          .contiguous()
          .clone()
          .view([6] as const);

        expect(result.shape).toEqual([6]);
        expect(await result.toArray()).toEqual([1, 4, 2, 5, 3, 6]);
      });

      it('should maintain properties through utility operations', async () => {
        const original = await tensor([1, 2, 3, 4] as const, { device, dtype: int32 });

        const cloned = await original.clone();
        expect(cloned.dtype).toBe(int32);
        expect(cloned.device).toBe(device);

        const viewed = await cloned.view([2, 2] as const);
        expect(viewed.dtype).toBe(int32);
        expect(viewed.device).toBe(device);

        const contiguous = await viewed.contiguous();
        expect(contiguous.dtype).toBe(int32);
        expect(contiguous.device).toBe(device);
      });
    });

    describe('memory layout optimization', () => {
      it('should optimize repeated contiguous calls', async () => {
        const original = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });

        const first = await original.contiguous();
        const second = await first.contiguous();
        const third = await second.contiguous();

        // All should be the same object (optimization)
        expect(first).toBe(original);
        expect(second).toBe(first);
        expect(third).toBe(second);
      });

      it('should handle view of contiguous correctly', async () => {
        const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

        // Make explicitly contiguous
        const contiguous = await original.contiguous();
        expect(contiguous).toBe(original); // Already was contiguous

        // View should work directly
        const viewed = await contiguous.view([2, 3] as const);
        expect(viewed.shape).toEqual([2, 3]);
      });
    });
  });
}
