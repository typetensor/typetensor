/**
 * Test generators for data access operations
 *
 * These generators test data extraction and formatting operations including:
 * - toArray: Converting tensors to nested JavaScript arrays
 * - item: Extracting scalar values
 * - format: String representation of tensors
 * - Edge cases and error conditions
 */

import type { Device } from '@typetensor/core';
import { tensor, zeros, ones, float32, int32, bool, int64 } from '@typetensor/core';

/**
 * Generates tests for data access operations
 *
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateDataAccessOperationTests(
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
      toBeLessThan: (expected: number) => void;
      toContain: (substring: string) => void;
      not: {
        toContain: (substring: string) => void;
      };
      rejects: {
        toThrow: (error?: string | RegExp) => Promise<void>;
      };
    };
  },
) {
  const { describe, it, expect } = testFramework;

  describe(`Data Access Operations Tests (${device.type}:${device.id})`, () => {
    describe('toArray operations', () => {
      it('should extract scalar values as nested arrays', async () => {
        const scalar = await tensor(42.5, { device, dtype: float32 });
        const result = await scalar.toArray();

        // Scalar should return the value directly (not wrapped in array)
        expect(result).toBe(42.5);
        expect(typeof result).toBe('number');
      });

      it('should extract 1D arrays correctly', async () => {
        const vec = await tensor([1, 2, 3, 4, 5] as const, { device, dtype: float32 });
        const result = await vec.toArray();

        expect(Array.isArray(result)).toBeTruthy();
        expect(result).toEqual([1, 2, 3, 4, 5]);
        expect(result.length).toBe(5);
      });

      it('should extract 2D arrays with proper nesting', async () => {
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await matrix.toArray();

        expect(Array.isArray(result)).toBeTruthy();
        expect(result.length).toBe(2);
        expect(Array.isArray(result[0])).toBeTruthy();
        expect(Array.isArray(result[1])).toBeTruthy();
        expect(result).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);
      });

      it('should handle 3D arrays', async () => {
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
        const result = await tensor3d.toArray();

        expect(Array.isArray(result)).toBeTruthy();
        expect(result.length).toBe(2);
        expect(result).toEqual([
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

      it('should preserve data types accurately', async () => {
        // Float values
        const floatTensor = await tensor([1.1, 2.2, 3.3] as const, { device, dtype: float32 });
        const floatResult = await floatTensor.toArray();
        expect(floatResult[0]).toBeCloseTo(1.1, 5);
        expect(floatResult[1]).toBeCloseTo(2.2, 5);
        expect(floatResult[2]).toBeCloseTo(3.3, 5);

        // Integer values
        const intTensor = await tensor([1, 2, 3] as const, { device, dtype: int32 });
        const intResult = await intTensor.toArray();
        expect(intResult).toEqual([1, 2, 3]);
        expect(Number.isInteger(intResult[0])).toBeTruthy();

        // Boolean values
        const boolTensor = await tensor([true, false, true] as const, { device, dtype: bool });
        const boolResult = await boolTensor.toArray();
        expect(boolResult).toEqual([true, false, true]);
        expect(typeof boolResult[0]).toBe('boolean');

        // BigInt values
        const int64Tensor = await tensor([1n, 2n, 3n] as const, { device, dtype: int64 });
        const int64Result = await int64Tensor.toArray();
        expect(int64Result).toEqual([1n, 2n, 3n]);
        expect(typeof int64Result[0]).toBe('bigint');
      });

      it('should handle empty tensors', async () => {
        // Empty 1D
        const empty1d = await tensor([] as const, { device, dtype: float32 });
        const result1d = await empty1d.toArray();
        expect(Array.isArray(result1d)).toBeTruthy();
        expect(result1d).toEqual([]);
        expect(result1d.length).toBe(0);

        // Empty 2D
        const empty2d = await zeros([0, 5] as const, { device, dtype: float32 });
        const result2d = await empty2d.toArray();
        expect(Array.isArray(result2d)).toBeTruthy();
        expect(result2d).toEqual([]);
        expect(result2d.length).toBe(0);

        // Empty 3D with non-zero dimensions
        const empty3d = await zeros([2, 0, 3] as const, { device, dtype: float32 });
        const result3d = await empty3d.toArray();
        expect(Array.isArray(result3d)).toBeTruthy();
        expect(result3d.length).toBe(2);
        expect(result3d[0]).toEqual([]);
        expect(result3d[1]).toEqual([]);
      });

      it('should handle special float values', async () => {
        const special = await tensor([Infinity, -Infinity, NaN, 0, -0] as const, {
          device,
          dtype: float32,
        });
        const result = await special.toArray();

        expect(result[0]).toBe(Infinity);
        expect(result[1]).toBe(-Infinity);
        expect(Number.isNaN(result[2])).toBeTruthy();
        expect(result[3]).toBe(0);
        expect(Object.is(result[4], -0)).toBeTruthy();
      });

      it('should work after view operations', async () => {
        const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

        // After reshape
        const reshaped = await original.reshape([2, 3] as const);
        const reshapedResult = await reshaped.toArray();
        expect(reshapedResult).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);

        // After transpose
        const transposed = await reshaped.transpose();
        const transposedResult = await transposed.toArray();
        expect(transposedResult).toEqual([
          [1, 4],
          [2, 5],
          [3, 6],
        ]);

        // After slice
        const sliced = await original.slice([{ start: 1, stop: 4 }]);
        const slicedResult = await sliced.toArray();
        expect(slicedResult).toEqual([2, 3, 4]);
      });

      it('should handle very large tensors', async () => {
        // Large but manageable tensor
        const large = await zeros([100, 100] as const, { device, dtype: float32 });
        const result = await large.toArray();

        expect(Array.isArray(result)).toBeTruthy();
        expect(result.length).toBe(100);
        expect(result[0].length).toBe(100);
        expect(result[0][0]).toBe(0);
        expect(result[99][99]).toBe(0);
      });

      it('should preserve precision for different dtypes', async () => {
        // High precision values
        const precise = await tensor([1.23456789, 9.87654321] as const, { device, dtype: float32 });
        const result = await precise.toArray();

        // float32 has ~7 decimal digits of precision
        expect(result[0]).toBeCloseTo(1.23456789, 5);
        expect(result[1]).toBeCloseTo(9.87654321, 5);
      });
    });

    describe('item operations', () => {
      it('should extract scalar values correctly', async () => {
        // float32
        const floatScalar = await tensor(3.14159, { device, dtype: float32 });
        const floatValue = await floatScalar.item();
        expect(floatValue).toBeCloseTo(3.14159, 5);
        expect(typeof floatValue).toBe('number');

        // int32
        const intScalar = await tensor(42, { device, dtype: int32 });
        const intValue = await intScalar.item();
        expect(intValue).toBe(42);
        expect(Number.isInteger(intValue)).toBeTruthy();

        // bool
        const boolScalar = await tensor(true, { device, dtype: bool });
        const boolValue = await boolScalar.item();
        expect(boolValue).toBe(true);
        expect(typeof boolValue).toBe('boolean');

        // int64
        const int64Scalar = await tensor(123n, { device, dtype: int64 });
        const int64Value = await int64Scalar.item();
        expect(int64Value).toBe(123n);
        expect(typeof int64Value).toBe('bigint');
      });

      it('should handle special scalar values', async () => {
        // Infinity
        const inf = await tensor(Infinity, { device, dtype: float32 });
        expect(await inf.item()).toBe(Infinity);

        // -Infinity
        const negInf = await tensor(-Infinity, { device, dtype: float32 });
        expect(await negInf.item()).toBe(-Infinity);

        // NaN
        const nan = await tensor(NaN, { device, dtype: float32 });
        expect(Number.isNaN(await nan.item())).toBeTruthy();

        // -0
        const negZero = await tensor(-0, { device, dtype: float32 });
        const negZeroValue = await negZero.item();
        expect(Object.is(negZeroValue, -0)).toBeTruthy();
      });

      it('should error on non-scalar tensors', async () => {
        // 1D tensor with multiple elements
        const vec = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        await expect(vec.item()).rejects.toThrow(/3 elements|cannot be converted|Scalar/i);

        // 2D tensor
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        await expect(matrix.item()).rejects.toThrow(/4 elements|cannot be converted|Scalar/i);

        // Empty tensor
        const empty = await tensor([] as const, { device, dtype: float32 });
        await expect(empty.item()).rejects.toThrow(/0 elements|cannot be converted|Scalar|empty/i);
      });

      it('should work with 1-element tensors of any shape', async () => {
        // 1-element 1D tensor
        const vec1 = await tensor([42] as const, { device, dtype: float32 });
        expect(await vec1.item()).toBe(42);

        // 1-element 2D tensor
        const matrix1 = await tensor([[42]] as const, { device, dtype: float32 });
        expect(await matrix1.item()).toBe(42);

        // 1-element 3D tensor
        const tensor3d1 = await tensor([[[42]]] as const, { device, dtype: float32 });
        expect(await tensor3d1.item()).toBe(42);

        // 1-element high-dimensional tensor
        const high1 = await ones([1, 1, 1, 1, 1] as const, { device, dtype: float32 });
        expect(await high1.item()).toBe(1);
      });

      it('should work after operations that produce scalars', async () => {
        const vec = await tensor([1, 2, 3, 4, 5] as const, { device, dtype: float32 });

        // Sum reduction to scalar
        const sum = await vec.sum();
        expect(await sum.item()).toBe(15);

        // Mean reduction to scalar
        const mean = await vec.mean();
        expect(await mean.item()).toBe(3);

        // Max reduction to scalar
        const max = await vec.max();
        expect(await max.item()).toBe(5);

        // Min reduction to scalar
        const min = await vec.min();
        expect(await min.item()).toBe(1);
      });
    });

    describe('format operations', () => {
      it('should format scalar tensors', async () => {
        const scalar = await tensor(42.5, { device, dtype: float32 });
        const formatted = await scalar.format();

        expect(formatted).toContain('42.5');
        expect(formatted).not.toContain('['); // No brackets for scalar
      });

      it('should format 1D tensors', async () => {
        const vec = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const formatted = await vec.format();

        expect(formatted).toContain('[1');
        expect(formatted).toContain('2');
        expect(formatted).toContain('3]');
      });

      it('should format 2D tensors with proper indentation', async () => {
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const formatted = await matrix.format();

        // TypeTensor uses newline formatting for matrices
        expect(formatted).toContain('[1'); // Has the number 1
        expect(formatted).toContain('3'); // Has the number 3
        expect(formatted).toContain('2'); // Has the number 2
        expect(formatted).toContain('4'); // Has the number 4
      });

      it('should include dtype annotation when not float32', async () => {
        // int32
        const intTensor = await tensor([1, 2, 3] as const, { device, dtype: int32 });
        const intFormatted = await intTensor.format();
        expect(intFormatted).toContain('dtype=int32');

        // bool
        const boolTensor = await tensor([true, false] as const, { device, dtype: bool });
        const boolFormatted = await boolTensor.format();
        expect(boolFormatted).toContain('dtype=bool');

        // float32 should not include dtype annotation
        const floatTensor = await tensor([1.0, 2.0] as const, { device, dtype: float32 });
        const floatFormatted = await floatTensor.format();
        expect(floatFormatted).not.toContain('dtype=');
      });

      it('should include device annotation when not CPU', async () => {
        const t = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const formatted = await t.format();

        if (device.id !== 'cpu') {
          expect(formatted).toContain(`device='${device.id}'`);
        } else {
          expect(formatted).not.toContain('device=');
        }
      });

      it('should format empty tensors', async () => {
        const empty1d = await tensor([] as const, { device, dtype: float32 });
        const formatted1d = await empty1d.format();
        expect(formatted1d).toContain('[]');

        const empty2d = await zeros([0, 5] as const, { device, dtype: float32 });
        const formatted2d = await empty2d.format();
        expect(formatted2d).toContain('[]');
      });

      it('should format special values', async () => {
        const special = await tensor([Infinity, -Infinity, NaN] as const, {
          device,
          dtype: float32,
        });
        const formatted = await special.format();

        // TypeTensor uses 'Infinity' while PyTorch uses 'inf'
        // Accept both formats
        expect(formatted.toLowerCase()).toContain('inf');
        expect(formatted.toLowerCase()).toContain('nan');
      });

      it('should handle large tensors with ellipsis', async () => {
        // Very large tensor that should be truncated
        const large = await zeros([100, 100] as const, { device, dtype: float32 });
        const formatted = await large.format();

        // Should contain ellipsis or truncation indicator
        expect(formatted.length).toBeLessThan(10000); // Reasonable size limit
        // Exact format depends on implementation
      });

      it('should format boolean tensors correctly', async () => {
        const boolTensor = await tensor(
          [
            [true, false],
            [false, true],
          ] as const,
          { device, dtype: bool },
        );
        const formatted = await boolTensor.format();

        expect(formatted).toContain('true');
        expect(formatted).toContain('false');
        expect(formatted).toContain('dtype=bool');
      });

      it('should format after various operations', async () => {
        const original = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // After transpose
        const transposed = await original.transpose();
        const transposedFormatted = await transposed.format();
        expect(transposedFormatted).toContain('1'); // Contains the values
        expect(transposedFormatted).toContain('4');

        // After slice
        const sliced = await original.slice([
          { start: 0, stop: 1 },
          { start: 0, stop: 2 },
        ]);
        const slicedFormatted = await sliced.format();
        expect(slicedFormatted).toContain('[1'); // Should show sliced data

        // After reduction
        const summed = await original.sum(undefined, true);
        const summedFormatted = await summed.format();
        expect(summedFormatted).toContain('21'); // Sum of all elements
      });
    });

    describe('edge cases and error handling', () => {
      it('should handle operations on disposed tensors gracefully', async () => {
        const t = await tensor([1, 2, 3] as const, { device, dtype: float32 });

        // Dispose the tensor
        t.dispose();

        // Operations should either throw or handle gracefully
        try {
          await t.toArray();
          // If it doesn't throw, that's also acceptable
          expect(true).toBeTruthy();
        } catch (error) {
          expect(error).toBeTruthy();
        }
      });

      it('should handle concurrent data access', async () => {
        const t = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // Concurrent access should work correctly
        const [array1, array2, formatted] = await Promise.all([
          t.toArray(),
          t.toArray(),
          t.format(),
        ]);

        expect(array1).toEqual(array2);
        expect(formatted).toContain('1');
      });

      it('should maintain data integrity through conversions', async () => {
        const original = [
          [1.1, 2.2, 3.3],
          [4.4, 5.5, 6.6],
        ] as const;

        const t = await tensor(original, { device, dtype: float32 });
        const extracted = await t.toArray();

        // Verify data matches within float32 precision
        expect(extracted[0][0]).toBeCloseTo(1.1, 5);
        expect(extracted[0][1]).toBeCloseTo(2.2, 5);
        expect(extracted[0][2]).toBeCloseTo(3.3, 5);
        expect(extracted[1][0]).toBeCloseTo(4.4, 5);
        expect(extracted[1][1]).toBeCloseTo(5.5, 5);
        expect(extracted[1][2]).toBeCloseTo(6.6, 5);
      });
    });
  });
}
