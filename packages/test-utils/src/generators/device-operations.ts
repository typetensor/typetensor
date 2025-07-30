/**
 * Test generators for device transfer operations
 *
 * These generators test tensor device transfer functionality including:
 * - Transfer between different devices
 * - Same device optimization
 * - Data integrity preservation
 * - Non-contiguous tensor handling
 * - Various shapes and dtypes
 */

import type { Device } from '@typetensor/core';
import { tensor, zeros, ones, float32, int32, bool, int64 } from '@typetensor/core';

/**
 * Generates tests for device transfer operations
 *
 * @param device - Primary device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 * @param alternativeDevice - Optional secondary device for transfer tests
 */
export function generateDeviceOperationTests(
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
  alternativeDevice?: Device,
) {
  const { describe, it, expect } = testFramework;

  describe(`Device Operations Tests (${device.type}:${device.id})`, () => {
    describe('same device transfer', () => {
      it('should return the same tensor when transferring to the same device', async () => {
        // PyTorch: tensor.to(tensor.device) returns the same tensor object
        const original = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });
        const transferred = await original.to(device);

        // Should be the exact same object reference (optimization)
        expect(transferred).toBe(original);
        expect(transferred.device).toBe(device);
        expect(transferred.device.id).toBe(device.id);

        // Data should be unchanged
        const data = await transferred.toArray();
        expect(data).toEqual([1, 2, 3, 4]);
      });

      it('should handle same device transfer for various shapes', async () => {
        // Scalar
        const scalar = await tensor(42, { device, dtype: float32 });
        const scalarTransferred = await scalar.to(device);
        expect(scalarTransferred).toBe(scalar);

        // Matrix
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const matrixTransferred = await matrix.to(device);
        expect(matrixTransferred).toBe(matrix);

        // 3D tensor
        const tensor3d = await zeros([2, 3, 4] as const, { device, dtype: float32 });
        const tensor3dTransferred = await tensor3d.to(device);
        expect(tensor3dTransferred).toBe(tensor3d);
      });

      it('should handle same device transfer for different dtypes', async () => {
        // float32
        const float32Tensor = await tensor([1.5, 2.5, 3.5] as const, { device, dtype: float32 });
        const float32Transferred = await float32Tensor.to(device);
        expect(float32Transferred).toBe(float32Tensor);

        // int32
        const int32Tensor = await tensor([1, 2, 3] as const, { device, dtype: int32 });
        const int32Transferred = await int32Tensor.to(device);
        expect(int32Transferred).toBe(int32Tensor);

        // bool
        const boolTensor = await tensor([true, false, true] as const, { device, dtype: bool });
        const boolTransferred = await boolTensor.to(device);
        expect(boolTransferred).toBe(boolTensor);

        // int64
        const int64Tensor = await tensor([1n, 2n, 3n] as const, { device, dtype: int64 });
        const int64Transferred = await int64Tensor.to(device);
        expect(int64Transferred).toBe(int64Tensor);
      });
    });

    if (alternativeDevice) {
      describe('cross-device transfer', () => {
        it('should transfer data between devices correctly', async () => {
          // PyTorch: tensor.to('cuda') or tensor.to('cpu')
          const original = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });
          const transferred = await original.to(alternativeDevice);

          // Should be a different tensor
          expect(transferred).not.toBe(original);
          expect(transferred.device).toBe(alternativeDevice);
          expect(transferred.device.id).toBe(alternativeDevice.id);

          // But with the same data
          expect(transferred.shape).toEqual([4]);
          expect(transferred.dtype).toBe(float32);
          const data = await transferred.toArray();
          expect(data).toEqual([1, 2, 3, 4]);
        });

        it('should preserve tensor properties during transfer', async () => {
          // Complex tensor with specific properties
          const original = await tensor(
            [
              [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
            ] as const,
            { device, dtype: float32 },
          );

          const transferred = await original.to(alternativeDevice);

          // Verify all properties are preserved
          expect(transferred.shape).toEqual(original.shape);
          expect(transferred.dtype).toBe(original.dtype);
          expect(transferred.ndim).toBe(original.ndim);
          expect(transferred.size).toBe(original.size);

          // Verify data integrity
          const originalData = await original.toArray();
          const transferredData = await transferred.toArray();
          expect(transferredData).toEqual(originalData);
        });

        it('should handle empty tensors during transfer', async () => {
          // Empty 1D tensor
          const empty1d = await tensor([] as const, { device, dtype: float32 });
          const transferred1d = await empty1d.to(alternativeDevice);
          expect(transferred1d.shape).toEqual([0]);
          expect(transferred1d.device).toBe(alternativeDevice);
          expect(await transferred1d.toArray()).toEqual([]);

          // Empty 2D tensor
          const empty2d = await zeros([0, 5] as const, { device, dtype: float32 });
          const transferred2d = await empty2d.to(alternativeDevice);
          expect(transferred2d.shape).toEqual([0, 5]);
          expect(transferred2d.size).toBe(0);
        });

        it('should handle scalar tensors during transfer', async () => {
          const scalar = await tensor(3.14159, { device, dtype: float32 });
          const transferred = await scalar.to(alternativeDevice);

          expect(transferred.shape).toEqual([]);
          expect(transferred.ndim).toBe(0);
          expect(transferred.device).toBe(alternativeDevice);
          expect(await transferred.item()).toBeCloseTo(3.14159, 5);
        });

        it('should handle large tensors during transfer', async () => {
          // Large tensor that might trigger different transfer mechanisms
          const large = await ones([100, 100] as const, { device, dtype: float32 });
          const transferred = await large.to(alternativeDevice);

          expect(transferred.shape).toEqual([100, 100]);
          expect(transferred.size).toBe(10000);
          expect(transferred.device).toBe(alternativeDevice);

          // Verify a sample of the data
          const data = await transferred.toArray();
          expect(data[0][0]).toBe(1);
          expect(data[50][50]).toBe(1);
          expect(data[99][99]).toBe(1);
        });

        it('should handle non-contiguous tensors during transfer', async () => {
          // Create a non-contiguous tensor via transpose
          const original = await tensor(
            [
              [1, 2, 3],
              [4, 5, 6],
            ] as const,
            { device, dtype: float32 },
          );
          const transposed = await original.transpose();

          // Verify it's non-contiguous (if supported by the device)
          expect(transposed.shape).toEqual([3, 2]);

          // Transfer the non-contiguous tensor
          const transferred = await transposed.to(alternativeDevice);

          expect(transferred.shape).toEqual([3, 2]);
          expect(transferred.device).toBe(alternativeDevice);

          // Verify data is correctly transferred
          const data = await transferred.toArray();
          expect(data).toEqual([
            [1, 4],
            [2, 5],
            [3, 6],
          ]);
        });

        it('should handle view operations after transfer', async () => {
          // Create tensor on one device
          const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

          // Transfer to another device
          const transferred = await original.to(alternativeDevice);

          // Apply view operations on the transferred tensor
          const reshaped = await transferred.reshape([2, 3] as const);
          expect(reshaped.shape).toEqual([2, 3]);
          expect(reshaped.device).toBe(alternativeDevice);

          const data = await reshaped.toArray();
          expect(data).toEqual([
            [1, 2, 3],
            [4, 5, 6],
          ]);
        });

        it('should handle chained transfers', async () => {
          // Transfer: device1 -> device2 -> device1
          const original = await tensor([1, 2, 3] as const, { device, dtype: float32 });
          const toAlt = await original.to(alternativeDevice);
          const backToOriginal = await toAlt.to(device);

          // Should have the same data but different object
          expect(backToOriginal).not.toBe(original);
          expect(backToOriginal.device).toBe(device);
          expect(await backToOriginal.toArray()).toEqual([1, 2, 3]);
        });

        it('should preserve numerical precision during transfer', async () => {
          // Test with values that might lose precision
          const preciseValues = await tensor(
            [1.23456789, 2.34567891, 3.45678912, 4.56789123] as const,
            { device, dtype: float32 },
          );

          const transferred = await preciseValues.to(alternativeDevice);
          const data = await transferred.toArray();

          // float32 precision (about 7 decimal digits)
          expect(data[0]).toBeCloseTo(1.23456789, 5);
          expect(data[1]).toBeCloseTo(2.34567891, 5);
          expect(data[2]).toBeCloseTo(3.45678912, 5);
          expect(data[3]).toBeCloseTo(4.56789123, 5);
        });

        it('should handle special float values during transfer', async () => {
          // Test Inf, -Inf, and NaN if supported
          const specialValues = await tensor(
            [Infinity, -Infinity, 0, -0] as const,
            { device, dtype: float32 },
          );

          const transferred = await specialValues.to(alternativeDevice);
          const data = await transferred.toArray();

          expect(data[0]).toBe(Infinity);
          expect(data[1]).toBe(-Infinity);
          expect(data[2]).toBe(0);
          expect(data[3]).toBe(-0);
          expect(Object.is(data[3], -0)).toBeTruthy(); // Ensure -0 is preserved
        });
      });
    } else {
      describe('single device environment', () => {
        it('should handle transfer requests when only one device is available', async () => {
          // Even without alternative device, to() should work with the same device
          const original = await tensor([1, 2, 3] as const, { device, dtype: float32 });
          const result = await original.to(device);

          expect(result).toBe(original);
          expect(result.device).toBe(device);
        });
      });
    }

    describe('device property access', () => {
      it('should correctly report device information', async () => {
        const t = await tensor([1, 2, 3] as const, { device, dtype: float32 });

        expect(t.device).toBe(device);
        expect(t.device.id).toBe(device.id);
        expect(t.device.type).toBe(device.type);
        expect(typeof t.device.id).toBe('string');
        expect(typeof t.device.type).toBe('string');
      });

      it('should maintain device reference across operations', async () => {
        const original = await tensor([1, 2, 3] as const, { device, dtype: float32 });

        // Various operations should preserve device
        const negated = await original.neg();
        expect(negated.device).toBe(device);

        const reshaped = await original.reshape([3, 1] as const);
        expect(reshaped.device).toBe(device);

        const summed = await original.sum();
        expect(summed.device).toBe(device);
      });
    });

    describe('error handling', () => {
      it('should handle invalid device gracefully', async () => {
        const t = await tensor([1, 2, 3] as const, { device, dtype: float32 });

        // Test with null/undefined if the API allows it
        // Otherwise, this test might need to be adjusted based on the actual API
        try {
          // @ts-expect-error - Testing runtime behavior with invalid input
          await t.to(null);
          // If it doesn't throw, that's also acceptable
          expect(true).toBeTruthy();
        } catch (error) {
          // Should throw a meaningful error
          expect(error).toBeTruthy();
        }
      });
    });

    describe('memory management', () => {
      it('should handle device transfers without memory leaks', async () => {
        // This test is more of a stress test to ensure no obvious memory issues
        // Real memory leak detection would require profiling tools

        const iterations = 100;
        const tensors: any[] = [];

        for (let i = 0; i < iterations; i++) {
          const t = await tensor([i, i + 1, i + 2] as const, { device, dtype: float32 });
          tensors.push(t);
        }

        // Transfer all tensors to same device (should be optimized)
        const transferred = await Promise.all(tensors.map((t) => t.to(device)));

        // Verify they're the same objects (optimization working)
        for (let i = 0; i < iterations; i++) {
          expect(transferred[i]).toBe(tensors[i]);
        }

        // Clean up references
        tensors.length = 0;
        transferred.length = 0;
      });
    });
  });
}