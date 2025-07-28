/**
 * Test generators for tensor creation operations
 *
 * These generators create standardized test suites that can be run against
 * any Device implementation to verify tensor creation functionality.
 */

import type { Device } from '@typetensor/core';
import { tensor, zeros, ones, eye, float32, int32, bool } from '@typetensor/core';

/**
 * Generates a comprehensive test suite for tensor creation operations
 *
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateTensorCreationTests(
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

  describe(`Tensor Creation Tests (${device.type}:${device.id})`, () => {
    describe('scalar creation', () => {
      it('should create scalar tensors with correct properties', async () => {
        // PyTorch: torch.tensor(3.14)
        // Output: tensor(3.1400)
        // shape: torch.Size([]), dtype: torch.float32
        const floatScalar = await tensor(3.14, { device, dtype: float32 });

        // Verify scalar properties
        expect(Array.isArray(floatScalar.shape)).toBe(true);
        expect(floatScalar.shape).toEqual([]);
        expect(floatScalar.ndim).toBe(0);
        expect(floatScalar.size).toBe(1);
        expect(floatScalar.dtype).toBe(float32);
        expect(floatScalar.device).toBe(device);

        // Verify data integrity
        const value = await floatScalar.item();
        expect(value).toBeCloseTo(3.14, 5); // float32 precision
        expect(typeof value).toBe('number');

        // Test other dtypes maintain their properties
        // PyTorch: torch.tensor(42, dtype=torch.int32)
        // Output: tensor(42, dtype=torch.int32)
        // shape: torch.Size([]), dtype: torch.int32
        const intScalar = await tensor(42, { device, dtype: int32 });
        expect(intScalar.shape).toEqual([]);
        expect(intScalar.ndim).toBe(0);
        expect(intScalar.size).toBe(1);
        expect(intScalar.dtype).toBe(int32);
        expect(await intScalar.item()).toBe(42);

        // PyTorch: torch.tensor(True)
        // Output: tensor(True)
        // shape: torch.Size([]), dtype: torch.bool
        const boolScalar = await tensor(true, { device, dtype: bool });
        expect(boolScalar.shape).toEqual([]);
        expect(boolScalar.dtype).toBe(bool);
        expect(await boolScalar.item()).toBe(true);
      });

      it('should extract scalar values with correct types', async () => {
        // PyTorch: torch.tensor(42.5).item()
        // Output: 42.5
        const floatScalar = await tensor(42.5, { device, dtype: float32 });
        const floatValue = await floatScalar.item();
        expect(floatValue).toBeCloseTo(42.5, 5); // float32 precision
        expect(typeof floatValue).toBe('number');

        // PyTorch: torch.tensor(-123, dtype=torch.int32).item()
        // Output: -123
        const intScalar = await tensor(-123, { device, dtype: int32 });
        const intValue = await intScalar.item();
        expect(intValue).toBe(-123); // int32 should be exact
        expect(typeof intValue).toBe('number');

        // PyTorch: torch.tensor(False).item()
        // Output: False
        const boolScalar = await tensor(false, { device, dtype: bool });
        const boolValue = await boolScalar.item();
        expect(boolValue).toBe(false);
        expect(typeof boolValue).toBe('boolean');
      });
    });

    describe('vector creation', () => {
      it('should create 1D tensors with data integrity', async () => {
        // PyTorch: torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        // Output: tensor([1., 2., 3., 4.])
        // shape: torch.Size([4]), dtype: torch.float32
        const originalData = [1, 2, 3, 4] as const;
        const vec = await tensor(originalData, { device, dtype: float32 });

        // Verify structural properties
        expect(Array.isArray(vec.shape)).toBe(true);
        expect(vec.shape).toEqual([4]);
        expect(vec.ndim).toBe(1);
        expect(vec.size).toBe(4);
        expect(vec.dtype).toBe(float32);
        expect(vec.device).toBe(device);

        // Verify strides for 1D tensor
        expect(Array.isArray(vec.strides)).toBe(true);
        expect(vec.strides).toEqual([1]);

        // Verify data integrity
        const extractedData = await vec.toArray();
        expect(Array.isArray(extractedData)).toBe(true);
        expect(extractedData).toEqual([1, 2, 3, 4]);
        expect(extractedData.length).toBe(4);

        // Verify individual elements
        for (let i = 0; i < extractedData.length; i++) {
          expect(typeof extractedData[i]).toBe('number');
          expect(extractedData[i]).toBe(originalData[i]);
        }
      });

      it('should handle empty vectors correctly', async () => {
        // PyTorch: torch.tensor([])
        // Output: tensor([])
        // shape: torch.Size([0]), dtype: torch.float32
        const empty = await tensor([] as const, { device, dtype: float32 });

        // Verify empty tensor properties
        expect(Array.isArray(empty.shape)).toBe(true);
        expect(empty.shape).toEqual([0]);
        expect(empty.ndim).toBe(1);
        expect(empty.size).toBe(0);
        expect(empty.dtype).toBe(float32);
        expect(empty.device).toBe(device);

        // Verify empty data
        const data = await empty.toArray();
        expect(Array.isArray(data)).toBe(true);
        expect(data).toEqual([]);
        expect(data.length).toBe(0);
      });
    });

    describe('matrix creation', () => {
      it('should create 2D tensors with correct layout', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        // Output: tensor([[1., 2., 3.],
        //                  [4., 5., 6.]])
        // shape: torch.Size([2, 3]), dtype: torch.float32
        const originalData = [
          [1, 2, 3],
          [4, 5, 6],
        ] as const;
        const matrix = await tensor(originalData, { device, dtype: float32 });

        // Verify matrix properties
        expect(Array.isArray(matrix.shape)).toBe(true);
        expect(matrix.shape).toEqual([2, 3]);
        expect(matrix.ndim).toBe(2);
        expect(matrix.size).toBe(6);
        expect(matrix.dtype).toBe(float32);
        expect(matrix.device).toBe(device);

        // Verify row-major strides (C-order)
        expect(Array.isArray(matrix.strides)).toBe(true);
        expect(matrix.strides).toEqual([3, 1]);

        // Verify data integrity and structure
        const extractedData = await matrix.toArray();
        expect(Array.isArray(extractedData)).toBe(true);
        expect(extractedData.length).toBe(2);
        expect(Array.isArray(extractedData[0])).toBe(true);
        expect(Array.isArray(extractedData[1])).toBe(true);
        expect(extractedData).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);

        // Verify individual row structure
        expect(extractedData[0].length).toBe(3);
        expect(extractedData[1].length).toBe(3);
      });

      it('should create square matrices', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        // Output: tensor([[1, 2],
        //                  [3, 4]], dtype=torch.int32)
        const square = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: int32 },
        );

        expect(square.shape).toEqual([2, 2]);
        expect(square.dtype).toBe(int32);
      });
    });

    describe('higher dimensional tensors', () => {
      it('should create 3D tensors', async () => {
        // PyTorch: torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
        // Output: tensor([[[1., 2.],
        //                   [3., 4.]],
        //                  [[5., 6.],
        //                   [7., 8.]]])
        // shape: torch.Size([2, 2, 2]), dtype: torch.float32
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

        expect(tensor3d.shape).toEqual([2, 2, 2]);
        expect(tensor3d.ndim).toBe(3);
        expect(tensor3d.size).toBe(8);
      });

      it('should create 4D tensors', async () => {
        // PyTorch: torch.tensor([[[[1, 2]], [[3, 4]]], [[[5, 6]], [[7, 8]]]], dtype=torch.float32)
        // shape: torch.Size([2, 2, 1, 2]), dtype: torch.float32
        const tensor4d = await tensor(
          [
            [[[1, 2]], [[3, 4]]],
            [[[5, 6]], [[7, 8]]],
          ] as const,
          { device, dtype: float32 },
        );

        expect(tensor4d.shape).toEqual([2, 2, 1, 2]);
        expect(tensor4d.ndim).toBe(4);
        expect(tensor4d.size).toBe(8);
      });
    });

    describe('special tensor creation', () => {
      it('should create zero tensors with all zeros', async () => {
        // PyTorch: torch.zeros(2, 3)
        // Output: tensor([[0., 0., 0.],
        //                  [0., 0., 0.]])
        // shape: torch.Size([2, 3]), dtype: torch.float32
        const zeroMatrix = await zeros([2, 3] as const, { device, dtype: float32 });

        // Verify tensor structure
        expect(Array.isArray(zeroMatrix.shape)).toBe(true);
        expect(zeroMatrix.shape).toEqual([2, 3]);
        expect(zeroMatrix.ndim).toBe(2);
        expect(zeroMatrix.size).toBe(6);
        expect(zeroMatrix.dtype).toBe(float32);
        expect(zeroMatrix.device).toBe(device);

        // Verify all values are zero
        const data = await zeroMatrix.toArray();
        expect(Array.isArray(data)).toBe(true);
        expect(data).toEqual([
          [0, 0, 0],
          [0, 0, 0],
        ]);

        // Verify each element is actually zero
        for (const row of data) {
          for (const value of row) {
            expect(value).toBe(0);
            expect(typeof value).toBe('number');
          }
        }
      });

      it('should create one tensors with all ones', async () => {
        // PyTorch: torch.ones(4, dtype=torch.int32)
        // Output: tensor([1, 1, 1, 1], dtype=torch.int32)
        // shape: torch.Size([4]), dtype: torch.int32
        const oneVector = await ones([4] as const, { device, dtype: int32 });

        // Verify tensor structure
        expect(Array.isArray(oneVector.shape)).toBe(true);
        expect(oneVector.shape).toEqual([4]);
        expect(oneVector.ndim).toBe(1);
        expect(oneVector.size).toBe(4);
        expect(oneVector.dtype).toBe(int32);
        expect(oneVector.device).toBe(device);

        // Verify all values are one
        const data = await oneVector.toArray();
        expect(Array.isArray(data)).toBe(true);
        expect(data).toEqual([1, 1, 1, 1]);
        expect(data.length).toBe(4);

        // Verify each element is actually one
        for (const value of data) {
          expect(value).toBe(1);
          expect(typeof value).toBe('number');
        }
      });

      it('should create proper identity matrices', async () => {
        // PyTorch: torch.eye(3)
        // Output: tensor([[1., 0., 0.],
        //                  [0., 1., 0.],
        //                  [0., 0., 1.]])
        // shape: torch.Size([3, 3]), dtype: torch.float32
        const identity = await eye(3, { device, dtype: float32 });

        // Verify identity matrix structure
        expect(Array.isArray(identity.shape)).toBe(true);
        expect(identity.shape).toEqual([3, 3]);
        expect(identity.ndim).toBe(2);
        expect(identity.size).toBe(9);
        expect(identity.dtype).toBe(float32);
        expect(identity.device).toBe(device);

        // Verify identity matrix data
        const data = await identity.toArray();
        expect(Array.isArray(data)).toBe(true);
        expect(data.length).toBe(3);
        expect(data).toEqual([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ]);

        // Verify diagonal and off-diagonal elements
        for (let i = 0; i < 3; i++) {
          const row = data[i];
          expect(Array.isArray(row)).toBe(true);
          expect(row!.length).toBe(3);
          for (let j = 0; j < 3; j++) {
            const element = row![j];
            expect(typeof element).toBe('number');
            if (i === j) {
              expect(element).toBe(1); // diagonal should be 1
            } else {
              expect(element).toBe(0); // off-diagonal should be 0
            }
          }
        }
      });

      it('should create scalar zeros and ones', async () => {
        // PyTorch: torch.zeros(())
        // Output: tensor(0.)
        // shape: torch.Size([]), dtype: torch.float32
        const zeroScalar = await zeros([] as const, { device, dtype: float32 });
        expect(zeroScalar.shape).toEqual([]);
        expect(await zeroScalar.item()).toBe(0);

        // PyTorch: torch.ones(())
        // Output: tensor(1.)
        // shape: torch.Size([]), dtype: torch.float32
        const oneScalar = await ones([] as const, { device, dtype: float32 });
        expect(oneScalar.shape).toEqual([]);
        expect(await oneScalar.item()).toBe(1);
      });
    });

    describe('error handling', () => {
      it('should throw on mismatched array dimensions', async () => {
        // PyTorch: torch.tensor([[1, 2, 3], [4, 5]])
        // Error: ValueError: expected sequence of length 3 at dim 1 (got 2)
        await expect(
          tensor(
            [
              [1, 2, 3],
              [4, 5], // Wrong length
            ] as const,
            { device, dtype: float32 },
          ),
        ).rejects.toThrow('Inconsistent array dimensions');
      });

      it('should throw on invalid eye matrix size', async () => {
        // PyTorch: torch.eye(-1)
        // Error: RuntimeError: n must be greater or equal to 0, got -1
        await expect(eye(-1, { device, dtype: float32 })).rejects.toThrow(
          'Invalid size for identity matrix: -1',
        );
      });
    });
  });
}
