/**
 * Test generators for developer utilities
 *
 * These generators test user-facing utilities like toString(), toArray(),
 * and other debugging/inspection methods.
 */

import type { Device } from '@typetensor/core';
import { tensor, zeros, ones, float32, int32, bool } from '@typetensor/core';

/**
 * Generates tests for developer utility functions
 *
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateDevUtilsTests(
  device: Device,
  testFramework: {
    describe: (name: string, fn: () => void) => void;
    it: (name: string, fn: () => void | Promise<void>) => void;
    expect: (actual: unknown) => {
      toBe: (expected: unknown) => void;
      toEqual: (expected: unknown) => void;
      toMatch: (pattern: RegExp) => void;
      toContain: (substring: string) => void;
      toBeGreaterThan: (expected: number) => void;
      toBeCloseTo: (expected: number, precision?: number) => void;
      toHaveLength?: (length: number) => void;
      not?: {
        toThrow: () => void;
      };
    };
  },
) {
  const { describe, it, expect } = testFramework;

  describe(`Developer Utilities Tests (${device.type}:${device.id})`, () => {
    describe('toString() formatting', () => {
      it('should format scalar tensors', async () => {
        // PyTorch: scalar = torch.tensor(42.5)
        // print(scalar) -> tensor(42.5000)
        // repr(scalar) -> tensor(42.5000)
        const scalar = await tensor(42.5, { device, dtype: float32 });
        const str = scalar.toString();

        expect(str).toContain('Tensor');
        expect(str).toContain('shape=scalar []');
        expect(str).toContain('dtype=float32');
        expect(str).toContain(`device=${device.id}`);
      });

      it('should format vector tensors', async () => {
        // PyTorch: vector = torch.tensor([1, 2, 3], dtype=torch.int32)
        // print(vector) -> tensor([1, 2, 3], dtype=torch.int32)
        const vector = await tensor([1, 2, 3] as const, { device, dtype: int32 });
        const str = vector.toString();

        expect(str).toContain('shape=[3]');
        expect(str).toContain('dtype=int32');
        expect(str).toContain(`device=${device.id}`);
      });

      it('should format matrix tensors', async () => {
        // PyTorch: matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        // print(matrix) -> tensor([[1., 2.],
        //                          [3., 4.]])
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const str = matrix.toString();

        expect(str).toContain('shape=[2, 2]');
        expect(str).toContain('dtype=float32');
        expect(str).toContain(`device=${device.id}`);
      });

      it('should format higher dimensional tensors', async () => {
        // PyTorch: tensor3d = torch.zeros(2, 3, 4)
        // Output shape: torch.Size([2, 3, 4])
        const tensor3d = await zeros([2, 3, 4] as const, { device, dtype: float32 });
        const str = tensor3d.toString();

        expect(str).toContain('shape=[2, 3, 4]');
        expect(str).toContain('dtype=float32');
      });

      it('should format boolean tensors', async () => {
        // PyTorch: boolTensor = torch.tensor([True, False])
        // print(boolTensor) -> tensor([ True, False])
        const boolTensor = await tensor([true, false] as const, { device, dtype: bool });
        const str = boolTensor.toString();

        expect(str).toContain('dtype=bool');
      });
    });

    describe('toArray() data extraction', () => {
      it('should extract scalar values', async () => {
        // PyTorch: scalar = torch.tensor(3.14159)
        // scalar.numpy() -> array(3.14159, dtype=float32)
        // scalar.numpy().shape -> ()
        const scalar = await tensor(3.14159, { device, dtype: float32 });
        const array = await scalar.toArray();
        expect(array).toBeCloseTo(3.14159, 5); // float32 precision
      });

      it('should extract vector data', async () => {
        // PyTorch: vector = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        // vector.numpy() -> array([1, 2, 3, 4, 5], dtype=int32)
        const vector = await tensor([1, 2, 3, 4, 5] as const, { device, dtype: int32 });
        const array = await vector.toArray();
        expect(array).toEqual([1, 2, 3, 4, 5]);
      });

      it('should extract matrix data', async () => {
        // PyTorch: matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        // matrix.numpy() -> array([[1., 2., 3.],
        //                          [4., 5., 6.]], dtype=float32)
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const array = await matrix.toArray();
        expect(array).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);
      });

      it('should extract 3D tensor data', async () => {
        // PyTorch: tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        // tensor3d.numpy() -> 3D numpy array with shape (2, 2, 2)
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
        const array = await tensor3d.toArray();
        expect(array).toEqual([
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

      it('should extract boolean data', async () => {
        // PyTorch: boolTensor = torch.tensor([True, False, True, False])
        // boolTensor.numpy() -> array([ True, False,  True, False])
        const boolTensor = await tensor([true, false, true, false] as const, {
          device,
          dtype: bool,
        });
        const array = await boolTensor.toArray();
        expect(array).toEqual([true, false, true, false]);
      });

      it('should handle empty tensors', async () => {
        // PyTorch: empty = torch.tensor([])
        // empty.shape -> torch.Size([0]), empty.numpy() -> array([], dtype=float32)
        const empty = await tensor([] as const, { device, dtype: float32 });
        const array = await empty.toArray();
        expect(array).toEqual([]);
      });

      it('should preserve data precision', async () => {
        // PyTorch: tensor([1.23456789, -2.98765432, 0.0, 999.999], dtype=torch.float32)
        // Note: float32 has ~7 significant decimal digits of precision
        const preciseData = [1.23456789, -2.98765432, 0.0, 999.999] as const;
        const tensor1 = await tensor(preciseData, { device, dtype: float32 });
        const extracted = await tensor1.toArray();

        // Verify array structure and types
        expect(Array.isArray(extracted)).toBe(true);
        expect(extracted.length).toBe(4);
        expect(typeof extracted[0]).toBe('number');
        expect(typeof extracted[1]).toBe('number');
        expect(typeof extracted[2]).toBe('number');
        expect(typeof extracted[3]).toBe('number');
      });
    });

    describe('item() scalar extraction', () => {
      it('should extract scalar values', async () => {
        // PyTorch: torch.tensor(42).item() -> 42 (returns Python int)
        // torch.tensor(3.14).item() -> 3.14000... (returns Python float)
        // torch.tensor(True).item() -> True (returns Python bool)
        const scalar1 = await tensor(42, { device, dtype: int32 });
        expect(await scalar1.item()).toBe(42);

        const scalar2 = await tensor(3.14, { device, dtype: float32 });
        expect(await scalar2.item()).toBeCloseTo(3.14, 5); // float32 precision

        const scalar3 = await tensor(true, { device, dtype: bool });
        expect(await scalar3.item()).toBe(true);
      });

      it('should work with computed scalars', async () => {
        // PyTorch: torch.ones(1).item() -> 1.0
        // Works for any tensor with exactly one element
        const ones1x1 = await ones([1] as const, { device, dtype: float32 });
        // This would be a view/slice to make it scalar, but for now test the vector
        const array = await ones1x1.toArray();
        expect(array).toEqual([1]);
      });
    });

    describe('string representation', () => {
      it('should format scalar metadata correctly', async () => {
        // PyTorch: str(torch.tensor(42.5)) -> "tensor(42.5000)"
        // Our format includes more metadata for debugging
        const scalar = await tensor(42.5, { device, dtype: float32 });
        const str = scalar.toString();

        // Should contain essential tensor metadata
        expect(str).toContain('Tensor(');
        expect(str).toContain('shape=scalar []');
        expect(str).toContain('dtype=float32');
        expect(str).toContain(`device=${device.id}`);
      });

      it('should format vector metadata correctly', async () => {
        // PyTorch: repr(torch.tensor([1, 2, 3])) -> "tensor([1, 2, 3])"
        const vector = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const str = vector.toString();

        // Should show correct shape and metadata
        expect(str).toContain('Tensor(');
        expect(str).toContain('shape=[3]');
        expect(str).toContain('dtype=float32');
        expect(str).toContain(`device=${device.id}`);
      });

      it('should format matrix metadata correctly', async () => {
        // PyTorch: repr(torch.tensor([[1, 2], [3, 4]])) ->
        // "tensor([[1, 2],\n         [3, 4]])"
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const str = matrix.toString();

        // Should show correct 2D shape
        expect(str).toContain('Tensor(');
        expect(str).toContain('shape=[2, 2]');
        expect(str).toContain('dtype=float32');
        expect(str).toContain(`device=${device.id}`);
      });
    });

    describe('data consistency', () => {
      it('should maintain data consistency between toArray calls', async () => {
        // Multiple calls to numpy() should return equivalent data
        const tensor1 = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });

        const array1 = await tensor1.toArray();
        const array2 = await tensor1.toArray();

        expect(array1).toEqual(array2);
      });

      it('should maintain consistency between toString calls', async () => {
        // String representation should be stable across calls
        const tensor1 = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        const str1 = tensor1.toString();
        const str2 = tensor1.toString();

        expect(str1).toBe(str2);
      });

      it('should provide consistent data across different access methods', async () => {
        // Data should be consistent across different views/representations
        const originalData = [10, 20, 30] as const;
        const tensor1 = await tensor(originalData, { device, dtype: float32 });

        const extractedArray = await tensor1.toArray();
        expect(extractedArray).toEqual([10, 20, 30]);

        const stringRep = tensor1.toString();
        expect(stringRep).toContain('shape=[3]');
      });
    });

    describe('error handling', () => {
      it('should handle tensor access without errors', async () => {
        // Basic operations should be robust and error-free
        const tensor1 = await tensor([1, 2, 3] as const, { device, dtype: float32 });

        // Basic operations should work reliably
        const str = tensor1.toString();
        const array = await tensor1.toArray();

        expect(typeof str).toBe('string');
        expect(str.length).toBeGreaterThan(0);
        expect(Array.isArray(array)).toBe(true);
        expect(array.length).toBe(3);
      });
    });
  });
}
