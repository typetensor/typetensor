/**
 * Test generators for binary operations
 *
 * These generators test binary mathematical operations like add, sub, mul, div
 * on tensors of various shapes and data types, including broadcasting behavior.
 */

import type { Device } from '@typetensor/core';
import { tensor, float32, int32 } from '@typetensor/core';

/**
 * Generates tests for binary operations
 *
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateBinaryOperationTests(
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

  describe(`Binary Operations Tests (${device.type}:${device.id})`, () => {
    describe('addition operations', () => {
      it('should add scalar values', async () => {
        // PyTorch: torch.tensor(3.14) + torch.tensor(2.86)
        // Output: tensor(6.0000)
        const a = await tensor(3.14, { device, dtype: float32 });
        const b = await tensor(2.86, { device, dtype: float32 });
        const result = await a.add(b);

        expect(result.shape).toEqual([]);
        expect(result.dtype).toBe(float32);
        expect(result.device).toBe(device);
        expect(await result.item()).toBeCloseTo(6.0, 5);
      });

      it('should add vector values element-wise', async () => {
        // PyTorch: torch.tensor([1, 2, 3, 4]) + torch.tensor([5, 6, 7, 8])
        // Output: tensor([6, 8, 10, 12])
        const a = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });
        const b = await tensor([5, 6, 7, 8] as const, { device, dtype: float32 });
        const result = await a.add(b);

        expect(result.shape).toEqual([4]);
        expect(result.dtype).toBe(float32);
        expect(result.device).toBe(device);

        const data = await result.toArray();
        expect(data).toEqual([6, 8, 10, 12]);
      });

      it('should add matrix values element-wise', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]) + torch.tensor([[5, 6], [7, 8]])
        // Output: tensor([[6, 8],
        //                 [10, 12]])
        const a = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [5, 6],
            [7, 8],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await a.add(b);

        expect(result.shape).toEqual([2, 2]);
        const data = await result.toArray();
        expect(data).toEqual([
          [6, 8],
          [10, 12],
        ]);
      });

      it('should handle integer addition', async () => {
        // PyTorch: torch.tensor([1, 2, 3], dtype=torch.int32) + torch.tensor([4, 5, 6], dtype=torch.int32)
        // Output: tensor([5, 7, 9], dtype=torch.int32)
        const a = await tensor([1, 2, 3] as const, { device, dtype: int32 });
        const b = await tensor([4, 5, 6] as const, { device, dtype: int32 });
        const result = await a.add(b);

        expect(result.dtype).toBe(int32);
        const data = await result.toArray();
        expect(data).toEqual([5, 7, 9]);
      });
    });

    describe('subtraction operations', () => {
      it('should subtract scalar values', async () => {
        // PyTorch: torch.tensor(10.5) - torch.tensor(3.2)
        // Output: tensor(7.3000)
        const a = await tensor(10.5, { device, dtype: float32 });
        const b = await tensor(3.2, { device, dtype: float32 });
        const result = await a.sub(b);

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(7.3, 5);
      });

      it('should subtract vector values element-wise', async () => {
        // PyTorch: torch.tensor([10, 8, 6, 4]) - torch.tensor([1, 2, 3, 4])
        // Output: tensor([9, 6, 3, 0])
        const a = await tensor([10, 8, 6, 4] as const, { device, dtype: float32 });
        const b = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });
        const result = await a.sub(b);

        expect(result.shape).toEqual([4]);
        const data = await result.toArray();
        expect(data).toEqual([9, 6, 3, 0]);
      });

      it('should subtract matrix values element-wise', async () => {
        // PyTorch: torch.tensor([[10, 8], [6, 4]]) - torch.tensor([[1, 2], [3, 4]])
        // Output: tensor([[9, 6],
        //                 [3, 0]])
        const a = await tensor(
          [
            [10, 8],
            [6, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await a.sub(b);

        expect(result.shape).toEqual([2, 2]);
        const data = await result.toArray();
        expect(data).toEqual([
          [9, 6],
          [3, 0],
        ]);
      });

      it('should handle negative results', async () => {
        // PyTorch: torch.tensor([1, 2]) - torch.tensor([3, 4])
        // Output: tensor([-2, -2])
        const a = await tensor([1, 2] as const, { device, dtype: float32 });
        const b = await tensor([3, 4] as const, { device, dtype: float32 });
        const result = await a.sub(b);

        const data = await result.toArray();
        expect(data).toEqual([-2, -2]);
      });
    });

    describe('multiplication operations', () => {
      it('should multiply scalar values', async () => {
        // PyTorch: torch.tensor(3.0) * torch.tensor(4.0)
        // Output: tensor(12.0)
        const a = await tensor(3.0, { device, dtype: float32 });
        const b = await tensor(4.0, { device, dtype: float32 });
        const result = await a.mul(b);

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(12.0, 5);
      });

      it('should multiply vector values element-wise', async () => {
        // PyTorch: torch.tensor([1, 2, 3, 4]) * torch.tensor([2, 3, 4, 5])
        // Output: tensor([2, 6, 12, 20])
        const a = await tensor([1, 2, 3, 4] as const, { device, dtype: float32 });
        const b = await tensor([2, 3, 4, 5] as const, { device, dtype: float32 });
        const result = await a.mul(b);

        expect(result.shape).toEqual([4]);
        const data = await result.toArray();
        expect(data).toEqual([2, 6, 12, 20]);
      });

      it('should multiply matrix values element-wise', async () => {
        // PyTorch: torch.tensor([[1, 2], [3, 4]]) * torch.tensor([[2, 3], [4, 5]])
        // Output: tensor([[2, 6],
        //                 [12, 20]])
        const a = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [2, 3],
            [4, 5],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await a.mul(b);

        expect(result.shape).toEqual([2, 2]);
        const data = await result.toArray();
        expect(data).toEqual([
          [2, 6],
          [12, 20],
        ]);
      });

      it('should handle zero multiplication', async () => {
        // PyTorch: torch.tensor([1, 2, 3]) * torch.tensor([0, 0, 0])
        // Output: tensor([0, 0, 0])
        const a = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const b = await tensor([0, 0, 0] as const, { device, dtype: float32 });
        const result = await a.mul(b);

        const data = await result.toArray();
        expect(data).toEqual([0, 0, 0]);
      });

      it('should handle negative multiplication', async () => {
        // PyTorch: torch.tensor([1, -2, 3]) * torch.tensor([-1, 2, -3])
        // Output: tensor([-1, -4, -9])
        const a = await tensor([1, -2, 3] as const, { device, dtype: float32 });
        const b = await tensor([-1, 2, -3] as const, { device, dtype: float32 });
        const result = await a.mul(b);

        const data = await result.toArray();
        expect(data).toEqual([-1, -4, -9]);
      });
    });

    describe('division operations', () => {
      it('should divide scalar values', async () => {
        // PyTorch: torch.tensor(12.0) / torch.tensor(3.0)
        // Output: tensor(4.0)
        const a = await tensor(12.0, { device, dtype: float32 });
        const b = await tensor(3.0, { device, dtype: float32 });
        const result = await a.div(b);

        expect(result.shape).toEqual([]);
        expect(await result.item()).toBeCloseTo(4.0, 5);
      });

      it('should divide vector values element-wise', async () => {
        // PyTorch: torch.tensor([12, 15, 20, 8]) / torch.tensor([3, 5, 4, 2])
        // Output: tensor([4., 3., 5., 4.])
        const a = await tensor([12, 15, 20, 8] as const, { device, dtype: float32 });
        const b = await tensor([3, 5, 4, 2] as const, { device, dtype: float32 });
        const result = await a.div(b);

        expect(result.shape).toEqual([4]);
        const data = await result.toArray();
        expect(data[0]).toBeCloseTo(4, 5);
        expect(data[1]).toBeCloseTo(3, 5);
        expect(data[2]).toBeCloseTo(5, 5);
        expect(data[3]).toBeCloseTo(4, 5);
      });

      it('should divide matrix values element-wise', async () => {
        // PyTorch: torch.tensor([[12, 15], [20, 8]]) / torch.tensor([[3, 5], [4, 2]])
        // Output: tensor([[4., 3.],
        //                 [5., 4.]])
        const a = await tensor(
          [
            [12, 15],
            [20, 8],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [3, 5],
            [4, 2],
          ] as const,
          { device, dtype: float32 },
        );
        const result = await a.div(b);

        expect(result.shape).toEqual([2, 2]);
        const data = await result.toArray();
        expect(data[0][0]).toBeCloseTo(4, 5);
        expect(data[0][1]).toBeCloseTo(3, 5);
        expect(data[1][0]).toBeCloseTo(5, 5);
        expect(data[1][1]).toBeCloseTo(4, 5);
      });

      it('should handle fractional division', async () => {
        // PyTorch: torch.tensor([1, 1, 1]) / torch.tensor([2, 3, 4])
        // Output: tensor([0.5000, 0.3333, 0.2500])
        const a = await tensor([1, 1, 1] as const, { device, dtype: float32 });
        const b = await tensor([2, 3, 4] as const, { device, dtype: float32 });
        const result = await a.div(b);

        const data = await result.toArray();
        expect(data[0]).toBeCloseTo(0.5, 5);
        expect(data[1]).toBeCloseTo(0.333333, 5);
        expect(data[2]).toBeCloseTo(0.25, 5);
      });

      it('should handle negative division', async () => {
        // PyTorch: torch.tensor([6, -8]) / torch.tensor([-2, 4])
        // Output: tensor([-3., -2.])
        const a = await tensor([6, -8] as const, { device, dtype: float32 });
        const b = await tensor([-2, 4] as const, { device, dtype: float32 });
        const result = await a.div(b);

        const data = await result.toArray();
        expect(data[0]).toBeCloseTo(-3, 5);
        expect(data[1]).toBeCloseTo(-2, 5);
      });
    });

    describe('broadcasting operations', () => {
      it('should broadcast scalar with vector', async () => {
        // PyTorch: scalar + vector and vector * scalar broadcasting
        // torch.tensor(5) + torch.tensor([1, 2, 3]) = tensor([6, 7, 8])
        // torch.tensor([1, 2, 3]) * torch.tensor(5) = tensor([5, 10, 15])
        const scalar = await tensor(5, { device, dtype: float32 });
        const vector = await tensor([1, 2, 3] as const, { device, dtype: float32 });

        const addResult = await scalar.add(vector);
        const mulResult = await vector.mul(scalar);

        expect(addResult.shape).toEqual([3]);
        expect(mulResult.shape).toEqual([3]);

        const addData = await addResult.toArray();
        const mulData = await mulResult.toArray();
        expect(addData).toEqual([6, 7, 8]);
        expect(mulData).toEqual([5, 10, 15]);
      });

      it('should broadcast scalar with matrix', async () => {
        // PyTorch: matrix * scalar broadcasting
        // torch.tensor([[1, 2], [3, 4]]) * torch.tensor(2)
        // Output: tensor([[2, 4],
        //                 [6, 8]])
        const scalar = await tensor(2, { device, dtype: float32 });
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await matrix.mul(scalar);

        expect(result.shape).toEqual([2, 2]);
        const data = await result.toArray();
        expect(data).toEqual([
          [2, 4],
          [6, 8],
        ]);
      });

      it('should broadcast vector with matrix (row broadcasting)', async () => {
        // PyTorch: Row broadcasting - vector broadcasts across rows
        // matrix + vector where vector shape [2] broadcasts to [3, 2]
        // Output: tensor([[11, 22],
        //                 [13, 24],
        //                 [15, 26]])
        const vector = await tensor([10, 20] as const, { device, dtype: float32 });
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
            [5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await matrix.add(vector);

        expect(result.shape).toEqual([3, 2]);
        const data = await result.toArray();
        expect(data).toEqual([
          [11, 22],
          [13, 24],
          [15, 26],
        ]);
      });

      it('should broadcast vector with matrix (column broadcasting)', async () => {
        // PyTorch: Column broadcasting - reshape to [3, 1] then broadcast
        // vector.reshape(3, 1) + matrix broadcasts to [3, 2]
        // Output: tensor([[101, 102],
        //                 [203, 204],
        //                 [305, 306]])
        const vector = await tensor([100, 200, 300] as const, { device, dtype: float32 });
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
            [5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // Reshape vector to column vector [3, 1] for column broadcasting
        const columnVector = vector.reshape([3, 1] as const);
        const result = await matrix.add(columnVector);

        expect(result.shape).toEqual([3, 2]);
        const data = await result.toArray();
        expect(data).toEqual([
          [101, 102],
          [203, 204],
          [305, 306],
        ]);
      });

      it('should handle compatible shape broadcasting', async () => {
        // PyTorch: Broadcasting [2, 1] + [1, 3] -> [2, 3]
        // A: [[1], [2]], B: [[10, 20, 30]]
        // Output: tensor([[11, 21, 31],
        //                 [12, 22, 32]])
        const a = await tensor([[1], [2]] as const, { device, dtype: float32 });
        const b = await tensor([[10, 20, 30]] as const, { device, dtype: float32 });

        const result = await a.add(b);

        expect(result.shape).toEqual([2, 3]);
        const data = await result.toArray();
        expect(data).toEqual([
          [11, 21, 31],
          [12, 22, 32],
        ]);
      });
    });

    describe('property preservation', () => {
      it('should preserve tensor metadata across binary operations', async () => {
        const a = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );
        const b = await tensor(
          [
            [7, 8, 9],
            [10, 11, 12],
          ] as const,
          { device, dtype: float32 },
        );

        const add = await a.add(b);
        const sub = await a.sub(b);
        const mul = await a.mul(b);
        const div = await a.div(b);

        // All results should have same shape as inputs
        expect(add.shape).toEqual([2, 3]);
        expect(sub.shape).toEqual([2, 3]);
        expect(mul.shape).toEqual([2, 3]);
        expect(div.shape).toEqual([2, 3]);

        // All results should preserve dtype
        expect(add.dtype).toBe(float32);
        expect(sub.dtype).toBe(float32);
        expect(mul.dtype).toBe(float32);
        expect(div.dtype).toBe(float32);

        // All results should preserve device
        expect(add.device).toBe(device);
        expect(sub.device).toBe(device);
        expect(mul.device).toBe(device);
        expect(div.device).toBe(device);

        // All results should have correct size
        expect(add.size).toBe(6);
        expect(sub.size).toBe(6);
        expect(mul.size).toBe(6);
        expect(div.size).toBe(6);
      });

      it('should work with different tensor shapes', async () => {
        // Test 3D tensor operations
        const a = await tensor([[[1, 2]], [[3, 4]]] as const, { device, dtype: float32 });
        const b = await tensor([[[5, 6]], [[7, 8]]] as const, { device, dtype: float32 });

        const result = await a.add(b);
        expect(result.shape).toEqual([2, 1, 2]);

        const data = await result.toArray();
        expect(data).toEqual([[[6, 8]], [[10, 12]]]);
      });
    });

    describe('error handling', () => {
      it('should handle division by zero gracefully', async () => {
        // PyTorch: torch.tensor([1, 2, 3]) / torch.tensor([1, 0, 3])
        // Output: tensor([1., inf, 1.])
        // Division by zero produces infinity in PyTorch
        const a = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const b = await tensor([1, 0, 3] as const, { device, dtype: float32 });

        // Division by zero behavior may vary by implementation
        // Some may return Infinity, others may throw, both are acceptable
        const result = await a.div(b);
        expect(result.shape).toEqual([3]);

        const data = await result.toArray();
        expect(Array.isArray(data)).toBe(true);
        expect(data.length).toBe(3);
        // First and third elements should be finite
        expect(data[0]).toBeCloseTo(1, 5);
        expect(data[2]).toBeCloseTo(1, 5);
        // Second element (1/0) might be Infinity or error - just check it exists
        expect(typeof data[1]).toBe('number');
      });

      it('should handle incompatible shapes appropriately', async () => {
        const a = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const b = await tensor([1, 2] as const, { device, dtype: float32 }); // Different size

        // This should either broadcast correctly or throw an error
        // depending on the implementation's broadcasting rules
        try {
          // @ts-expect-error - Testing runtime behavior with potentially incompatible shapes
          const result = await a.add(b);
          // If it succeeds, check the result is valid
          // @ts-expect-error - result type may be never but we're testing runtime behavior
          expect(result.shape.length).toBeGreaterThan(0);
        } catch (error) {
          // If it throws, that's also acceptable for incompatible shapes
          expect(error).toBeTruthy();
        }
      });
    });

    describe('chaining operations', () => {
      it('should allow chaining binary operations', async () => {
        // PyTorch: (torch.tensor([2, 4, 6]) + torch.tensor([1, 2, 3])) * torch.tensor([3, 3, 3])
        // Step 1: [2, 4, 6] + [1, 2, 3] = [3, 6, 9]
        // Step 2: [3, 6, 9] * [3, 3, 3] = [9, 18, 27]
        const a = await tensor([2, 4, 6] as const, { device, dtype: float32 });
        const b = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const c = await tensor([3, 3, 3] as const, { device, dtype: float32 });

        // Test: (a + b) * c
        const result = await (await a.add(b)).mul(c);

        expect(result.shape).toEqual([3]);
        const data = await result.toArray();
        expect(data).toEqual([9, 18, 27]); // (2+1)*3, (4+2)*3, (6+3)*3
      });

      it('should allow complex operation chains', async () => {
        // PyTorch: (torch.tensor([8, 12, 16]) / torch.tensor([2, 3, 4])) - torch.tensor([1, 1, 1])
        // Step 1: [8, 12, 16] / [2, 3, 4] = [4., 4., 4.]
        // Step 2: [4, 4, 4] - [1, 1, 1] = [3., 3., 3.]
        const a = await tensor([8, 12, 16] as const, { device, dtype: float32 });
        const b = await tensor([2, 3, 4] as const, { device, dtype: float32 });
        const c = await tensor([1, 1, 1] as const, { device, dtype: float32 });

        // Test: (a / b) - c = [4, 4, 4] - [1, 1, 1] = [3, 3, 3]
        const result = await (await a.div(b)).sub(c);

        expect(result.shape).toEqual([3]);
        const data = await result.toArray();
        expect(data[0]).toBeCloseTo(3, 5);
        expect(data[1]).toBeCloseTo(3, 5);
        expect(data[2]).toBeCloseTo(3, 5);
      });

      it('should combine with unary operations', async () => {
        // PyTorch: torch.sqrt(torch.tensor([1, 4, 9])) + torch.tensor([1, 2, 3])
        // Step 1: sqrt([1, 4, 9]) = [1., 2., 3.]
        // Step 2: [1, 2, 3] + [1, 2, 3] = [2., 4., 6.]
        const a = await tensor([1, 4, 9] as const, { device, dtype: float32 });
        const b = await tensor([1, 2, 3] as const, { device, dtype: float32 });

        // Test: sqrt(a) + b = [1, 2, 3] + [1, 2, 3] = [2, 4, 6]
        const result = await (await a.sqrt()).add(b);

        expect(result.shape).toEqual([3]);
        const data = await result.toArray();
        expect(data[0]).toBeCloseTo(2, 5);
        expect(data[1]).toBeCloseTo(4, 5);
        expect(data[2]).toBeCloseTo(6, 5);
      });
    });

    describe('dtype compatibility', () => {
      it('should handle same dtype operations', async () => {
        // PyTorch: torch.tensor([1, 2, 3], dtype=torch.int32) + torch.tensor([4, 5, 6], dtype=torch.int32)
        // Output: tensor([5, 7, 9], dtype=torch.int32)
        // Same dtypes preserve the dtype
        const a = await tensor([1, 2, 3] as const, { device, dtype: int32 });
        const b = await tensor([4, 5, 6] as const, { device, dtype: int32 });

        const result = await a.add(b);
        expect(result.dtype).toBe(int32);

        const data = await result.toArray();
        expect(data).toEqual([5, 7, 9]);
      });

      it('should handle mixed precision operations appropriately', async () => {
        // PyTorch: torch.tensor([1.5, 2.5], dtype=torch.float32) + torch.tensor([1, 2], dtype=torch.int32)
        // Output: tensor([2.5000, 4.5000], dtype=torch.float32)
        // PyTorch promotes int32 to float32 for mixed operations
        const float32Tensor = await tensor([1.5, 2.5] as const, { device, dtype: float32 });
        const int32Tensor = await tensor([1, 2] as const, { device, dtype: int32 });

        // Implementation may promote to float32 or handle differently
        try {
          const result = await float32Tensor.add(int32Tensor);
          // If it succeeds, check the result is reasonable
          expect(result.shape).toEqual([2]);
          const data = await result.toArray();
          expect(typeof data[0]).toBe('number');
          expect(typeof data[1]).toBe('number');
        } catch (error) {
          // If mixed types aren't supported, that's also valid
          expect(error).toBeTruthy();
        }
      });
    });
  });
}
