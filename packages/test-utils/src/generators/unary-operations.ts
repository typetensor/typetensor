/**
 * Test generators for unary operations
 *
 * These generators test unary mathematical operations like neg, abs, sin, cos,
 * exp, log, sqrt, and square on tensors of various shapes and data types.
 */

import type { Device } from '@typetensor/core';
import { tensor, float32, int32 } from '@typetensor/core';

/**
 * Generates tests for unary operations
 *
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateUnaryOperationTests(
  device: Device,
  testFramework: {
    describe: (name: string, fn: () => void) => void;
    it: (name: string, fn: () => void | Promise<void>) => void;
    expect: (actual: unknown) => {
      toBe: (expected: unknown) => void;
      toEqual: (expected: unknown) => void;
      toBeCloseTo: (expected: number, precision?: number) => void;
      toThrow: (error?: string | RegExp) => void;
      rejects: {
        toThrow: (error?: string | RegExp) => Promise<void>;
      };
    };
  },
) {
  const { describe, it, expect } = testFramework;

  describe(`Unary Operations Tests (${device.type}:${device.id})`, () => {
    describe('negation operations', () => {
      it('should negate scalar values', async () => {
        // PyTorch: -torch.tensor(3.14)
        // Output: tensor(-3.1400)
        const scalar = await tensor(3.14, { device, dtype: float32 });
        const negated = await scalar.neg();

        expect(scalar.shape).toEqual([]);
        expect(negated.shape).toEqual([]);
        expect(negated.dtype).toBe(float32);
        expect(negated.device).toBe(device);

        const result = await negated.item();
        expect(result).toBeCloseTo(-3.14, 5);
      });

      it('should negate vector values', async () => {
        // PyTorch: -torch.tensor([1, -2, 3, -4])
        // Output: tensor([-1, 2, -3, 4])
        const vector = await tensor([1, -2, 3, -4] as const, { device, dtype: float32 });
        const negated = await vector.neg();

        expect(vector.shape).toEqual([4]);
        expect(negated.shape).toEqual([4]);
        expect(negated.dtype).toBe(float32);
        expect(negated.device).toBe(device);

        const result = await negated.toArray();
        expect(result).toEqual([-1, 2, -3, 4]);
      });

      it('should negate matrix values', async () => {
        // PyTorch: -torch.tensor([[1, -2], [-3, 4]])
        // Output: tensor([[-1, 2], [3, -4]])
        const matrix = await tensor([
          [1, -2],
          [-3, 4]
        ] as const, { device, dtype: float32 });
        const negated = await matrix.neg();

        expect(matrix.shape).toEqual([2, 2]);
        expect(negated.shape).toEqual([2, 2]);

        const result = await negated.toArray();
        expect(result).toEqual([
          [-1, 2],
          [3, -4]
        ]);
      });

      it('should handle integer negation', async () => {
        // PyTorch: -torch.tensor([1, -2, 3], dtype=torch.int32)
        // Output: tensor([-1, 2, -3], dtype=torch.int32)
        const intTensor = await tensor([1, -2, 3] as const, { device, dtype: int32 });
        const negated = await intTensor.neg();

        expect(negated.dtype).toBe(int32);
        const result = await negated.toArray();
        expect(result).toEqual([-1, 2, -3]);
      });
    });

    describe('absolute value operations', () => {
      it('should compute absolute values for scalars', async () => {
        // PyTorch: torch.abs(torch.tensor(-3.14)) = tensor(3.1400)
        //          torch.abs(torch.tensor(2.71)) = tensor(2.7100)
        const negative = await tensor(-3.14, { device, dtype: float32 });
        const positive = await tensor(2.71, { device, dtype: float32 });

        const absNegative = await negative.abs();
        const absPositive = await positive.abs();

        expect(await absNegative.item()).toBeCloseTo(3.14, 5);
        expect(await absPositive.item()).toBeCloseTo(2.71, 5);
      });

      it('should compute absolute values for vectors', async () => {
        // PyTorch: torch.abs(torch.tensor([1, -2, 0, -4.5, 3.2]))
        // Output: tensor([1.0000, 2.0000, 0.0000, 4.5000, 3.2000])
        const vector = await tensor([1, -2, 0, -4.5, 3.2] as const, { device, dtype: float32 });
        const absVector = await vector.abs();

        expect(absVector.shape).toEqual([5]);
        expect(absVector.dtype).toBe(float32);

        const result = await absVector.toArray();
        expect(result[0]).toBeCloseTo(1, 5);
        expect(result[1]).toBeCloseTo(2, 5);
        expect(result[2]).toBeCloseTo(0, 5);
        expect(result[3]).toBeCloseTo(4.5, 5);
        expect(result[4]).toBeCloseTo(3.2, 5);
      });

      it('should compute absolute values for matrices', async () => {
        // PyTorch: torch.abs(torch.tensor([[-1.5, 2.3], [0, -4.7]]))
        // Output: tensor([[1.5000, 2.3000],
        //                 [0.0000, 4.7000]])
        const matrix = await tensor([
          [-1.5, 2.3],
          [0, -4.7]
        ] as const, { device, dtype: float32 });
        const absMatrix = await matrix.abs();

        const result = await absMatrix.toArray();
        expect(result[0][0]).toBeCloseTo(1.5, 5);
        expect(result[0][1]).toBeCloseTo(2.3, 5);
        expect(result[1][0]).toBeCloseTo(0, 5);
        expect(result[1][1]).toBeCloseTo(4.7, 5);
      });
    });

    describe('trigonometric operations', () => {
      it('should compute sine values', async () => {
        // PyTorch: torch.sin(torch.tensor([0, π/2, π, 3π/2]))
        // Output: tensor([0.0000e+00, 1.0000e+00, -8.7423e-08, -1.0000e+00])
        // Note: Small floating point errors for π values
        const angles = await tensor([0, Math.PI/2, Math.PI, 3*Math.PI/2] as const, { device, dtype: float32 });
        const sines = await angles.sin();

        expect(sines.shape).toEqual([4]);
        const result = await sines.toArray();
        
        expect(result[0]).toBeCloseTo(0, 5);      // sin(0) = 0
        expect(result[1]).toBeCloseTo(1, 5);      // sin(π/2) = 1
        expect(result[2]).toBeCloseTo(0, 5);      // sin(π) = 0
        expect(result[3]).toBeCloseTo(-1, 5);     // sin(3π/2) = -1
      });

      it('should compute cosine values', async () => {
        // PyTorch: torch.cos(torch.tensor([0, π/2, π, 3π/2]))
        // Output: tensor([1.0000e+00, -4.3711e-08, -1.0000e+00, 1.1925e-08])
        // Note: Small floating point errors for π/2 and 3π/2
        const angles = await tensor([0, Math.PI/2, Math.PI, 3*Math.PI/2] as const, { device, dtype: float32 });
        const cosines = await angles.cos();

        expect(cosines.shape).toEqual([4]);
        const result = await cosines.toArray();
        
        expect(result[0]).toBeCloseTo(1, 5);      // cos(0) = 1
        expect(result[1]).toBeCloseTo(0, 5);      // cos(π/2) = 0
        expect(result[2]).toBeCloseTo(-1, 5);     // cos(π) = -1
        expect(result[3]).toBeCloseTo(0, 5);      // cos(3π/2) = 0
      });

      it('should handle trigonometric functions on matrices', async () => {
        // PyTorch: Matrix [[0, π/4], [π/2, π]]
        // sin: [[0.0000, 0.7071], [1.0000, -8.7423e-08]]
        // cos: [[1.0000, 0.7071], [-4.3711e-08, -1.0000]]
        const matrix = await tensor([
          [0, Math.PI/4],
          [Math.PI/2, Math.PI]
        ] as const, { device, dtype: float32 });

        const sines = await matrix.sin();
        const cosines = await matrix.cos();

        expect(sines.shape).toEqual([2, 2]);
        expect(cosines.shape).toEqual([2, 2]);

        const sinResult = await sines.toArray();
        const cosResult = await cosines.toArray();

        // Check some key values
        expect(sinResult[0][0]).toBeCloseTo(0, 5);           // sin(0)
        expect(sinResult[1][0]).toBeCloseTo(1, 5);           // sin(π/2)
        expect(cosResult[0][0]).toBeCloseTo(1, 5);           // cos(0)
        expect(cosResult[1][1]).toBeCloseTo(-1, 5);          // cos(π)
      });
    });

    describe('exponential and logarithmic operations', () => {
      it('should compute exponential values', async () => {
        // PyTorch: torch.exp(torch.tensor([0, 1, 2, -1]))
        // Output: tensor([1.0000, 2.7183, 7.3891, 0.3679])
        const values = await tensor([0, 1, 2, -1] as const, { device, dtype: float32 });
        const exponentials = await values.exp();

        expect(exponentials.shape).toEqual([4]);
        const result = await exponentials.toArray();

        expect(result[0]).toBeCloseTo(1, 5);           // e^0 = 1
        expect(result[1]).toBeCloseTo(Math.E, 5);      // e^1 = e
        expect(result[2]).toBeCloseTo(Math.E * Math.E, 5); // e^2
        expect(result[3]).toBeCloseTo(1/Math.E, 5);    // e^(-1) = 1/e
      });

      it('should compute logarithmic values', async () => {
        // PyTorch: torch.log(torch.tensor([1, e, e^2, 1/e]))
        // Output: tensor([0.0000, 1.0000, 2.0000, -1.0000])
        const values = await tensor([1, Math.E, Math.E * Math.E, 1/Math.E] as const, { device, dtype: float32 });
        const logarithms = await values.log();

        expect(logarithms.shape).toEqual([4]);
        const result = await logarithms.toArray();

        expect(result[0]).toBeCloseTo(0, 5);      // ln(1) = 0
        expect(result[1]).toBeCloseTo(1, 5);      // ln(e) = 1
        expect(result[2]).toBeCloseTo(2, 5);      // ln(e^2) = 2
        expect(result[3]).toBeCloseTo(-1, 5);     // ln(1/e) = -1
      });

      it('should handle exp/log on matrices', async () => {
        // PyTorch: torch.exp(torch.tensor([[0, 1], [2, -1]]))
        // Output: tensor([[1.0000, 2.7183],
        //                 [7.3891, 0.3679]])
        const matrix = await tensor([
          [0, 1],
          [2, -1]
        ] as const, { device, dtype: float32 });

        const exponentials = await matrix.exp();
        const result = await exponentials.toArray();

        expect(result[0][0]).toBeCloseTo(1, 5);           // e^0
        expect(result[0][1]).toBeCloseTo(Math.E, 5);      // e^1
        expect(result[1][0]).toBeCloseTo(Math.E * Math.E, 5); // e^2
        expect(result[1][1]).toBeCloseTo(1/Math.E, 5);    // e^(-1)
      });
    });

    describe('square root operations', () => {
      it('should compute square roots for scalars', async () => {
        // PyTorch: torch.sqrt(torch.tensor(9.0))
        // Output: tensor(3.0)
        const scalar = await tensor(9, { device, dtype: float32 });
        const sqrt = await scalar.sqrt();

        expect(await sqrt.item()).toBeCloseTo(3, 5);
      });

      it('should compute square roots for vectors', async () => {
        // PyTorch: torch.sqrt(torch.tensor([1, 4, 9, 16, 25]))
        // Output: tensor([1., 2., 3., 4., 5.])
        const vector = await tensor([1, 4, 9, 16, 25] as const, { device, dtype: float32 });
        const sqrts = await vector.sqrt();

        expect(sqrts.shape).toEqual([5]);
        const result = await sqrts.toArray();

        expect(result[0]).toBeCloseTo(1, 5);
        expect(result[1]).toBeCloseTo(2, 5);
        expect(result[2]).toBeCloseTo(3, 5);
        expect(result[3]).toBeCloseTo(4, 5);
        expect(result[4]).toBeCloseTo(5, 5);
      });

      it('should compute square roots for matrices', async () => {
        // PyTorch: torch.sqrt(torch.tensor([[1, 4], [9, 16]]))
        // Output: tensor([[1., 2.],
        //                 [3., 4.]])
        const matrix = await tensor([
          [1, 4],
          [9, 16]
        ] as const, { device, dtype: float32 });
        const sqrts = await matrix.sqrt();

        const result = await sqrts.toArray();
        expect(result[0][0]).toBeCloseTo(1, 5);
        expect(result[0][1]).toBeCloseTo(2, 5);
        expect(result[1][0]).toBeCloseTo(3, 5);
        expect(result[1][1]).toBeCloseTo(4, 5);
      });
    });

    describe('square operations', () => {
      it('should compute squares for scalars', async () => {
        // PyTorch: torch.square(torch.tensor(3.0))
        // Output: tensor(9.0)
        const scalar = await tensor(3, { device, dtype: float32 });
        const squared = await scalar.square();

        expect(await squared.item()).toBeCloseTo(9, 5);
      });

      it('should compute squares for vectors', async () => {
        // PyTorch: torch.square(torch.tensor([1, -2, 3, -4, 5]))
        // Output: tensor([1., 4., 9., 16., 25.])
        const vector = await tensor([1, -2, 3, -4, 5] as const, { device, dtype: float32 });
        const squares = await vector.square();

        expect(squares.shape).toEqual([5]);
        const result = await squares.toArray();

        expect(result[0]).toBeCloseTo(1, 5);
        expect(result[1]).toBeCloseTo(4, 5);
        expect(result[2]).toBeCloseTo(9, 5);
        expect(result[3]).toBeCloseTo(16, 5);
        expect(result[4]).toBeCloseTo(25, 5);
      });

      it('should compute squares for matrices', async () => {
        // PyTorch: torch.square(torch.tensor([[1, -2], [3, -4]]))
        // Output: tensor([[1., 4.],
        //                 [9., 16.]])
        const matrix = await tensor([
          [1, -2],
          [3, -4]
        ] as const, { device, dtype: float32 });
        const squares = await matrix.square();

        const result = await squares.toArray();
        expect(result[0][0]).toBeCloseTo(1, 5);
        expect(result[0][1]).toBeCloseTo(4, 5);
        expect(result[1][0]).toBeCloseTo(9, 5);
        expect(result[1][1]).toBeCloseTo(16, 5);
      });
    });

    describe('property preservation', () => {
      it('should preserve tensor metadata across unary operations', async () => {
        const original = await tensor([
          [1, 2, 3],
          [4, 5, 6]
        ] as const, { device, dtype: float32 });

        const neg = await original.neg();
        const abs = await original.abs();
        const sqrt = await original.sqrt();

        // All results should have same shape, dtype, device
        expect(neg.shape).toEqual([2, 3]);
        expect(abs.shape).toEqual([2, 3]);
        expect(sqrt.shape).toEqual([2, 3]);

        expect(neg.dtype).toBe(float32);
        expect(abs.dtype).toBe(float32);
        expect(sqrt.dtype).toBe(float32);

        expect(neg.device).toBe(device);
        expect(abs.device).toBe(device);
        expect(sqrt.device).toBe(device);

        expect(neg.size).toBe(6);
        expect(abs.size).toBe(6);
        expect(sqrt.size).toBe(6);
      });

      it('should work with different tensor shapes', async () => {
        // Test scalar
        const scalar = await tensor(4, { device, dtype: float32 });
        const scalarSqrt = await scalar.sqrt();
        expect(scalarSqrt.shape).toEqual([]);
        expect(await scalarSqrt.item()).toBeCloseTo(2, 5);

        // Test 3D tensor
        const tensor3d = await tensor([
          [[1, 4]],
          [[9, 16]]
        ] as const, { device, dtype: float32 });
        const tensor3dSqrt = await tensor3d.sqrt();
        expect(tensor3dSqrt.shape).toEqual([2, 1, 2]);
        
        const result3d = await tensor3dSqrt.toArray();
        expect(result3d[0][0][0]).toBeCloseTo(1, 5);
        expect(result3d[0][0][1]).toBeCloseTo(2, 5);
        expect(result3d[1][0][0]).toBeCloseTo(3, 5);
        expect(result3d[1][0][1]).toBeCloseTo(4, 5);
      });
    });

    describe('error handling', () => {
      it('should handle mathematical domain errors gracefully', async () => {
        // PyTorch: torch.sqrt(torch.tensor([-1, -4, -9]))
        // Output: tensor([nan, nan, nan])
        // Note: Square root of negative numbers produces NaN
        const negativeValues = await tensor([-1, -4, -9] as const, { device, dtype: float32 });
        
        // Note: Implementation may handle this differently (NaN, error, etc.)
        // This test ensures it doesn't crash and produces some result
        const sqrts = await negativeValues.sqrt();
        expect(sqrts.shape).toEqual([3]);
        
        // The actual values might be NaN - that's acceptable behavior
        const result = await sqrts.toArray();
        expect(Array.isArray(result)).toBe(true);
        expect(result.length).toBe(3);
      });

      it('should handle log of non-positive numbers gracefully', async () => {
        // PyTorch: torch.log(torch.tensor([0, -1, -5]))
        // Output: tensor([-inf, nan, nan])
        // log(0) = -inf, log(negative) = nan
        const problematicValues = await tensor([0, -1, -5] as const, { device, dtype: float32 });
        
        // Implementation may return -Infinity for log(0), NaN for log(negative)
        const logs = await problematicValues.log();
        expect(logs.shape).toEqual([3]);
        
        const result = await logs.toArray();
        expect(Array.isArray(result)).toBe(true);
        expect(result.length).toBe(3);
      });
    });

    describe('chaining operations', () => {
      it('should allow chaining unary operations', async () => {
        // PyTorch: torch.sqrt(torch.square(torch.tensor([1, 4, 9])))
        // Output: tensor([1., 4., 9.])
        // square then sqrt returns to original for positive values
        const original = await tensor([1, 4, 9] as const, { device, dtype: float32 });
        const chained = await (await original.square()).sqrt();

        expect(chained.shape).toEqual([3]);
        const result = await chained.toArray();
        
        expect(result[0]).toBeCloseTo(1, 5);
        expect(result[1]).toBeCloseTo(4, 5);
        expect(result[2]).toBeCloseTo(9, 5);
      });

      it('should allow complex operation chains', async () => {
        // PyTorch: torch.square(torch.sqrt(torch.abs(torch.tensor([-4, -9, -16]))))
        // Output: tensor([4., 9., 16.])
        // abs → sqrt → square preserves absolute values
        const original = await tensor([-4, -9, -16] as const, { device, dtype: float32 });
        const chained = await (await (await original.abs()).sqrt()).square();

        expect(chained.shape).toEqual([3]);
        const result = await chained.toArray();
        
        expect(result[0]).toBeCloseTo(4, 5);
        expect(result[1]).toBeCloseTo(9, 5);
        expect(result[2]).toBeCloseTo(16, 5);
      });
    });
  });
}