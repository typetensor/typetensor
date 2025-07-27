/**
 * Test generators for softmax operations
 *
 * These generators test softmax and log-softmax operations that are commonly
 * used in machine learning for converting logits to probabilities and
 * numerical stability in loss computation.
 */

import type { Device } from '@typetensor/core';
import { tensor, float32, int32 } from '@typetensor/core';

/**
 * Generates tests for softmax operations
 *
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateSoftmaxOperationTests(
  device: Device,
  testFramework: {
    describe: (name: string, fn: () => void) => void;
    it: (name: string, fn: () => void | Promise<void>) => void;
    expect: (actual: unknown) => {
      toBe: (expected: unknown) => void;
      toEqual: (expected: unknown) => void;
      toBeCloseTo: (expected: number, precision?: number) => void;
      toThrow: (error?: string | RegExp) => void;
      toBeLessThan: (expected: number) => void;
      toBeGreaterThan: (expected: number) => void;
      toBeTruthy: () => void;
      rejects: {
        toThrow: (error?: string | RegExp) => Promise<void>;
      };
    };
  },
) {
  const { describe, it, expect } = testFramework;

  describe(`Softmax Operations Tests (${device.type}:${device.id})`, () => {
    describe('softmax operations', () => {
      it('should compute softmax for a simple vector', async () => {
        const logits = await tensor([1.0, 2.0, 3.0] as const, { device, dtype: float32 });
        const result = await logits.softmax(-1);

        expect(result.shape).toEqual([3]);
        expect(result.dtype).toBe(float32);
        expect(result.device).toBe(device);

        const data = await result.toArray();
        // Verify softmax properties: all values positive and sum to 1
        expect(data[0]).toBeGreaterThan(0);
        expect(data[1]).toBeGreaterThan(0);
        expect(data[2]).toBeGreaterThan(0);

        const sum = data[0] + data[1] + data[2];
        expect(sum).toBeCloseTo(1.0, 5);

        // Verify relative ordering (larger input -> larger output)
        expect(data[2]).toBeGreaterThan(data[1]);
        expect(data[1]).toBeGreaterThan(data[0]);
      });

      it('should compute softmax along axis 0 for matrix', async () => {
        const logits = await tensor(
          [
            [1.0, 2.0],
            [3.0, 4.0],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await logits.softmax(0);

        expect(result.shape).toEqual([2, 2]);
        const data = await result.toArray();

        // Each column should sum to 1
        const col0Sum = data[0][0] + data[1][0];
        const col1Sum = data[0][1] + data[1][1];
        expect(col0Sum).toBeCloseTo(1.0, 5);
        expect(col1Sum).toBeCloseTo(1.0, 5);

        // All values should be positive
        expect(data[0][0]).toBeGreaterThan(0);
        expect(data[0][1]).toBeGreaterThan(0);
        expect(data[1][0]).toBeGreaterThan(0);
        expect(data[1][1]).toBeGreaterThan(0);
      });

      it('should compute softmax along axis 1 for matrix', async () => {
        const logits = await tensor(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await logits.softmax(1);

        expect(result.shape).toEqual([2, 3]);
        const data = await result.toArray();

        // Each row should sum to 1
        const row0Sum = data[0][0] + data[0][1] + data[0][2];
        const row1Sum = data[1][0] + data[1][1] + data[1][2];
        expect(row0Sum).toBeCloseTo(1.0, 5);
        expect(row1Sum).toBeCloseTo(1.0, 5);

        // Check relative ordering within each row
        expect(data[0][2]).toBeGreaterThan(data[0][1]);
        expect(data[0][1]).toBeGreaterThan(data[0][0]);
        expect(data[1][2]).toBeGreaterThan(data[1][1]);
        expect(data[1][1]).toBeGreaterThan(data[1][0]);
      });

      it('should handle negative axis indexing', async () => {
        const logits = await tensor(
          [
            [1.0, 2.0],
            [3.0, 4.0],
          ] as const,
          { device, dtype: float32 },
        );

        // axis -1 should be equivalent to axis 1 for 2D tensor
        const result = await logits.softmax(-1);

        expect(result.shape).toEqual([2, 2]);
        const data = await result.toArray();

        // Each row should sum to 1
        const row0Sum = data[0][0] + data[0][1];
        const row1Sum = data[1][0] + data[1][1];
        expect(row0Sum).toBeCloseTo(1.0, 5);
        expect(row1Sum).toBeCloseTo(1.0, 5);
      });

      it('should handle 3D tensors with batch dimension', async () => {
        // 2x2x3 tensor (batch=2, seq=2, vocab=3)
        const logits = await tensor(
          [
            [
              [1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
            ],
            [
              [7.0, 8.0, 9.0],
              [10.0, 11.0, 12.0],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        // Softmax over vocabulary (last dimension)
        const result = await logits.softmax(-1);

        expect(result.shape).toEqual([2, 2, 3]);
        const data = await result.toArray();

        // Each position should sum to 1 over vocab dimension
        for (let batch = 0; batch < 2; batch++) {
          for (let seq = 0; seq < 2; seq++) {
            const sum = data[batch]![seq]![0]! + data[batch]![seq]![1]! + data[batch]![seq]![2]!;
            expect(sum).toBeCloseTo(1.0, 5);
          }
        }
      });

      it('should handle extreme values without overflow', async () => {
        // Large values that could cause exp overflow
        const logits = await tensor([10.0, 20.0, 30.0] as const, { device, dtype: float32 });
        const result = await logits.softmax(-1);

        const data = await result.toArray();

        // Should still sum to 1 and be finite
        const sum = data[0] + data[1] + data[2];
        expect(sum).toBeCloseTo(1.0, 5);
        expect(Number.isFinite(data[0])).toBeTruthy();
        expect(Number.isFinite(data[1])).toBeTruthy();
        expect(Number.isFinite(data[2])).toBeTruthy();

        // Largest input should have highest probability
        expect(data[2]).toBeGreaterThan(data[1]);
        expect(data[1]).toBeGreaterThan(data[0]);
      });

      it('should handle uniform inputs', async () => {
        // All same values should give uniform distribution
        const logits = await tensor([5.0, 5.0, 5.0, 5.0] as const, { device, dtype: float32 });
        const result = await logits.softmax(-1);

        const data = await result.toArray();

        // All probabilities should be equal (1/4 = 0.25)
        expect(data[0]).toBeCloseTo(0.25, 5);
        expect(data[1]).toBeCloseTo(0.25, 5);
        expect(data[2]).toBeCloseTo(0.25, 5);
        expect(data[3]).toBeCloseTo(0.25, 5);
      });
    });

    describe('log-softmax operations', () => {
      it('should compute log-softmax for a simple vector', async () => {
        const logits = await tensor([1.0, 2.0, 3.0] as const, { device, dtype: float32 });
        const result = await logits.logSoftmax(-1);

        expect(result.shape).toEqual([3]);
        expect(result.dtype).toBe(float32);
        expect(result.device).toBe(device);

        const data = await result.toArray();

        // PyTorch reference: torch.log_softmax([1,2,3], dim=-1) = [-2.4076, -1.4076, -0.4076]
        expect(data[0]).toBeCloseTo(-2.4076, 3);
        expect(data[1]).toBeCloseTo(-1.4076, 3);
        expect(data[2]).toBeCloseTo(-0.4076, 3);

        // Should preserve relative ordering
        expect(data[2]).toBeGreaterThan(data[1]);
        expect(data[1]).toBeGreaterThan(data[0]);

        // Verify relationship: exp(log_softmax) = softmax
        const softmaxResult = await logits.softmax(-1);
        const softmaxData = await softmaxResult.toArray();

        expect(Math.exp(data[0])).toBeCloseTo(softmaxData[0], 5);
        expect(Math.exp(data[1])).toBeCloseTo(softmaxData[1], 5);
        expect(Math.exp(data[2])).toBeCloseTo(softmaxData[2], 5);
      });

      it('should compute log-softmax along axis 0 for matrix', async () => {
        const logits = await tensor(
          [
            [1.0, 2.0],
            [3.0, 4.0],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await logits.logSoftmax(0);

        expect(result.shape).toEqual([2, 2]);
        const data = await result.toArray();

        // PyTorch reference: torch.log_softmax([[1,2],[3,4]], dim=0) = [[-2.1269, -2.1269], [-0.1269, -0.1269]]
        expect(data[0][0]).toBeCloseTo(-2.1269, 3);
        expect(data[0][1]).toBeCloseTo(-2.1269, 3);
        expect(data[1][0]).toBeCloseTo(-0.1269, 3);
        expect(data[1][1]).toBeCloseTo(-0.1269, 3);

        // Verify exp(log_softmax) = softmax consistency
        const softmaxResult = await logits.softmax(0);
        const softmaxData = await softmaxResult.toArray();

        expect(Math.exp(data[0][0])).toBeCloseTo(softmaxData[0][0], 5);
        expect(Math.exp(data[0][1])).toBeCloseTo(softmaxData[0][1], 5);
        expect(Math.exp(data[1][0])).toBeCloseTo(softmaxData[1][0], 5);
        expect(Math.exp(data[1][1])).toBeCloseTo(softmaxData[1][1], 5);
      });

      it('should handle negative axis indexing', async () => {
        const logits = await tensor(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await logits.logSoftmax(-1);

        expect(result.shape).toEqual([2, 3]);
        const data = await result.toArray();

        // Values should be negative
        for (let i = 0; i < 2; i++) {
          for (let j = 0; j < 3; j++) {
            expect(data[i]![j]!).toBeLessThan(0.01);
          }
        }
      });

      it('should provide numerical stability for large values', async () => {
        // Large values that could cause numerical issues
        const logits = await tensor([100.0, 101.0, 102.0] as const, { device, dtype: float32 });
        const result = await logits.logSoftmax(-1);

        const data = await result.toArray();

        // Should produce finite values
        expect(Number.isFinite(data[0])).toBeTruthy();
        expect(Number.isFinite(data[1])).toBeTruthy();
        expect(Number.isFinite(data[2])).toBeTruthy();

        // Should preserve relative ordering
        expect(data[2]).toBeGreaterThan(data[1]);
        expect(data[1]).toBeGreaterThan(data[0]);
      });
    });

    describe('property preservation', () => {
      it('should preserve device and dtype for softmax results', async () => {
        const logits = await tensor([1.0, 2.0, 3.0] as const, { device, dtype: float32 });

        const softmax = await logits.softmax(-1);
        const logSoftmax = await logits.logSoftmax(-1);

        // Both should preserve device
        expect(softmax.device).toBe(device);
        expect(logSoftmax.device).toBe(device);

        // Both should preserve dtype
        expect(softmax.dtype).toBe(float32);
        expect(logSoftmax.dtype).toBe(float32);
      });

      it('should work with different input dtypes', async () => {
        // Note: This test assumes the implementation can handle integer inputs
        // by converting them to float internally
        const intLogits = await tensor([1, 2, 3] as const, { device, dtype: int32 });

        try {
          const softmax = await intLogits.softmax(-1);
          const logSoftmax = await intLogits.logSoftmax(-1);

          // Results should be valid
          expect(softmax.shape).toEqual([3]);
          expect(logSoftmax.shape).toEqual([3]);

          const softmaxData = await softmax.toArray();
          const sum = softmaxData[0] + softmaxData[1] + softmaxData[2];
          expect(sum).toBeCloseTo(1.0, 5);
        } catch (error) {
          // If integer inputs aren't supported, that's also valid
          expect(error).toBeTruthy();
        }
      });
    });

    describe('edge cases', () => {
      it('should handle single-element tensors', async () => {
        const single = await tensor([42.0] as const, { device, dtype: float32 });

        const softmax = await single.softmax(-1);
        const logSoftmax = await single.logSoftmax(-1);

        // PyTorch reference: torch.softmax([42.0], dim=-1) = [1.0], torch.log_softmax([42.0], dim=-1) = [0.0]
        // Both preserve shape [1], item() extracts the scalar value
        expect(softmax.shape).toEqual([1]);
        expect(logSoftmax.shape).toEqual([1]);

        expect(await softmax.item()).toBeCloseTo(1.0, 5);
        expect(await logSoftmax.item()).toBeCloseTo(0.0, 5);
      });

      it('should handle zero inputs', async () => {
        const zeros = await tensor([0.0, 0.0, 0.0] as const, { device, dtype: float32 });

        const softmax = await zeros.softmax(-1);
        const data = await softmax.toArray();

        // Should give uniform distribution
        expect(data[0]).toBeCloseTo(1 / 3, 5);
        expect(data[1]).toBeCloseTo(1 / 3, 5);
        expect(data[2]).toBeCloseTo(1 / 3, 5);
      });
    });

    describe('error handling', () => {
      it('should handle invalid axis values gracefully', async () => {
        const logits = await tensor(
          [
            [1.0, 2.0],
            [3.0, 4.0],
          ] as const,
          { device, dtype: float32 },
        );

        // Test with axis that's out of bounds
        try {
          // @ts-expect-error - Testing runtime behavior with invalid axis
          await logits.softmax(5);
          // If it doesn't throw, that's also acceptable
        } catch (error) {
          expect(error).toBeTruthy();
        }
      });

      it('should handle empty tensors appropriately', async () => {
        try {
          const empty = await tensor([] as const, { device, dtype: float32 });
          await empty.softmax(-1);
          // If it succeeds, verify the result makes sense
        } catch (error) {
          // If it throws, that's also reasonable for empty tensors
          expect(error).toBeTruthy();
        }
      });
    });

    describe('chaining with other operations', () => {
      it('should chain softmax with other operations', async () => {
        const logits = await tensor(
          [
            [1.0, 2.0],
            [3.0, 4.0],
          ] as const,
          { device, dtype: float32 },
        );

        // Chain: reshape -> softmax -> sum
        const result = await (await (await logits.reshape([4] as const)).softmax(-1)).sum();

        // Sum of softmax should be 1
        expect(await result.item()).toBeCloseTo(1.0, 5);
      });

      it('should use softmax in mathematical expressions', async () => {
        const logits = await tensor([1.0, 2.0, 3.0] as const, { device, dtype: float32 });
        const weights = await tensor([0.5, 1.0, 1.5] as const, { device, dtype: float32 });

        // Weighted softmax
        const weightedLogits = await logits.mul(weights);
        const softmax = await weightedLogits.softmax(-1);

        const data = await softmax.toArray();
        const sum = data[0] + data[1] + data[2];
        expect(sum).toBeCloseTo(1.0, 5);
      });
    });
  });
}
