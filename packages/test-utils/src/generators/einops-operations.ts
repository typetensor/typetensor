/**
 * Test generators for einops operations
 *
 * These generators test einops-style tensor rearrangement operations using
 * intuitive string patterns for dimension manipulation. Tests cover basic
 * operations, common ML patterns, edge cases, and error handling.
 */

import type { Device } from '@typetensor/core';
import { tensor, float32, int32, rearrange } from '@typetensor/core';

/**
 * Generates tests for einops operations
 *
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateEinopsOperationTests(
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

  describe(`Einops Operations Tests (${device.type}:${device.id})`, () => {
    describe('basic axis operations', () => {
      it('should handle identity operations', async () => {
        // PyTorch einops: rearrange(tensor, 'a b c -> a b c')
        // Identity operation should not change the tensor
        const t = await tensor(
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

        const result = await rearrange(t, 'a b c -> a b c');

        expect(result.shape).toEqual([2, 2, 2]);
        expect(result.dtype).toBe(float32);
        expect(result.device).toBe(device);

        const data = await result.toArray();
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

      it('should transpose 2D tensors', async () => {
        // PyTorch einops: rearrange(tensor([[1,2,3],[4,5,6]]), 'h w -> w h')
        // Output: tensor([[1, 4],
        //                 [2, 5],
        //                 [3, 6]])
        const matrix = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        const transposed = await rearrange(matrix, 'height width -> width height');

        expect(transposed.shape).toEqual([3, 2]);
        const data = await transposed.toArray();
        expect(data).toEqual([
          [1, 4],
          [2, 5],
          [3, 6],
        ]);
      });

      it('should handle 3D permutations', async () => {
        // PyTorch einops: rearrange(tensor, 'a b c -> c b a')
        const t = await tensor(
          [
            [
              [1, 2],
              [3, 4],
              [5, 6],
            ],
            [
              [7, 8],
              [9, 10],
              [11, 12],
            ],
          ] as const,
          { device, dtype: int32 },
        );

        const permuted = await rearrange(t, 'a b c -> c b a');

        expect(permuted.shape).toEqual([2, 3, 2]);
        const data = await permuted.toArray();
        expect(data[0][0][0]).toBe(1);
        expect(data[0][0][1]).toBe(7);
        expect(data[1][0][0]).toBe(2);
        expect(data[1][0][1]).toBe(8);
      });

      it('should handle axis reordering', async () => {
        // All axes from input must appear in output with same names
        const t = await tensor(
          [
            [
              [1, 2, 3],
              [4, 5, 6],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        // Reorder dimensions - same axes, different order
        const reordered = await rearrange(t, 'batch seq dim -> seq batch dim');

        expect(reordered.shape).toEqual([2, 1, 3]);
        const data = await reordered.toArray();
        expect(data).toEqual([[[1, 2, 3]], [[4, 5, 6]]]);
      });
    });

    describe('dimension addition', () => {
      it('should add dimension at the beginning', async () => {
        // PyTorch einops: rearrange(tensor([[1,2],[3,4]]), 'h w -> 1 h w')
        // Output shape: torch.Size([1, 2, 2])
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        const with_batch = await rearrange(matrix, 'h w -> 1 h w');

        expect(with_batch.shape).toEqual([1, 2, 2]);
        const data = await with_batch.toArray();
        expect(data).toEqual([
          [
            [1, 2],
            [3, 4],
          ],
        ]);
      });

      it('should add dimension in the middle', async () => {
        // PyTorch einops: rearrange(tensor([[1,2],[3,4]]), 'h w -> h 1 w')
        const matrix = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        const expanded = await rearrange(matrix, 'h w -> h 1 w');

        expect(expanded.shape).toEqual([2, 1, 2]);
        const data = await expanded.toArray();
        expect(data).toEqual([[[1, 2]], [[3, 4]]]);
      });

      it('should add multiple dimensions', async () => {
        // PyTorch einops: rearrange(tensor([1,2,3]), 'w -> 1 1 w 1')
        const vector = await tensor([1, 2, 3] as const, { device, dtype: float32 });

        const expanded = await rearrange(vector, 'w -> 1 1 w 1');

        expect(expanded.shape).toEqual([1, 1, 3, 1]);
        const data = await expanded.toArray();
        expect(data).toEqual([[[[1], [2], [3]]]]);
      });
    });

    describe('dimension removal', () => {
      it('should remove single dimensions', async () => {
        // PyTorch einops: rearrange(tensor([[[1,2],[3,4]]]), '1 h w -> h w')
        const t = await tensor(
          [
            [
              [1, 2],
              [3, 4],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        const squeezed = await rearrange(t, '1 h w -> h w');

        expect(squeezed.shape).toEqual([2, 2]);
        const data = await squeezed.toArray();
        expect(data).toEqual([
          [1, 2],
          [3, 4],
        ]);
      });

      it('should remove multiple single dimensions', async () => {
        // PyTorch einops: tensor with shape [1, 2, 1, 3, 1] -> [2, 3]
        const t = await tensor([[[[[1], [2], [3]]], [[[4], [5], [6]]]]] as const, {
          device,
          dtype: float32,
        });

        const squeezed = await rearrange(t, '1 h 1 w 1 -> h w');

        expect(squeezed.shape).toEqual([2, 3]);
        const data = await squeezed.toArray();
        expect(data).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);
      });
    });

    describe('dimension composition (merging)', () => {
      it('should merge adjacent dimensions', async () => {
        // PyTorch einops: rearrange(tensor of shape [2, 3, 4], 'a b c -> (a b) c')
        // Output shape: [6, 4]
        const t = await tensor(
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

        const merged = await rearrange(t, 'a b c -> (a b) c');

        expect(merged.shape).toEqual([6, 4]);
        const data = await merged.toArray();
        expect(data[0]).toEqual([1, 2, 3, 4]);
        expect(data[1]).toEqual([5, 6, 7, 8]);
        expect(data[2]).toEqual([9, 10, 11, 12]);
        expect(data[3]).toEqual([13, 14, 15, 16]);
      });

      it('should merge multiple dimension groups', async () => {
        // PyTorch einops: rearrange(tensor of shape [2, 3, 4, 5], 'a b c d -> (a b) (c d)')
        const t = await tensor(
          [
            [
              [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
              ],
              [
                [21, 22, 23, 24, 25],
                [26, 27, 28, 29, 30],
                [31, 32, 33, 34, 35],
                [36, 37, 38, 39, 40],
              ],
              [
                [41, 42, 43, 44, 45],
                [46, 47, 48, 49, 50],
                [51, 52, 53, 54, 55],
                [56, 57, 58, 59, 60],
              ],
            ],
            [
              [
                [61, 62, 63, 64, 65],
                [66, 67, 68, 69, 70],
                [71, 72, 73, 74, 75],
                [76, 77, 78, 79, 80],
              ],
              [
                [81, 82, 83, 84, 85],
                [86, 87, 88, 89, 90],
                [91, 92, 93, 94, 95],
                [96, 97, 98, 99, 100],
              ],
              [
                [101, 102, 103, 104, 105],
                [106, 107, 108, 109, 110],
                [111, 112, 113, 114, 115],
                [116, 117, 118, 119, 120],
              ],
            ],
          ] as const,
          { device, dtype: int32 },
        );

        const merged = await rearrange(t, 'a b c d -> (a b) (c d)');

        expect(merged.shape).toEqual([6, 20]);
        expect(merged.size).toBe(120);
      });

      it('should fully flatten tensor', async () => {
        // PyTorch einops: rearrange(tensor of shape [2, 3, 4], 'a b c -> (a b c)')
        const t = await tensor(
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

        const flattened = await rearrange(t, 'a b c -> (a b c)');

        expect(flattened.shape).toEqual([24]);
        const data = await flattened.toArray();
        expect(data[0]).toBe(1);
        expect(data[23]).toBe(24);
      });
    });

    describe('dimension decomposition (splitting)', () => {
      it('should split dimension with explicit sizes', async () => {
        // PyTorch einops: rearrange(tensor([[1,2,3,4,5,6],[7,8,9,10,11,12]]),
        //                          'batch (h w) -> batch h w', h=2, w=3)
        const t = await tensor(
          [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
          ] as const,
          { device, dtype: float32 },
        );

        const split = await rearrange(t, 'batch (h w) -> batch h w', { h: 2, w: 3 });

        expect(split.shape).toEqual([2, 2, 3]);
        const data = await split.toArray();
        expect(data).toEqual([
          [
            [1, 2, 3],
            [4, 5, 6],
          ],
          [
            [7, 8, 9],
            [10, 11, 12],
          ],
        ]);
      });

      it('should split dimension with one size inferred', async () => {
        // PyTorch einops: rearrange(tensor([1,2,3,4,5,6,7,8,9,10,11,12]),
        //                          '(h w) -> h w', h=3)
        // w is inferred as 4
        const t = await tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as const, {
          device,
          dtype: float32,
        });

        const split = await rearrange(t, '(h w) -> h w', { h: 3 });

        expect(split.shape).toEqual([3, 4]);
        const data = await split.toArray();
        expect(data).toEqual([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12],
        ]);
      });

      it('should handle multiple splits', async () => {
        // PyTorch einops: tensor of shape [24] split into [2, 3, 4]
        const t = await tensor(
          [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          ] as const,
          { device, dtype: int32 },
        );

        const split = await rearrange(t, '(a b c) -> a b c', { a: 2, b: 3, c: 4 });

        expect(split.shape).toEqual([2, 3, 4]);
        expect(split.size).toBe(24);
      });
    });

    describe('complex rearrangements', () => {
      it('should handle split and reorder simultaneously', async () => {
        // PyTorch einops: 
        // >>> t = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]], dtype=torch.float32)
        // >>> result = rearrange(t, 'batch (h w) c -> batch c h w', h=2)
        // >>> result.shape
        // torch.Size([1, 3, 2, 2])
        // >>> result
        // tensor([[[[ 1.,  4.],
        //           [ 7., 10.]],
        //
        //          [[ 2.,  5.],
        //           [ 8., 11.]],
        //
        //          [[ 3.,  6.],
        //           [ 9., 12.]]]])
        const t = await tensor(
          [
            [
              [1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await rearrange(t, 'batch (h w) c -> batch c h w', { h: 2 });

        expect(result.shape).toEqual([1, 3, 2, 2]);
        const data = await result.toArray();
        expect(data).toEqual([
          [
            [
              [1, 4],
              [7, 10],
            ],
            [
              [2, 5],
              [8, 11],
            ],
            [
              [3, 6],
              [9, 12],
            ],
          ],
        ]);
      });

      it('should handle merge and reorder simultaneously', async () => {
        // PyTorch einops: rearrange(tensor, 'b h w c -> b (h w c)')
        const t = await tensor(
          [
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
          ] as const,
          { device, dtype: float32 },
        );

        const result = await rearrange(t, 'b h w c -> b (h w c)');

        expect(result.shape).toEqual([1, 8]);
        const data = await result.toArray();
        expect(data).toEqual([[1, 2, 3, 4, 5, 6, 7, 8]]);
      });
    });

    describe('common ML patterns', () => {
      it('should convert CHW to HWC format', async () => {
        // PyTorch einops: rearrange(image_chw, 'c h w -> h w c')
        const chw = await tensor(
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

        const hwc = await rearrange(chw, 'c h w -> h w c');

        expect(hwc.shape).toEqual([2, 2, 3]);
        const data = await hwc.toArray();
        expect(data).toEqual([
          [
            [1, 5, 9],
            [2, 6, 10],
          ],
          [
            [3, 7, 11],
            [4, 8, 12],
          ],
        ]);
      });

      it('should prepare multi-head attention', async () => {
        // PyTorch einops: rearrange(embeddings, 'b s (h d) -> b h s d', h=4)
        const embeddings = await tensor(
          [
            [
              [1, 2, 3, 4, 5, 6, 7, 8],
              [9, 10, 11, 12, 13, 14, 15, 16],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        const multihead = await rearrange(
          embeddings,
          'batch seq (heads dim) -> batch heads seq dim',
          {
            heads: 4,
          },
        );

        expect(multihead.shape).toEqual([1, 4, 2, 2]);
        const data = await multihead.toArray();
        expect(data).toEqual([
          [
            [
              [1, 2],
              [9, 10],
            ],
            [
              [3, 4],
              [11, 12],
            ],
            [
              [5, 6],
              [13, 14],
            ],
            [
              [7, 8],
              [15, 16],
            ],
          ],
        ]);
      });

      it('should handle patch extraction pattern', async () => {
        // PyTorch einops: Extract 2x2 patches from 4x4 image
        // rearrange(image, '(h ph) (w pw) c -> h w (ph pw c)', ph=2, pw=2)
        const image = await tensor(
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
          ] as const,
          { device, dtype: float32 },
        );

        const patches = await rearrange(image, '(h ph) (w pw) -> h w (ph pw)', { ph: 2, pw: 2 });

        expect(patches.shape).toEqual([2, 2, 4]);
        const data = await patches.toArray();
        expect(data).toEqual([
          [
            [1, 2, 5, 6],
            [3, 4, 7, 8],
          ],
          [
            [9, 10, 13, 14],
            [11, 12, 15, 16],
          ],
        ]);
      });

      it('should handle grouped convolution pattern', async () => {
        // PyTorch einops: rearrange(tensor, 'b (g c) h w -> b g c h w', g=2)
        const t = await tensor(
          [
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
              [
                [13, 14],
                [15, 16],
              ],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        const grouped = await rearrange(t, 'b (g c) h w -> b g c h w', { g: 2 });

        expect(grouped.shape).toEqual([1, 2, 2, 2, 2]);
      });
    });

    describe('edge cases', () => {
      it('should handle scalar tensors', async () => {
        // PyTorch einops: rearrange(torch.tensor(42), '-> 1')
        const scalar = await tensor(42, { device, dtype: float32 });

        const expanded = await rearrange(scalar, '-> 1');

        expect(expanded.shape).toEqual([1]);
        const data = await expanded.toArray();
        expect(data).toEqual([42]);
      });

      it('should handle single element tensors', async () => {
        // PyTorch einops: various operations on [1] shaped tensor
        const single = await tensor([42] as const, { device, dtype: float32 });

        const as_matrix = await rearrange(single, '1 -> 1 1');
        expect(as_matrix.shape).toEqual([1, 1]);

        const as_3d = await rearrange(single, '1 -> 1 1 1');
        expect(as_3d.shape).toEqual([1, 1, 1]);
      });

      it('should handle high rank tensors', async () => {
        // PyTorch einops: 5D tensor operations
        const t5d = await tensor(
          [
            [
              [
                [[1], [2]],
                [[3], [4]],
              ],
              [
                [[5], [6]],
                [[7], [8]],
              ],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        const rearranged = await rearrange(t5d, 'a b c d e -> e d c b a');
        expect(rearranged.shape).toEqual([1, 2, 2, 2, 1]);
      });

      it('should preserve dtype and device', async () => {
        const t = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: int32 },
        );

        const result = await rearrange(t, 'h w -> w h');
        expect(result.dtype).toBe(int32);
        expect(result.device).toBe(device);
      });
    });

    describe('type and data integrity', () => {
      it('should preserve exact values for integers', async () => {
        const t = await tensor(
          [
            [1000000, 2000000],
            [3000000, 4000000],
          ] as const,
          { device, dtype: int32 },
        );

        const rearranged = await rearrange(t, 'h w -> (h w)');
        const data = await rearranged.toArray();
        expect(data).toEqual([1000000, 2000000, 3000000, 4000000]);
      });

      it('should preserve floating point values', async () => {
        const t = await tensor(
          [
            [1.234567, 2.345678],
            [3.456789, 4.56789],
          ] as const,
          { device, dtype: float32 },
        );

        const rearranged = await rearrange(t, 'h w -> w h');
        const data = await rearranged.toArray();
        expect(data[0][0]).toBeCloseTo(1.234567, 5);
        expect(data[1][0]).toBeCloseTo(2.345678, 5);
      });

      it('should handle special values', async () => {
        const t = await tensor(
          [
            [Infinity, -Infinity],
            [0, -0],
          ] as const,
          { device, dtype: float32 },
        );

        const rearranged = await rearrange(t, 'h w -> (h w)');
        const data = await rearranged.toArray();
        expect(data[0]).toBe(Infinity);
        expect(data[1]).toBe(-Infinity);
        expect(data[2]).toBe(0);
        expect(data[3]).toBe(-0);
      });
    });

    describe('chained operations', () => {
      it('should chain multiple rearrange operations', async () => {
        // PyTorch einops:
        // >>> import einops
        // >>> from einops import rearrange
        // >>> t = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        // >>> step1 = rearrange(t, 'h w -> w h')  # [4, 2]
        // >>> step2 = rearrange(step1, 'a b -> b a')  # [2, 4]
        // >>> step3 = rearrange(step2, 'x y -> y x')  # [4, 2]
        // >>> step3.shape
        // torch.Size([4, 2])
        const t = await tensor(
          [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ] as const,
          { device, dtype: float32 },
        );

        const result = await rearrange(
          await rearrange(await rearrange(t, 'h w -> w h'), 'a b -> b a'),
          'x y -> y x',
        );

        expect(result.shape).toEqual([4, 2]);
        const data = await result.toArray();
        expect(data).toEqual([
          [1, 5],
          [2, 6],
          [3, 7],
          [4, 8],
        ]);
      });

      it('should work with other tensor operations', async () => {
        // PyTorch einops:
        // >>> t = torch.tensor([[1, 2], [3, 4]])
        // >>> transposed = t.T  # [[1, 3], [2, 4]]
        // >>> rearranged = rearrange(transposed, 'h w -> (h w)')
        // >>> rearranged
        // tensor([1, 3, 2, 4])
        const t = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        const transposed = t.transpose();
        const rearranged = await rearrange(transposed, 'h w -> (h w)');

        expect(rearranged.shape).toEqual([4]);
        const data = await rearranged.toArray();
        expect(data).toEqual([1, 3, 2, 4]);
      });
    });

    describe('error handling', () => {
      it('should throw on mismatched element counts', async () => {
        const t = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        await expect(rearrange(t, '2 3 -> 2 4')).rejects.toThrow();
      });

      it('should throw on ambiguous decomposition without sizes', async () => {
        const t = await tensor([1, 2, 3, 4, 5, 6] as const, { device, dtype: float32 });

        await expect(rearrange(t, '(a b) -> a b')).rejects.toThrow();
      });

      it('should throw on invalid size specifications', async () => {
        const t = await tensor([1, 2, 3, 4, 5] as const, { device, dtype: float32 });

        await expect(rearrange(t, '(a b) -> a b', { a: 2 })).rejects.toThrow(); // 5 doesn't divide by 2
      });

      it('should throw on unknown axes', async () => {
        const t = await tensor(
          [
            [1, 2],
            [3, 4],
          ] as const,
          { device, dtype: float32 },
        );

        await expect(rearrange(t, 'h w -> h w z')).rejects.toThrow(); // z is not defined
      });
    });

    describe('advanced patterns', () => {
      it('should handle batch matrix multiplication preparation', async () => {
        // Prepare for bmm: convert [batch, seq1, seq2, dim] to [batch*seq1, seq2, dim]
        const t = await tensor(
          [
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
          ] as const,
          { device, dtype: float32 },
        );

        const bmm_ready = await rearrange(t, 'batch seq1 seq2 dim -> (batch seq1) seq2 dim');

        expect(bmm_ready.shape).toEqual([2, 2, 2]);
        const data = await bmm_ready.toArray();
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

      it('should handle positional encoding preparation', async () => {
        // Flatten batch and sequence for positional encoding
        const t = await tensor(
          [
            [
              [1, 2, 3, 4],
              [5, 6, 7, 8],
            ],
            [
              [9, 10, 11, 12],
              [13, 14, 15, 16],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        const flat = await rearrange(t, 'batch seq dim -> (batch seq) dim');
        expect(flat.shape).toEqual([4, 4]);

        // And back
        const unflat = await rearrange(flat, '(batch seq) dim -> batch seq dim', { batch: 2 });
        expect(unflat.shape).toEqual([2, 2, 4]);
      });

      it('should handle im2col style transformation', async () => {
        // Flatten spatial dimensions for convolution
        const feature_maps = await tensor(
          [
            [
              [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
              ],
              [
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
              ],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        const im2col = await rearrange(
          feature_maps,
          'batch channels height width -> batch channels (height width)',
        );

        expect(im2col.shape).toEqual([1, 2, 9]);
      });

      it('should handle squeeze and unsqueeze patterns', async () => {
        const t = await tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ] as const,
          { device, dtype: float32 },
        );

        // Add multiple dimensions then remove them
        const expanded = await rearrange(t, 'h w -> 1 h 1 w 1');
        expect(expanded.shape).toEqual([1, 2, 1, 3, 1]);

        const squeezed = await rearrange(expanded, '1 h 1 w 1 -> h w');
        expect(squeezed.shape).toEqual([2, 3]);

        const data = await squeezed.toArray();
        expect(data).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);
      });
    });

    describe('repeated axis names', () => {
      it('should handle repeated axis names correctly', async () => {
        // PyTorch einops: allows repeated names if they refer to different positions
        const t = await tensor(
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

        const result = await rearrange(t, 'h1 h2 w -> h2 h1 w');

        expect(result.shape).toEqual([2, 2, 2]);
        const data = await result.toArray();
        expect(data).toEqual([
          [
            [1, 2],
            [5, 6],
          ],
          [
            [3, 4],
            [7, 8],
          ],
        ]);
      });
    });

    describe('complex real-world patterns', () => {
      it('should handle vision transformer patch embedding', async () => {
        // Convert image to patches for ViT
        // [batch, height, width, channels] -> [batch, num_patches, patch_size]
        const image = await tensor(
          [
            [
              [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
              ],
              [
                [13, 14, 15],
                [16, 17, 18],
                [19, 20, 21],
                [22, 23, 24],
              ],
              [
                [25, 26, 27],
                [28, 29, 30],
                [31, 32, 33],
                [34, 35, 36],
              ],
              [
                [37, 38, 39],
                [40, 41, 42],
                [43, 44, 45],
                [46, 47, 48],
              ],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        // Extract 2x2 patches
        const patches = await rearrange(image, 'batch (h ph) (w pw) c -> batch (h w) (ph pw c)', {
          ph: 2,
          pw: 2,
        });

        expect(patches.shape).toEqual([1, 4, 12]);
      });

      it('should handle BERT-style attention mask broadcasting prep', async () => {
        // Prepare attention mask for broadcasting
        const mask = await tensor(
          [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
          ] as const,
          { device, dtype: float32 },
        );

        // Add dimensions for num_heads and seq_len broadcast
        const broadcast_mask = await rearrange(mask, 'batch seq -> batch 1 1 seq');

        expect(broadcast_mask.shape).toEqual([2, 1, 1, 4]);
      });

      it('should handle conv2d to linear layer transition', async () => {
        // Common pattern in CNNs before FC layers
        const conv_output = await tensor(
          [
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
            ],
            [
              [
                [13, 14],
                [15, 16],
              ],
              [
                [17, 18],
                [19, 20],
              ],
              [
                [21, 22],
                [23, 24],
              ],
            ],
          ] as const,
          { device, dtype: float32 },
        );

        // Flatten all spatial dimensions
        const fc_input = await rearrange(
          conv_output,
          'batch channels height width -> batch (channels height width)',
        );

        expect(fc_input.shape).toEqual([2, 12]);
        const data = await fc_input.toArray();
        expect(data[0]).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
      });
    });
  });
}
