/**
 * TypeTensor Einops: Type-Safe Tensor Operations with Einstein Notation
 *
 * This example demonstrates all einops operations: rearrange, reduce, and repeat.
 * These functions provide intuitive tensor manipulations using Einstein notation patterns
 * with full compile-time validation.
 *
 * Run with: bun run examples/05-einops.ts
 */

import { tensor, float32 } from '@typetensor/core';
import { rearrange, reduce, repeat } from '@typetensor/einops';
import { cpu } from '@typetensor/backend-cpu';

async function main(): Promise<void> {
  console.log('TypeTensor Einops: Type-Safe Einstein Operations\n');
  console.log('='.repeat(50));

  // 1. Simple transpose
  const matrix = await tensor(
    //  ^ matrix: Tensor<Shape<[2, 3]>>
    [
      [1, 2, 3],
      [4, 5, 6],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  const transposed = await rearrange(matrix, 'h w -> w h');
  //    ^ transposed: Tensor<Shape<[3, 2]>>
  // TypeScript knows the output shape at compile time!

  console.log('Transpose [2,3] → [3,2]:');
  console.log(await transposed.format());

  // 2. Invalid pattern detection at compile time
  rearrange(matrix, 'h w -> h w c');
  //                ^ error: [Einops ❌] Axis Error: Unknown axis 'c' in output. Available axes: ['h', 'w']
  // TypeScript catches invalid einops patterns before runtime!

  // 3. Flatten and reshape with composite patterns
  const tensor3d = await tensor(
    //  ^ tensor3d: Tensor<Shape<[2, 3, 2]>>
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
    { device: cpu, dtype: float32 },
  );

  const flattened = await rearrange(tensor3d, 'batch seq features -> (batch seq) features');
  //    ^ flattened: Tensor<Shape<[6, 2]>>
  // Composite pattern (batch seq) flattens from [2,3,2] to [6,2]

  console.log('\nFlatten [2,3,2] → [6,2]:');
  console.log(await flattened.format());

  // 4. Split dimensions with axis values
  const flat = await tensor([[1, 2, 3, 4, 5, 6, 7, 8]] as const, { device: cpu, dtype: float32 });
  //    ^ flat: Tensor<Shape<[1, 8]>>

  const split = await rearrange(flat, 'batch (heads dim) -> batch heads dim', { heads: 4 });
  //    ^ split: Tensor<Shape<[1, 4, 2]>>
  // TypeScript infers dim=2 from 8/4=2

  console.log('\nSplit [1,8] → [1,4,2] with heads=4:');
  console.log(await split.format());

  // Invalid split - TypeScript catches this!
  rearrange(flat, 'batch (heads dim) -> batch heads dim', { heads: 3 });
  //               ^ error: [Einops ❌] Shape Error: Pattern 'batch (heads dim) -> batch heads dim' produces fractional dimensions. Composite axes must divide evenly. Use integer axis values: rearrange(tensor, pattern, {axis: integer})

  const data = await tensor(
    [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ] as const,
    { device: cpu, dtype: float32 },
  );
  //    ^ data: Tensor<Shape<[3, 4]>>

  // Sum along rows (keep columns)
  const sumRows = await reduce(data, 'rows cols -> cols', 'sum');
  //    ^ sumRows: Tensor<Shape<[4]>>
  // Result: [15, 18, 21, 24] - sum of each column

  console.log('Sum rows [3,4] → [4]:');
  console.log(await sumRows.format());

  // Mean along columns (keep rows)
  const meanCols = await reduce(data, 'rows cols -> rows', 'mean');
  //    ^ meanCols: Tensor<Shape<[3]>>
  // Result: [2.5, 6.5, 10.5] - mean of each row

  console.log('\nMean cols [3,4] → [3]:');
  console.log(await meanCols.format());

  // Global max (reduce all dimensions)
  const globalMax = await reduce(data, 'rows cols -> ', 'max');
  //    ^ globalMax: Tensor<Shape<[]>> (scalar)
  // Empty output pattern creates a scalar

  console.log('\nGlobal max [3,4] → scalar:');
  console.log(await globalMax.format());

  // Keep dimensions example
  const sumKeepDims = await reduce(data, 'rows cols -> rows 1', 'sum', true);
  //    ^ sumKeepDims: Tensor<Shape<[3, 1]>>
  // Using '1' in pattern keeps dimension with size 1

  console.log('\nSum cols keeping dims [3,4] → [3,1]:');
  console.log(await sumKeepDims.format());

  // Invalid reduction - cannot create new axes
  reduce(data, 'h w -> h w c', 'sum');
  //            ^ error: [Reduce ❌] Axis Error: Cannot create new axis 'c' in reduce output. Available input axes: ['h', 'w']. Reduce can only preserve or remove axes

  const vector = await tensor([1, 2, 3] as const, { device: cpu, dtype: float32 });
  //    ^ vector: Tensor<Shape<[3]>>

  // Add new dimension
  const expanded = await repeat(vector, 'w -> h w', { h: 4 });
  //    ^ expanded: Tensor<Shape<[4, 3]>>
  // Creates 4 copies of the vector

  console.log('Expand [3] → [4,3] with h=4:');
  console.log(await expanded.format());

  // Repeat along existing dimension
  const matrix2d = await tensor(
    [
      [1, 2],
      [3, 4],
    ] as const,
    { device: cpu, dtype: float32 },
  );
  //    ^ matrix2d: Tensor<Shape<[2, 2]>>

  const repeated = await repeat(matrix2d, 'h w -> (h repeat) w', { repeat: 3 });
  //    ^ repeated: Tensor<Shape<[6, 2]>>
  // Each row is repeated 3 times: [1,2], [1,2], [1,2], [3,4], [3,4], [3,4]

  console.log('\nRepeat rows [2,2] → [6,2] with repeat=3:');
  console.log(await repeated.format());

  // Upsampling pattern
  const small = await tensor(
    [
      [1, 2],
      [3, 4],
    ] as const,
    { device: cpu, dtype: float32 },
  );
  //    ^ small: Tensor<Shape<[2, 2]>>

  const upsampled = await repeat(small, 'h w -> (h h2) (w w2)', { h2: 2, w2: 2 });
  //    ^ upsampled: Tensor<Shape<[4, 4]>>
  // 2x2 upsampling creates [[1,1,2,2], [1,1,2,2], [3,3,4,4], [3,3,4,4]]

  console.log('\nUpsample [2,2] → [4,4] with 2x2:');
  console.log(await upsampled.format());

  // Invalid repeat - missing required axis
  repeat(vector, 'w -> h w');
  //              ^ error: [Repeat ❌] Axis Error: New axis 'h' requires explicit size. Specify: repeat(tensor, pattern, {h: number})

  // Invalid repeat - composite pattern in input
  repeat(matrix2d, '(h w) -> h w c', { c: 3 });
  //               ^ error: [Repeat ❌] Shape Error: Repeat pattern expects 1 dimensions but tensor has 2
}

// Run the example
main().catch(console.error);
