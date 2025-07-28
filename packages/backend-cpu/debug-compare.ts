import { tensor, float32, rearrange } from '@typetensor/core';
import { cpu } from './src';

async function compareApproaches() {
  console.log('=== Comparing Permute vs Rearrange ===\n');
  
  // Test 1: Simple 2D transpose
  const matrix = await tensor(
    [
      [1, 2],
      [3, 4],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Original matrix:');
  console.log(await matrix.toArray());
  
  // Using permute directly
  const permuted = matrix.permute([1, 0] as const);
  console.log('\nUsing permute([1, 0]):');
  console.log(await permuted.toArray());
  
  // Using rearrange
  const rearranged = await rearrange(matrix, 'h w -> w h');
  console.log('\nUsing rearrange("h w -> w h"):');
  console.log(await rearranged.toArray());
  
  // Test 2: After transpose, flatten
  console.log('\n--- After transpose, flatten ---');
  const transposed = matrix.transpose();
  console.log('Transposed:');
  console.log(await transposed.toArray());
  
  // Flatten the transposed tensor
  const flattenedDirect = transposed.reshape([4] as const);
  console.log('\nFlattened with reshape:');
  console.log(await flattenedDirect.toArray());
  
  const flattenedRearrange = await rearrange(transposed, 'h w -> (h w)');
  console.log('\nFlattened with rearrange:');
  console.log(await flattenedRearrange.toArray());
}

compareApproaches();