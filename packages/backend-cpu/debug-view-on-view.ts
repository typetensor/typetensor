import { tensor, float32 } from '@typetensor/core';
import { cpu } from './src';

async function debugViewOnView() {
  console.log('=== Testing View-on-View Operations ===\n');
  
  // Create a simple 2x2 matrix
  const matrix = await tensor(
    [
      [1, 2],
      [3, 4],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Original matrix:');
  console.log(await matrix.toArray());
  console.log('Shape:', matrix.shape);
  console.log('Strides:', matrix.strides);
  
  // First view: transpose
  const transposed = matrix.transpose();
  console.log('\nAfter transpose:');
  console.log(await transposed.toArray());
  console.log('Shape:', transposed.shape);
  console.log('Strides:', transposed.strides);
  
  // Second view: reshape (flatten)
  const flattened = transposed.reshape([4] as const);
  console.log('\nAfter reshape (view on view):');
  console.log(await flattened.toArray());
  console.log('Expected: [1, 3, 2, 4]');
  console.log('Shape:', flattened.shape);
  console.log('Strides:', flattened.strides);
  
  // For comparison: contiguous copy then reshape
  console.log('\n--- With contiguous copy ---');
  // We don't have contiguous() in the API, but let's simulate by creating new tensor
  const transposedData = await transposed.toArray();
  const contiguous = await tensor(transposedData, { device: cpu, dtype: float32 });
  console.log('Contiguous copy:');
  console.log(await contiguous.toArray());
  
  const flattenedContiguous = contiguous.reshape([4] as const);
  console.log('\nFlattened contiguous:');
  console.log(await flattenedContiguous.toArray());
  console.log('This gives the correct result!');
  
  // Test with permute too
  console.log('\n--- Testing with permute ---');
  const permuted = matrix.permute([1, 0] as const);
  console.log('Permuted:');
  console.log(await permuted.toArray());
  console.log('Strides:', permuted.strides);
  
  const flattenedPermuted = permuted.reshape([4] as const);
  console.log('\nFlattened permuted:');
  console.log(await flattenedPermuted.toArray());
  console.log('Expected: [1, 3, 2, 4]');
}

debugViewOnView();