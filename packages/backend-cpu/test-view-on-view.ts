import { tensor, float32 } from '@typetensor/core';
import { cpu } from './src';

async function testViewOnView() {
  console.log('=== Testing View-on-View Failures ===\n');
  
  // Test 1: Reshape transposed matrix
  console.log('Test 1: Reshape transposed matrix');
  const matrix = await tensor(
    [
      [1, 2],
      [3, 4],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  const transposed = matrix.transpose();
  const reshaped = transposed.reshape([4] as const);
  
  console.log('Expected: [1, 3, 2, 4]');
  console.log('Actual:', await reshaped.toArray());
  
  // Test 2: Flatten permuted tensor
  console.log('\nTest 2: Flatten permuted tensor');
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
    { device: cpu, dtype: float32 },
  );

  const permuted = tensor3d.permute([2, 0, 1] as const);
  const flattened = await permuted.flatten();
  
  console.log('Expected: [1, 5, 3, 7, 2, 6, 4, 8]');
  console.log('Actual:', await flattened.toArray());
  
  // Test 3: Maintain strides
  console.log('\nTest 3: Strides through operations');
  const original = await tensor(
    [
      [1, 2, 3],
      [4, 5, 6],
    ] as const,
    { device: cpu, dtype: float32 },
  );
  
  console.log('Original strides:', original.strides);
  const transposed2 = original.transpose();
  console.log('Transposed strides:', transposed2.strides);
  const reshaped2 = transposed2.reshape([6] as const);
  console.log('Reshaped strides:', reshaped2.strides);
  console.log('Expected data: [1, 4, 2, 5, 3, 6]');
  console.log('Actual data:', await reshaped2.toArray());
}

testViewOnView();