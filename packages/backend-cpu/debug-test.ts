import { tensor, float32 } from '@typetensor/core';
import { cpu } from './src/index';

async function debugReshapeError() {
  console.log('=== Testing Reshape Error ===');
  const tensor12 = await tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as const, { device: cpu, dtype: float32 });
  
  console.log('Tensor shape:', tensor12.shape);
  console.log('Tensor size:', tensor12.size);
  
  try {
    // This should throw an error: 12 elements cannot be reshaped to 15 elements (3*5)
    const invalid = tensor12.reshape([3, 5] as const);
    console.log('ERROR: reshape did not throw! Result shape:', invalid.shape);
  } catch (error) {
    console.log('SUCCESS: reshape threw error:', (error as Error).message);
  }
}

async function debugTransposeFlatten() {
  console.log('\n=== Testing Transpose + Flatten Chain ===');
  
  // Test the exact case from our failing test
  const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device: cpu, dtype: float32 });
  console.log('Original shape:', original.shape);
  console.log('Original data:', await original.toArray());
  
  const reshaped = original.reshape([2, 3] as const);
  console.log('After reshape([2,3]) shape:', reshaped.shape);
  console.log('After reshape([2,3]) data:', await reshaped.toArray());
  
  const transposed = reshaped.transpose();
  console.log('After transpose() shape:', transposed.shape);
  console.log('After transpose() strides:', transposed.strides);
  console.log('After transpose() c_contiguous:', transposed.layout.c_contiguous);
  console.log('After transpose() data:', await transposed.toArray());
  
  const flattened = await transposed.flatten();
  console.log('After flatten() shape:', flattened.shape);
  console.log('After flatten() strides:', flattened.strides);
  console.log('After flatten() data:', await flattened.toArray());
  
  console.log('Expected from PyTorch/NumPy: [1, 4, 2, 5, 3, 6]');
}

// Run tests
await debugReshapeError();
await debugTransposeFlatten();