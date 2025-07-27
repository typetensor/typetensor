import { tensor, float32 } from '@typetensor/core';
import { cpu } from './src/index';

async function testReshapeError() {
  console.log('=== Testing Reshape Error Behavior ===');
  
  const tensor12 = await tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as const, {
    device: cpu,
    dtype: float32,
  });
  
  console.log('Tensor size:', tensor12.size);
  console.log('Tensor shape:', tensor12.shape);
  
  try {
    console.log('Attempting reshape([3, 5])...');
    const result = tensor12.reshape([3, 5] as const);
    console.log('ERROR: reshape did not throw!');
    console.log('Result shape:', result.shape);
    console.log('Result data:', await result.toArray());
  } catch (error) {
    console.log('SUCCESS: reshape threw error:', (error as Error).message);
  }
  
  // Test the function wrapper approach
  console.log('\nTesting function wrapper approach...');
  try {
    const fn = () => {
      return tensor12.reshape([3, 5] as const);
    };
    const result = fn();
    console.log('Function wrapper did not throw, returned:', typeof result);
  } catch (error) {
    console.log('Function wrapper threw:', (error as Error).message);
  }
}

await testReshapeError();