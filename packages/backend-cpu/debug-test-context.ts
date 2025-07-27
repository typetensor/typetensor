import { tensor, float32 } from '@typetensor/core';
import { cpu } from './src/index';

async function debugTestContext() {
  console.log('=== DEBUGGING TEST CONTEXT DIFFERENCES ===\n');
  
  // Recreate the exact same scenario as in the test
  console.log('1. Creating tensor exactly like the test...');
  const tensor12 = await tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as const, {
    device: cpu,
    dtype: float32,
  });
  
  console.log('  Tensor shape:', tensor12.shape);
  console.log('  Tensor size:', tensor12.size);
  console.log('  Tensor device:', tensor12.device);
  console.log('  Tensor dtype:', tensor12.dtype);
  
  // Test direct call
  console.log('\n2. Testing direct reshape call...');
  try {
    const result = tensor12.reshape([3, 5] as const);
    console.log('  ❌ Direct call did NOT throw!');
    console.log('  Result shape:', result.shape);
  } catch (error) {
    console.log('  ✅ Direct call threw:', (error as Error).message);
  }
  
  // Test function wrapper (like in the test)
  console.log('\n3. Testing function wrapper (like test)...');
  try {
    const fn = () => {
      tensor12.reshape([3, 5] as const);
    };
    fn();
    console.log('  ❌ Function wrapper did NOT throw!');
  } catch (error) {
    console.log('  ✅ Function wrapper threw:', (error as Error).message);
  }
  
  // Test expect-style wrapper
  console.log('\n4. Testing expect-style behavior...');
  
  // Simulate what expect(() => ...).toThrow() does
  let threwError = false;
  let caughtError: Error | null = null;
  
  try {
    const testFunction = () => {
      return tensor12.reshape([3, 5] as const);
    };
    testFunction();
  } catch (error) {
    threwError = true;
    caughtError = error as Error;
  }
  
  console.log('  Did throw?', threwError);
  if (caughtError) {
    console.log('  Error message:', caughtError.message);
  }
  
  // Test if there's a difference with type annotation
  console.log('\n5. Testing type annotation differences...');
  try {
    // This is what the test has with @ts-expect-error
    (tensor12 as any).reshape([3, 5] as const);
    console.log('  ❌ Type cast version did NOT throw!');
  } catch (error) {
    console.log('  ✅ Type cast version threw:', (error as Error).message);
  }
}

await debugTestContext();