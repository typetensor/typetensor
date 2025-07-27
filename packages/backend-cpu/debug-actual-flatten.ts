import { tensor, float32 } from '@typetensor/core';
import { cpu } from './src/index';

async function debugActualFlatten() {
  console.log('=== DEBUGGING ACTUAL FLATTEN METHOD ===\n');
  
  // Create transposed tensor
  const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device: cpu, dtype: float32 });
  const reshaped = original.reshape([2, 3] as const);
  const transposed = reshaped.transpose();
  
  console.log('Transposed tensor c_contiguous:', transposed.layout.c_contiguous);
  console.log('Transposed tensor data:', await transposed.toArray());
  
  // Try calling flatten and see what happens
  console.log('\nCalling transposed.flatten()...');
  const flattened = await transposed.flatten();
  
  console.log('Result shape:', flattened.shape);
  console.log('Result data:', await flattened.toArray());
  console.log('Expected: [1, 4, 2, 5, 3, 6]');
  
  // Let's add some debug logs to see what's happening inside our method
  console.log('\n=== MANUAL REPRODUCTION OF THE METHOD ===');
  
  // Step 1: Check contiguity
  console.log('Step 1: Is c_contiguous?', transposed.layout.c_contiguous === true);
  
  if (transposed.layout.c_contiguous !== true) {
    console.log('Step 2: Creating contiguous copy...');
    
    // Step 2a: Get logical data
    const logicalData = await transposed.toArray();
    console.log('  Logical data:', logicalData);
    
    // Step 2b: Flatten (using same method as in class)
    function _flattenNestedArray(arr: any): any[] {
      const result: any[] = [];
      
      function flatten(item: any): void {
        if (Array.isArray(item)) {
          for (const subItem of item) {
            flatten(subItem);
          }
        } else {
          result.push(item);
        }
      }
      
      flatten(arr);
      return result;
    }
    
    const flatData = _flattenNestedArray(logicalData);
    console.log('  Flattened data:', flatData);
    
    // Step 2c: Create new tensor (exactly like our method)
    console.log('  Creating new tensor with:');
    console.log('    flatData:', flatData);
    console.log('    device:', transposed.device);
    console.log('    dtype:', transposed.dtype);
    console.log('    shape:', transposed.shape);
    
    const contiguousCopy = await tensor(flatData as any, {
      device: transposed.device,
      dtype: transposed.dtype,
      shape: transposed.shape,
    });
    
    console.log('  New tensor data:', await contiguousCopy.toArray());
    console.log('  New tensor c_contiguous:', contiguousCopy.layout.c_contiguous);
    console.log('  New tensor strides:', contiguousCopy.strides);
    
    // Step 2d: Reshape to flatten
    console.log('  Reshaping to [6]...');
    const finalFlattened = contiguousCopy._reshapeUnsafe([6] as const);
    console.log('  Final result:', await finalFlattened.toArray());
  }
}

await debugActualFlatten();