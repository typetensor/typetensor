import { tensor, float32 } from '@typetensor/core';
import { cpu } from './src/index';

async function investigateRootCause() {
  console.log('=== ROOT CAUSE INVESTIGATION ===\n');
  
  // Create the problematic chain
  const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device: cpu, dtype: float32 });
  const reshaped = original.reshape([2, 3] as const);
  const transposed = reshaped.transpose();
  
  console.log('1. BASIC STATE:');
  console.log('Original data:', await original.toArray());
  console.log('Reshaped data:', await reshaped.toArray());
  console.log('Transposed data:', await transposed.toArray());
  console.log('Transposed strides:', transposed.strides);
  console.log('Transposed c_contiguous:', transposed.layout.c_contiguous);
  console.log();
  
  console.log('2. TESTING DATA READING:');
  // Test if toArray() on transposed gives correct logical view
  const transposedArray = await transposed.toArray();
  console.log('Transposed toArray() result:', transposedArray);
  console.log('Expected transposed view: [[1,4],[2,5],[3,6]]');
  console.log('Does toArray() work correctly?', JSON.stringify(transposedArray) === JSON.stringify([[1,4],[2,5],[3,6]]));
  console.log();
  
  console.log('3. TESTING CLONE:');
  const cloned = await transposed.clone();
  console.log('Cloned data:', await cloned.toArray());
  console.log('Cloned strides:', cloned.strides);
  console.log('Cloned c_contiguous:', cloned.layout.c_contiguous);
  console.log('Clone preserves view layout?', JSON.stringify(cloned.strides) === JSON.stringify(transposed.strides));
  console.log();
  
  console.log('4. TESTING RESHAPE ON CLONE:');
  const clonedFlattened = cloned._reshapeUnsafe([6] as const);
  console.log('Clone reshaped to [6] data:', await clonedFlattened.toArray());
  console.log('Clone reshaped strides:', clonedFlattened.strides);
  console.log();
  
  console.log('5. TESTING DEVICE DATA READING:');
  // Directly inspect what the device returns
  const deviceBuffer = await transposed.data.device.readData(transposed.data);
  console.log('Raw device buffer length:', deviceBuffer.byteLength);
  console.log('Expected float32 values:', deviceBuffer.byteLength / 4);
  
  // Convert buffer to float32 array to see raw data
  const float32View = new Float32Array(deviceBuffer);
  console.log('Raw device float32 data:', Array.from(float32View));
  console.log('Is raw data in original order?', Array.from(float32View).join(',') === '1,2,3,4,5,6');
  console.log();
  
  console.log('6. TESTING DIFFERENT TENSOR CREATION:');
  // Create a tensor directly with the expected flatten result
  const expectedFlattened = await tensor([1, 4, 2, 5, 3, 6] as const, { device: cpu, dtype: float32 });
  console.log('Expected flatten result:', await expectedFlattened.toArray());
  console.log();
  
  console.log('7. MANUAL STRIDE CALCULATION:');
  // Manually follow the strides to see what we should get
  console.log('Transposed shape:', transposed.shape); // [3, 2]
  console.log('Transposed strides:', transposed.strides); // [1, 3]
  
  // For shape [3,2] with strides [1,3], element [i,j] is at offset: i*1 + j*3
  console.log('Manual stride mapping:');
  console.log('[0,0] -> offset 0*1 + 0*3 = 0 -> value should be 1');
  console.log('[0,1] -> offset 0*1 + 1*3 = 3 -> value should be 4');
  console.log('[1,0] -> offset 1*1 + 0*3 = 1 -> value should be 2');
  console.log('[1,1] -> offset 1*1 + 1*3 = 4 -> value should be 5');
  console.log('[2,0] -> offset 2*1 + 0*3 = 2 -> value should be 3');
  console.log('[2,1] -> offset 2*1 + 1*3 = 5 -> value should be 6');
  console.log('So flatten order should be: [1,4,2,5,3,6]');
}

await investigateRootCause();