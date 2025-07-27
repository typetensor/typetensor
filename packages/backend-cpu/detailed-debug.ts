import { tensor, float32 } from '@typetensor/core';
import { cpu } from './src/index';

async function detailedDebug() {
  console.log('=== DETAILED STEP-BY-STEP DEBUG ===\n');
  
  // Step 1: Create transposed tensor
  const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device: cpu, dtype: float32 });
  const reshaped = original.reshape([2, 3] as const);
  const transposed = reshaped.transpose();
  
  console.log('1. TRANSPOSED TENSOR STATE:');
  console.log('  Shape:', transposed.shape);
  console.log('  Strides:', transposed.strides);
  console.log('  c_contiguous:', transposed.layout.c_contiguous);
  console.log('  Data via toArray():', await transposed.toArray());
  console.log();
  
  // Step 2: Debug what toArray() returns
  console.log('2. WHAT TOARRAY() GIVES US:');
  const logicalData = await transposed.toArray();
  console.log('  Logical data:', logicalData);
  console.log('  Type:', typeof logicalData);
  console.log('  Is array?', Array.isArray(logicalData));
  console.log('  Length:', (logicalData as any).length);
  console.log();
  
  // Step 3: Debug flattening
  console.log('3. FLATTENING THE LOGICAL DATA:');
  
  // Manual flatten to see what we get
  function manualFlatten(arr: any): any[] {
    const result: any[] = [];
    console.log('  Processing item:', arr);
    
    function flatten(item: any): void {
      console.log('    Flatten called with:', item, 'Is array?', Array.isArray(item));
      if (Array.isArray(item)) {
        for (let i = 0; i < item.length; i++) {
          console.log(`      Processing item[${i}]:`, item[i]);
          flatten(item[i]);
        }
      } else {
        console.log('      Adding to result:', item);
        result.push(item);
      }
    }
    
    flatten(arr);
    return result;
  }
  
  const flatData = manualFlatten(logicalData);
  console.log('  Final flat data:', flatData);
  console.log('  Expected flat data: [1, 4, 2, 5, 3, 6]');
  console.log('  Does flat data match expected?', JSON.stringify(flatData) === JSON.stringify([1, 4, 2, 5, 3, 6]));
  console.log();
  
  // Step 4: Test tensor creation with flat data
  console.log('4. CREATING NEW TENSOR FROM FLAT DATA:');
  const newTensor = await tensor(flatData as any, {
    device: cpu,
    dtype: float32,
    shape: [3, 2] as const, // Same shape as transposed
  });
  
  console.log('  New tensor shape:', newTensor.shape);
  console.log('  New tensor data:', await newTensor.toArray());
  console.log('  New tensor c_contiguous:', newTensor.layout.c_contiguous);
  console.log();
  
  // Step 5: Test flatten on the new tensor
  console.log('5. FLATTENING THE NEW TENSOR:');
  const flattened = await newTensor.flatten();
  console.log('  Flattened shape:', flattened.shape);
  console.log('  Flattened data:', await flattened.toArray());
  console.log('  Should be: [1, 4, 2, 5, 3, 6]');
  console.log();
  
  // Step 6: Debug why our _createContiguousCopy isn't working
  console.log('6. DEBUGGING _createContiguousCopy:');
  
  // Let's manually reproduce what _createContiguousCopy should do
  console.log('  Step 6a: Getting logical data again...');
  const logicalData2 = await transposed.toArray();
  console.log('  Logical data:', logicalData2);
  
  console.log('  Step 6b: Flattening...');
  const flatData2 = manualFlatten(logicalData2);
  console.log('  Flat data:', flatData2);
  
  console.log('  Step 6c: Creating tensor with same shape as original transposed...');
  const reconstructed = await tensor(flatData2 as any, {
    device: cpu,
    dtype: float32,
    shape: transposed.shape, // This should be [3, 2]
  });
  
  console.log('  Reconstructed shape:', reconstructed.shape);
  console.log('  Reconstructed data:', await reconstructed.toArray());
  console.log('  Reconstructed strides:', reconstructed.strides);
  console.log('  Reconstructed c_contiguous:', reconstructed.layout.c_contiguous);
  
  console.log('  Step 6d: Flattening reconstructed...');
  const finalFlattened = await reconstructed.flatten();
  console.log('  Final flattened data:', await finalFlattened.toArray());
  console.log('  Is this correct?', JSON.stringify(await finalFlattened.toArray()) === JSON.stringify([1, 4, 2, 5, 3, 6]));
}

await detailedDebug();