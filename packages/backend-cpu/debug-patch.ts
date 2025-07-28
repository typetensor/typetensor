import { tensor, float32, rearrange } from '@typetensor/core';
import { cpu } from './src';

async function debugPatchExtraction() {
  console.log('=== Testing Patch Extraction ===');
  
  const image = await tensor(
    [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Input shape:', image.shape);
  console.log('Input data:', await image.toArray());

  try {
    const patches = await rearrange(image, '(h ph) (w pw) -> h w (ph pw)', { ph: 2, pw: 2 });
    console.log('\nOutput shape:', patches.shape);
    console.log('Output data:', await patches.toArray());
    
    console.log('\nExpected output (from PyTorch):');
    console.log('[');
    console.log('  [');
    console.log('    [1, 2, 5, 6],');
    console.log('    [3, 4, 7, 8],');
    console.log('  ],');
    console.log('  [');
    console.log('    [9, 10, 13, 14],');
    console.log('    [11, 12, 15, 16],');
    console.log('  ]');
    console.log(']');
  } catch (error) {
    console.error('Error:', error);
  }
}

async function debugTransposeFlatten() {
  console.log('\n=== Testing Transpose + Flatten ===');
  
  const t = await tensor(
    [
      [1, 2],
      [3, 4],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Input shape:', t.shape);
  console.log('Input data:', await t.toArray());

  const transposed = t.transpose();
  console.log('\nTransposed shape:', transposed.shape);
  console.log('Transposed data:', await transposed.toArray());

  const flattened = await rearrange(transposed, 'h w -> (h w)');
  console.log('\nFlattened shape:', flattened.shape);
  console.log('Flattened data:', await flattened.toArray());
  console.log('Expected (from PyTorch): [1, 3, 2, 4]');
}

debugPatchExtraction().then(() => debugTransposeFlatten());