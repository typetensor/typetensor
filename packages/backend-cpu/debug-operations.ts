import { tensor, float32 } from '@typetensor/core';
import { cpu } from './src';
import { parse } from '../core/src/einops/scanner';
import { AxisResolver } from '../core/src/einops/axis-resolver';

// Helper to visualize tensor reshaping
function visualizeTensor(data: number[], shape: number[]): string {
  if (shape.length === 1) {
    return `[${data.join(', ')}]`;
  } else if (shape.length === 2) {
    const rows = [];
    for (let i = 0; i < shape[0]; i++) {
      const row = [];
      for (let j = 0; j < shape[1]; j++) {
        row.push(data[i * shape[1] + j]);
      }
      rows.push(`  [${row.join(', ')}]`);
    }
    return `[\n${rows.join(',\n')}\n]`;
  } else if (shape.length === 3) {
    const blocks = [];
    for (let i = 0; i < shape[0]; i++) {
      const rows = [];
      for (let j = 0; j < shape[1]; j++) {
        const row = [];
        for (let k = 0; k < shape[2]; k++) {
          row.push(data[i * shape[1] * shape[2] + j * shape[2] + k]);
        }
        rows.push(`    [${row.join(', ')}]`);
      }
      blocks.push(`  [\n${rows.join(',\n')}\n  ]`);
    }
    return `[\n${blocks.join(',\n')}\n]`;
  }
  return 'Too many dimensions to visualize';
}

async function analyzeOperations() {
  console.log('=== Analyzing Patch Extraction Operations ===\n');
  
  // Create test tensor
  const image = await tensor(
    [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
      [13, 14, 15, 16],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Input tensor [4, 4]:');
  const inputData = await image.toArray();
  const flatInput = inputData.flat();
  console.log(visualizeTensor(flatInput, [4, 4]));
  
  // What should happen:
  console.log('\n=== Expected Transformation ===');
  console.log('Pattern: (h ph) (w pw) -> h w (ph pw) with ph=2, pw=2');
  console.log('\nStep 1: Conceptually reshape [4, 4] to [h, ph, w, pw] = [2, 2, 2, 2]');
  console.log('This breaks each dimension into blocks');
  
  console.log('\nStep 2: Permute to [h, w, ph, pw] = [2, 2, 2, 2]');
  console.log('This groups the patches together');
  
  console.log('\nStep 3: Reshape to [h, w, (ph pw)] = [2, 2, 4]');
  console.log('This flattens each patch');
  
  // Let's see what permutation we need
  console.log('\n=== Permutation Analysis ===');
  console.log('Original axes order: [h, ph, w, pw] (after expanding composites)');
  console.log('Target axes order: [h, w, ph, pw]');
  console.log('Permutation needed: [0, 2, 1, 3]');
  
  // Manually compute what the result should be
  console.log('\n=== Manual Computation ===');
  
  // First reshape to [2, 2, 2, 2]
  const reshaped4d = [];
  for (let h = 0; h < 2; h++) {
    for (let ph = 0; ph < 2; ph++) {
      for (let w = 0; w < 2; w++) {
        for (let pw = 0; pw < 2; pw++) {
          const idx = h * 8 + ph * 4 + w * 2 + pw;
          reshaped4d.push(flatInput[idx]);
        }
      }
    }
  }
  
  console.log('After reshape to [2, 2, 2, 2]:');
  console.log('reshaped4d =', reshaped4d);
  
  // Apply permutation [0, 2, 1, 3]
  const permuted = [];
  for (let h = 0; h < 2; h++) {
    for (let w = 0; w < 2; w++) {
      for (let ph = 0; ph < 2; ph++) {
        for (let pw = 0; pw < 2; pw++) {
          // Original index in [h, ph, w, pw] layout
          const oldIdx = h * 8 + ph * 4 + w * 2 + pw;
          permuted.push(reshaped4d[oldIdx]);
        }
      }
    }
  }
  
  console.log('\nAfter permute to [h, w, ph, pw]:');
  console.log('permuted =', permuted);
  
  console.log('\nFinal result [2, 2, 4]:');
  console.log(visualizeTensor(permuted, [2, 2, 4]));
}

analyzeOperations();