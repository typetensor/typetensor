import { parse } from '../core/src/einops/scanner';
import { isSimpleAxis, isCompositeAxis } from '../core/src/einops/ast';
import type { AxisPattern } from '../core/src/einops/ast';

// Copy the flattenPatternToAxisNames function
function flattenPatternToAxisNames(patterns: readonly AxisPattern[]): string[] {
  const names: string[] = [];
  
  for (const pattern of patterns) {
    if (isSimpleAxis(pattern)) {
      names.push(pattern.name);
    } else if (isCompositeAxis(pattern)) {
      names.push(...flattenPatternToAxisNames(pattern.axes));
    }
  }
  
  return names;
}

// Copy the computeGeneralPermutation function
function computeGeneralPermutation(
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
): number[] | null {
  const inputNames = flattenPatternToAxisNames(inputPatterns);
  const outputNames = flattenPatternToAxisNames(outputPatterns);
  
  console.log('Input names:', inputNames);
  console.log('Output names:', outputNames);
  
  if (inputNames.length !== outputNames.length) {
    return null;
  }
  
  const permutation: number[] = [];
  
  for (const outputName of outputNames) {
    const inputIndex = inputNames.indexOf(outputName);
    if (inputIndex === -1) {
      return null;
    }
    permutation.push(inputIndex);
  }
  
  const isIdentity = permutation.every((val, idx) => val === idx);
  return isIdentity ? null : permutation;
}

// Test patch extraction pattern
const pattern = '(h ph) (w pw) -> h w (ph pw)';
const ast = parse(pattern);

console.log('=== Testing Permutation Computation ===');
console.log('Pattern:', pattern);

const permutation = computeGeneralPermutation(ast.input, ast.output);
console.log('\nComputed permutation:', permutation);
console.log('Expected permutation: [0, 2, 1, 3]');

// Test simple flatten pattern
console.log('\n=== Testing Simple Flatten ===');
const pattern2 = 'h w -> (h w)';
const ast2 = parse(pattern2);
const permutation2 = computeGeneralPermutation(ast2.input, ast2.output);
console.log('Pattern:', pattern2);
console.log('Computed permutation:', permutation2);