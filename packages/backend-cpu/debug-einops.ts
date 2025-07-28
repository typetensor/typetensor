// Import internal modules directly since they're not exported
import { parse } from '../core/src/einops/scanner';
import { AxisResolver } from '../core/src/einops/axis-resolver';

// Debug patch extraction pattern
const pattern = '(h ph) (w pw) -> h w (ph pw)';
const inputShape = [4, 4];
const axes = { ph: 2, pw: 2 };

console.log('=== Debugging Einops Pattern ===');
console.log('Pattern:', pattern);
console.log('Input shape:', inputShape);
console.log('Provided axes:', axes);

// Step 1: Parse
const ast = parse(pattern);
console.log('\nParsed AST:');
console.log('Input patterns:', JSON.stringify(ast.input, null, 2));
console.log('Output patterns:', JSON.stringify(ast.output, null, 2));

// Step 2: Resolve axes
const resolver = new AxisResolver();
const resolved = resolver.resolvePattern(ast, inputShape, axes);
console.log('\nResolved axes:');
console.log('Axis dimensions:', Object.fromEntries(resolved.axisDimensions));
console.log('Output shape:', resolved.outputShape);

// Let's manually trace what SHOULD happen:
console.log('\n=== Manual Analysis ===');
console.log('Input: [4, 4] representing a 4x4 matrix');
console.log('Pattern breaks it down as:');
console.log('  - First dim: (h ph) where h=2, ph=2, so 4 = 2*2');
console.log('  - Second dim: (w pw) where w=2, pw=2, so 4 = 2*2');
console.log('\nOutput should be [h, w, (ph pw)] = [2, 2, 4]');
console.log('\nThe 2x2 patches should be extracted as:');
console.log('Patch [0,0]: elements at positions (0,0), (0,1), (1,0), (1,1) → [1, 2, 5, 6]');
console.log('Patch [0,1]: elements at positions (0,2), (0,3), (1,2), (1,3) → [3, 4, 7, 8]');
console.log('Patch [1,0]: elements at positions (2,0), (2,1), (3,0), (3,1) → [9, 10, 13, 14]');
console.log('Patch [1,1]: elements at positions (2,2), (2,3), (3,2), (3,3) → [11, 12, 15, 16]');