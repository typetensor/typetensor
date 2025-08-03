import type { ValidRepeatPattern } from './src/einops/type-shape-resolver-repeat';

// Test ellipsis patterns
type Test1 = ValidRepeatPattern<'... -> (... r)', readonly [4, 4], { r: 2 }>;
type Test2 = ValidRepeatPattern<'... -> batch ...', readonly [3, 4, 5], { batch: 2 }>;
type Test3 = ValidRepeatPattern<'batch ... c -> batch ... c d', readonly [2, 3, 4, 5], { d: 6 }>;

// Force TypeScript to show us the resolved types
const test1: never = {} as Test1;
const test2: never = {} as Test2;
const test3: never = {} as Test3;