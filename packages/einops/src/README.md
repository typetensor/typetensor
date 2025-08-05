# TypeTensor Einops Implementation

This module implements a type-safe version of einops-style tensor operations, heavily inspired by ArkType's string template parsing techniques.

## Overview

Einops (Einstein Operations) provides a powerful notation for tensor manipulations using string patterns like `"batch height width channels -> batch channels height width"`. Our implementation brings compile-time type safety to these operations using TypeScript's template literal types.

## Key Learnings from ArkType

### 1. Scanner-Based Architecture

ArkType uses a scanner class (`ark/type/parser/shift/scanner.ts`) that extends a base Scanner to parse strings character by character:

```typescript
// From ark/type/parser/shift/scanner.ts
export class ArkTypeScanner<lookahead extends string = string> extends Scanner<lookahead> {
  shiftUntilNextTerminator(): string {
    this.shiftUntilNonWhitespace();
    return this.shiftUntil(() => this.lookahead in ArkTypeScanner.terminatingChars);
  }

  static terminatingChars = {
    '<': 1,
    '>': 1,
    '=': 1,
    '|': 1,
    '&': 1,
    ')': 1,
    '[': 1,
    '%': 1,
    ',': 1,
    ':': 1,
    '?': 1,
    '#': 1,
    ...whitespaceChars,
  } as const;
}
```

For einops, we need similar terminating characters but simpler:

- `->` : Pattern separator
- `(` `)` : Composite dimension markers
- ` ` : Axis separator (whitespace is significant!)
- `...` : Ellipsis for variable dimensions

### 2. Type-Level State Machine

ArkType maintains parse state through recursive type transformations (`ark/type/parser/reduce/static.ts`):

```typescript
// From ark/type/parser/reduce/static.ts
export type StaticState = {
  root: unknown;
  branches: BranchState;
  groups: BranchState[];
  finalizer: ArkTypeScanner.FinalizingLookahead | ErrorMessage | undefined;
  scanned: string;
  unscanned: string;
};

// State transitions via conditional types
export type setRoot<s extends StaticState, root, unscanned extends string = s['unscanned']> = from<{
  root: root;
  branches: s['branches'];
  groups: s['groups'];
  finalizer: s['finalizer'];
  scanned: updateScanned<s['scanned'], s['unscanned'], unscanned>;
  unscanned: unscanned;
}>;
```

For einops, our state needs to track:

- Current position in pattern
- Axes seen on input side
- Whether we're inside composite parentheses
- Whether we've passed the `->` operator

### 3. Template Literal Type Magic

ArkType uses template literal types for string manipulation (`ark/util/strings.ts`):

```typescript
// From ark/util/strings.ts
export type firstChar<s extends string> = s extends `${infer head}${string}` ? head : '';

export type charsAfterFirst<s extends string> = s extends `${string}${infer tail}` ? tail : '';

export type lastChar<s extends string> = s extends `${infer head}${infer tail}`
  ? tail extends ''
    ? head
    : lastChar<tail>
  : s;
```

And more complex parsing from scanner:

```typescript
// From ark/type/parser/shift/scanner.ts
export type shiftUntil<
  unscanned extends string,
  terminator extends string,
  scanned extends string = '',
> =
  unscanned extends shift<infer lookahead, infer nextUnscanned>
    ? lookahead extends terminator
      ? scanned extends `${infer base}${EscapeChar}`
        ? shiftUntil<nextUnscanned, terminator, `${base}${lookahead}`>
        : [scanned, unscanned]
      : shiftUntil<nextUnscanned, terminator, `${scanned}${lookahead}`>
    : [scanned, ''];
```

### 4. Error Handling at Type Level

ArkType provides detailed error messages (`ark/type/parser/string.ts`):

```typescript
// From ark/type/parser/string.ts
export const parseString = (def: string, ctx: BaseParseContext): InnerParseResult => {
  // ... parsing logic ...
  if (s.finalizer === '>') throwParseError(writeUnexpectedCharacterMessage('>'));
  return node;
};

// Type-level error handling
type fullStringParse<s extends StaticState, $, args> = extractFinalizedResult<
  parseUntilFinalizer<s, $, args>
>;

export type extractFinalizedResult<s extends StaticState> = s['finalizer'] extends ''
  ? s['root']
  : s['finalizer'] extends ErrorMessage
    ? s['finalizer']
    : s['finalizer'] extends '?'
      ? [s['root'], '?']
      : s['finalizer'] extends '='
        ? parseDefault<s['root'], s['unscanned']>
        : ErrorMessage<writeUnexpectedCharacterMessage<s['finalizer'] & string>>;
```

### 5. Dynamic State Management

ArkType uses a DynamicState class for runtime parsing (`ark/type/parser/reduce/dynamic.ts`):

```typescript
// The dynamic state maintains both runtime values and type information
export class DynamicState extends DynamicStateWithRoot {
  // Manages the parsing process, tracking position, operators, etc.
}
```

## Einops-Specific Implementation Strategy

### Phase 1: Scanner Implementation ✅ COMPLETED

Create an `EinopsScanner` that can tokenize our specific syntax:

```typescript
type EinopsToken =
  | { type: 'axis'; name: string; position: Position }
  | { type: 'arrow'; position: Position }
  | { type: 'lparen'; position: Position }
  | { type: 'rparen'; position: Position }
  | { type: 'ellipsis'; position: Position }
  | { type: 'singleton'; position: Position }
  | { type: 'whitespace'; position: Position };
```

**Implementation**: [`scanner.ts`](./scanner.ts) - Complete scanner with position tracking and error handling
**Token Types**: [`types.ts`](./types.ts) - All 7 token types with discriminated unions
**Testing**: 48 runtime tests ([`scanner.test.ts`](./scanner.test.ts)) + type tests ([`scanner.test-d.ts`](./scanner.test-d.ts))

### Phase 2: Parser State Machine ✅ COMPLETED

Track einops-specific state:

```typescript
type EinopsParseState = {
  input: AxisPattern[];
  output: AxisPattern[];
  currentSide: 'input' | 'output';
  compositeStack: CompositeGroup[];
  seenAxes: Set<string>;
  errors: string[];
};
```

**AST Implementation**: [`ast.ts`](./ast.ts) - Complete AST structure with 4 pattern types
**Runtime Parser**: [`parser.ts`](./parser.ts) - Tokens → AST conversion with validation
**Testing**: 77 tests (42 AST + 35 parser) with full pattern coverage

### Phase 3: Type-Level Validation ✅ COMPLETED

Following ArkType's pattern of compile-time validation:

```typescript
type ValidateEinopsPattern<Pattern extends string> =
  ParsePattern<Pattern> extends [infer Input, infer Output]
    ? ValidateAxes<Input, Output> extends true
      ? Pattern
      : ErrorMessage<'Invalid axis configuration'>
    : ErrorMessage<'Invalid pattern syntax'>;
```

**Type Parser**: [`type-parser.ts`](./type-parser.ts) - Template literal parsing at compile time
**Validation**: [`validation.ts`](./validation.ts) - Axis validation with error messages
**Testing**: 120+ type tests across parser, validation, and integration

### Phase 4: Runtime Integration ✅ COMPLETED

Bridge compile-time parsing with runtime execution:

```typescript
function rearrange<Pattern extends string>(
  tensor: Tensor,
  pattern: ValidateEinopsPattern<Pattern>,
  axes?: AxisSizes,
): RearrangeResult<typeof tensor, Pattern>;
```

**Axis Resolution**: [`axis-resolver.ts`](./axis-resolver.ts) - Dimension mapping and shape computation
**Rearrange API**: [`rearrange.ts`](./rearrange.ts) - User-facing einops function
**Testing**: 107 tests (60 axis resolution + 47 rearrange integration)

## Implementation Status

### ✅ Phase 1: Scanner & Tokenizer (COMPLETED)
   - **Scanner**: [`scanner.ts`](./scanner.ts) - Character-by-character parsing with position tracking
   - **Token Types**: [`types.ts`](./types.ts) - All 7 einops token types with discriminated unions
   - **Testing**: 48 runtime tests + comprehensive type tests
   - **Pattern Support**: Simple, composite, ellipsis, and singleton patterns

### ✅ Phase 2: AST & Runtime Parser (COMPLETED)
   - **AST Structure**: [`ast.ts`](./ast.ts) - 4 pattern types (simple, composite, ellipsis, singleton)
   - **Runtime Parser**: [`parser.ts`](./parser.ts) - Tokens → AST conversion with validation
   - **Testing**: 77 tests (42 AST + 35 parser) with full coverage
   - **Error Handling**: 5 specialized error classes with position tracking

### ✅ Phase 3: Type-Level Parser (COMPLETED)
   - **Type Parser**: [`type-parser.ts`](./type-parser.ts) - Template literal parsing at compile time
   - **Validation**: [`validation.ts`](./validation.ts) - Type-level axis validation
   - **Testing**: 120+ type tests across parser, validation, and integration
   - **Features**: Nested parentheses, ellipsis support, descriptive errors

### ✅ Phase 4: Axis Resolution (COMPLETED)
   - **Axis Resolver**: [`axis-resolver.ts`](./axis-resolver.ts) - Dimension mapping and shape computation
   - **Features**: Composite resolution, ellipsis handling, singleton support
   - **Testing**: 60 comprehensive tests covering all scenarios
   - **Algorithm**: Matches PyTorch einops behavior for provided axes

### ✅ Phase 5: Basic Rearrange Implementation (COMPLETED)
   - **Rearrange API**: [`rearrange.ts`](./rearrange.ts) - User-facing einops function
   - **Operation Planning**: Basic reshape and transpose operations
   - **Testing**: 47 integration tests demonstrating real-world usage
   - **Export**: Integrated into main TypeTensor exports

### ⏳ Phase 6: Advanced Operations (PLANNED)

#### Phase 6A: `reduce` Operation
   - **Parser Extensions**: Handle reduction patterns with axis removal
   - **AST Support**: Add reduction operation type (sum/mean/max/min/prod)
   - **Type-Level**: `ReduceShape<Input, Pattern, Op>` for compile-time validation
   - **Axis Resolution**: Support split-then-reduce patterns like `'(h 2) (w 2) c -> h w c'`
   - **Integration**: Map to TypeTensor's reduction operations
   - **Test Cases**: Simple, partial, and split-reduce patterns

#### Phase 6B: `repeat` Operation
   - **Parser Extensions**: Distinguish new axes from repeated existing axes
   - **AST Support**: Add repeat count validation to pattern types
   - **Type-Level**: `RepeatShape<Input, Pattern, Counts>` for shape computation
   - **Memory Strategy**: Determine view vs copy for different repeat patterns
   - **Integration**: Use TypeTensor's expand and broadcast operations
   - **Test Cases**: New axis creation, axis repetition, multiple expansions

#### Phase 6C: `einsum` Operation
   - **New Parser**: Multi-tensor pattern syntax `'i j, j k -> i k'`
   - **AST Extensions**: `EinsumAST` with contraction index tracking
   - **Type-Level**: Index validation and output shape inference
   - **Optimization**: Contraction path planning for efficiency
   - **Integration**: Map to TypeTensor's matmul and tensor contraction ops
   - **Test Cases**: Matrix multiply, batch operations, trace, outer product

#### Phase 6D: Operation Planning Optimization
   - **Fusion**: Detect and merge consecutive reshape/permute operations
   - **View Guarantees**: Document when operations return views vs copies
   - **Backend Optimization**: Allow backends to optimize operation sequences
   - **Performance**: Minimize memory allocations and data movement

## Key Files to Reference in ArkType

1. **Scanner Implementation**: `ark/type/parser/shift/scanner.ts`
2. **State Management**: `ark/type/parser/reduce/static.ts`
3. **String Parsing**: `ark/type/parser/string.ts`
4. **Template Literal Utils**: `ark/util/strings.ts`
5. **Error Handling**: `ark/type/parser/shift/operator/operator.ts`
6. **Type-Level AST**: `ark/type/parser/ast/`

## Design Principles (from ArkType)

1. **Type-First Development**: Every runtime operation has a type-level equivalent
2. **Detailed Error Messages**: Parse errors should guide users to the solution
3. **Composable Architecture**: Each component should be independently testable
4. **Performance**: Compile-time parsing eliminates runtime overhead where possible

## Working Examples

### Currently Working ✅
```typescript
// Rearrange operations
const reshaped = rearrange(tensor, 'batch (h w) c -> batch h w c');
const transposed = rearrange(tensor, 'h w c -> c h w');
const flattened = rearrange(tensor, 'batch h w c -> batch (h w c)');
```

### Planned Operations 🔄
```typescript
// Reduce (Phase 6A)
const averaged = reduce(tensor, 'batch h w c -> batch c', 'mean');
const pooled = reduce(tensor, 'batch (h 2) (w 2) c -> batch h w c', 'max');

// Repeat (Phase 6B)
const expanded = repeat(tensor, 'h w -> h w c', { c: 3 });
const duplicated = repeat(tensor, 'h w c -> h (repeat w) c', { repeat: 2 });

// Einsum (Phase 6C)
const matmul = einsum([a, b], 'batch i j, batch j k -> batch i k');
const trace = einsum([matrix], 'i i ->');
```

## Key Implementation Achievements

### 🎯 Type-Level Parser
Successfully implemented complete type-level parsing using TypeScript's template literal types:

1. **Character-by-Character Parsing**: Following ArkType's approach
2. **Nested Parentheses Support**: Using `ts-arithmetic` for depth tracking
3. **Complete Pattern Support**: All einops patterns work at compile time
4. **Clear Error Messages**: Invalid patterns produce descriptive compile-time errors
5. **Consistent AST**: Type-level and runtime parsers produce identical structures

### 🔧 Modular Architecture
Each component is independently testable and follows single responsibility:
- Scanner handles tokenization
- Parser builds AST from tokens
- Axis Resolver maps dimensions
- Rearrange orchestrates operations

## Next Steps

1. ✅ **Scanner implementation** - Complete with all 7 token types
2. ✅ **AST types and runtime parser** - Full tokens → AST pipeline
3. ✅ **Type-level parser** - Compile-time pattern validation
4. ✅ **Axis resolution** - Dimension mapping and shape computation
5. ✅ **Basic rearrange** - Working API with operation planning
6. ⏳ **Advanced operations**:
   - 6A: `reduce` - Aggregation operations with type safety
   - 6B: `repeat` - Dimension expansion and broadcasting
   - 6C: `einsum` - Einstein summation notation
   - 6D: Optimization - Operation fusion and performance

This implementation brings the elegance of einops to TypeScript with complete type safety, inspired by ArkType's groundbreaking approach to string template parsing.

## Operation-Specific Implementation Plans

### `reduce` Operation Specification

**Pattern Syntax**:
- Basic: `'b h w c -> b c'` (reduce h,w dimensions)
- Partial: `'b h w c -> b h c'` (reduce only w)
- Split-reduce: `'b (h h2) (w w2) c -> b h w c'` (spatial pooling)
- With ellipsis: `'batch ... h w -> batch ...'` (reduce last 2 dims)

**Implementation Requirements**:
1. **Parser Changes**:
   - Detect axes present in input but missing in output
   - Validate split patterns have all parts specified
   - Handle reduction with singletons: `'h w 1 -> h w'`

2. **Type System**:
   ```typescript
   type ReductionOp = 'sum' | 'mean' | 'max' | 'min' | 'prod';
   
   type ReduceShape<
     InputShape extends Shape,
     Pattern extends string,
     Op extends ReductionOp
   > = ComputeReducedShape<ParsePattern<Pattern>, InputShape>;
   ```

3. **Runtime Mapping**:
   - `sum` → `tensor.sum(axes, keepDims)`
   - `mean` → `tensor.mean(axes, keepDims)`
   - `max` → `tensor.max(axes, keepDims)`
   - `min` → `tensor.min(axes, keepDims)`
   - `prod` → `tensor.prod(axes, keepDims)`

### `repeat` Operation Specification

**Pattern Syntax**:
- New axis: `'h w -> h w c'` (requires c value)
- Expand existing: `'h w c -> h (w repeat) c'` (requires repeat value)
- Multiple new: `'h w -> h w c1 c2'` (requires c1, c2 values)
- With composites: `'h w -> (h h2) (w w2)'` (requires h2, w2)

**Implementation Requirements**:
1. **Parser Changes**:
   - Track new axes not in input pattern
   - Detect repeat multipliers in composite patterns
   - Validate all new axes have provided sizes

2. **Type System**:
   ```typescript
   type RepeatCounts<Pattern extends string> = 
     ExtractNewAxes<Pattern> extends infer NewAxes
       ? { [K in NewAxes]: number }
       : never;
   
   type RepeatShape<
     InputShape extends Shape,
     Pattern extends string,
     Counts extends RepeatCounts<Pattern>
   > = ComputeExpandedShape<ParsePattern<Pattern>, InputShape, Counts>;
   ```

3. **Operation Strategy**:
   - New axis at end: Use `unsqueeze` then `expand`
   - New axis in middle: `unsqueeze` at position then `expand`
   - Existing axis repeat: `repeat_interleave` or manual expansion

### `einsum` Operation Specification

**Pattern Syntax**:
- Basic: `'i j, j k -> i k'` (matrix multiply)
- Batch: `'bij, bjk -> bik'` (batched matmul)
- Trace: `'ii ->'` (diagonal sum)
- Outer: `'i, j -> ij'` (outer product)
- Complex: `'pqrs, tuqvr -> pstuv'` (tensor contraction)

**Implementation Requirements**:
1. **New Parser**:
   ```typescript
   class EinsumParser {
     parseMultiPattern(pattern: string): EinsumAST {
       // Split by comma for inputs, arrow for output
       // Track index usage across patterns
       // Identify contraction vs free indices
     }
   }
   ```

2. **Contraction Analysis**:
   - Free indices: Appear in output
   - Contraction indices: In input but not output
   - Batch indices: Same position in all tensors
   - Invalid: Index in output but not input

3. **Operation Planning**:
   - Optimize contraction path (minimize intermediate sizes)
   - Map to TypeTensor operations:
     - 2D: `matmul`
     - Batch: `batchMatmul`
     - General: Sequence of `transpose`, `reshape`, `matmul`, `sum`

### Testing Strategy for New Operations

1. **Type Tests** (`*.test-d.ts`):
   - Valid patterns compile
   - Invalid patterns show errors
   - Shape inference correctness
   - Required parameters validation

2. **Runtime Tests** (`*.test.ts`):
   - Correctness against NumPy einops
   - Edge cases (empty, scalar, singleton)
   - Performance benchmarks
   - Memory usage (view vs copy)

3. **Integration Tests**:
   - Chain operations: `rearrange` → `reduce`
   - Complex real-world patterns
   - Backend compatibility

## Current Implementation Summary

### 📊 **Project Statistics**
- **Total Tests**: 232+ tests (172 runtime + 60+ type tests)
- **Implementation Files**: 13 core modules + comprehensive test suite
- **Pattern Support**: Complete einops syntax (simple, composite, ellipsis, singleton)
- **Type Safety**: Full compile-time validation with descriptive error messages
- **User API**: Working `rearrange()` function ready for use

### 🔗 **Implementation Pipeline**
- **Runtime**: String → Scanner → Tokens → Parser → AST → Axis Resolution → Tensor Operations
- **Type-level**: Template literal → Type Parser → Type AST → Validation → Compile-time errors

### ✅ **Completed Components**
1. **Scanner** - Tokenizes pattern strings with position tracking
2. **Parser** - Converts tokens to AST with error handling
3. **Type Parser** - Compile-time pattern validation
4. **Axis Resolver** - Maps axes to dimensions and computes shapes
5. **Rearrange** - User-facing API with operation planning

## TypeTensor Integration Deep Dive

After analyzing the TypeTensor codebase, here's a comprehensive guide on how our einops implementation will integrate with the existing architecture:

### Core Architecture Overview

TypeTensor follows a sophisticated type-level programming approach with:

1. **Storage Transformations**: Lazy operations described at the type level
2. **Shape System**: Compile-time shape validation and arithmetic
3. **View Operations**: Memory-efficient tensor transformations
4. **Device Abstraction**: Backend-agnostic execution model

### Key Integration Points

#### 1. Tensor Class (`tensor/tensor.ts`)

The main Tensor class uses a storage transformation pattern:

```typescript
export class Tensor<S extends AnyStorageTransformation = AnyStorageTransformation> {
  constructor(
    private readonly transform: S,
    private readonly data: DeviceData,
  ) {}
  
  // Critical methods for einops:
  reshape<NewShape extends readonly number[]>(shape: ValidReshapeShape<...>): Tensor<ReshapeOp<...>>
  permute<Axes extends readonly number[]>(axes: Axes): Tensor<PermuteOp<...>>
  view<NewShape extends readonly (number | -1)[]>(shape: NewShape): Tensor<View<...>>
  transpose(): Tensor<TransposeOp<S['__output']>>
}
```

Our einops operations will compose these primitives to achieve complex transformations.

#### 2. Shape System (`shape/types.ts` & `shape/runtime.ts`)

TypeTensor's shape system provides:

```typescript
// Type-level shape operations
export type Product<T extends Shape> // Total elements
export type CanReshape<From extends Shape, To extends Shape> // Validate reshape
export type Permute<T extends Shape, Order extends readonly number[]> // Permute dimensions
export type BroadcastShapes<A extends Shape, B extends Shape> // For reduce operations

// Runtime shape utilities
export class RuntimeShape<S extends Shape = Shape> {
  canReshapeTo(newShape: number[]): boolean
  transpose(axes?: number[]): RuntimeShape
  broadcastWith(other: RuntimeShape): RuntimeShape
}
```

We'll leverage these for:
- Validating einops patterns against tensor shapes
- Computing intermediate shapes during transformations
- Broadcasting for reduction operations

#### 3. Storage System (`storage/layout.ts` & `storage/view.ts`)

The storage transformation pattern is key:

```typescript
export interface StorageTransformation<
  OpType extends AllOperationTypes,
  Output extends TensorStorage<...>,
  Inputs extends readonly TensorStorage<...>[]
> {
  readonly __op: OpType;
  readonly __output: Output;
  readonly __inputs: Inputs;
}

// View operations we'll use
export interface ReshapeOp<Input extends AnyTensorStorage, NewShape extends Shape>
export interface PermuteOp<Input extends AnyTensorStorage, Axes extends readonly number[]>
```

Our einops operations will create these transformation objects.

#### 4. Type Safety Patterns

TypeTensor uses sophisticated type-level validation:

```typescript
// Shape validation with error messages
type ValidReshapeShape<Current extends Shape, New extends readonly number[]> = 
  number extends New['length']
    ? `[TypeTensor ❌] Shape must use 'as const' → reshape([2, 3] as const)`
    : Product<Current> extends Product<New>
      ? New
      : `[TypeTensor ❌] Cannot reshape: ${Product<Current>} ≠ ${Product<New>} elements`;

// Error types
export interface ShapeError<Message extends string, Context = unknown> {
  readonly __error: 'ShapeError';
  readonly message: Message;
  readonly context: Context;
}
```

We'll follow this pattern for einops errors.

### Einops Implementation Architecture

#### Core Components (Implemented)

```typescript
// Scanner: Tokenizes pattern strings
class EinopsScanner extends Scanner<string> {
  scanTokens(): Token[]
}

// Parser: Converts tokens to AST
class EinopsParser {
  parse(tokens: Token[]): EinopsAST
}

// Type Parser: Compile-time validation
type ParsePattern<Pattern extends string> = ...

// Axis Resolver: Maps dimensions
class AxisResolver {
  resolvePattern(ast: EinopsAST, shape: number[]): ResolvedPattern
}

// Rearrange: User-facing API
function rearrange<Pattern extends string>(
  tensor: Tensor,
  pattern: Pattern,
  axes?: AxisSizes
): Promise<Tensor>
```

#### Extended Architecture (Planned)

```typescript
// Reduce-specific components
interface ReduceAST extends EinopsAST {
  operation: 'sum' | 'mean' | 'max' | 'min' | 'prod';
  keepDims?: boolean;
}

type ValidateReducePattern<Pattern, Shape> = ...
type ComputeReduceShape<Input, Pattern, Op> = ...

// Repeat-specific components
interface RepeatAST extends EinopsAST {
  newAxes: Set<string>;
  repeatCounts: Map<string, number>;
}

type ValidateRepeatPattern<Pattern, Shape> = ...
type ComputeRepeatShape<Input, Pattern, Counts> = ...

// Einsum-specific components
interface EinsumAST {
  inputs: EinsumPattern[];
  output: EinsumPattern;
  contractionIndices: Set<string>;
}

type ValidateEinsumPattern<Pattern, Shapes> = ...
type ComputeEinsumShape<Pattern, InputShapes> = ...
```

#### Operation-Specific Planning

```typescript
// Extended operation types
type EinopsOperation = 
  | ReshapeOp 
  | PermuteOp 
  | ReduceOp    // New
  | RepeatOp    // New
  | EinsumOp;   // New

interface ReduceOp {
  type: 'reduce';
  axes: number[];
  operation: 'sum' | 'mean' | 'max' | 'min' | 'prod';
  keepDims: boolean;
}

interface RepeatOp {
  type: 'repeat';
  axis: number;
  count: number;
  mode: 'new' | 'expand';
}

interface EinsumOp {
  type: 'einsum';
  contractionAxes: number[][];
  outputOrder: number[];
}

// Extended planner for each operation type
class ReducePlanner extends OperationPlanner { ... }
class RepeatPlanner extends OperationPlanner { ... }
class EinsumPlanner extends OperationPlanner { ... }
```

#### Phase 4: Integration API

```typescript
// Main einops functions
export async function rearrange<
  Pattern extends string,
  S extends AnyStorageTransformation
>(
  tensor: Tensor<S>,
  pattern: ValidatePattern<Pattern, S['__output']['__shape']>,
  axes?: AxisSizes<Pattern>
): Promise<Tensor<ComputeEinopsTransformation<Pattern, S>>> {
  const parsed = parsePattern(pattern);
  const operations = planOperations(parsed, tensor.shape, axes);
  
  let result: Tensor = tensor;
  for (const op of operations) {
    switch (op.type) {
      case 'reshape':
        result = result.reshape(op.params.shape as const);
        break;
      case 'permute':
        result = result.permute(op.params.axes as const);
        break;
      // ... other operations
    }
  }
  
  return result as Tensor<ComputeEinopsTransformation<Pattern, S>>;
}

export async function reduce<
  Pattern extends string,
  S extends AnyStorageTransformation
>(
  tensor: Tensor<S>,
  pattern: ValidateReducePattern<Pattern, S['__output']['__shape']>,
  reduction: 'mean' | 'sum' | 'max' | 'min'
): Promise<Tensor<ComputeReduceTransformation<Pattern, S>>> {
  // Similar implementation with reduction operations
}
```

### Implementation Dependencies

```typescript
// Our einops module will depend on:
import type { Tensor } from '../tensor';
import type { 
  Shape, 
  Product, 
  CanReshape, 
  Permute, 
  ShapeToString,
  ShapeError 
} from '../shape/types';
import type { 
  StorageTransformation,
  AnyStorageTransformation,
  ComputeStrides
} from '../storage/layout';
import type { ReshapeOp, PermuteOp } from '../storage/view';
import { RuntimeShape, formatShape } from '../shape/runtime';
import { computeStrides, computeSize } from '../tensor/utils';
```

### Error Handling Strategy

Following TypeTensor's error patterns:

```typescript
// Type-level errors
type EinopsError<Message extends string, Context = unknown> = ShapeError<
  `[Einops] ${Message}`,
  Context & { operation: 'einops' }
>;

// Runtime errors with helpful messages
function throwEinopsError(message: string, context?: unknown): never {
  const errorMsg = `[Einops] ${message}`;
  if (context) {
    console.error('Error context:', context);
  }
  throw new Error(errorMsg);
}

// Pattern-specific error messages
type InvalidAxisError<Axis extends string> = EinopsError<
  `Unknown axis '${Axis}' in output pattern`,
  { hint: "All output axes must appear in input pattern or be '1' for singleton dimensions" }
>;
```

### Testing Strategy

```typescript
// Type-level tests (.test-d.ts files)
import { expectType } from '@typetensor/testing';

// Test pattern parsing
expectType<ParsePattern<"b h w c -> b c h w">>({
  input: [
    { type: 'simple', name: 'b' },
    { type: 'simple', name: 'h' },
    { type: 'simple', name: 'w' },
    { type: 'simple', name: 'c' }
  ],
  output: [
    { type: 'simple', name: 'b' },
    { type: 'simple', name: 'c' },
    { type: 'simple', name: 'h' },
    { type: 'simple', name: 'w' }
  ]
});

// Runtime tests (.test.ts files)
describe('einops rearrange', () => {
  it('should handle basic transpose', async () => {
    const tensor = await createTensor([[1, 2], [3, 4]]);
    const result = await rearrange(tensor, "h w -> w h");
    expect(await result.toArray()).toEqual([[1, 3], [2, 4]]);
  });
});
```

### Performance Considerations

1. **Compile-time parsing**: Pattern validation happens at compile time
2. **Lazy evaluation**: Operations are composed but not executed until needed
3. **View operations**: Use tensor views whenever possible to avoid copies
4. **Device optimization**: Let backends optimize operation sequences

### Next Steps

1. Implement the base scanner using TypeTensor's patterns
2. Create the type-level parser with proper error messages
3. Build the operation planner
4. Integrate with Tensor methods
5. Add comprehensive tests

This integration ensures our einops implementation is:
- Fully type-safe at compile time
- Compatible with TypeTensor's architecture
- Efficient through view operations
- Extensible for future einops operations

## Research Findings: Addressing Implementation Unknowns

### 1. Scanner Implementation Details

Based on ArkType's Scanner class (`ark/util/scanner.ts`), here's our concrete implementation approach:

```typescript
export class EinopsScanner extends Scanner<string> {
  // Track multi-character operators
  shiftArrow(): boolean {
    if (this.lookahead === "-" && this.nextLookahead === ">") {
      this.shift(); // consume '-'
      this.shift(); // consume '>'
      return true;
    }
    return false;
  }

  shiftEllipsis(): boolean {
    if (this.lookahead === "." && 
        this.chars[this.i + 1] === "." && 
        this.chars[this.i + 2] === ".") {
      this.jumpForward(3);
      return true;
    }
    return false;
  }

  // Handle whitespace as significant delimiter
  isAxisDelimiter(): boolean {
    return this.lookahead in whitespaceChars || 
           this.lookahead === "(" || 
           this.lookahead === ")" ||
           this.lookahead === "";
  }

  // Shift until we hit a delimiter, collecting axis name
  shiftAxisName(): string {
    return this.shiftUntil((scanner, shifted) => {
      return scanner.isAxisDelimiter() || 
             scanner.lookahead === "-" || // Could be arrow start
             scanner.lookahead === ".";   // Could be ellipsis start
    });
  }
}
```

### 2. TypeScript Recursion Limits & Workarounds

Based on research, TypeScript has these limits:
- **Instantiation depth**: 50 (standard recursion)
- **Tail recursion**: 1000 (hard limit)
- **Type instantiation count**: Variable, typically ~500,000

**Strategies for einops patterns:**

```typescript
// Use deferred instantiation for deep recursion
type ParsePatternDeferred<Pattern extends string> = {
  // Defer instantiation by wrapping in object
  result: ParsePatternImpl<Pattern>;
}["result"];

// Break recursion every N steps
type ParseWithCounter<
  Pattern extends string,
  Counter extends number = 0
> = Counter extends 10 
  ? { continue: ParseWithCounter<Pattern, 0> }["continue"]
  : ActualParse<Pattern, Counter>;

// For axis validation, use accumulator pattern
type ValidateAxes<
  Axes extends string[],
  Seen extends Record<string, true> = {}
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends string
    ? Head extends keyof Seen
      ? `Duplicate axis: ${Head}`
      : Tail extends string[]
        ? ValidateAxes<Tail, Seen & { [K in Head]: true }>
        : never
    : never
  : true;
```

### 3. Formal Grammar Specification

Based on einops documentation analysis:

```ebnf
# EBNF Grammar for Einops Patterns
pattern         ::= input_pattern "->" output_pattern
input_pattern   ::= axis_list
output_pattern  ::= axis_list

axis_list       ::= axis_element (whitespace axis_element)*
axis_element    ::= simple_axis | composite_axis | ellipsis | singleton
simple_axis     ::= identifier
composite_axis  ::= "(" axis_list ")"
ellipsis        ::= "..."
singleton       ::= "1"

identifier      ::= letter (letter | digit | "_")*
letter          ::= "a".."z" | "A".."Z"
digit           ::= "0".."9"
whitespace      ::= " " | "\t" | "\n"
```

### 4. Complete AST Structure

```typescript
// Full AST with metadata for error reporting
interface EinopsAST {
  readonly input: readonly AxisPattern[];
  readonly output: readonly AxisPattern[];
  readonly metadata: ASTMetadata;
}

interface ASTMetadata {
  readonly originalPattern: string;
  readonly arrowPosition: Position;
  readonly inputTokenCount: number;
  readonly outputTokenCount: number;
}

type AxisPattern = 
  | SimpleAxis
  | CompositeAxis
  | EllipsisAxis
  | SingletonAxis;

interface SimpleAxis {
  type: "simple";
  name: string;
  position: TokenPosition;
}

interface CompositeAxis {
  type: "composite";
  axes: AxisPattern[];
  position: TokenPosition;
}

interface EllipsisAxis {
  type: "ellipsis";
  position: TokenPosition;
}

interface SingletonAxis {
  type: "singleton";
  position: TokenPosition;
}
```

**Implementation**: [`ast.ts`](./ast.ts) - Complete AST with 7 utility functions and full test coverage

### 5. Axis Resolution Algorithm

```typescript
class AxisResolver {
  resolvePattern(
    ast: EinopsAST,
    inputShape: number[],
    providedAxes?: Record<string, number>
  ): ResolvedPattern {
    // Step 1: Build axis dimension map from input
    const axisDimensions = this.mapAxesToDimensions(
      ast.input, 
      inputShape,
      providedAxes
    );

    // Step 2: Validate all output axes are known
    this.validateOutputAxes(ast.output, axisDimensions);

    // Step 3: Compute output shape
    const outputShape = this.computeOutputShape(
      ast.output,
      axisDimensions
    );

    return { axisDimensions, outputShape };
  }
}
```

**Implementation**: [`axis-resolver.ts`](./axis-resolver.ts) - Complete with composite resolution, ellipsis handling, and 60 tests

### 6. Operation Planning Algorithm

```typescript
interface PlannedOperation {
  type: "reshape" | "permute" | "expand" | "squeeze" | "reduce" | "repeat";
  params: any;
  resultShape: number[];
}

class OperationPlanner {
  planOperations(
    inputShape: number[],
    outputShape: number[],
    inputPattern: AxisPattern[],
    outputPattern: AxisPattern[],
    axisDims: Map<string, number>,
    operation?: 'rearrange' | 'reduce' | 'repeat' | 'einsum'
  ): PlannedOperation[] {
    // Current: Basic reshape and permute for rearrange
    // Planned: Extended for reduce, repeat, and einsum operations
  }
}
```

**Current**: Basic operation planning in [`rearrange.ts`](./rearrange.ts)
**Planned Extensions**:
- **Reduce**: Plan aggregation axes and operation type
- **Repeat**: Plan expansion strategy (view vs copy)
- **Einsum**: Plan contraction path optimization

### 7. Edge Cases Handling

```typescript
// Scalar handling
if (inputShape.length === 0) {
  if (outputPattern.length === 1 && outputPattern[0].type === "singleton") {
    // Scalar -> [1]
    return [{ type: "reshape", params: { shape: [1] }}];
  }
}

// Empty tensor handling  
if (inputShape.includes(0)) {
  // Preserve zeros through transformations
  const zeroIndices = inputShape.map((d, i) => d === 0 ? i : -1)
    .filter(i => i >= 0);
  // Track through operations...
}

// Identity patterns
if (patternEquals(inputPattern, outputPattern)) {
  return []; // No operations needed
}

// Ellipsis identity
if (inputPattern.length === 1 && inputPattern[0].type === "ellipsis" &&
    outputPattern.length === 1 && outputPattern[0].type === "ellipsis") {
  return []; // No operations needed
}
```

### 8. Error Handling Strategy

```typescript
class EinopsError extends Error {
  constructor(
    message: string,
    public readonly pattern: string,
    public readonly position?: TokenPosition,
    public readonly context?: Record<string, unknown>
  ) {
    super(EinopsError.formatMessage(message, pattern, position));
  }

  static formatMessage(
    message: string, 
    pattern: string,
    position?: TokenPosition
  ): string {
    if (position) {
      const pointer = " ".repeat(position.start) + "^".repeat(position.end - position.start);
      return `${message}\n  ${pattern}\n  ${pointer}`;
    }
    return `${message} in pattern: ${pattern}`;
  }
}

// Usage
throw new EinopsError(
  "Unknown axis 'z' in output",
  "x y -> x y z",
  { start: 10, end: 11 },
  { availableAxes: ["x", "y"] }
);
```

### 9. Performance Optimization Strategy

```typescript
// Cache parsed patterns
const patternCache = new Map<string, EinopsAST>();

// Optimize operation sequences
class OperationOptimizer {
  optimize(operations: PlannedOperation[]): PlannedOperation[] {
    // Merge consecutive reshapes
    const optimized = this.mergeReshapes(operations);
    
    // Eliminate identity permutations
    return this.removeIdentityOps(optimized);
  }
  
  // Pre-compile patterns for repeated use
  compile(pattern: string): CompiledPattern {
    const ast = parsePattern(pattern);
    return {
      ast,
      apply: (tensor: Tensor) => {
        // Optimized execution path
      }
    };
  }
}
```

### 10. Remaining Open Questions

1. **Symbolic dimensions**: How to handle patterns like `"(h w) c -> h w c"` where h,w are unknown?
   - Solution: Require explicit axis values or use divisibility constraints

2. **Multiple ellipses**: Should we support patterns like `"... a ... -> ..."`?
   - Decision: No, follow Python einops - single ellipsis only

3. **Axis name validation**: Allow unicode? Hyphens?
   - Decision: Follow JavaScript identifier rules for consistency

4. **Maximum pattern length**: Practical limits?
   - Set to 1000 characters to prevent DoS

5. **View guarantees**: When can we guarantee view vs copy?
   - Document clearly: only simple reshapes and permutations of contiguous tensors
