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

### Phase 1: Scanner Implementation

Create an `EinopsScanner` that can tokenize our specific syntax:

```typescript
type EinopsToken =
  | { type: 'axis'; name: string; position: Position }      // ✅ IMPLEMENTED
  | { type: 'arrow'; position: Position }                   // ✅ IMPLEMENTED
  | { type: 'lparen'; position: Position }                  // ✅ IMPLEMENTED
  | { type: 'rparen'; position: Position }                  // ✅ IMPLEMENTED
  | { type: 'ellipsis'; position: Position }                // ✅ IMPLEMENTED
  | { type: 'singleton'; position: Position }               // ✅ IMPLEMENTED
  | { type: 'whitespace'; position: Position };             // ✅ IMPLEMENTED
```

#### ✅ Progress: FULL COMPLETION
**Files**: [`scanner.ts`](./scanner.ts) | [`types.ts`](./types.ts) | [`scanner.test.ts`](./scanner.test.ts) | [`scanner.test-d.ts`](./scanner.test-d.ts)

**✅ Completed:**
- ✅ **Basic Scanner**: Character-by-character parsing with position tracking
- ✅ **Core Token Types**: `AxisToken`, `ArrowToken`, `WhitespaceToken` with position info
- ✅ **Composite Tokens**: `LparenToken`, `RparenToken` for patterns like `"(h w) c -> h w c"`
- ✅ **Ellipsis Tokens**: `EllipsisToken` for patterns like `"batch ... -> ..."`
- ✅ **Singleton Tokens**: `SingletonToken` for patterns like `"h w 1 -> h w"`
- ✅ **Error Handling**: Comprehensive error classes with helpful messages and position highlighting
- ✅ **Comprehensive Testing**: 48 runtime tests + type tests covering all implemented functionality
- ✅ **Complex Patterns**: Handles all einops syntax including `"(batch seq) embed"`, `"((a b) c) d"`, `"batch ... -> ..."`, `"h w 1 -> h w"`

### Phase 2: Parser State Machine

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

### Phase 3: Type-Level Validation

Following ArkType's pattern of compile-time validation:

```typescript
type ValidateEinopsPattern<Pattern extends string> =
  ParsePattern<Pattern> extends [infer Input, infer Output]
    ? ValidateAxes<Input, Output> extends true
      ? Pattern
      : ErrorMessage<'Invalid axis configuration'>
    : ErrorMessage<'Invalid pattern syntax'>;
```

### Phase 4: Runtime Integration

Bridge compile-time parsing with runtime execution:

```typescript
function rearrange<Pattern extends string>(
  tensor: Tensor,
  pattern: ValidateEinopsPattern<Pattern>,
  axes?: AxisSizes,
): RearrangeResult<typeof tensor, Pattern>;
```

## Implementation Status

### ✅ Phase 1: Scanner & Tokenizer (FULLY COMPLETED)
   - ✅ **Scanner Implementation**: [`scanner.ts`](./scanner.ts) - Complete scanner with all token types
   - ✅ **Token Types**: [`types.ts`](./types.ts) - Complete token definitions including all einops syntax
   - ✅ **Runtime Tests**: [`scanner.test.ts`](./scanner.test.ts) - 48 comprehensive tests covering all functionality
   - ✅ **Type Tests**: [`scanner.test-d.ts`](./scanner.test-d.ts) - Full compile-time type safety validation

   **Supports**: 
   - ✅ Simple patterns: `"a"`, `"h w -> w h"`, `"batch height width channels -> batch channels height width"`
   - ✅ Composite patterns: `"(h w) c -> h w c"`, `"(batch seq) embed"`, `"((a b) c) d"`
   - ✅ Ellipsis patterns: `"batch ... -> ..."`, `"... channels -> channels ..."`
   - ✅ Singleton patterns: `"h w 1 -> h w"`, `"batch 1 height -> batch height 1"`

### ⏳ Phase 2: AST & Type-Level Parser (READY TO START)
   - 🔄 Define AST types for einops patterns
   - ⏳ Implement pattern parsing using template literals
   - ⏳ Add axis validation
   - ⏳ Create error messages

### ⏳ Phase 3: Runtime Parser (PLANNED)
   - ⏳ Build AST from patterns
   - ⏳ Validate dimensions
   - ⏳ Generate operation sequence

### ⏳ Phase 4: Integration (PLANNED)
   - ⏳ Connect with tensor operations
   - ⏳ Add full test coverage
   - ⏳ Document API

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

## Example Implementation Goals

```typescript
// Type-safe at compile time
const reshaped = rearrange(tensor, 'batch (h w) c -> batch h w c');
//                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//                                   Pattern validated at compile time

// Clear error messages
const invalid = rearrange(tensor, 'batch h w -> batch h w unknown');
//                                 Error: Axis 'unknown' not found in input pattern

// Complex transformations
const reduced = reduce(tensor, 'batch ... h w c -> batch ... c', 'mean');
//                              Ellipsis handling for variable dimensions
```

## Next Steps

1. ✅ **Complete scanner implementation** → **FULLY COMPLETE** with all 7 token types (axis, arrow, whitespace, lparen, rparen, ellipsis, singleton)
2. 🔄 **Define AST types for einops patterns** → **READY TO START** (scanner provides complete foundation)
3. ⏳ Implement type-level parser using template literals
4. ⏳ Build runtime validator and operation planner
5. ⏳ Connect to tensor operations with full einops API

This implementation will bring the elegance of einops to TypeScript with the type safety inspired by ArkType's groundbreaking approach to string template parsing.

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

#### Phase 1: Type-Level Parser

```typescript
// Parser state following ArkType's pattern
type EinopsParseState<
  Input extends AxisPattern[] = [],
  Output extends AxisPattern[] = [],
  SeenAxes extends Record<string, number> = {},
  CurrentSide extends 'input' | 'output' = 'input',
  CompositeStack extends CompositeGroup[] = []
> = {
  input: Input;
  output: Output;
  seenAxes: SeenAxes;
  currentSide: CurrentSide;
  compositeStack: CompositeStack;
};

// Axis pattern types
type AxisPattern = 
  | SimpleAxis<string>
  | CompositeAxis<AxisPattern[]>
  | EllipsisAxis
  | SingletonAxis;

interface SimpleAxis<Name extends string> {
  type: 'simple';
  name: Name;
}

interface CompositeAxis<Axes extends AxisPattern[]> {
  type: 'composite';
  axes: Axes;
}
```

#### Phase 2: Pattern Validation

```typescript
// Validate pattern against tensor shape
type ValidatePattern<
  Pattern extends string,
  InputShape extends Shape
> = ParseEinopsPattern<Pattern> extends [infer Input, infer Output] ?
  Input extends AxisPattern[] ?
    Output extends AxisPattern[] ?
      ValidateInputAxes<Input, InputShape> extends true ?
        ValidateOutputAxes<Output, Input> extends true ?
          Pattern
        : EinopsError<"Output contains unknown axes", {output: Output, input: Input}>
      : EinopsError<"Input pattern doesn't match tensor shape", {pattern: Input, shape: InputShape}>
    : never
  : never
: EinopsError<"Invalid pattern syntax", {pattern: Pattern}>;
```

#### Phase 3: Operation Planning

```typescript
// Plan the sequence of tensor operations
interface EinopsOperation {
  type: 'reshape' | 'permute' | 'expand' | 'reduce';
  params: unknown;
}

class EinopsPlanner {
  planOperations(
    parsed: ParsedPattern,
    inputShape: Shape,
    axisValues?: Record<string, number>
  ): EinopsOperation[] {
    // 1. Resolve composite axes to simple axes
    // 2. Determine reshape operations for merging/splitting
    // 3. Calculate permutation for axis reordering
    // 4. Handle expansions and reductions
    return operations;
  }
}
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
  input: AxisPattern[];
  output: AxisPattern[];
  metadata: {
    originalPattern: string;
    inputTokens: TokenInfo[];
    outputTokens: TokenInfo[];
    arrowPosition: number;
  };
}

interface TokenInfo {
  type: "axis" | "lparen" | "rparen" | "arrow" | "ellipsis" | "whitespace";
  value: string;
  start: number;
  end: number;
}

type AxisPattern = 
  | SimpleAxis
  | CompositeAxis
  | EllipsisAxis
  | SingletonAxis;

interface SimpleAxis {
  type: "simple";
  name: string;
  position: TokenPosition; // For error reporting
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

interface TokenPosition {
  start: number;
  end: number;
}
```

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

  private mapAxesToDimensions(
    patterns: AxisPattern[],
    shape: number[],
    provided?: Record<string, number>
  ): Map<string, number> {
    const dimensions = new Map<string, number>();
    let shapeIndex = 0;

    for (const pattern of patterns) {
      switch (pattern.type) {
        case "simple":
          if (provided?.[pattern.name]) {
            dimensions.set(pattern.name, provided[pattern.name]);
          } else {
            dimensions.set(pattern.name, shape[shapeIndex++]);
          }
          break;
        
        case "composite":
          // Composite dimension is product of inner axes
          const compositeDim = shape[shapeIndex++];
          this.resolveComposite(pattern, compositeDim, dimensions, provided);
          break;
        
        case "ellipsis":
          // Consume remaining dimensions
          // Implementation depends on output pattern
          break;
      }
    }

    return dimensions;
  }

  private resolveComposite(
    composite: CompositeAxis,
    totalDim: number,
    dimensions: Map<string, number>,
    provided?: Record<string, number>
  ): void {
    const knownAxes = composite.axes.filter(
      a => a.type === "simple" && provided?.[a.name]
    );
    
    if (knownAxes.length === composite.axes.length - 1) {
      // Can infer the unknown dimension
      const knownProduct = knownAxes.reduce(
        (prod, axis) => prod * provided![axis.name], 
        1
      );
      const unknownDim = totalDim / knownProduct;
      
      if (!Number.isInteger(unknownDim)) {
        throw new Error(`Cannot evenly split dimension ${totalDim}`);
      }
      
      // Set the inferred dimension
      for (const axis of composite.axes) {
        if (axis.type === "simple" && !dimensions.has(axis.name)) {
          dimensions.set(axis.name, unknownDim);
        }
      }
    } else {
      throw new Error("Cannot infer multiple unknown dimensions in composite");
    }
  }
}
```

### 6. Operation Planning Algorithm

```typescript
interface PlannedOperation {
  type: "reshape" | "permute" | "expand" | "squeeze";
  params: any;
  resultShape: number[];
}

class OperationPlanner {
  planOperations(
    inputShape: number[],
    outputShape: number[],
    inputPattern: AxisPattern[],
    outputPattern: AxisPattern[],
    axisDims: Map<string, number>
  ): PlannedOperation[] {
    const operations: PlannedOperation[] = [];
    
    // Step 1: Flatten composites in input
    const flatInput = this.flattenComposites(inputPattern, inputShape);
    if (!this.shapeEquals(flatInput.shape, inputShape)) {
      operations.push({
        type: "reshape",
        params: { shape: flatInput.shape },
        resultShape: flatInput.shape
      });
    }
    
    // Step 2: Compute permutation
    const permutation = this.computePermutation(
      flatInput.axes,
      outputPattern
    );
    
    if (!this.isIdentityPermutation(permutation)) {
      const permutedShape = permutation.map(i => flatInput.shape[i]);
      operations.push({
        type: "permute",
        params: { axes: permutation },
        resultShape: permutedShape
      });
    }
    
    // Step 3: Final reshape if needed
    if (!this.shapeEquals(permutedShape || flatInput.shape, outputShape)) {
      operations.push({
        type: "reshape",
        params: { shape: outputShape },
        resultShape: outputShape
      });
    }
    
    return operations;
  }
}
```

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
