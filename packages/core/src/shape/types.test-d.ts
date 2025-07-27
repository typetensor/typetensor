/**
 * Type tests for the shape system
 *
 * These tests validate that our type-level shape operations work correctly
 * at compile time. They use expectTypeOf to ensure type safety and catch
 * regressions in our type system.
 */

import { expectTypeOf } from 'expect-type';
import type {
  Shape,
  DynamicShape,
  SymbolicShape,
  SymbolicDim,
  Product,
  Length,
  IsEmpty,
  Head,
  Tail,
  Last,
  Init,
  Concat,
  Reverse,
  Take,
  Drop,
  Permute,
  At,
  Transpose,
  Squeeze,
  Unsqueeze,
  CanBroadcast,
  BroadcastShapes,
  Equals,
  IsAssignableTo,
  CanReshape,
  IsMatMulCompatible,
  MatMulShape,
  ShapeError,
  ShapeMismatchError,
  IncompatibleShapes,
  TupleOf,
  Shape1D,
  Shape2D,
  Shape3D,
  Shape4D,
  BatchShape,
  ImageShape,
  SequenceShape,
  AttentionShape,
  SlicedShape,
} from './types';

// =============================================================================
// Basic Shape Types
// =============================================================================
{
  expectTypeOf<Shape>().toEqualTypeOf<readonly number[]>();
  expectTypeOf<readonly [2, 3, 4]>().toExtend<Shape>();
}

{
  // eslint-disable-next-line @typescript-eslint/no-redundant-type-constituents
  expectTypeOf<DynamicShape>().toEqualTypeOf<readonly (number | -1)[]>();
  expectTypeOf<[2, -1, 4]>().toExtend<DynamicShape>();
  expectTypeOf<readonly [2, -1, 4]>().toExtend<DynamicShape>();
}

{
  type TestSymbolic = SymbolicDim<'batch'>;
  expectTypeOf<readonly [TestSymbolic, 224, 224, 3]>().toExtend<SymbolicShape>();
}

{
  expectTypeOf<Shape1D>().toEqualTypeOf<readonly [number]>();
  expectTypeOf<Shape2D>().toEqualTypeOf<readonly [number, number]>();
  expectTypeOf<Shape3D>().toEqualTypeOf<readonly [number, number, number]>();
  expectTypeOf<Shape4D>().toEqualTypeOf<readonly [number, number, number, number]>();

  expectTypeOf<ImageShape>().toEqualTypeOf<readonly [number, number, number, number]>();
  expectTypeOf<SequenceShape>().toEqualTypeOf<readonly [number, number]>();
  expectTypeOf<AttentionShape>().toEqualTypeOf<readonly [number, number, number]>();
}

// =============================================================================
// Shape Arithmetic and Utilities
// =============================================================================

{
  expectTypeOf<Product<[]>>().toEqualTypeOf<1>();
  expectTypeOf<Product<[5]>>().toEqualTypeOf<5>();
  expectTypeOf<Product<[2, 3]>>().toEqualTypeOf<6>();
  expectTypeOf<Product<[2, 3, 4]>>().toEqualTypeOf<24>();
  expectTypeOf<Product<[0, 5]>>().toEqualTypeOf<0>(); // Zero dimension
}

{
  expectTypeOf<Length<[]>>().toEqualTypeOf<0>();
  expectTypeOf<Length<[2, 3, 4]>>().toEqualTypeOf<3>();

  expectTypeOf<IsEmpty<[]>>().toEqualTypeOf<true>();
  expectTypeOf<IsEmpty<[2, 3]>>().toEqualTypeOf<false>();

  expectTypeOf<Head<[2, 3, 4]>>().toEqualTypeOf<2>();
  expectTypeOf<Tail<[2, 3, 4]>>().toExtend<readonly [3, 4]>();
  expectTypeOf<Last<[2, 3, 4]>>().toEqualTypeOf<4>();
  expectTypeOf<Init<[2, 3, 4]>>().toExtend<readonly [2, 3]>();
}

{
  expectTypeOf<Concat<[2, 3], [4, 5]>>().toEqualTypeOf<readonly [2, 3, 4, 5]>();
  expectTypeOf<Reverse<[2, 3, 4]>>().toEqualTypeOf<readonly [4, 3, 2]>();

  expectTypeOf<Take<[2, 3, 4, 5], 2>>().toEqualTypeOf<readonly [2, 3]>();
  expectTypeOf<Drop<[2, 3, 4, 5], 2>>().toExtend<readonly [4, 5]>();

  expectTypeOf<At<[2, 3, 4], 1>>().toEqualTypeOf<3>();
}

{
  expectTypeOf<Permute<[2, 3, 4], [2, 0, 1]>>().toEqualTypeOf<readonly [4, 2, 3]>();
  expectTypeOf<Permute<[2, 3, 4], [1, 2, 0]>>().toEqualTypeOf<readonly [3, 4, 2]>();

  expectTypeOf<Transpose<[2, 3]>>().toEqualTypeOf<readonly [3, 2]>();
  expectTypeOf<Transpose<[2, 3, 4]>>().toEqualTypeOf<readonly [2, 4, 3]>();
  expectTypeOf<Transpose<[1, 2, 3, 4]>>().toEqualTypeOf<readonly [1, 2, 4, 3]>();
}

{
  expectTypeOf<Squeeze<[2, 1, 3, 1]>>().toEqualTypeOf<readonly [2, 3]>();
  expectTypeOf<Squeeze<[1, 1, 1]>>().toEqualTypeOf<readonly []>();
  expectTypeOf<Squeeze<[2, 3, 4]>>().toEqualTypeOf<readonly [2, 3, 4]>();

  expectTypeOf<Unsqueeze<[2, 3], 0>>().toEqualTypeOf<readonly [1, 2, 3]>();
  expectTypeOf<Unsqueeze<[2, 3], 1>>().toEqualTypeOf<readonly [2, 1, 3]>();
  expectTypeOf<Unsqueeze<[2, 3], 2>>().toEqualTypeOf<readonly [2, 3, 1]>();
}

// =============================================================================
// Broadcasting and Compatibility
// =============================================================================

{
  // Compatible shapes
  expectTypeOf<CanBroadcast<[1, 3], [2, 1]>>().toEqualTypeOf<true>();
  expectTypeOf<CanBroadcast<[5, 1, 3], [1, 3]>>().toEqualTypeOf<true>();
  expectTypeOf<CanBroadcast<[2, 3], [3]>>().toEqualTypeOf<true>();
  expectTypeOf<CanBroadcast<[], [2, 3]>>().toEqualTypeOf<true>(); // Scalar

  // Incompatible shapes
  expectTypeOf<CanBroadcast<[2, 3], [4, 5]>>().toEqualTypeOf<false>();
  expectTypeOf<CanBroadcast<[2, 3], [2, 4]>>().toEqualTypeOf<false>();
}

{
  expectTypeOf<BroadcastShapes<[1, 3], [2, 1]>>().toEqualTypeOf<readonly [2, 3]>();
  expectTypeOf<BroadcastShapes<[5, 1, 3], [1, 3]>>().toEqualTypeOf<readonly [5, 1, 3]>();
  expectTypeOf<BroadcastShapes<[2, 3], [3]>>().toEqualTypeOf<readonly [2, 3]>();
  expectTypeOf<BroadcastShapes<[], [2, 3]>>().toEqualTypeOf<readonly [2, 3]>();

  // These should be never (incompatible)
  expectTypeOf<BroadcastShapes<[2, 3], [4, 5]>>().toEqualTypeOf<never>();
  expectTypeOf<BroadcastShapes<[2, 3], [2, 4]>>().toEqualTypeOf<never>();
}

{
  expectTypeOf<Equals<[2, 3], [2, 3]>>().toEqualTypeOf<true>();
  expectTypeOf<Equals<[2, 3], [3, 2]>>().toEqualTypeOf<false>();
  expectTypeOf<Equals<[], []>>().toEqualTypeOf<true>();

  expectTypeOf<IsAssignableTo<[2, 3], readonly number[]>>().toEqualTypeOf<true>();
  expectTypeOf<IsAssignableTo<readonly [2, 3], Shape>>().toEqualTypeOf<true>();
}

// =============================================================================
// Shape Validation and Constraints
// =============================================================================

{
  // Valid reshapes (same total elements)
  expectTypeOf<CanReshape<[2, 6], [3, 4]>>().toEqualTypeOf<true>(); // 12 = 12
  expectTypeOf<CanReshape<[2, 3, 4], [6, 4]>>().toEqualTypeOf<true>(); // 24 = 24
  expectTypeOf<CanReshape<[24], [2, 3, 4]>>().toEqualTypeOf<true>(); // 24 = 24

  // Invalid reshapes (different total elements)
  expectTypeOf<CanReshape<[2, 3], [4, 5]>>().toEqualTypeOf<false>(); // 6 ≠ 20
  expectTypeOf<CanReshape<[2, 3], [2, 4]>>().toEqualTypeOf<false>(); // 6 ≠ 8
}

{
  // Compatible for matrix multiplication
  expectTypeOf<IsMatMulCompatible<[2, 3], [3, 4]>>().toEqualTypeOf<true>();
  expectTypeOf<IsMatMulCompatible<[5, 2, 3], [5, 3, 4]>>().toEqualTypeOf<true>();
  expectTypeOf<IsMatMulCompatible<[10], [10, 5]>>().toEqualTypeOf<true>();

  // Incompatible for matrix multiplication
  expectTypeOf<IsMatMulCompatible<[2, 3], [4, 5]>>().toEqualTypeOf<false>();
  expectTypeOf<IsMatMulCompatible<[2, 3], [2, 4]>>().toEqualTypeOf<false>();
}

{
  expectTypeOf<MatMulShape<[2, 3], [3, 4]>>().toEqualTypeOf<readonly [2, 4]>();
  expectTypeOf<MatMulShape<[5, 2, 3], [5, 3, 4]>>().toEqualTypeOf<readonly [5, 2, 4]>();
  expectTypeOf<MatMulShape<[10], [10, 5]>>().toEqualTypeOf<readonly [5]>();

  // Batched matrix multiplication
  expectTypeOf<MatMulShape<[32, 2, 3], [32, 3, 4]>>().toEqualTypeOf<readonly [32, 2, 4]>();
  expectTypeOf<MatMulShape<[1, 32, 2, 3], [1, 32, 3, 4]>>().toEqualTypeOf<readonly [1, 32, 2, 4]>();

  // Incompatible should be never
  expectTypeOf<MatMulShape<[2, 3], [4, 5]>>().toEqualTypeOf<never>();
}

// =============================================================================
// Error Types and Messages
// =============================================================================

{
  type TestError = ShapeError<'Test error message', { context: 'test' }>;
  expectTypeOf<TestError>().toMatchTypeOf<{
    readonly __error: 'ShapeError';
    readonly message: string;
    readonly context: unknown;
  }>();
}

{
  type MismatchError = ShapeMismatchError<[2, 3], [4, 5]>;
  expectTypeOf<MismatchError>().toMatchTypeOf<
    ShapeError<
      string,
      {
        expected: readonly [2, 3];
        actual: readonly [4, 5];
      }
    >
  >();
}

{
  type IncompatibleShapesType = IncompatibleShapes<[2, 3], [4, 5]>;
  // IncompatibleShapes is `never` with an error message attached
  expectTypeOf<IncompatibleShapesType>().toEqualTypeOf<never>();
}

// =============================================================================
// Utility Types
// =============================================================================

{
  expectTypeOf<TupleOf<1, 3>>().toEqualTypeOf<readonly [1, 1, 1]>();
  expectTypeOf<TupleOf<'x', 2>>().toEqualTypeOf<readonly ['x', 'x']>();
  expectTypeOf<TupleOf<boolean, 0>>().toMatchTypeOf<readonly []>();
}

{
  expectTypeOf<BatchShape<[224, 224, 3]>>().toEqualTypeOf<readonly [number, 224, 224, 3]>();
  expectTypeOf<BatchShape<[10]>>().toEqualTypeOf<readonly [number, 10]>();
}

// =============================================================================
// Slicing Operations
// =============================================================================

{
  // Basic slicing with null (full dimension)
  expectTypeOf<SlicedShape<[10, 20, 30], [null, null, null]>>().toEqualTypeOf<
    readonly [10, 20, 30]
  >();
  expectTypeOf<SlicedShape<[5, 10], [null, null]>>().toEqualTypeOf<readonly [5, 10]>();
}

{
  // Slicing with SliceSpec
  type Original = readonly [10, 20, 30];

  // Simple slices
  expectTypeOf<SlicedShape<Original, [{ start: 0; stop: 5 }, null, null]>>().toEqualTypeOf<
    readonly [5, 20, 30]
  >();
  expectTypeOf<SlicedShape<Original, [null, { start: 5; stop: 15 }, null]>>().toEqualTypeOf<
    readonly [10, 10, 30]
  >();
  expectTypeOf<SlicedShape<Original, [null, null, { start: 10; stop: 20 }]>>().toEqualTypeOf<
    readonly [10, 20, 10]
  >();
}

{
  // Slicing with step
  type Original = readonly [20, 30, 40];

  expectTypeOf<
    SlicedShape<Original, [{ start: 0; stop: 10; step: 2 }, null, null]>
  >().toEqualTypeOf<readonly [5, 30, 40]>(); // ceil((10-0)/2) = 5

  expectTypeOf<
    SlicedShape<Original, [null, { start: 0; stop: 30; step: 3 }, null]>
  >().toEqualTypeOf<readonly [20, 10, 40]>(); // ceil((30-0)/3) = 10

  expectTypeOf<
    SlicedShape<Original, [null, null, { start: 5; stop: 25; step: 5 }]>
  >().toEqualTypeOf<readonly [20, 30, 4]>(); // ceil((25-5)/5) = 4
}

{
  // Integer indexing (removes dimensions)
  type Original = readonly [10, 20, 30, 40];

  expectTypeOf<SlicedShape<Original, [5]>>().toEqualTypeOf<readonly [20, 30, 40]>();
  expectTypeOf<SlicedShape<Original, [5, 10]>>().toEqualTypeOf<readonly [30, 40]>();
  expectTypeOf<SlicedShape<Original, [5, 10, 15]>>().toEqualTypeOf<readonly [40]>();
  expectTypeOf<SlicedShape<Original, [5, 10, 15, 20]>>().toEqualTypeOf<readonly []>();
}

{
  // Mixed slicing and indexing
  type Original = readonly [10, 20, 30, 40];

  expectTypeOf<SlicedShape<Original, [5, null, { start: 0; stop: 10 }]>>().toEqualTypeOf<
    readonly [20, 10, 40]
  >();

  expectTypeOf<
    SlicedShape<Original, [{ start: 0; stop: 5 }, 10, null, { start: 10; stop: 30; step: 2 }]>
  >().toEqualTypeOf<readonly [5, 30, 10]>(); // [5, removed, 30, ceil((30-10)/2)=10]

  expectTypeOf<SlicedShape<Original, [null, 5, null, 10]>>().toEqualTypeOf<readonly [10, 30]>();
}

{
  // Partial indexing (fewer indices than dimensions)
  type Original = readonly [10, 20, 30, 40];

  expectTypeOf<SlicedShape<Original, [5]>>().toEqualTypeOf<readonly [20, 30, 40]>();
  expectTypeOf<SlicedShape<Original, [{ start: 0; stop: 5 }]>>().toEqualTypeOf<
    readonly [5, 20, 30, 40]
  >();
  expectTypeOf<SlicedShape<Original, [null]>>().toEqualTypeOf<readonly [10, 20, 30, 40]>();
}

{
  // Edge cases
  expectTypeOf<SlicedShape<[], []>>().toEqualTypeOf<readonly []>(); // Scalar
  expectTypeOf<SlicedShape<[10], [null]>>().toEqualTypeOf<readonly [10]>();
  expectTypeOf<SlicedShape<[10], [5]>>().toEqualTypeOf<readonly []>();
  expectTypeOf<SlicedShape<[10], [{ start: 2; stop: 8 }]>>().toEqualTypeOf<readonly [6]>();
}

{
  // Default values in SliceSpec
  type Original = readonly [20, 30];

  // Only stop specified (start defaults to 0)
  expectTypeOf<SlicedShape<Original, [{ stop: 10 }, null]>>().toEqualTypeOf<readonly [10, 30]>();

  // Only start specified (stop would be runtime-dependent)
  expectTypeOf<SlicedShape<Original, [{ start: 5 }, null]>>().toEqualTypeOf<readonly [15, 30]>();

  // Step without start/stop (would slice entire dimension with step)
  expectTypeOf<SlicedShape<Original, [{ step: 2 }, null]>>().toEqualTypeOf<readonly [10, 30]>();
}

{
  // Complex example: CNN feature extraction
  type ImageBatch = readonly [32, 224, 224, 3]; // [batch, height, width, channels]

  // Get center crop: [32, 112:224, 112:224, 3]
  type CenterCrop = SlicedShape<
    ImageBatch,
    [null, { start: 112; stop: 224 }, { start: 112; stop: 224 }, null]
  >;
  expectTypeOf<CenterCrop>().toEqualTypeOf<readonly [32, 112, 112, 3]>();

  // Get every other pixel: [32, 0:224:2, 0:224:2, 3]
  type Downsampled = SlicedShape<
    ImageBatch,
    [null, { start: 0; stop: 224; step: 2 }, { start: 0; stop: 224; step: 2 }, null]
  >;
  expectTypeOf<Downsampled>().toEqualTypeOf<readonly [32, 112, 112, 3]>();

  // Select single batch item and red channel: [0, :, :, 0]
  type SingleRed = SlicedShape<ImageBatch, [0, null, null, 0]>;
  expectTypeOf<SingleRed>().toEqualTypeOf<readonly [224, 224]>();
}

// =============================================================================
// Complex Integration Tests
// =============================================================================

{
  // Linear layer: [batch, in_features] -> [batch, out_features]
  type LinearLayerInput = readonly [number, 784];
  type LinearLayerOutput = readonly [number, 128];

  // Verify the shapes are compatible for a linear transformation
  expectTypeOf<LinearLayerInput>().toExtend<Shape>();
  expectTypeOf<LinearLayerOutput>().toExtend<Shape>();

  // Weight matrix should be [out_features, in_features]
  type WeightShape = readonly [128, 784];
  expectTypeOf<
    IsMatMulCompatible<LinearLayerInput, Transpose<WeightShape>>
  >().toEqualTypeOf<true>();
}

{
  // Input: [batch, height, width, channels]
  type ConvInput = readonly [32, 224, 224, 3];
  // Kernel: [kernel_height, kernel_width, in_channels, out_channels]
  type ConvKernel = readonly [3, 3, 3, 64];

  expectTypeOf<ConvInput>().toExtend<ImageShape>();
  expectTypeOf<ConvKernel>().toExtend<Shape4D>();

  // Output channels should match kernel out_channels
  expectTypeOf<Last<ConvKernel>>().toEqualTypeOf<64>();
}

{
  // Attention: [batch, seq_len, features]
  type AttentionInput = readonly [32, 128, 768];

  expectTypeOf<AttentionInput>().toExtend<AttentionShape>();

  // Query, Key, Value should have same shape
  expectTypeOf<AttentionInput>().toEqualTypeOf<AttentionInput>();
  expectTypeOf<AttentionInput>().toEqualTypeOf<AttentionInput>();

  // Attention weights: [batch, seq_len, seq_len]
  type AttentionWeights = readonly [32, 128, 128];
  expectTypeOf<IsMatMulCompatible<AttentionWeights, AttentionInput>>().toEqualTypeOf<true>();
}

{
  // Batch normalization: [batch, features] + [features] -> [batch, features]
  type BatchInput = readonly [32, 256];
  type BNParams = readonly [256];

  expectTypeOf<CanBroadcast<BatchInput, BNParams>>().toEqualTypeOf<true>();
  expectTypeOf<BroadcastShapes<BatchInput, BNParams>>().toEqualTypeOf<BatchInput>();

  // Layer normalization: [batch, seq, features] + [features] -> [batch, seq, features]
  type LNInput = readonly [32, 128, 768];
  type LNParams = readonly [768];

  expectTypeOf<CanBroadcast<LNInput, LNParams>>().toEqualTypeOf<true>();
  expectTypeOf<BroadcastShapes<LNInput, LNParams>>().toEqualTypeOf<LNInput>();
}

{
  // Chain: Input -> Linear -> ReLU -> Linear -> Output
  type Input = readonly [32, 784]; // MNIST batch
  type Hidden = readonly [32, 128]; // Hidden layer
  type Output = readonly [32, 10]; // 10 classes

  // First linear: input -> hidden
  type W1 = readonly [128, 784];
  expectTypeOf<IsMatMulCompatible<Input, Transpose<W1>>>().toEqualTypeOf<true>();
  expectTypeOf<MatMulShape<Input, Transpose<W1>>>().toEqualTypeOf<Hidden>();

  // ReLU preserves shape
  expectTypeOf<Hidden>().toEqualTypeOf<Hidden>();

  // Second linear: hidden -> output
  type W2 = readonly [10, 128];
  expectTypeOf<IsMatMulCompatible<Hidden, Transpose<W2>>>().toEqualTypeOf<true>();
  expectTypeOf<MatMulShape<Hidden, Transpose<W2>>>().toEqualTypeOf<Output>();
}

// =============================================================================
// Edge Cases and Boundary Conditions
// =============================================================================

{
  // Scalar (empty shape)
  expectTypeOf<Product<[]>>().toEqualTypeOf<1>();
  expectTypeOf<Length<[]>>().toEqualTypeOf<0>();
  expectTypeOf<IsEmpty<[]>>().toEqualTypeOf<true>();

  // Broadcasting with scalars
  expectTypeOf<CanBroadcast<[], [2, 3]>>().toEqualTypeOf<true>();
  expectTypeOf<BroadcastShapes<[], [2, 3]>>().toEqualTypeOf<readonly [2, 3]>();
}

{
  // Transpose with custom axes validation
  type Shape3D = readonly [2, 3, 4];

  // Valid permutations
  type ValidPerm1 = Transpose<Shape3D, [2, 0, 1]>;
  type ValidPerm2 = Transpose<Shape3D, [1, 2, 0]>;

  expectTypeOf<ValidPerm1>().toEqualTypeOf<readonly [4, 2, 3]>();
  expectTypeOf<ValidPerm2>().toEqualTypeOf<readonly [3, 4, 2]>();

  // Default transpose (swap last two)
  type DefaultTranspose = Transpose<Shape3D>;
  expectTypeOf<DefaultTranspose>().toEqualTypeOf<readonly [2, 4, 3]>();
}

{
  // Multiple squeezable dimensions
  type MultiOnes = readonly [1, 2, 1, 3, 1];
  type Squeezed = Squeeze<MultiOnes>;
  expectTypeOf<Squeezed>().toEqualTypeOf<readonly [2, 3]>();

  // Unsqueeze at boundaries
  type Original = readonly [2, 3];
  type UnsqueezedStart = Unsqueeze<Original, 0>;
  type UnsqueezedEnd = Unsqueeze<Original, 2>;

  expectTypeOf<UnsqueezedStart>().toEqualTypeOf<readonly [1, 2, 3]>();
  expectTypeOf<UnsqueezedEnd>().toEqualTypeOf<readonly [2, 3, 1]>();
}

{
  expectTypeOf<Transpose<[5]>>().toExtend<readonly [5]>();
  expectTypeOf<Squeeze<[1]>>().toExtend<readonly []>();
  expectTypeOf<Unsqueeze<[], 0>>().toExtend<readonly [1]>();
}

{
  expectTypeOf<Squeeze<[1, 1, 1]>>().toEqualTypeOf<readonly []>();
  expectTypeOf<Squeeze<[2, 1, 3, 1, 4]>>().toEqualTypeOf<readonly [2, 3, 4]>();

  // Broadcasting with ones
  expectTypeOf<CanBroadcast<[1, 1, 5], [3, 4, 1]>>().toEqualTypeOf<true>();
  expectTypeOf<BroadcastShapes<[1, 1, 5], [3, 4, 1]>>().toEqualTypeOf<readonly [3, 4, 5]>();
}

{
  // This should work (rank 8)
  type MaxRankShape = readonly [1, 2, 3, 4, 5, 6, 7, 8];
  expectTypeOf<MaxRankShape>().toExtend<Shape>();
  expectTypeOf<Length<MaxRankShape>>().toEqualTypeOf<8>();
}

// =============================================================================
// Performance and Compiler Limits
// =============================================================================

{
  // Complex but reasonable operations
  type ComplexShape = readonly [2, 3, 4, 5];
  type ComplexOp = Transpose<Reverse<Take<Drop<ComplexShape, 1>, 2>>>;

  expectTypeOf<ComplexOp>().toExtend<Shape>();
}

{
  // Test Product with larger numbers
  type LargeProduct1 = Product<[100, 100]>; // 10,000
  type LargeProduct2 = Product<[50, 50, 50]>; // 125,000
  type LargeProduct3 = Product<[10, 10, 10, 10]>; // 10,000

  expectTypeOf<LargeProduct1>().toEqualTypeOf<10000>();
  expectTypeOf<LargeProduct2>().toEqualTypeOf<125000>();
  expectTypeOf<LargeProduct3>().toEqualTypeOf<10000>();
}

{
  // Test permutation at maximum rank
  type MaxRankShape = readonly [1, 2, 3, 4, 5, 6, 7, 8];
  type ReversePermutation = Permute<MaxRankShape, [7, 6, 5, 4, 3, 2, 1, 0]>;
  type IdentityPermutation = Permute<MaxRankShape, [0, 1, 2, 3, 4, 5, 6, 7]>;

  expectTypeOf<ReversePermutation>().toEqualTypeOf<readonly [8, 7, 6, 5, 4, 3, 2, 1]>();
  expectTypeOf<IdentityPermutation>().toEqualTypeOf<MaxRankShape>();
}

{
  type TestShape = readonly [2, 3, 4];

  // Valid accesses
  expectTypeOf<At<TestShape, 0>>().toEqualTypeOf<2>();
  expectTypeOf<At<TestShape, 2>>().toEqualTypeOf<4>();

  // Out of bounds - TypeScript's keyof includes numeric indices beyond array length
  // so these actually return undefined | 2 | 3 | 4 for the union of possible values
  type OutOfBounds1 = At<TestShape, 3>;
  type OutOfBounds2 = At<TestShape, -1>;

  // These should not be assignable to concrete numbers
  expectTypeOf<OutOfBounds1>().toEqualTypeOf<never>();
  expectTypeOf<OutOfBounds2>().toEqualTypeOf<never>();
}

{
  // Common CNN shapes
  type ImageBatch = readonly [32, 224, 224, 3];
  type PoolOutput = readonly [32, 112, 112, 64];
  type FlattenOutput = readonly [32, 802816]; // 112*112*64

  expectTypeOf<ImageBatch>().toExtend<ImageShape>();
  expectTypeOf<CanReshape<PoolOutput, FlattenOutput>>().toEqualTypeOf<true>();

  // Common NLP shapes
  type TokenSequence = readonly [32, 512]; // batch, seq_len
  type EmbeddingOutput = readonly [32, 512, 768]; // batch, seq_len, embed_dim
  type AttentionOutput = readonly [32, 512, 768]; // same as input

  expectTypeOf<TokenSequence>().toExtend<SequenceShape>();
  expectTypeOf<EmbeddingOutput>().toExtend<AttentionShape>();
  expectTypeOf<AttentionOutput>().toExtend<AttentionShape>();
}

// =============================================================================
// Type-Level Function Composition
// =============================================================================

{
  type Original = readonly [2, 3, 4, 5];

  // Chain of transformations
  type Step1 = Reverse<Original>; // [5, 4, 3, 2]
  type Step2 = Take<Step1, 2>; // [5, 4]
  type Step3 = Transpose<Step2>; // [4, 5]

  expectTypeOf<Step1>().toEqualTypeOf<readonly [5, 4, 3, 2]>();
  expectTypeOf<Step2>().toEqualTypeOf<readonly [5, 4]>();
  expectTypeOf<Step3>().toEqualTypeOf<readonly [4, 5]>();

  // Verify composition
  type Composed = Transpose<Take<Reverse<Original>, 2>>;
  expectTypeOf<Composed>().toEqualTypeOf<Step3>();
}

{
  type Shape1 = readonly [1, 3, 1];
  type Shape2 = readonly [2, 1, 4];
  type Broadcast = BroadcastShapes<Shape1, Shape2>; // [2, 3, 4]
  type Squeezed = Squeeze<Concat<Broadcast, [1, 1]>>; // [2, 3, 4] + [1, 1] -> [2, 3, 4, 1, 1] -> [2, 3, 4]

  expectTypeOf<Broadcast>().toEqualTypeOf<readonly [2, 3, 4]>();
  expectTypeOf<Squeezed>().toEqualTypeOf<readonly [2, 3, 4]>();
}
