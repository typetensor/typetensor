/**
 * Type integration tests for the shape system
 *
 * These tests validate that all shape type exports work together correctly
 * and provide comprehensive type-level integration testing.
 */

import { describe, it } from 'bun:test';
import { expectTypeOf } from 'expect-type';
import type {
  Shape,
  Shape1D,
  Shape2D,
  Shape3D,
  Shape4D,
  ImageShape,
  SequenceShape,
  AttentionShape,
  BatchShape,
  Product,
  CanBroadcast,
  BroadcastShapes,
  IsMatMulCompatible,
  MatMulShape,
  CanReshape,
  Transpose,
  Squeeze,
  Unsqueeze,
} from './index';

// =============================================================================
// Type System Integration Tests
// =============================================================================

describe('Type System Integration', () => {
  it('should export all core shape types', () => {
    // Basic shapes
    expectTypeOf<Shape>().toEqualTypeOf<readonly number[]>();
    expectTypeOf<Shape1D>().toEqualTypeOf<readonly [number]>();
    expectTypeOf<Shape2D>().toEqualTypeOf<readonly [number, number]>();
    expectTypeOf<Shape3D>().toEqualTypeOf<readonly [number, number, number]>();
    expectTypeOf<Shape4D>().toEqualTypeOf<readonly [number, number, number, number]>();

    // Specialized shapes
    expectTypeOf<ImageShape>().toEqualTypeOf<readonly [number, number, number, number]>();
    expectTypeOf<SequenceShape>().toEqualTypeOf<readonly [number, number]>();
    expectTypeOf<AttentionShape>().toEqualTypeOf<readonly [number, number, number]>();
  });

  it('should handle neural network type inference end-to-end', () => {
    // Complete neural network type flow
    type InputBatch = readonly [32, 784]; // MNIST
    type LinearWeight = readonly [128, 784];
    type Bias = readonly [128];

    // Linear transformation
    type LinearOutput = MatMulShape<InputBatch, Transpose<LinearWeight>>;
    expectTypeOf<LinearOutput>().toEqualTypeOf<readonly [32, 128]>();

    // Bias addition (broadcasting)
    expectTypeOf<CanBroadcast<LinearOutput, Bias>>().toEqualTypeOf<true>();
    type BiasedOutput = BroadcastShapes<LinearOutput, Bias>;
    expectTypeOf<BiasedOutput>().toEqualTypeOf<readonly [32, 128]>();

    // Second layer
    type OutputWeight = readonly [10, 128];
    type FinalOutput = MatMulShape<BiasedOutput, Transpose<OutputWeight>>;
    expectTypeOf<FinalOutput>().toEqualTypeOf<readonly [32, 10]>();
  });

  it('should handle convolutional network type patterns', () => {
    // Input image batch
    type ImageBatch = readonly [32, 224, 224, 3];
    expectTypeOf<ImageBatch>().toExtend<ImageShape>();

    // Convolution kernel
    type ConvKernel = readonly [3, 3, 3, 64];
    expectTypeOf<ConvKernel>().toExtend<Shape4D>();

    // Feature maps after conv (simplified - just check structure)
    type FeatureMaps = readonly [32, 222, 222, 64];
    expectTypeOf<FeatureMaps>().toExtend<ImageShape>();

    // Global average pooling (reduce spatial dimensions)
    type PooledFeatures = readonly [32, 64];
    expectTypeOf<CanReshape<FeatureMaps, PooledFeatures>>().toEqualTypeOf<false>(); // Different total size
  });

  it('should handle attention mechanism type flow', () => {
    // Transformer attention shapes
    type SeqInput = readonly [16, 128, 768]; // [batch, seq_len, embed_dim]
    expectTypeOf<SeqInput>().toExtend<AttentionShape>();

    // Multi-head reshaping
    type NumHeads = 12;
    type HeadDim = 64; // 768 / 12
    type MultiHeadShape = readonly [16, 128, NumHeads, HeadDim];

    // Check total elements match for reshape
    type OriginalSize = Product<SeqInput>;
    type ReshapedSize = Product<MultiHeadShape>;
    expectTypeOf<OriginalSize>().toEqualTypeOf<ReshapedSize>();
    expectTypeOf<CanReshape<SeqInput, MultiHeadShape>>().toEqualTypeOf<true>();
  });

  it('should handle batch processing patterns', () => {
    // Batch normalization
    type BatchFeatures = readonly [64, 256];
    type BNParams = readonly [256];

    expectTypeOf<CanBroadcast<BatchFeatures, BNParams>>().toEqualTypeOf<true>();
    type NormalizedFeatures = BroadcastShapes<BatchFeatures, BNParams>;
    expectTypeOf<NormalizedFeatures>().toEqualTypeOf<BatchFeatures>();

    // Layer normalization (different axis)
    type LNInput = readonly [32, 128, 768];
    type LNParams = readonly [768];

    expectTypeOf<CanBroadcast<LNInput, LNParams>>().toEqualTypeOf<true>();
    type LayerNormOutput = BroadcastShapes<LNInput, LNParams>;
    expectTypeOf<LayerNormOutput>().toEqualTypeOf<LNInput>();
  });

  it('should handle tensor manipulation type flows', () => {
    // Squeeze/unsqueeze operations
    type TensorWithOnes = readonly [32, 1, 64, 1];
    type Squeezed = Squeeze<TensorWithOnes>;
    expectTypeOf<Squeezed>().toEqualTypeOf<readonly [32, 64]>();

    type Unsqueezed = Unsqueeze<Squeezed, 1>;
    expectTypeOf<Unsqueezed>().toEqualTypeOf<readonly [32, 1, 64]>();

    // Transpose operations
    type Matrix = readonly [128, 256];
    type Transposed = Transpose<Matrix>;
    expectTypeOf<Transposed>().toEqualTypeOf<readonly [256, 128]>();

    // Matrix multiplication with transposed
    expectTypeOf<IsMatMulCompatible<Matrix, Transposed>>().toEqualTypeOf<true>();
    type MatMulResult = MatMulShape<Matrix, Transposed>;
    expectTypeOf<MatMulResult>().toEqualTypeOf<readonly [128, 128]>();
  });

  it('should handle batch shape patterns', () => {
    // Batch dimension patterns
    type UnbatchedImage = readonly [224, 224, 3];
    type BatchedImage = BatchShape<UnbatchedImage>;
    expectTypeOf<BatchedImage>().toEqualTypeOf<readonly [number, 224, 224, 3]>();

    type UnbatchedSequence = readonly [128, 768];
    type BatchedSequence = BatchShape<UnbatchedSequence>;
    expectTypeOf<BatchedSequence>().toEqualTypeOf<readonly [number, 128, 768]>();
  });

  it('should handle complex type compositions', () => {
    // Complex operation chain types
    type Input = readonly [32, 3, 224, 224]; // NCHW format
    type Permuted = Transpose<Input, [0, 2, 3, 1]>; // NCHW -> NHWC
    expectTypeOf<Permuted>().toEqualTypeOf<readonly [32, 224, 224, 3]>();

    // Flatten spatial dimensions
    type SpatialFlattened = readonly [32, 150528]; // 224*224*3
    expectTypeOf<CanReshape<Permuted, SpatialFlattened>>().toEqualTypeOf<true>();

    // Dense layer
    type DenseWeight = readonly [1000, 150528];
    type DenseOutput = MatMulShape<SpatialFlattened, Transpose<DenseWeight>>;
    expectTypeOf<DenseOutput>().toEqualTypeOf<readonly [32, 1000]>();
  });

  it('should validate type system constraints', () => {
    // Matrix multiplication constraints
    expectTypeOf<IsMatMulCompatible<[2, 3], [3, 4]>>().toEqualTypeOf<true>();
    expectTypeOf<IsMatMulCompatible<[2, 3], [4, 5]>>().toEqualTypeOf<false>();

    // Broadcasting constraints
    expectTypeOf<CanBroadcast<[1, 3], [2, 1]>>().toEqualTypeOf<true>();
    expectTypeOf<CanBroadcast<[2, 3], [4, 5]>>().toEqualTypeOf<false>();

    // Reshape constraints
    expectTypeOf<CanReshape<[2, 6], [3, 4]>>().toEqualTypeOf<true>();
    expectTypeOf<CanReshape<[2, 3], [4, 5]>>().toEqualTypeOf<false>();
  });
});
