/**
 * Type tests for the symbolic shape system
 *
 * These tests validate that our symbolic shape types work correctly
 * at compile time with TypeScript.
 */

import { describe, it } from 'bun:test';
import { expectTypeOf } from 'expect-type';
import type { SymbolicDim, SymbolicShape, ResolvedShape, PartialShape } from './types';
import type {
  ConstraintType,
  SymbolicConstraint,
  DimensionBinding,
  ResolutionContext,
} from './symbolic';

// =============================================================================
// Symbolic Type Definitions
// =============================================================================

describe('Symbolic Type Definitions', () => {
  it('should define SymbolicDim correctly', () => {
    type BatchDim = SymbolicDim<'batch'>;
    type SeqLenDim = SymbolicDim<'seq_len'>;

    // These should be different types even with same structure
    expectTypeOf<BatchDim>().not.toEqualTypeOf<SeqLenDim>();

    // Should have symbolic marker
    expectTypeOf<BatchDim>().toMatchTypeOf<{ __symbolic: string }>();
  });

  it('should define SymbolicShape as array of symbolic dims and numbers', () => {
    type BatchDim = SymbolicDim<'batch'>;
    type FeaturesDim = SymbolicDim<'features'>;

    type TestSymbolicShape = SymbolicShape;

    // Should accept mixed symbolic and numeric dimensions
    expectTypeOf<readonly [BatchDim, 128, FeaturesDim]>().toExtend<TestSymbolicShape>();
    expectTypeOf<readonly [32, 768]>().toExtend<TestSymbolicShape>();
    expectTypeOf<readonly [BatchDim]>().toExtend<TestSymbolicShape>();
  });

  it('should define constraint types correctly', () => {
    expectTypeOf<ConstraintType>().toEqualTypeOf<'eq' | 'gt' | 'lt' | 'gte' | 'lte' | 'ne'>();
  });

  it('should define SymbolicConstraint interface', () => {
    type BatchDim = SymbolicDim<'batch'>;

    type TestConstraint = SymbolicConstraint;

    expectTypeOf<TestConstraint>().toMatchTypeOf<{
      readonly id: string;
      readonly left: SymbolicDim | number;
      readonly right: SymbolicDim | number;
      readonly type: ConstraintType;
      readonly description?: string;
    }>();

    // Should accept symbolic dimensions or numbers
    const constraint: TestConstraint = {
      id: 'test',
      left: {} as BatchDim,
      right: 32,
      type: 'gte',
    };
    expectTypeOf(constraint).toMatchTypeOf<TestConstraint>();
  });
});

// =============================================================================
// Resolution Type Safety
// =============================================================================

describe('Resolution Type Safety', () => {
  it('should define ResolvedShape correctly', () => {
    // ResolvedShape is a branded type - should be Shape with __resolved brand
    expectTypeOf<ResolvedShape>().toMatchTypeOf<{ readonly __resolved: true }>();

    // Should not extend plain arrays directly - needs to be branded
    expectTypeOf<readonly [32, 768]>().not.toExtend<ResolvedShape>();
    expectTypeOf<readonly []>().not.toExtend<ResolvedShape>();
  });

  it('should define PartialShape for partially resolved shapes', () => {
    // PartialShape is a branded type - should be DynamicShape with __partial brand
    expectTypeOf<PartialShape>().toMatchTypeOf<{ readonly __partial: true }>();

    // Should not extend plain arrays directly - needs to be branded
    expectTypeOf<readonly [32, -1, 768]>().not.toExtend<PartialShape>();
    expectTypeOf<readonly [32, 768]>().not.toExtend<PartialShape>();
  });

  it('should define ResolutionContext interface', () => {
    expectTypeOf<ResolutionContext>().toMatchTypeOf<{
      readonly bindings: Map<string, number>;
      readonly constraints: SymbolicConstraint[];
      readonly strictMode: boolean;
    }>();
  });

  it('should define DimensionBinding interface', () => {
    type BatchDim = SymbolicDim<'batch'>;

    expectTypeOf<DimensionBinding>().toMatchTypeOf<{
      readonly dimension: SymbolicDim;
      readonly value: number;
      readonly source: 'explicit' | 'inferred' | 'constraint';
    }>();

    const binding: DimensionBinding = {
      dimension: {} as BatchDim,
      value: 32,
      source: 'explicit',
    };
    expectTypeOf(binding).toMatchTypeOf<DimensionBinding>();
  });
});

// =============================================================================
// Type-Level Shape Operations with Symbolic Dimensions
// =============================================================================

describe('Symbolic Shape Operations', () => {
  it('should handle symbolic dimensions in type-level operations', () => {
    type BatchDim = SymbolicDim<'batch'>;
    type SeqLenDim = SymbolicDim<'seq_len'>;

    // Symbolic shapes should extend base shape types
    type SymbolicInputShape = readonly [BatchDim, SeqLenDim, 768];
    expectTypeOf<SymbolicInputShape>().toExtend<SymbolicShape>();

    // Should be different from concrete shapes
    type ConcreteShape = readonly [32, 128, 768];
    expectTypeOf<SymbolicInputShape>().not.toEqualTypeOf<ConcreteShape>();
  });

  it('should distinguish between different symbolic dimension types', () => {
    type BatchDim = SymbolicDim<'batch'>;
    type FeaturesDim = SymbolicDim<'features'>;
    type VocabDim = SymbolicDim<'vocab'>;

    // Different named symbolic dimensions should be different types
    expectTypeOf<BatchDim>().not.toEqualTypeOf<FeaturesDim>();
    expectTypeOf<FeaturesDim>().not.toEqualTypeOf<VocabDim>();
    expectTypeOf<BatchDim>().not.toEqualTypeOf<VocabDim>();

    // But same names should be the same type
    type AnotherBatchDim = SymbolicDim<'batch'>;
    expectTypeOf<BatchDim>().toEqualTypeOf<AnotherBatchDim>();
  });

  it('should handle nested symbolic shape compositions', () => {
    type BatchDim = SymbolicDim<'batch'>;
    type SeqLenDim = SymbolicDim<'seq_len'>;
    type EmbedDim = SymbolicDim<'embed_dim'>;
    type NumHeadsDim = SymbolicDim<'num_heads'>;
    type HeadDim = SymbolicDim<'head_dim'>;

    // Multi-head attention shapes
    type InputShape = readonly [BatchDim, SeqLenDim, EmbedDim];
    type MultiHeadShape = readonly [BatchDim, SeqLenDim, NumHeadsDim, HeadDim];

    expectTypeOf<InputShape>().toExtend<SymbolicShape>();
    expectTypeOf<MultiHeadShape>().toExtend<SymbolicShape>();
    expectTypeOf<InputShape>().not.toEqualTypeOf<MultiHeadShape>();
  });

  it('should handle transformer architecture type patterns', () => {
    type BatchDim = SymbolicDim<'batch'>;
    type SeqLenDim = SymbolicDim<'seq_len'>;
    type EmbedDim = SymbolicDim<'embed_dim'>;
    type VocabDim = SymbolicDim<'vocab_size'>;
    type FFNDim = SymbolicDim<'ffn_dim'>;

    // Transformer layer shapes
    type EmbeddingInputShape = readonly [BatchDim, SeqLenDim]; // Token indices
    type EmbeddingOutputShape = readonly [BatchDim, SeqLenDim, EmbedDim];
    type AttentionOutputShape = readonly [BatchDim, SeqLenDim, EmbedDim]; // Same as input
    type FFNIntermediateShape = readonly [BatchDim, SeqLenDim, FFNDim];
    type LogitsShape = readonly [BatchDim, SeqLenDim, VocabDim];

    // All should be valid symbolic shapes
    expectTypeOf<EmbeddingInputShape>().toExtend<SymbolicShape>();
    expectTypeOf<EmbeddingOutputShape>().toExtend<SymbolicShape>();
    expectTypeOf<AttentionOutputShape>().toExtend<SymbolicShape>();
    expectTypeOf<FFNIntermediateShape>().toExtend<SymbolicShape>();
    expectTypeOf<LogitsShape>().toExtend<SymbolicShape>();

    // Attention input and output should be same type
    expectTypeOf<EmbeddingOutputShape>().toEqualTypeOf<AttentionOutputShape>();
  });

  it('should handle CNN architecture type patterns', () => {
    type BatchDim = SymbolicDim<'batch'>;
    type HeightDim = SymbolicDim<'height'>;
    type WidthDim = SymbolicDim<'width'>;
    type ChannelsDim = SymbolicDim<'channels'>;
    type FiltersDim = SymbolicDim<'filters'>;

    // CNN layer shapes
    type ImageInputShape = readonly [BatchDim, HeightDim, WidthDim, ChannelsDim];
    type ConvOutputShape = readonly [BatchDim, HeightDim, WidthDim, FiltersDim];
    type PooledShape = readonly [BatchDim, HeightDim, WidthDim, FiltersDim]; // Size changes, type same
    type FlattenedShape = readonly [BatchDim, FiltersDim]; // Spatial dims flattened

    expectTypeOf<ImageInputShape>().toExtend<SymbolicShape>();
    expectTypeOf<ConvOutputShape>().toExtend<SymbolicShape>();
    expectTypeOf<PooledShape>().toExtend<SymbolicShape>();
    expectTypeOf<FlattenedShape>().toExtend<SymbolicShape>();

    // Different channel types should be different
    expectTypeOf<ImageInputShape>().not.toEqualTypeOf<ConvOutputShape>();
    expectTypeOf<ConvOutputShape>().toEqualTypeOf<PooledShape>();
  });
});

// =============================================================================
// Advanced Symbolic Type Patterns
// =============================================================================

describe('Advanced Symbolic Type Patterns', () => {
  it('should handle conditional symbolic types', () => {
    type BatchDim = SymbolicDim<'batch'>;
    type SeqLenDim = SymbolicDim<'seq_len'>;

    // Different shapes based on model type
    type EncoderShape<T extends 'encoder' | 'decoder'> = T extends 'encoder'
      ? readonly [BatchDim, SeqLenDim, 768]
      : readonly [BatchDim, SeqLenDim, 512];

    type EncoderInputType = EncoderShape<'encoder'>;
    type DecoderInputType = EncoderShape<'decoder'>;

    expectTypeOf<EncoderInputType>().toEqualTypeOf<readonly [BatchDim, SeqLenDim, 768]>();
    expectTypeOf<DecoderInputType>().toEqualTypeOf<readonly [BatchDim, SeqLenDim, 512]>();
    expectTypeOf<EncoderInputType>().not.toEqualTypeOf<DecoderInputType>();
  });

  it('should handle symbolic dimension constraints in types', () => {
    type BatchDim = SymbolicDim<'batch'>;
    type EmbedDim = SymbolicDim<'embed_dim'>;
    type NumHeadsDim = SymbolicDim<'num_heads'>;
    type HeadDim = SymbolicDim<'head_dim'>;

    // Constraint: embed_dim = num_heads * head_dim (type-level representation)
    interface MultiHeadAttentionShapes {
      input: readonly [BatchDim, 128, EmbedDim];
      query: readonly [BatchDim, 128, EmbedDim];
      key: readonly [BatchDim, 128, EmbedDim];
      value: readonly [BatchDim, 128, EmbedDim];
      multiHead: readonly [BatchDim, 128, NumHeadsDim, HeadDim];
      output: readonly [BatchDim, 128, EmbedDim];
    }

    type Shapes = MultiHeadAttentionShapes;

    // Input and output should have same type
    expectTypeOf<Shapes['input']>().toEqualTypeOf<Shapes['output']>();
    expectTypeOf<Shapes['input']>().toEqualTypeOf<Shapes['query']>();
    expectTypeOf<Shapes['query']>().toEqualTypeOf<Shapes['key']>();
    expectTypeOf<Shapes['key']>().toEqualTypeOf<Shapes['value']>();

    // Multi-head should be different structure
    expectTypeOf<Shapes['input']>().not.toEqualTypeOf<Shapes['multiHead']>();
  });

  it('should handle dynamic dimension relationships', () => {
    type BatchDim = SymbolicDim<'batch'>;
    type InputLenDim = SymbolicDim<'input_len'>;
    type OutputLenDim = SymbolicDim<'output_len'>;
    type VocabDim = SymbolicDim<'vocab'>;

    // Seq2Seq model shapes
    type EncoderInput = readonly [BatchDim, InputLenDim, VocabDim];
    type EncoderOutput = readonly [BatchDim, InputLenDim, 512]; // Hidden size
    type DecoderInput = readonly [BatchDim, OutputLenDim, VocabDim];
    type DecoderOutput = readonly [BatchDim, OutputLenDim, VocabDim];

    expectTypeOf<EncoderInput>().toExtend<SymbolicShape>();
    expectTypeOf<EncoderOutput>().toExtend<SymbolicShape>();
    expectTypeOf<DecoderInput>().toExtend<SymbolicShape>();
    expectTypeOf<DecoderOutput>().toExtend<SymbolicShape>();

    // Input and output vocab should match
    expectTypeOf<EncoderInput>().not.toEqualTypeOf<EncoderOutput>();
    expectTypeOf<DecoderInput>().toEqualTypeOf<DecoderOutput>();
  });

  it('should handle generic symbolic dimension utilities', () => {
    // Utility type to extract symbolic dimensions from a shape
    type ExtractSymbolicDims<T extends SymbolicShape> = {
      [K in keyof T]: T[K] extends SymbolicDim ? T[K] : never;
    };

    type BatchDim = SymbolicDim<'batch'>;
    type SeqLenDim = SymbolicDim<'seq_len'>;

    type TestShape = readonly [BatchDim, 128, SeqLenDim, 768];
    type ExtractedDims = ExtractSymbolicDims<TestShape>;

    // Should extract only symbolic dimensions
    expectTypeOf<ExtractedDims>().toEqualTypeOf<readonly [BatchDim, never, SeqLenDim, never]>();
  });
});
