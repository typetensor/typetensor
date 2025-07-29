/**
 * Integration tests for the shape system
 *
 * These tests validate that all shape modules work together correctly
 * and provide comprehensive integration testing.
 */

import { describe, it, expect } from 'bun:test';
import {
  createShape,
  reshape,
  SCALAR_SHAPE,
  SHAPE_PATTERNS,
  isValidShape,
  assertValidShape,
  assertShapesCompatible,
  BroadcastManager,
  canBroadcast,
  broadcastShapes,
} from './index';
import type { Shape } from './types';

// =============================================================================
// Integration Tests
// =============================================================================

describe('Shape System Integration', () => {
  it('should work end-to-end with neural network operations', () => {
    // Create input batch
    const batchSize = 32;
    const inputFeatures = 784;
    const hiddenFeatures = 128;

    const inputShape = createShape(batchSize, inputFeatures);
    const weightsShape = createShape(hiddenFeatures, inputFeatures);
    const biasShape = createShape(hiddenFeatures);

    // Validate shapes
    expect(inputShape.dims).toEqual([32, 784]);
    expect(weightsShape.dims).toEqual([128, 784]);
    expect(biasShape.dims).toEqual([128]);

    // Check matrix multiplication compatibility
    const weightsTransposed = weightsShape.transpose();
    expect(inputShape.canMatMulWith(weightsTransposed)).toBe(true);

    // Check broadcasting compatibility for bias addition
    const linearOutput = inputShape.matMul(weightsTransposed);
    expect(linearOutput.canBroadcastWith(biasShape)).toBe(true);

    // Compute final shape
    const finalShape = linearOutput.broadcastWith(biasShape);
    expect(finalShape.dims).toEqual([32, 128]);
  });

  it('should handle convolution-like operations', () => {
    // Image batch: [batch, height, width, channels]
    const imageShape = createShape(32, 224, 224, 3);

    // Kernel: [out_channels, kernel_height, kernel_width, in_channels]
    const kernelShape = createShape(64, 3, 3, 3);

    expect(imageShape.rank).toBe(4);
    expect(kernelShape.rank).toBe(4);

    // Check input/output channel compatibility
    expect(imageShape.dim(-1)).toBe(kernelShape.dim(-1)); // in_channels match

    // Simulate conv output shape (simplified)
    const outputHeight = 222; // 224 - 3 + 1 (assuming no padding)
    const outputWidth = 222;
    const outputChannels = kernelShape.dim(0);

    const convOutputShape = createShape(32, outputHeight, outputWidth, outputChannels);
    expect(convOutputShape.dims).toEqual([32, 222, 222, 64]);
  });

  it('should handle broadcasting in batch operations', () => {
    // Batch normalization scenario
    const batchInput = createShape(32, 256); // [batch, features]
    const bnScale = createShape(256); // [features]
    const bnBias = createShape(256); // [features]

    // Check broadcasting compatibility
    expect(batchInput.canBroadcastWith(bnScale)).toBe(true);
    expect(batchInput.canBroadcastWith(bnBias)).toBe(true);

    // Compute broadcast shapes
    const scaleResult = batchInput.broadcastWith(bnScale);
    const biasResult = batchInput.broadcastWith(bnBias);

    expect(scaleResult.dims).toEqual([32, 256]);
    expect(biasResult.dims).toEqual([32, 256]);
  });

  it('should handle attention mechanism shapes', () => {
    const batchSize = 16;
    const seqLen = 128;
    const embedDim = 768;
    const numHeads = 12;
    const headDim = embedDim / numHeads;

    // Input: [batch, seq_len, embed_dim]
    const inputShape = createShape(batchSize, seqLen, embedDim);

    // QKV projections: [embed_dim, embed_dim]
    const qkvWeight = createShape(embedDim, embedDim);

    // Check matrix multiplication
    expect(inputShape.canMatMulWith(qkvWeight.transpose())).toBe(true);

    // Reshape for multi-head attention
    // [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
    const reshapedSize = batchSize * seqLen * numHeads * headDim;
    const originalSize = inputShape.size;
    expect(reshapedSize).toBe(originalSize);

    const multiHeadShape = createShape(batchSize, seqLen, numHeads, headDim);
    expect(inputShape.canReshapeTo([...multiHeadShape.dims])).toBe(true);
  });

  it('should integrate with utility functions', () => {
    // Test shape patterns
    const scalarShape = SHAPE_PATTERNS.scalar();
    const vectorShape = SHAPE_PATTERNS.vector(10);
    const matrixShape = SHAPE_PATTERNS.matrix(5, 8);
    const imageShape = SHAPE_PATTERNS.image2d(224, 224);

    expect(scalarShape).toEqual(SCALAR_SHAPE);
    expect(vectorShape).toEqual([10]);
    expect(matrixShape).toEqual([5, 8]);
    expect(imageShape).toEqual([224, 224, 3]);

    // Test validation
    expect(isValidShape(vectorShape)).toBe(true);
    expect(isValidShape([-1, 5])).toBe(false);

    // Test assertions
    expect(() => {
      assertValidShape(matrixShape);
    }).not.toThrow();
    expect(() => {
      assertValidShape([2.5, 3]);
    }).toThrow();

    // Test broadcasting compatibility
    expect(() => {
      assertShapesCompatible([2, 1], [1, 3], 'element-wise');
    }).not.toThrow();
    expect(() => {
      assertShapesCompatible([2, 3], [4, 5], 'element-wise');
    }).toThrow();
  });

  it('should handle complex reshape scenarios', () => {
    // Flatten convolutional output for dense layer
    const convOutput = createShape(32, 56, 56, 128); // After pooling
    const flattenedSize = 56 * 56 * 128;
    const flattenedShape = [32, flattenedSize];

    expect(convOutput.canReshapeTo(flattenedShape)).toBe(true);

    // Use reshape utility
    const reshapedDims = reshape(convOutput.dims, [-1, flattenedSize]);
    expect(reshapedDims).toEqual([32, flattenedSize]);

    // Dense layer
    const denseWeight = createShape(256, flattenedSize);
    const flattenedTensor = createShape(...reshapedDims);

    expect(flattenedTensor.canMatMulWith(denseWeight.transpose())).toBe(true);
  });
});

// =============================================================================
// Performance Integration Tests
// =============================================================================

describe('Performance Integration', () => {
  it('should handle large tensor operations efficiently', () => {
    const largeShape1 = createShape(1000, 1000);
    const largeShape2 = createShape(1000, 500);

    // These operations should complete quickly
    const startTime = performance.now();

    expect(largeShape1.canMatMulWith(largeShape2)).toBe(true);
    expect(largeShape1.size).toBe(1000000);
    expect(largeShape2.size).toBe(500000);

    // Stride calculations should be cached
    const strides1 = largeShape1.strides;
    const strides2 = largeShape1.strides;
    expect(strides1).toBe(strides2); // Same reference

    const endTime = performance.now();
    expect(endTime - startTime).toBeLessThan(100); // Should be very fast
  });

  it('should handle broadcasting with large shapes', () => {
    const shape1: Shape = [1000, 1];
    const shape2: Shape = [1, 500];

    const startTime = performance.now();

    expect(canBroadcast(shape1, shape2)).toBe(true);
    const result = broadcastShapes(shape1, shape2);
    expect(result).toEqual([1000, 500]);

    const endTime = performance.now();
    expect(endTime - startTime).toBeLessThan(50); // Should be very fast
  });

  it('should handle complex broadcasting scenarios', () => {
    // Multi-tensor broadcasting
    const shapes: Shape[] = [
      [1, 32, 1, 16],
      [8, 1, 4, 1],
      [1, 1, 4, 16],
    ];

    const context = BroadcastManager.createContext(shapes);
    expect(context.outputShape).toEqual([8, 32, 4, 16]);
    expect(context.strategy).toBe('general');
  });
});

// =============================================================================
// Error Handling Integration
// =============================================================================

describe('Error Handling Integration', () => {
  it('should provide detailed error messages across modules', () => {
    try {
      assertShapesCompatible([2, 3, 4], [5, 6, 7], 'matrix multiplication');
    } catch (error) {
      const message = (error as Error).message;
      expect(message).toContain('matrix multiplication');
      expect(message).toContain('[2, 3, 4]');
      expect(message).toContain('[5, 6, 7]');
    }

    // Invalid reshape
    expect(() => {
      reshape([2, 3], [4, 5]);
    }).toThrow(/Cannot reshape tensor/);

    // Invalid dimension access
    const shape = createShape(2, 3);
    expect(() => {
      shape.dim(5);
    }).toThrow(/out of bounds/);
  });

  it('should handle edge cases consistently', () => {
    // Empty shapes
    const emptyShape = createShape();
    expect(emptyShape.isScalar).toBe(true);
    expect(emptyShape.size).toBe(1);

    // Zero dimensions
    const zeroShape = createShape(0, 5);
    expect(zeroShape.size).toBe(0);
    expect(zeroShape.rank).toBe(2);

    // Large dimensions (should not overflow)
    expect(() => createShape(1e8, 1e8)).toThrow(/exceeds maximum safe size/);
  });
});
