/**
 * Runtime tests for the shape system
 *
 * These tests validate the actual runtime behavior of the RuntimeShape class
 * and utility functions.
 */

import { describe, it, expect, beforeEach } from 'bun:test';
import {
  RuntimeShape,
  isValidShape,
  isStaticShape,
  hasSymbolicDimensions,
  assertValidShape,
  assertShapesCompatible,
  createShape,
  reshape,
  SCALAR_SHAPE,
  SHAPE_PATTERNS,
} from './runtime';
import type { DynamicShape, SymbolicShape, SymbolicDim } from './types';

// =============================================================================
// RuntimeShape Class Tests
// =============================================================================

describe('RuntimeShape', () => {
  describe('Construction and Basic Properties', () => {
    it('should create a RuntimeShape from valid dimensions', () => {
      const shape = new RuntimeShape([2, 3, 4]);

      expect(shape.dims).toEqual([2, 3, 4]);
      expect(shape.rank).toBe(3);
      expect(shape.size).toBe(24);
      expect(shape.strides).toEqual([12, 4, 1]);
    });

    it('should handle scalar shapes', () => {
      const scalar = new RuntimeShape([]);

      expect(scalar.dims).toEqual([]);
      expect(scalar.rank).toBe(0);
      expect(scalar.size).toBe(1);
      expect(scalar.strides).toEqual([]);
      expect(scalar.isScalar).toBe(true);
      expect(scalar.isVector).toBe(false);
      expect(scalar.isMatrix).toBe(false);
    });

    it('should handle vector shapes', () => {
      const vector = new RuntimeShape([5]);

      expect(vector.rank).toBe(1);
      expect(vector.size).toBe(5);
      expect(vector.isScalar).toBe(false);
      expect(vector.isVector).toBe(true);
      expect(vector.isMatrix).toBe(false);
    });

    it('should handle matrix shapes', () => {
      const matrix = new RuntimeShape([3, 4]);

      expect(matrix.rank).toBe(2);
      expect(matrix.size).toBe(12);
      expect(matrix.isScalar).toBe(false);
      expect(matrix.isVector).toBe(false);
      expect(matrix.isMatrix).toBe(true);
    });

    it('should validate dimensions during construction', () => {
      expect(() => new RuntimeShape([-1, 3, 4])).toThrow('Invalid dimension -1');
      expect(() => new RuntimeShape([2.5, 3, 4])).toThrow('Invalid dimension 2.5');
      expect(() => new RuntimeShape([1, 2, 3, 4, 5, 6, 7, 8, 9])).toThrow(
        'exceeds maximum supported rank',
      );
      expect(() => new RuntimeShape([1e10, 1e10])).toThrow('exceeds maximum safe size');
    });
  });

  describe('Dimension Access', () => {
    let shape: RuntimeShape;

    beforeEach(() => {
      shape = new RuntimeShape([2, 3, 4, 5]);
    });

    it('should get dimensions by positive index', () => {
      expect(shape.dim(0)).toBe(2);
      expect(shape.dim(1)).toBe(3);
      expect(shape.dim(2)).toBe(4);
      expect(shape.dim(3)).toBe(5);
    });

    it('should get dimensions by negative index', () => {
      expect(shape.dim(-1)).toBe(5);
      expect(shape.dim(-2)).toBe(4);
      expect(shape.dim(-3)).toBe(3);
      expect(shape.dim(-4)).toBe(2);
    });

    it('should throw on out-of-bounds access', () => {
      expect(() => shape.dim(4)).toThrow('out of bounds');
      expect(() => shape.dim(-5)).toThrow('out of bounds');
    });
  });

  describe('Shape Manipulation', () => {
    let shape: RuntimeShape;

    beforeEach(() => {
      shape = new RuntimeShape([2, 1, 3, 1, 4]);
    });

    it('should squeeze all size-1 dimensions', () => {
      const squeezed = shape.squeeze();
      expect(squeezed.dims).toEqual([2, 3, 4]);
      expect(squeezed.rank).toBe(3);
      expect(squeezed.size).toBe(24);
    });

    it('should squeeze specific dimensions', () => {
      const squeezed = shape.squeeze([1, 3]);
      expect(squeezed.dims).toEqual([2, 3, 4]);
    });

    it('should fail to squeeze non-unit dimensions', () => {
      expect(() => shape.squeeze([0])).toThrow('Cannot squeeze dimension 0 with size 2');
    });

    it('should unsqueeze at specific positions', () => {
      const original = new RuntimeShape([2, 3]);

      const unsqueezed0 = original.unsqueeze(0);
      expect(unsqueezed0.dims).toEqual([1, 2, 3]);

      const unsqueezed1 = original.unsqueeze(1);
      expect(unsqueezed1.dims).toEqual([2, 1, 3]);

      const unsqueezed2 = original.unsqueeze(2);
      expect(unsqueezed2.dims).toEqual([2, 3, 1]);
    });

    it('should transpose dimensions', () => {
      const matrix = new RuntimeShape([3, 4]);
      const transposed = matrix.transpose();
      expect(transposed.dims).toEqual([4, 3]);
    });

    it('should transpose with custom axes', () => {
      const tensor = new RuntimeShape([2, 3, 4, 5]);
      const transposed = tensor.transpose([3, 1, 0, 2]);
      expect(transposed.dims).toEqual([5, 3, 2, 4]);
    });

    it('should validate transpose axes', () => {
      const tensor = new RuntimeShape([2, 3, 4]);

      expect(() => tensor.transpose([0, 1])).toThrow(
        'Transpose axes length 2 must match tensor rank 3',
      );
      expect(() => tensor.transpose([0, 1, 1])).toThrow('Transpose axes must be unique');
      expect(() => tensor.transpose([0, 1, 3])).toThrow('Transpose axis 3 out of bounds');
    });
  });

  describe('Index Conversion', () => {
    let shape: RuntimeShape;

    beforeEach(() => {
      shape = new RuntimeShape([2, 3, 4]);
    });

    it('should convert linear index to multi-dimensional indices', () => {
      expect(shape.unravel(0)).toEqual([0, 0, 0]);
      expect(shape.unravel(1)).toEqual([0, 0, 1]);
      expect(shape.unravel(4)).toEqual([0, 1, 0]);
      expect(shape.unravel(12)).toEqual([1, 0, 0]);
      expect(shape.unravel(23)).toEqual([1, 2, 3]);
    });

    it('should convert multi-dimensional indices to linear index', () => {
      expect(shape.ravel([0, 0, 0])).toBe(0);
      expect(shape.ravel([0, 0, 1])).toBe(1);
      expect(shape.ravel([0, 1, 0])).toBe(4);
      expect(shape.ravel([1, 0, 0])).toBe(12);
      expect(shape.ravel([1, 2, 3])).toBe(23);
    });

    it('should handle negative indices in ravel', () => {
      expect(shape.ravel([0, 0, -1])).toBe(3);
      expect(shape.ravel([-1, -1, -1])).toBe(23);
    });

    it('should validate indices bounds', () => {
      expect(() => shape.unravel(-1)).toThrow('Index -1 out of bounds');
      expect(() => shape.unravel(24)).toThrow('Index 24 out of bounds');

      expect(() => shape.ravel([0, 0])).toThrow('Expected 3 indices, got 2');
      expect(() => shape.ravel([2, 0, 0])).toThrow('Index 2 out of bounds for dimension 0');
    });
  });

  describe('Shape Comparison and Validation', () => {
    it('should check shape equality', () => {
      const shape1 = new RuntimeShape([2, 3, 4]);
      const shape2 = new RuntimeShape([2, 3, 4]);
      const shape3 = new RuntimeShape([2, 3, 5]);

      expect(shape1.equals(shape2)).toBe(true);
      expect(shape1.equals(shape3)).toBe(false);
    });

    it('should validate reshape compatibility', () => {
      const shape = new RuntimeShape([2, 3, 4]);

      expect(shape.canReshapeTo([6, 4])).toBe(true);
      expect(shape.canReshapeTo([24])).toBe(true);
      expect(shape.canReshapeTo([2, 12])).toBe(true);
      expect(shape.canReshapeTo([8, 3])).toBe(true);

      expect(shape.canReshapeTo([2, 3, 5])).toBe(false);
      expect(shape.canReshapeTo([25])).toBe(false);
    });

    it('should check broadcasting compatibility', () => {
      const shape1 = new RuntimeShape([2, 1, 4]);
      const shape2 = new RuntimeShape([3, 1]);
      const shape3 = new RuntimeShape([2, 3, 5]); // Incompatible last dimension

      expect(shape1.canBroadcastWith(shape2)).toBe(true);
      expect(shape1.canBroadcastWith(shape3)).toBe(false);
    });

    it('should compute broadcast shapes', () => {
      const shape1 = new RuntimeShape([2, 1, 4]);
      const shape2 = new RuntimeShape([3, 1]);
      const broadcasted = shape1.broadcastWith(shape2);

      expect(broadcasted.dims).toEqual([2, 3, 4]);
    });
  });

  describe('Static Methods', () => {
    it('should create shapes from arrays', () => {
      const shape = RuntimeShape.from([2, 3, 4]);
      expect(shape.dims).toEqual([2, 3, 4]);
    });

    it('should validate shape arrays', () => {
      expect(RuntimeShape.validate([2, 3, 4])).toBe(true);
      expect(RuntimeShape.validate([-1, 3, 4])).toBe(false);
      expect(RuntimeShape.validate([2.5, 3, 4])).toBe(false);
      expect(RuntimeShape.validate([1, 2, 3, 4, 5, 6, 7, 8, 9])).toBe(false);
    });

    it('should check shape equality', () => {
      expect(RuntimeShape.equals([2, 3, 4], [2, 3, 4])).toBe(true);
      expect(RuntimeShape.equals([2, 3, 4], [2, 3, 5])).toBe(false);
      expect(RuntimeShape.equals([2, 3], [2, 3, 4])).toBe(false);
    });

    it('should compute shape products', () => {
      expect(RuntimeShape.product([])).toBe(1);
      expect(RuntimeShape.product([5])).toBe(5);
      expect(RuntimeShape.product([2, 3, 4])).toBe(24);
      expect(RuntimeShape.product([0, 5])).toBe(0);
    });

    it('should check broadcasting compatibility', () => {
      expect(RuntimeShape.canBroadcast([2, 1], [1, 3])).toBe(true);
      expect(RuntimeShape.canBroadcast([2, 3], [4, 5])).toBe(false);
      expect(RuntimeShape.canBroadcast([], [2, 3])).toBe(true);
    });

    it('should compute broadcast shapes', () => {
      expect(RuntimeShape.broadcastShapes([2, 1], [1, 3])).toEqual([2, 3]);
      expect(RuntimeShape.broadcastShapes([5, 1, 3], [1, 3])).toEqual([5, 1, 3]);
      expect(RuntimeShape.broadcastShapes([], [2, 3])).toEqual([2, 3]);

      expect(() => RuntimeShape.broadcastShapes([2, 3], [4, 5])).toThrow('Cannot broadcast');
    });

    it('should handle empty tensor broadcasting correctly', () => {
      // Empty tensor with scalar (NumPy compatible)
      expect(RuntimeShape.broadcastShapes([0], [])).toEqual([0]);
      expect(RuntimeShape.broadcastShapes([], [0])).toEqual([0]);
      
      // Empty tensor with non-empty should fail (NumPy compatible)
      expect(() => RuntimeShape.broadcastShapes([0], [3])).toThrow('Cannot broadcast');
      expect(() => RuntimeShape.broadcastShapes([3], [0])).toThrow('Cannot broadcast');
      
      // Broadcasting with 1 works (NumPy compatible) - 0 always wins
      expect(RuntimeShape.broadcastShapes([2, 0], [1])).toEqual([2, 0]);
      expect(RuntimeShape.broadcastShapes([0], [1])).toEqual([0]);
      expect(RuntimeShape.broadcastShapes([1], [0])).toEqual([0]);
      
      // Multiple dimensions with zeros and incompatible sizes should fail
      expect(() => RuntimeShape.broadcastShapes([0, 3], [5, 1])).toThrow('Cannot broadcast');
      expect(() => RuntimeShape.broadcastShapes([2, 0, 4], [1, 3, 1])).toThrow('Cannot broadcast');
    });

    it('should handle edge cases in padLeft', () => {
      // Should not create negative padding
      expect(RuntimeShape.padLeft([1, 2, 3], 2)).toEqual([1, 2, 3]);
      expect(RuntimeShape.padLeft([1, 2, 3], 5)).toEqual([1, 1, 1, 2, 3]);
      expect(RuntimeShape.padLeft([], 3)).toEqual([1, 1, 1]);
    });

    it('should infer shapes from nested arrays', () => {
      expect(RuntimeShape.inferFromNestedArray(42)).toEqual([]);
      expect(RuntimeShape.inferFromNestedArray([1, 2, 3])).toEqual([3]);
      expect(
        RuntimeShape.inferFromNestedArray([
          [1, 2],
          [3, 4],
        ]),
      ).toEqual([2, 2]);
      expect(RuntimeShape.inferFromNestedArray([[[1]], [[2]]])).toEqual([2, 1, 1]);
    });
  });
});

// =============================================================================
// Type Guards and Validation Tests
// =============================================================================

describe('Type Guards and Validation', () => {
  describe('isValidShape', () => {
    it('should validate correct shapes', () => {
      expect(isValidShape([])).toBe(true);
      expect(isValidShape([2, 3, 4])).toBe(true);
      expect(isValidShape([0, 5])).toBe(true);
    });

    it('should reject invalid shapes', () => {
      expect(isValidShape(null)).toBe(false);
      expect(isValidShape('not an array')).toBe(false);
      expect(isValidShape([2.5, 3])).toBe(false);
      expect(isValidShape([-1, 3])).toBe(false);
      expect(isValidShape([1, 2, 3, 4, 5, 6, 7, 8, 9])).toBe(false);
    });
  });

  describe('isStaticShape', () => {
    it('should identify static shapes', () => {
      expect(isStaticShape([2, 3, 4] as DynamicShape)).toBe(true);
      expect(isStaticShape([0, 5] as DynamicShape)).toBe(true);
    });

    it('should identify dynamic shapes', () => {
      expect(isStaticShape([2, -1, 4] as DynamicShape)).toBe(false);
      expect(isStaticShape([-1] as DynamicShape)).toBe(false);
    });
  });

  describe('hasSymbolicDimensions', () => {
    it('should identify symbolic shapes', () => {
      const symbolicDim: SymbolicDim<'batch'> = { __symbolic: 'batch' };
      const symbolicShape: SymbolicShape = [symbolicDim, 224, 224, 3];

      expect(hasSymbolicDimensions(symbolicShape)).toBe(true);
      expect(hasSymbolicDimensions([2, 3, 4] as SymbolicShape)).toBe(false);
    });
  });

  describe('Assertion Functions', () => {
    it('should assert valid shapes', () => {
      expect(() => {
        assertValidShape([2, 3, 4]);
      }).not.toThrow();
      expect(() => {
        assertValidShape([-1, 3]);
      }).toThrow('Invalid shape');
      expect(() => {
        assertValidShape([2.5, 3], 'Custom error');
      }).toThrow('Custom error');
    });

    it('should assert shape compatibility', () => {
      expect(() => {
        assertShapesCompatible([2, 1], [1, 3], 'addition');
      }).not.toThrow();
      expect(() => {
        assertShapesCompatible([2, 3], [4, 5], 'addition');
      }).toThrow(/not compatible for addition/);

      // Check for enhanced error messages
      try {
        assertShapesCompatible([2, 3], [4, 5], 'multiplication');
      } catch (error) {
        const message = (error as Error).message;
        expect(message).toContain('Shapes [2, 3] and [4, 5]');
        expect(message).toContain('Broadcasting rule');
        expect(message).toContain('Incompatible dimensions');
        expect(message).toContain('Dimension 0: 2 vs 4');
        expect(message).toContain('Dimension 1: 3 vs 5');
      }
    });
  });
});

// =============================================================================
// Utility Functions Tests
// =============================================================================

describe('Utility Functions', () => {
  describe('Shape Patterns', () => {
    it('should create common shape patterns', () => {
      expect(SHAPE_PATTERNS.scalar()).toEqual([]);
      expect(SHAPE_PATTERNS.vector(5)).toEqual([5]);
      expect(SHAPE_PATTERNS.matrix(3, 4)).toEqual([3, 4]);
      expect(SHAPE_PATTERNS.image2d(224, 224)).toEqual([224, 224, 3]);
      expect(SHAPE_PATTERNS.image2d(224, 224, 1)).toEqual([224, 224, 1]);
      expect(SHAPE_PATTERNS.batch(32, 10)).toEqual([32, 10]);
      expect(SHAPE_PATTERNS.sequence(32, 128)).toEqual([32, 128]);
    });
  });

  describe('createShape', () => {
    it('should create RuntimeShape instances', () => {
      const shape = createShape(2, 3, 4);
      expect(shape).toBeInstanceOf(RuntimeShape);
      expect(shape.dims).toEqual([2, 3, 4]);
    });
  });

  describe('reshape', () => {
    it('should reshape with compatible dimensions', () => {
      expect(reshape([2, 6], [3, 4])).toEqual([3, 4]);
      expect(reshape([24], [2, 3, 4])).toEqual([2, 3, 4]);
    });

    it('should handle -1 dimension inference', () => {
      expect(reshape([2, 6], [-1, 4])).toEqual([3, 4]);
      expect(reshape([24], [2, -1, 4])).toEqual([2, 3, 4]);
    });

    it('should throw on incompatible reshapes', () => {
      expect(() => reshape([2, 3], [4, 5])).toThrow('Cannot reshape tensor');
      expect(() => reshape([11], [2, -1, 3])).toThrow('Cannot reshape tensor'); // 11 cannot be divided by 2*3=6
    });

    it('should validate all dimensions before processing', () => {
      expect(() => reshape([2, 3], [-2, 3])).toThrow(
        'Invalid dimension -2 at index 0: must be a non-negative integer or -1',
      );

      expect(() => reshape([2, 3], [2.5, 3])).toThrow(
        'Invalid dimension 2.5 at index 0: must be a non-negative integer or -1',
      );
    });

    it('should handle zero dimensions correctly', () => {
      expect(() => reshape([0, 5], [-1])).not.toThrow();
      expect(reshape([0, 5], [-1])).toEqual([0]);

      expect(() => reshape([2, 3], [0, -1])).toThrow(
        'Cannot infer dimension when other dimensions have size 0',
      );
    });
  });

  describe('Constants', () => {
    it('should define scalar shape constant', () => {
      expect(SCALAR_SHAPE).toEqual([]);
    });
  });
});

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

describe('Edge Cases and Error Handling', () => {
  describe('Empty and Zero-sized Shapes', () => {
    it('should handle scalar operations', () => {
      const scalar = new RuntimeShape([]);
      expect(scalar.size).toBe(1);
      expect(scalar.strides).toEqual([]);
    });

    it('should handle zero-sized dimensions', () => {
      const shape = new RuntimeShape([0, 5]);
      expect(shape.size).toBe(0);
      expect(shape.strides).toEqual([5, 1]);
    });
  });

  describe('Large Shapes', () => {
    it('should handle maximum rank shapes', () => {
      const maxRankShape = new RuntimeShape([1, 2, 1, 2, 1, 2, 1, 2]);
      expect(maxRankShape.rank).toBe(8);
      expect(maxRankShape.size).toBe(16);
    });

    it('should reject shapes exceeding maximum rank', () => {
      expect(() => new RuntimeShape([1, 2, 3, 4, 5, 6, 7, 8, 9])).toThrow(
        'exceeds maximum supported rank',
      );
    });

    it('should reject shapes with total size exceeding safe limits', () => {
      // Try to create a shape that would exceed MAX_SAFE_INTEGER
      expect(() => new RuntimeShape([1e10, 1e10])).toThrow('exceeds maximum safe size');
    });
  });

  describe('Memory and Performance', () => {
    it('should handle reasonably large tensors', () => {
      const largeShape = new RuntimeShape([100, 100]);
      expect(largeShape.size).toBe(10000);

      const indices = largeShape.unravel(5050);
      expect(largeShape.ravel(indices)).toBe(5050);
    });

    it('should reuse stride calculations', () => {
      const shape = new RuntimeShape([2, 3, 4]);
      const strides1 = shape.strides;
      const strides2 = shape.strides;

      expect(strides1).toBe(strides2); // Same reference
      expect(strides1).toEqual([12, 4, 1]);
    });
  });
});
