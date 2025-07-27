import { describe, it, expect } from 'bun:test';
import {
  computeStrides,
  computeSize,
  normalizeSliceIndex,
  computeSlicedShape,
  computeSlicedStrides,
  validateSliceIndices,
  computeTransposedStrides,
  computeTransposedShape,
  computePermutedStrides,
  computePermutedShape,
  validatePermutationAxes,
  normalizePermutationAxes,
} from './utils';

describe('tensor/utils', () => {
  describe('computeStrides', () => {
    it('should compute C-order strides for 1D shape', () => {
      expect(computeStrides([10])).toEqual([1]);
    });

    it('should compute C-order strides for 2D shape', () => {
      expect(computeStrides([3, 4])).toEqual([4, 1]);
    });

    it('should compute C-order strides for 3D shape', () => {
      expect(computeStrides([2, 3, 4])).toEqual([12, 4, 1]);
    });

    it('should compute C-order strides for 4D shape', () => {
      expect(computeStrides([2, 3, 4, 5])).toEqual([60, 20, 5, 1]);
    });

    it('should handle empty shape (scalar)', () => {
      expect(computeStrides([])).toEqual([]);
    });

    it('should handle shape with zeros', () => {
      expect(computeStrides([2, 0, 3])).toEqual([0, 3, 1]);
    });

    it('should handle large shapes', () => {
      expect(computeStrides([100, 200, 300])).toEqual([60000, 300, 1]);
    });
  });

  describe('computeSize', () => {
    it('should compute size for 1D shape', () => {
      expect(computeSize([10])).toBe(10);
    });

    it('should compute size for 2D shape', () => {
      expect(computeSize([3, 4])).toBe(12);
    });

    it('should compute size for 3D shape', () => {
      expect(computeSize([2, 3, 4])).toBe(24);
    });

    it('should return 1 for empty shape (scalar)', () => {
      expect(computeSize([])).toBe(1);
    });

    it('should return 0 for shape with zeros', () => {
      expect(computeSize([2, 0, 3])).toBe(0);
    });

    it('should handle large numbers', () => {
      expect(computeSize([1000, 1000])).toBe(1000000);
    });
  });

  describe('normalizeSliceIndex', () => {
    it('should return positive index unchanged if within bounds', () => {
      expect(normalizeSliceIndex(5, 10)).toBe(5);
    });

    it('should clamp positive index to dimension size', () => {
      expect(normalizeSliceIndex(15, 10)).toBe(10);
    });

    it('should handle negative indices', () => {
      expect(normalizeSliceIndex(-1, 10)).toBe(9);
      expect(normalizeSliceIndex(-2, 10)).toBe(8);
      expect(normalizeSliceIndex(-10, 10)).toBe(0);
    });

    it('should clamp negative indices that go beyond bounds', () => {
      expect(normalizeSliceIndex(-15, 10)).toBe(0);
    });

    it('should handle index at boundaries', () => {
      expect(normalizeSliceIndex(0, 10)).toBe(0);
      expect(normalizeSliceIndex(10, 10)).toBe(10);
    });
  });

  describe('computeSlicedShape', () => {
    describe('basic slicing', () => {
      it('should handle single dimension slice with start and stop', () => {
        expect(computeSlicedShape([10], [{ start: 2, stop: 8 }])).toEqual([6]);
      });

      it('should handle integer indexing (removes dimension)', () => {
        expect(computeSlicedShape([10, 20], [5])).toEqual([20]);
        expect(computeSlicedShape([10, 20, 30], [5, 10])).toEqual([30]);
      });

      it('should handle null indices (keeps dimension)', () => {
        expect(computeSlicedShape([10, 20], [null, null])).toEqual([10, 20]);
        expect(computeSlicedShape([10, 20, 30], [null, null, null])).toEqual([10, 20, 30]);
      });

      it('should handle partial indexing', () => {
        expect(computeSlicedShape([10, 20, 30], [5])).toEqual([20, 30]);
        expect(computeSlicedShape([10, 20, 30], [{ start: 2, stop: 8 }])).toEqual([6, 20, 30]);
      });
    });

    describe('step slicing', () => {
      it('should handle positive step', () => {
        expect(computeSlicedShape([10], [{ start: 0, stop: 10, step: 2 }])).toEqual([5]);
        expect(computeSlicedShape([20], [{ start: 0, stop: 20, step: 3 }])).toEqual([7]); // ceil(20/3)
      });

      it('should handle negative step', () => {
        expect(computeSlicedShape([10], [{ start: 9, stop: -1, step: -1 }])).toEqual([10]);
        expect(computeSlicedShape([20], [{ start: 15, stop: 5, step: -2 }])).toEqual([5]);
      });

      it('should handle large steps', () => {
        expect(computeSlicedShape([20], [{ step: 5 }])).toEqual([4]);
        expect(computeSlicedShape([100], [{ step: 10 }])).toEqual([10]);
      });

      it('should throw for zero step', () => {
        expect(() => computeSlicedShape([10], [{ step: 0 }])).toThrow('Slice step cannot be zero');
      });
    });

    describe('multi-dimensional slicing', () => {
      it('should handle mixed indices', () => {
        expect(computeSlicedShape([10, 20, 30], [5, { start: 5, stop: 15 }, null])).toEqual([
          10, 30,
        ]);
      });

      it('should handle complex slicing from type tests', () => {
        // From view.test-d.ts test 15
        expect(
          computeSlicedShape([8, 12, 16], [{ step: 2 }, { start: 2, stop: 10 }, { step: 3 }]),
        ).toEqual([4, 8, 6]);
      });
    });

    describe('edge cases', () => {
      it('should handle empty slices', () => {
        expect(computeSlicedShape([10], [{ start: 5, stop: 5 }])).toEqual([0]);
        expect(computeSlicedShape([10], [{ start: 0, stop: 0 }])).toEqual([0]);
      });

      it('should handle reversed slices with positive step', () => {
        expect(computeSlicedShape([10], [{ start: 8, stop: 2 }])).toEqual([0]);
      });

      it('should handle out of bounds indices', () => {
        expect(computeSlicedShape([10], [{ start: 15, stop: 20 }])).toEqual([0]);
        expect(computeSlicedShape([10], [{ start: -15, stop: 5 }])).toEqual([5]);
      });

      it('should handle negative indices in slices', () => {
        expect(computeSlicedShape([10], [{ start: -8, stop: -2 }])).toEqual([6]);
        expect(computeSlicedShape([20], [{ start: -5 }])).toEqual([5]);
      });

      it('should handle default values', () => {
        expect(computeSlicedShape([10], [{}])).toEqual([10]); // start=0, stop=10, step=1
        expect(computeSlicedShape([10], [{ start: 5 }])).toEqual([5]); // stop=10, step=1
        expect(computeSlicedShape([10], [{ stop: 5 }])).toEqual([5]); // start=0, step=1
      });
    });
  });

  describe('computeSlicedStrides', () => {
    it('should handle integer index (removes stride)', () => {
      expect(computeSlicedStrides([20, 1], [5])).toEqual([1]);
      expect(computeSlicedStrides([60, 20, 1], [5, 10])).toEqual([1]);
    });

    it('should handle null (preserves stride)', () => {
      expect(computeSlicedStrides([20, 1], [null, null])).toEqual([20, 1]);
      expect(computeSlicedStrides([60, 20, 1], [null, null, null])).toEqual([60, 20, 1]);
    });

    it('should handle SliceSpec (multiplies by step)', () => {
      expect(computeSlicedStrides([20, 1], [{ step: 2 }])).toEqual([40, 1]);
      expect(computeSlicedStrides([20, 1], [{ step: 3 }, null])).toEqual([60, 1]);
    });

    it('should handle partial indexing (preserves remaining strides)', () => {
      expect(computeSlicedStrides([60, 20, 1], [5])).toEqual([20, 1]);
      expect(computeSlicedStrides([60, 20, 1], [{ step: 2 }])).toEqual([120, 20, 1]);
    });

    it('should handle complex case from type tests', () => {
      // From view.test-d.ts test 15
      expect(
        computeSlicedStrides([192, 16, 1], [{ step: 2 }, { start: 2, stop: 10 }, { step: 3 }]),
      ).toEqual([384, 16, 3]);
    });

    it('should handle mixed indices', () => {
      expect(computeSlicedStrides([100, 10, 1], [null, 5, { step: 2 }])).toEqual([100, 2]);
    });

    it('should handle default step value', () => {
      expect(computeSlicedStrides([20, 1], [{ start: 2, stop: 8 }])).toEqual([20, 1]);
    });
  });

  describe('validateSliceIndices', () => {
    it('should pass for valid indices within bounds', () => {
      expect(() => validateSliceIndices([10, 20], [5, 10])).not.toThrow();
      expect(() => validateSliceIndices([10, 20], [null, null])).not.toThrow();
      expect(() => validateSliceIndices([10, 20], [{ start: 0, stop: 5 }])).not.toThrow();
    });

    it('should pass for partial indexing', () => {
      expect(() => validateSliceIndices([10, 20, 30], [5])).not.toThrow();
      expect(() => validateSliceIndices([10, 20, 30], [])).not.toThrow();
    });

    it('should throw for too many indices', () => {
      expect(() => validateSliceIndices([10, 20], [5, 10, 15])).toThrow(
        'Too many indices for tensor of dimension 2: got 3 indices',
      );
    });

    it('should throw for integer index out of bounds', () => {
      expect(() => validateSliceIndices([10, 20], [15])).toThrow(
        'Index 15 is out of bounds for dimension 0 with size 10',
      );
      expect(() => validateSliceIndices([10, 20], [-15])).toThrow(
        'Index -15 is out of bounds for dimension 0 with size 10',
      );
    });

    it('should handle negative indices correctly', () => {
      expect(() => validateSliceIndices([10, 20], [-1, -2])).not.toThrow();
      expect(() => validateSliceIndices([10, 20], [-10, -20])).not.toThrow();
    });

    it('should handle mixed valid and invalid indices', () => {
      expect(() => validateSliceIndices([10, 20], [5, 25])).toThrow(
        'Index 25 is out of bounds for dimension 1 with size 20',
      );
    });

    it('should not validate SliceSpec bounds (done at runtime)', () => {
      expect(() => validateSliceIndices([10], [{ start: 15, stop: 20 }])).not.toThrow();
    });
  });

  describe('transpose utilities', () => {
    describe('computeTransposedStrides', () => {
      it('should swap last two strides for 2D tensor', () => {
        expect(computeTransposedStrides([3, 4], [4, 1])).toEqual([1, 4]);
      });

      it('should swap last two strides for 3D tensor', () => {
        expect(computeTransposedStrides([2, 3, 4], [12, 4, 1])).toEqual([12, 1, 4]);
      });

      it('should swap last two strides for 4D tensor', () => {
        expect(computeTransposedStrides([2, 3, 4, 5], [60, 20, 5, 1])).toEqual([60, 20, 1, 5]);
      });

      it('should return unchanged for 1D tensor', () => {
        expect(computeTransposedStrides([10], [1])).toEqual([1]);
      });

      it('should return unchanged for scalar', () => {
        expect(computeTransposedStrides([], [])).toEqual([]);
      });
    });

    describe('computeTransposedShape', () => {
      it('should swap last two dimensions for 2D tensor', () => {
        expect(computeTransposedShape([3, 4])).toEqual([4, 3]);
      });

      it('should swap last two dimensions for 3D tensor', () => {
        expect(computeTransposedShape([2, 3, 4])).toEqual([2, 4, 3]);
      });

      it('should swap last two dimensions for 4D tensor', () => {
        expect(computeTransposedShape([2, 3, 4, 5])).toEqual([2, 3, 5, 4]);
      });

      it('should return unchanged for 1D tensor', () => {
        expect(computeTransposedShape([10])).toEqual([10]);
      });

      it('should return unchanged for scalar', () => {
        expect(computeTransposedShape([])).toEqual([]);
      });
    });
  });

  describe('permute utilities', () => {
    describe('computePermutedStrides', () => {
      it('should permute strides according to axes', () => {
        expect(computePermutedStrides([12, 4, 1], [2, 0, 1])).toEqual([1, 12, 4]);
        expect(computePermutedStrides([12, 4, 1], [1, 2, 0])).toEqual([4, 1, 12]);
        expect(computePermutedStrides([12, 4, 1], [0, 1, 2])).toEqual([12, 4, 1]); // identity
      });

      it('should handle 2D permutation', () => {
        expect(computePermutedStrides([20, 1], [1, 0])).toEqual([1, 20]);
        expect(computePermutedStrides([20, 1], [0, 1])).toEqual([20, 1]); // identity
      });

      it('should handle 4D permutation (NHWC to NCHW)', () => {
        // [batch, height, width, channels] -> [batch, channels, height, width]
        const nhwcStrides = [150528, 672, 3, 1]; // [32, 224, 224, 3]
        const axes = [0, 3, 1, 2];
        expect(computePermutedStrides(nhwcStrides, axes)).toEqual([150528, 1, 672, 3]);
      });

      it('should throw for invalid axis', () => {
        expect(() => computePermutedStrides([12, 4, 1], [0, 1, 3])).toThrow('Invalid stride for axis 3');
      });
    });

    describe('computePermutedShape', () => {
      it('should permute shape according to axes', () => {
        expect(computePermutedShape([2, 3, 4], [2, 0, 1])).toEqual([4, 2, 3]);
        expect(computePermutedShape([2, 3, 4], [1, 2, 0])).toEqual([3, 4, 2]);
        expect(computePermutedShape([2, 3, 4], [0, 1, 2])).toEqual([2, 3, 4]); // identity
      });

      it('should handle 2D permutation', () => {
        expect(computePermutedShape([10, 20], [1, 0])).toEqual([20, 10]);
        expect(computePermutedShape([10, 20], [0, 1])).toEqual([10, 20]); // identity
      });

      it('should handle 4D permutation (NHWC to NCHW)', () => {
        // [batch, height, width, channels] -> [batch, channels, height, width]
        expect(computePermutedShape([32, 224, 224, 3], [0, 3, 1, 2])).toEqual([32, 3, 224, 224]);
      });

      it('should throw for invalid axis', () => {
        expect(() => computePermutedShape([2, 3, 4], [0, 1, 3])).toThrow('Invalid dimension for axis 3');
      });
    });

    describe('validatePermutationAxes', () => {
      it('should accept valid permutations', () => {
        expect(() => validatePermutationAxes(3, [2, 0, 1])).not.toThrow();
        expect(() => validatePermutationAxes(3, [0, 1, 2])).not.toThrow(); // identity
        expect(() => validatePermutationAxes(2, [1, 0])).not.toThrow();
        expect(() => validatePermutationAxes(4, [3, 2, 1, 0])).not.toThrow(); // reverse
      });

      it('should accept negative indices', () => {
        expect(() => validatePermutationAxes(3, [-1, 0, 1])).not.toThrow();
        expect(() => validatePermutationAxes(3, [-3, -2, -1])).not.toThrow();
        expect(() => validatePermutationAxes(2, [-1, -2])).not.toThrow();
      });

      it('should throw for wrong length', () => {
        expect(() => validatePermutationAxes(3, [0, 1])).toThrow(
          'Permutation axes length 2 must match tensor rank 3'
        );
        expect(() => validatePermutationAxes(2, [0, 1, 2])).toThrow(
          'Permutation axes length 3 must match tensor rank 2'
        );
      });

      it('should throw for duplicate axes', () => {
        expect(() => validatePermutationAxes(3, [0, 0, 1])).toThrow('Duplicate axis 0');
        expect(() => validatePermutationAxes(3, [0, 1, 1])).toThrow('Duplicate axis 1');
        expect(() => validatePermutationAxes(3, [2, 1, -1])).toThrow('Duplicate axis -1'); // -1 = 2
      });

      it('should throw for out of bounds axes', () => {
        expect(() => validatePermutationAxes(3, [0, 1, 3])).toThrow('Axis 3 is out of bounds');
        expect(() => validatePermutationAxes(3, [-4, 0, 1])).toThrow('Axis -4 is out of bounds');
        expect(() => validatePermutationAxes(2, [0, 2])).toThrow('Axis 2 is out of bounds');
      });
    });

    describe('normalizePermutationAxes', () => {
      it('should convert negative indices to positive', () => {
        expect(normalizePermutationAxes([0, -1, 1], 3)).toEqual([0, 2, 1]);
        expect(normalizePermutationAxes([-3, -2, -1], 3)).toEqual([0, 1, 2]);
        expect(normalizePermutationAxes([-2, -1], 2)).toEqual([0, 1]);
      });

      it('should leave positive indices unchanged', () => {
        expect(normalizePermutationAxes([0, 1, 2], 3)).toEqual([0, 1, 2]);
        expect(normalizePermutationAxes([2, 0, 1], 3)).toEqual([2, 0, 1]);
      });

      it('should handle mixed positive and negative indices', () => {
        expect(normalizePermutationAxes([0, -1], 2)).toEqual([0, 1]);
        expect(normalizePermutationAxes([-2, 1], 2)).toEqual([0, 1]);
        expect(normalizePermutationAxes([1, -3, 2], 3)).toEqual([1, 0, 2]);
      });
    });
  });
});
