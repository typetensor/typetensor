import { describe, it, expect } from 'bun:test';
import {
  computeStrides,
  computeSize,
  normalizeSliceIndex,
  computeSlicedShape,
  computeSlicedStrides,
  validateSliceIndices,
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
});
