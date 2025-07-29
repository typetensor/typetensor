/**
 * Runtime tests for the broadcasting system
 *
 * These tests validate the actual runtime behavior of broadcasting operations,
 * including the BroadcastManager, BinaryBroadcaster, and ReductionBroadcaster classes.
 */

import { describe, it, expect } from 'bun:test';
import {
  BroadcastManager,
  BinaryBroadcaster,
  ReductionBroadcaster,
  canBroadcast,
  broadcastShapes,
  broadcastBinaryOp,
} from './broadcasting';
import type { Shape } from './types';

// =============================================================================
// Broadcasting Manager Tests
// =============================================================================

describe('BroadcastManager', () => {
  describe('Strategy Analysis', () => {
    it('should identify scalar broadcasting', () => {
      expect(BroadcastManager.analyze([[], [2, 3]])).toBe('scalar');
      expect(BroadcastManager.analyze([[2, 3], []])).toBe('scalar');
    });

    it('should identify vector broadcasting', () => {
      expect(BroadcastManager.analyze([[5], [2, 3]])).toBe('vector');
      expect(BroadcastManager.analyze([[2, 3], [5]])).toBe('vector');
      expect(BroadcastManager.analyze([[3], [5]])).toBe('vector');
    });

    it('should identify general broadcasting', () => {
      expect(
        BroadcastManager.analyze([
          [2, 1, 4],
          [1, 3, 1],
        ]),
      ).toBe('general');
      expect(
        BroadcastManager.analyze([
          [1, 2, 3],
          [4, 1, 3],
          [4, 2, 1],
        ]),
      ).toBe('general');
    });

    it('should validate input requirements', () => {
      expect(() => BroadcastManager.analyze([])).toThrow('requires at least 2 shapes');
      expect(() => BroadcastManager.analyze([[2, 3]])).toThrow('requires at least 2 shapes');
    });
  });

  describe('Shape Broadcasting', () => {
    it('should broadcast multiple shapes', () => {
      expect(
        BroadcastManager.broadcastShapes([
          [2, 1],
          [1, 3],
        ]),
      ).toEqual([2, 3]);
      expect(
        BroadcastManager.broadcastShapes([
          [1, 2, 3],
          [4, 1, 3],
          [4, 2, 1],
        ]),
      ).toEqual([4, 2, 3]);
    });

    it('should handle single shape input', () => {
      expect(BroadcastManager.broadcastShapes([[2, 3, 4]])).toEqual([2, 3, 4]);
    });

    it('should validate empty input', () => {
      expect(() => BroadcastManager.broadcastShapes([])).toThrow('Cannot broadcast empty array');
    });
  });

  describe('Broadcasting Context Creation', () => {
    it('should create complete broadcasting context', () => {
      const shapes: Shape[] = [
        [2, 1],
        [1, 3],
      ];
      const context = BroadcastManager.createContext(shapes);

      expect(context.strategy).toBe('general');
      expect(context.inputShapes).toEqual(shapes);
      expect(context.outputShape).toEqual([2, 3]);
      expect(context.expansions).toHaveLength(2);
    });

    it('should reject broadcasting that exceeds size limits', () => {
      // Number.MAX_SAFE_INTEGER is 2^53 - 1 ≈ 9e15, so we need shapes that multiply to exceed it
      const hugeShape1: Shape = [100000000, 100000000]; // 10^16 > MAX_SAFE_INTEGER
      const hugeShape2: Shape = [100000000, 1];

      expect(() => BroadcastManager.createContext([hugeShape1, hugeShape2])).toThrow(
        'exceeds maximum safe size',
      );
    });
  });

  describe('Broadcasting Execution', () => {
    it('should execute scalar broadcasting', () => {
      const shapes: Shape[] = [[], [2, 3]];
      const context = BroadcastManager.createContext(shapes);
      const inputs = [new Float32Array([5]), new Float32Array([1, 2, 3, 4, 5, 6])];

      const result = BroadcastManager.execute(context, inputs, (a, b) => a + b, Float32Array);

      expect(Array.from(result)).toEqual([6, 7, 8, 9, 10, 11]);
    });

    it('should execute vector broadcasting', () => {
      const shapes: Shape[] = [[3], [2, 3]];
      const context = BroadcastManager.createContext(shapes);
      const inputs = [new Float32Array([1, 2, 3]), new Float32Array([10, 20, 30, 40, 50, 60])];

      const result = BroadcastManager.execute(context, inputs, (a, b) => a + b, Float32Array);

      expect(Array.from(result)).toEqual([11, 22, 33, 41, 52, 63]);
    });

    it('should execute general broadcasting', () => {
      const shapes: Shape[] = [
        [2, 1],
        [1, 3],
      ];
      const context = BroadcastManager.createContext(shapes);
      const inputs = [new Float32Array([1, 2]), new Float32Array([10, 20, 30])];

      const result = BroadcastManager.execute(context, inputs, (a, b) => a * b, Float32Array);

      expect(Array.from(result)).toEqual([10, 20, 30, 20, 40, 60]);
    });
  });
});

// =============================================================================
// Binary Broadcaster Tests
// =============================================================================

describe('BinaryBroadcaster', () => {
  describe('Identical Shapes', () => {
    it('should handle identical shapes efficiently', () => {
      const shape: Shape = [2, 3];
      const input1 = new Float32Array([1, 2, 3, 4, 5, 6]);
      const input2 = new Float32Array([10, 20, 30, 40, 50, 60]);

      const { result, shape: outputShape } = BinaryBroadcaster.execute(
        shape,
        input1,
        shape,
        input2,
        (a, b) => a + b,
        Float32Array,
      );

      expect(outputShape).toEqual([2, 3]);
      expect(Array.from(result)).toEqual([11, 22, 33, 44, 55, 66]);
    });
  });

  describe('Scalar Broadcasting', () => {
    it('should handle left scalar efficiently', () => {
      const scalarShape: Shape = [];
      const tensorShape: Shape = [2, 3];
      const scalar = new Float32Array([5]);
      const tensor = new Float32Array([1, 2, 3, 4, 5, 6]);

      const { result, shape } = BinaryBroadcaster.execute(
        scalarShape,
        scalar,
        tensorShape,
        tensor,
        (a, b) => a * b,
        Float32Array,
      );

      expect(shape).toEqual([2, 3]);
      expect(Array.from(result)).toEqual([5, 10, 15, 20, 25, 30]);
    });

    it('should handle right scalar efficiently', () => {
      const tensorShape: Shape = [2, 3];
      const scalarShape: Shape = [];
      const tensor = new Float32Array([1, 2, 3, 4, 5, 6]);
      const scalar = new Float32Array([10]);

      const { result, shape } = BinaryBroadcaster.execute(
        tensorShape,
        tensor,
        scalarShape,
        scalar,
        (a, b) => a + b,
        Float32Array,
      );

      expect(shape).toEqual([2, 3]);
      expect(Array.from(result)).toEqual([11, 12, 13, 14, 15, 16]);
    });
  });

  describe('General Broadcasting', () => {
    it('should handle complex broadcasting patterns', () => {
      const shape1: Shape = [2, 1, 3];
      const shape2: Shape = [1, 4, 1];
      const input1 = new Float32Array([1, 2, 3, 4, 5, 6]);
      const input2 = new Float32Array([10, 20, 30, 40]);

      const { result, shape } = BinaryBroadcaster.execute(
        shape1,
        input1,
        shape2,
        input2,
        (a, b) => a + b,
        Float32Array,
      );

      expect(shape).toEqual([2, 4, 3]);
      expect(result.length).toBe(24);
    });
  });
});

// =============================================================================
// Reduction Broadcaster Tests
// =============================================================================

describe('ReductionBroadcaster', () => {
  describe('Shape Computation', () => {
    it('should compute reduced shapes correctly', () => {
      expect(ReductionBroadcaster.getReducedShape([2, 3, 4])).toEqual([]);
      expect(ReductionBroadcaster.getReducedShape([2, 3, 4], undefined, true)).toEqual([1, 1, 1]);
      expect(ReductionBroadcaster.getReducedShape([2, 3, 4], [1])).toEqual([2, 4]);
      expect(ReductionBroadcaster.getReducedShape([2, 3, 4], [1], true)).toEqual([2, 1, 4]);
      expect(ReductionBroadcaster.getReducedShape([2, 3, 4], [0, 2])).toEqual([3]);
    });

    it('should handle negative axes', () => {
      expect(ReductionBroadcaster.getReducedShape([2, 3, 4], [-1])).toEqual([2, 3]);
      expect(ReductionBroadcaster.getReducedShape([2, 3, 4], [-2, -1])).toEqual([2]);
    });

    it('should validate axes bounds', () => {
      expect(() => ReductionBroadcaster.getReducedShape([2, 3], [3])).toThrow('out of bounds');
      expect(() => ReductionBroadcaster.getReducedShape([2, 3], [-3])).toThrow('out of bounds');
    });
  });

  describe('Reduction Execution', () => {
    it('should reduce all elements', () => {
      const input = new Float32Array([1, 2, 3, 4, 5, 6]);
      const { result, shape } = ReductionBroadcaster.reduce([2, 3], input, (a, b) => a + b, 0);

      expect(shape).toEqual([]);
      expect(Array.from(result)).toEqual([21]);
    });

    it('should reduce along specific axes', () => {
      const input = new Float32Array([1, 2, 3, 4, 5, 6]);
      const { result, shape } = ReductionBroadcaster.reduce([2, 3], input, (a, b) => a + b, 0, [1]);

      expect(shape).toEqual([2]);
      expect(Array.from(result)).toEqual([6, 15]); // [1+2+3, 4+5+6]
    });

    it('should reduce with keepDims', () => {
      const input = new Float32Array([1, 2, 3, 4, 5, 6]);
      const { result, shape } = ReductionBroadcaster.reduce(
        [2, 3],
        input,
        (a, b) => a + b,
        0,
        [1],
        true,
      );

      expect(shape).toEqual([2, 1]);
      expect(Array.from(result)).toEqual([6, 15]);
    });
  });
});

// =============================================================================
// Convenience Functions Tests
// =============================================================================

describe('Convenience Functions', () => {
  describe('canBroadcast', () => {
    it('should check broadcasting compatibility', () => {
      expect(canBroadcast([2, 1], [1, 3])).toBe(true);
      expect(canBroadcast([2, 3], [4, 5])).toBe(false);
      expect(canBroadcast([])).toBe(true);
      expect(canBroadcast([2, 3])).toBe(true);
    });
  });

  describe('broadcastShapes', () => {
    it('should compute broadcast shapes', () => {
      expect(broadcastShapes([2, 1], [1, 3])).toEqual([2, 3]);
      expect(broadcastShapes([5, 1, 3], [1, 3])).toEqual([5, 1, 3]);
    });
  });

  describe('broadcastBinaryOp', () => {
    it('should execute binary operations with broadcasting', () => {
      const shape1: Shape = [2, 1];
      const shape2: Shape = [1, 3];
      const input1 = new Float32Array([2, 3]);
      const input2 = new Float32Array([10, 20, 30]);

      const { result, shape } = broadcastBinaryOp(shape1, input1, shape2, input2, (a, b) => a * b);

      expect(shape).toEqual([2, 3]);
      expect(Array.from(result)).toEqual([20, 40, 60, 30, 60, 90]);
    });
  });
});

// =============================================================================
// Broadcasting Edge Cases
// =============================================================================

describe('Broadcasting Edge Cases', () => {
  it('should handle scalar with any shape', () => {
    expect(canBroadcast([], [1, 2, 3, 4, 5, 6, 7, 8])).toBe(true);
    expect(broadcastShapes([], [2, 3, 4])).toEqual([2, 3, 4]);
  });

  it('should handle identical shapes', () => {
    const shape: Shape = [2, 3, 4];
    expect(canBroadcast(shape, shape)).toBe(true);
    expect(broadcastShapes(shape, shape)).toEqual(shape);
  });

  it('should handle [2, 1, 2] with [2, 1] broadcasting', () => {
    // This is the case from our failing CPU test
    // NumPy: np.broadcast_shapes((2, 1, 2), (2, 1)) = (2, 2, 2)
    expect(canBroadcast([2, 1, 2], [2, 1])).toBe(true);
    expect(broadcastShapes([2, 1, 2], [2, 1])).toEqual([2, 2, 2]);
  });

  it('should handle various shape combinations with different ranks', () => {
    // Test cases that verify NumPy-compatible broadcasting
    expect(broadcastShapes([2, 1, 2], [2, 1])).toEqual([2, 2, 2]);
    expect(broadcastShapes([2, 1, 2], [1])).toEqual([2, 1, 2]);
    expect(broadcastShapes([2, 1, 2], [2])).toEqual([2, 1, 2]);
    expect(broadcastShapes([5, 4], [1])).toEqual([5, 4]);
    expect(broadcastShapes([5, 4], [4])).toEqual([5, 4]);
    expect(broadcastShapes([15, 3, 5], [15, 1, 5])).toEqual([15, 3, 5]);
    expect(broadcastShapes([15, 3, 5], [3, 5])).toEqual([15, 3, 5]);
    expect(broadcastShapes([15, 3, 5], [3, 1])).toEqual([15, 3, 5]);
  });
});
