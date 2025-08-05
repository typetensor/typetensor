/**
 * Runtime tests for the einops AST types and utilities
 *
 * These tests validate the type guards, utility functions, and AST construction
 * helpers work correctly at runtime.
 */

import { describe, it, expect } from 'bun:test';
import {
  isSimpleAxis,
  isCompositeAxis,
  isEllipsisAxis,
  isSingletonAxis,
  getAxisNames,
  hasEllipsis,
  getCompositeDepth,
  countSimpleAxes,
  getUniqueAxisNames,
  hasSingleton,
} from './ast';
import type {
  SimpleAxis,
  CompositeAxis,
  EllipsisAxis,
  SingletonAxis,
  AxisPattern,
  EinopsAST,
} from './ast';

// =============================================================================
// Test Data Helpers
// =============================================================================

const createSimpleAxis = (name: string, start = 0, end = name.length): SimpleAxis => ({
  type: 'simple',
  name,
  position: { start, end },
});

const createCompositeAxis = (axes: AxisPattern[], start = 0, end = 10): CompositeAxis => ({
  type: 'composite',
  axes,
  position: { start, end },
});

const createEllipsisAxis = (start = 0, end = 3): EllipsisAxis => ({
  type: 'ellipsis',
  position: { start, end },
});

const createSingletonAxis = (start = 0, end = 1): SingletonAxis => ({
  type: 'singleton',
  position: { start, end },
});

// =============================================================================
// Type Guard Tests
// =============================================================================

describe('AST Type Guards', () => {
  describe('isSimpleAxis', () => {
    it('should return true for SimpleAxis', () => {
      const axis = createSimpleAxis('batch');
      expect(isSimpleAxis(axis)).toBe(true);
    });

    it('should return false for other axis types', () => {
      expect(isSimpleAxis(createCompositeAxis([]))).toBe(false);
      expect(isSimpleAxis(createEllipsisAxis())).toBe(false);
      expect(isSimpleAxis(createSingletonAxis())).toBe(false);
    });

    it('should narrow type correctly', () => {
      const axis: AxisPattern = createSimpleAxis('height');
      
      if (isSimpleAxis(axis)) {
        expect(axis.name).toBe('height');
        expect(axis.type).toBe('simple');
      } else {
        throw new Error('Type guard failed');
      }
    });
  });

  describe('isCompositeAxis', () => {
    it('should return true for CompositeAxis', () => {
      const axis = createCompositeAxis([createSimpleAxis('h'), createSimpleAxis('w')]);
      expect(isCompositeAxis(axis)).toBe(true);
    });

    it('should return false for other axis types', () => {
      expect(isCompositeAxis(createSimpleAxis('batch'))).toBe(false);
      expect(isCompositeAxis(createEllipsisAxis())).toBe(false);
      expect(isCompositeAxis(createSingletonAxis())).toBe(false);
    });

    it('should narrow type correctly', () => {
      const innerAxes = [createSimpleAxis('h'), createSimpleAxis('w')];
      const axis: AxisPattern = createCompositeAxis(innerAxes);
      
      if (isCompositeAxis(axis)) {
        expect(axis.axes).toEqual(innerAxes);
        expect(axis.type).toBe('composite');
      } else {
        throw new Error('Type guard failed');
      }
    });
  });

  describe('isEllipsisAxis', () => {
    it('should return true for EllipsisAxis', () => {
      const axis = createEllipsisAxis();
      expect(isEllipsisAxis(axis)).toBe(true);
    });

    it('should return false for other axis types', () => {
      expect(isEllipsisAxis(createSimpleAxis('batch'))).toBe(false);
      expect(isEllipsisAxis(createCompositeAxis([]))).toBe(false);
      expect(isEllipsisAxis(createSingletonAxis())).toBe(false);
    });

    it('should narrow type correctly', () => {
      const axis: AxisPattern = createEllipsisAxis();
      
      if (isEllipsisAxis(axis)) {
        expect(axis.type).toBe('ellipsis');
        expect(axis.position).toEqual({ start: 0, end: 3 });
      } else {
        throw new Error('Type guard failed');
      }
    });
  });

  describe('isSingletonAxis', () => {
    it('should return true for SingletonAxis', () => {
      const axis = createSingletonAxis();
      expect(isSingletonAxis(axis)).toBe(true);
    });

    it('should return false for other axis types', () => {
      expect(isSingletonAxis(createSimpleAxis('batch'))).toBe(false);
      expect(isSingletonAxis(createCompositeAxis([]))).toBe(false);
      expect(isSingletonAxis(createEllipsisAxis())).toBe(false);
    });

    it('should narrow type correctly', () => {
      const axis: AxisPattern = createSingletonAxis();
      
      if (isSingletonAxis(axis)) {
        expect(axis.type).toBe('singleton');
        expect(axis.position).toEqual({ start: 0, end: 1 });
      } else {
        throw new Error('Type guard failed');
      }
    });
  });
});

// =============================================================================
// Utility Function Tests
// =============================================================================

describe('AST Utility Functions', () => {
  describe('getAxisNames', () => {
    it('should extract names from simple axes', () => {
      const patterns = [
        createSimpleAxis('batch'),
        createSimpleAxis('height'),
        createSimpleAxis('width'),
      ];
      
      expect(getAxisNames(patterns)).toEqual(['batch', 'height', 'width']);
    });

    it('should extract names from composite axes', () => {
      const patterns = [
        createCompositeAxis([
          createSimpleAxis('h'),
          createSimpleAxis('w'),
        ]),
        createSimpleAxis('channels'),
      ];
      
      expect(getAxisNames(patterns)).toEqual(['h', 'w', 'channels']);
    });

    it('should handle nested composites', () => {
      const patterns = [
        createCompositeAxis([
          createCompositeAxis([
            createSimpleAxis('a'),
            createSimpleAxis('b'),
          ]),
          createSimpleAxis('c'),
        ]),
      ];
      
      expect(getAxisNames(patterns)).toEqual(['a', 'b', 'c']);
    });

    it('should ignore ellipsis and singleton axes', () => {
      const patterns = [
        createSimpleAxis('batch'),
        createEllipsisAxis(),
        createSingletonAxis(),
        createSimpleAxis('channels'),
      ];
      
      expect(getAxisNames(patterns)).toEqual(['batch', 'channels']);
    });

    it('should return empty array for no simple axes', () => {
      const patterns = [
        createEllipsisAxis(),
        createSingletonAxis(),
      ];
      
      expect(getAxisNames(patterns)).toEqual([]);
    });
  });

  describe('hasEllipsis', () => {
    it('should return true when ellipsis is present', () => {
      const patterns = [
        createSimpleAxis('batch'),
        createEllipsisAxis(),
        createSimpleAxis('channels'),
      ];
      
      expect(hasEllipsis(patterns)).toBe(true);
    });

    it('should return true when ellipsis is in composite', () => {
      const patterns = [
        createCompositeAxis([
          createSimpleAxis('batch'),
          createEllipsisAxis(),
        ]),
      ];
      
      expect(hasEllipsis(patterns)).toBe(true);
    });

    it('should return false when no ellipsis present', () => {
      const patterns = [
        createSimpleAxis('batch'),
        createSingletonAxis(),
        createCompositeAxis([
          createSimpleAxis('h'),
          createSimpleAxis('w'),
        ]),
      ];
      
      expect(hasEllipsis(patterns)).toBe(false);
    });

    it('should return false for empty array', () => {
      expect(hasEllipsis([])).toBe(false);
    });
  });

  describe('getCompositeDepth', () => {
    it('should return 1 for simple composite', () => {
      const composite = createCompositeAxis([
        createSimpleAxis('h'),
        createSimpleAxis('w'),
      ]);
      
      expect(getCompositeDepth(composite)).toBe(1);
    });

    it('should return 2 for nested composite', () => {
      const composite = createCompositeAxis([
        createCompositeAxis([
          createSimpleAxis('a'),
          createSimpleAxis('b'),
        ]),
        createSimpleAxis('c'),
      ]);
      
      expect(getCompositeDepth(composite)).toBe(2);
    });

    it('should return 3 for deeply nested composite', () => {
      const composite = createCompositeAxis([
        createCompositeAxis([
          createCompositeAxis([
            createSimpleAxis('a'),
          ]),
        ]),
      ]);
      
      expect(getCompositeDepth(composite)).toBe(3);
    });

    it('should handle mixed nesting levels', () => {
      const composite = createCompositeAxis([
        createCompositeAxis([
          createCompositeAxis([createSimpleAxis('a')]), // depth 3
          createSimpleAxis('b'), // depth 1
        ]),
        createCompositeAxis([createSimpleAxis('c')]), // depth 2
      ]);
      
      expect(getCompositeDepth(composite)).toBe(3);
    });
  });

  describe('countSimpleAxes', () => {
    it('should count simple axes correctly', () => {
      const patterns = [
        createSimpleAxis('batch'),
        createSimpleAxis('height'),
        createSimpleAxis('width'),
      ];
      
      expect(countSimpleAxes(patterns)).toBe(3);
    });

    it('should count axes in composites', () => {
      const patterns = [
        createCompositeAxis([
          createSimpleAxis('h'),
          createSimpleAxis('w'),
        ]),
        createSimpleAxis('channels'),
      ];
      
      expect(countSimpleAxes(patterns)).toBe(3);
    });

    it('should ignore non-simple axes', () => {
      const patterns = [
        createSimpleAxis('batch'),
        createEllipsisAxis(),
        createSingletonAxis(),
        createCompositeAxis([
          createSimpleAxis('h'),
          createEllipsisAxis(),
          createSimpleAxis('w'),
        ]),
      ];
      
      expect(countSimpleAxes(patterns)).toBe(3);
    });

    it('should return 0 for empty array', () => {
      expect(countSimpleAxes([])).toBe(0);
    });
  });

  describe('getUniqueAxisNames', () => {
    it('should return unique names only', () => {
      const patterns = [
        createSimpleAxis('batch'),
        createSimpleAxis('height'),
        createSimpleAxis('batch'), // duplicate
        createCompositeAxis([
          createSimpleAxis('height'), // duplicate
          createSimpleAxis('width'),
        ]),
      ];
      
      const unique = getUniqueAxisNames(patterns);
      expect(unique).toHaveLength(3);
      expect(unique).toEqual(expect.arrayContaining(['batch', 'height', 'width']));
    });

    it('should preserve order of first occurrence', () => {
      const patterns = [
        createSimpleAxis('c'),
        createSimpleAxis('a'),
        createSimpleAxis('b'),
        createSimpleAxis('a'), // duplicate
      ];
      
      expect(getUniqueAxisNames(patterns)).toEqual(['c', 'a', 'b']);
    });
  });

  describe('hasSingleton', () => {
    it('should return true when singleton is present', () => {
      const patterns = [
        createSimpleAxis('batch'),
        createSingletonAxis(),
        createSimpleAxis('channels'),
      ];
      
      expect(hasSingleton(patterns)).toBe(true);
    });

    it('should return true when singleton is in composite', () => {
      const patterns = [
        createCompositeAxis([
          createSimpleAxis('batch'),
          createSingletonAxis(),
        ]),
      ];
      
      expect(hasSingleton(patterns)).toBe(true);
    });

    it('should return false when no singleton present', () => {
      const patterns = [
        createSimpleAxis('batch'),
        createEllipsisAxis(),
        createCompositeAxis([
          createSimpleAxis('h'),
          createSimpleAxis('w'),
        ]),
      ];
      
      expect(hasSingleton(patterns)).toBe(false);
    });

    it('should return false for empty array', () => {
      expect(hasSingleton([])).toBe(false);
    });
  });
});

// =============================================================================
// Complex Pattern Tests
// =============================================================================

describe('Complex AST Patterns', () => {
  describe('Realistic einops patterns', () => {
    it('should handle simple transpose pattern', () => {
      const input = [
        createSimpleAxis('h'),
        createSimpleAxis('w'),
      ];
      const output = [
        createSimpleAxis('w'),
        createSimpleAxis('h'),
      ];
      
      expect(getAxisNames(input)).toEqual(['h', 'w']);
      expect(getAxisNames(output)).toEqual(['w', 'h']);
      expect(getUniqueAxisNames([...input, ...output])).toEqual(['h', 'w']);
    });

    it('should handle composite dimension splitting', () => {
      const input = [
        createCompositeAxis([
          createSimpleAxis('h'),
          createSimpleAxis('w'),
        ]),
        createSimpleAxis('c'),
      ];
      const output = [
        createSimpleAxis('h'),
        createSimpleAxis('w'),
        createSimpleAxis('c'),
      ];
      
      expect(countSimpleAxes(input)).toBe(3);
      expect(countSimpleAxes(output)).toBe(3);
      expect(getUniqueAxisNames([...input, ...output])).toEqual(['h', 'w', 'c']);
    });

    it('should handle ellipsis patterns', () => {
      const input = [
        createSimpleAxis('batch'),
        createEllipsisAxis(),
        createSimpleAxis('channels'),
      ];
      const output = [
        createSimpleAxis('channels'),
        createSimpleAxis('batch'),
        createEllipsisAxis(),
      ];
      
      expect(hasEllipsis(input)).toBe(true);
      expect(hasEllipsis(output)).toBe(true);
      expect(getAxisNames(input)).toEqual(['batch', 'channels']);
      expect(getAxisNames(output)).toEqual(['channels', 'batch']);
    });

    it('should handle singleton patterns', () => {
      const input = [
        createSimpleAxis('h'),
        createSimpleAxis('w'),
        createSingletonAxis(),
      ];
      const output = [
        createSimpleAxis('h'),
        createSimpleAxis('w'),
      ];
      
      expect(hasSingleton(input)).toBe(true);
      expect(hasSingleton(output)).toBe(false);
      expect(countSimpleAxes(input)).toBe(2);
      expect(countSimpleAxes(output)).toBe(2);
    });
  });

  describe('Complex nested structures', () => {
    it('should handle deeply nested composites', () => {
      const pattern = createCompositeAxis([
        createCompositeAxis([
          createCompositeAxis([
            createSimpleAxis('a'),
            createSimpleAxis('b'),
          ]),
          createSimpleAxis('c'),
        ]),
        createSimpleAxis('d'),
      ]);
      
      expect(getCompositeDepth(pattern)).toBe(3);
      expect(getAxisNames([pattern])).toEqual(['a', 'b', 'c', 'd']);
      expect(countSimpleAxes([pattern])).toBe(4);
    });

    it('should handle mixed pattern types in composites', () => {
      const pattern = createCompositeAxis([
        createSimpleAxis('batch'),
        createEllipsisAxis(),
        createSingletonAxis(),
        createCompositeAxis([
          createSimpleAxis('h'),
          createSimpleAxis('w'),
        ]),
      ]);
      
      expect(hasEllipsis([pattern])).toBe(true);
      expect(hasSingleton([pattern])).toBe(true);
      expect(getAxisNames([pattern])).toEqual(['batch', 'h', 'w']);
      expect(countSimpleAxes([pattern])).toBe(3);
    });
  });

  describe('AST construction', () => {
    it('should create valid AST structure', () => {
      const ast: EinopsAST = {
        input: [
          createCompositeAxis([
            createSimpleAxis('h'),
            createSimpleAxis('w'),
          ]),
          createSimpleAxis('c'),
        ],
        output: [
          createSimpleAxis('h'),
          createSimpleAxis('w'),
          createSimpleAxis('c'),
        ],
        metadata: {
          originalPattern: '(h w) c -> h w c',
          arrowPosition: { start: 7, end: 9 },
          inputTokenCount: 7,
          outputTokenCount: 5,
        },
      };
      
      expect(ast.input).toHaveLength(2);
      expect(ast.output).toHaveLength(3);
      expect(ast.metadata.originalPattern).toBe('(h w) c -> h w c');
      
      expect(getAxisNames(ast.input)).toEqual(['h', 'w', 'c']);
      expect(getAxisNames(ast.output)).toEqual(['h', 'w', 'c']);
    });
  });
});