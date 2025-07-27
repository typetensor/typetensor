/**
 * Runtime tests for dtype/promotion.ts
 *
 * Tests for type promotion logic and promotion matrix
 */

import { describe, it, expect } from 'bun:test';
import {
  promoteTypes,
  canPromoteTypes,
  promoteMultipleTypes,
  findCommonType,
  computeResultType,
  computeUnaryResultType,
  validatePromotionMatrix,
  analyzePromotion,
  getPromotionRules,
  toPromotedDType,
} from './promotion';
import { getDType } from './runtime';
import { int32, float32, int8, uint8 } from './constants';

describe('Type Promotion', () => {
  describe('promoteTypes', () => {
    it('should promote same types to themselves', () => {
      const int32 = getDType('int32');
      const promoted = promoteTypes(int32, int32);
      expect(promoted).toBe(int32);
    });

    it('should promote bool to any other type', () => {
      const bool = getDType('bool');
      const int32 = getDType('int32');
      const float64 = getDType('float64');

      expect(promoteTypes(bool, int32)).toBe(int32);
      expect(promoteTypes(int32, bool)).toBe(int32);
      expect(promoteTypes(bool, float64)).toBe(float64);
    });

    it('should promote integers by size and signedness', () => {
      const int8 = getDType('int8');
      const int16 = getDType('int16');
      const uint8 = getDType('uint8');

      // Same signedness - promote to larger
      expect(promoteTypes(int8, int16)).toBe(int16);

      // Mixed signedness - promote to larger signed
      const promoted = promoteTypes(int8, uint8);
      expect(promoted.name).toBe('int16');
      expect(promoted.signed).toBe(true);
    });

    it('should handle mixed signedness promotion correctly', () => {
      expect(promoteTypes(getDType('int8'), getDType('uint8')).name).toBe('int16');
      expect(promoteTypes(getDType('int16'), getDType('uint16')).name).toBe('int32');
      expect(promoteTypes(getDType('int32'), getDType('uint32')).name).toBe('int64');
      expect(promoteTypes(getDType('int64'), getDType('uint64')).name).toBe('float64');
    });

    it('should promote integers to floats appropriately', () => {
      const int32 = getDType('int32');
      const float32 = getDType('float32');
      const float64 = getDType('float64');

      // int32 precision requires float64
      expect(promoteTypes(int32, float32)).toBe(float64);

      // Small integers can use float32
      const int16 = getDType('int16');
      expect(promoteTypes(int16, float32)).toBe(float32);
    });

    it('should maintain promotion symmetry', () => {
      const dtypeNames = [
        'bool',
        'int8',
        'uint8',
        'int16',
        'uint16',
        'int32',
        'uint32',
        'int64',
        'uint64',
        'float32',
        'float64',
      ] as const;

      for (const nameA of dtypeNames) {
        for (const nameB of dtypeNames) {
          const dtypeA = getDType(nameA);
          const dtypeB = getDType(nameB);
          const promotedAB = promoteTypes(dtypeA, dtypeB);
          const promotedBA = promoteTypes(dtypeB, dtypeA);

          expect(promotedAB.name).toBe(promotedBA.name);
        }
      }
    });

    it('should handle special float promotion cases', () => {
      // Large integers with float32 need float64
      expect(promoteTypes(getDType('int32'), getDType('float32')).name).toBe('float64');
      expect(promoteTypes(getDType('uint32'), getDType('float32')).name).toBe('float64');
      expect(promoteTypes(getDType('int64'), getDType('float32')).name).toBe('float64');
      expect(promoteTypes(getDType('uint64'), getDType('float32')).name).toBe('float64');
    });
  });

  describe('canPromoteTypes', () => {
    it('should return true for all valid promotions', () => {
      expect(canPromoteTypes(getDType('int32'), getDType('float32'))).toBe(true);
      expect(canPromoteTypes(getDType('bool'), getDType('int64'))).toBe(true);
      expect(canPromoteTypes(getDType('float32'), getDType('float64'))).toBe(true);
    });

    it('should always return true for dtype pairs', () => {
      // In our system, all dtypes can be promoted (no invalid promotions)
      const dtypeNames = [
        'bool',
        'int8',
        'uint8',
        'int16',
        'uint16',
        'int32',
        'uint32',
        'int64',
        'uint64',
        'float32',
        'float64',
      ] as const;

      for (const nameA of dtypeNames) {
        for (const nameB of dtypeNames) {
          expect(canPromoteTypes(getDType(nameA), getDType(nameB))).toBe(true);
        }
      }
    });
  });

  describe('promoteMultipleTypes', () => {
    it('should promote multiple types to common type', () => {
      const bool = getDType('bool');
      const int8 = getDType('int8');
      const float32 = getDType('float32');

      const result = promoteMultipleTypes([bool, int8, float32]);
      expect(result).toBe(float32);
    });

    it('should handle empty array', () => {
      expect(() => promoteMultipleTypes([])).toThrow('Cannot promote empty array');
    });

    it('should handle single element', () => {
      const float64 = getDType('float64');
      expect(promoteMultipleTypes([float64])).toBe(float64);
    });

    it('should promote complex chains correctly', () => {
      // bool → int8 → int16 → float32
      const dtypes = [getDType('bool'), getDType('int8'), getDType('int16'), getDType('float32')];
      const result = promoteMultipleTypes(dtypes);
      expect(result.name).toBe('float32');
    });

    it('should handle mixed signedness in chains', () => {
      // int8 + uint8 → int16, then int16 + uint16 → int32
      const dtypes = [getDType('int8'), getDType('uint8'), getDType('uint16')];
      const result = promoteMultipleTypes(dtypes);
      expect(result.name).toBe('int32');
    });
  });

  describe('findCommonType', () => {
    it('should find common type from values', () => {
      const values = [true, 42, 3.14];
      const commonType = findCommonType(values);
      // 3.14 cannot be exactly represented as float32, so float64 is needed
      expect(commonType.name).toBe('float64');
    });

    it('should use float32 when values are exactly representable', () => {
      const values = [true, 42, 1.5]; // 1.5 is exactly representable in float32
      const commonType = findCommonType(values);
      expect(commonType.name).toBe('float32');
    });

    it('should handle integer values', () => {
      expect(findCommonType([1, 2, 3]).name).toBe('int8');
      // [1,2] are int8, 128 is uint8. int8+uint8 = int16 (matches NumPy/PyTorch)
      expect(findCommonType([1, 2, 128]).name).toBe('int16');
      expect(findCommonType([1, 2, 256]).name).toBe('int16');
      expect(findCommonType([1, 2, -129]).name).toBe('int16');
      expect(findCommonType([1, 2, 65536]).name).toBe('int32');
    });

    it('should handle float values', () => {
      expect(findCommonType([1.5, 2.5]).name).toBe('float32');
      expect(findCommonType([1.5, Number.MAX_VALUE]).name).toBe('float64');
    });

    it('should handle bigint values', () => {
      expect(findCommonType([1n, 2n, 3n]).name).toBe('uint64');
      // Mixed uint64 and int64 promotes to float64 (matches NumPy behavior)
      expect(findCommonType([1n, -2n, 3n]).name).toBe('float64');
      expect(findCommonType([1n, -2n, 3.14]).name).toBe('float64'); // Mixed bigint and float
      // All negative bigints stay as int64
      expect(findCommonType([-1n, -2n, -3n]).name).toBe('int64');
    });

    it('should handle mixed types', () => {
      expect(findCommonType([true, 1]).name).toBe('int8');
      expect(findCommonType([true, 1, 2.5]).name).toBe('float32');
      // int8 + uint64 = float64 (no integer type can hold full range of both)
      expect(findCommonType([1, 2, 3n]).name).toBe('float64');
    });

    it('should handle special values', () => {
      expect(findCommonType([NaN]).name).toBe('float32');
      expect(findCommonType([Infinity]).name).toBe('float32');
      expect(findCommonType([Number.MAX_SAFE_INTEGER + 1]).name).toBe('float64');
    });

    it('should throw for empty array', () => {
      expect(() => findCommonType([])).toThrow('Cannot find common type for empty array');
    });

    it('should throw for invalid values', () => {
      expect(() => findCommonType(['string'])).toThrow('Cannot determine DType');
      expect(() => findCommonType([null])).toThrow('Cannot determine DType');
    });
  });

  describe('computeResultType', () => {
    it('should compute binary operation result types', () => {
      const int32 = getDType('int32');
      const float32 = getDType('float32');
      const result = computeResultType(int32, float32);
      expect(result.name).toBe('float64');
    });

    it('should match promoteTypes behavior', () => {
      const dtypeNames = ['bool', 'int8', 'uint8', 'float32', 'float64'] as const;

      for (const nameA of dtypeNames) {
        for (const nameB of dtypeNames) {
          const dtypeA = getDType(nameA);
          const dtypeB = getDType(nameB);
          const promoted = promoteTypes(dtypeA, dtypeB);
          const computed = computeResultType(dtypeA, dtypeB);

          expect(computed).toBe(promoted);
        }
      }
    });
  });

  describe('computeUnaryResultType', () => {
    it('should preserve type for most unary operations', () => {
      const int32 = getDType('int32');
      expect(computeUnaryResultType(int32, 'abs')).toBe(int32);
      expect(computeUnaryResultType(int32, 'neg')).toBe(int32);
      expect(computeUnaryResultType(int32, 'sign')).toBe(int32);
    });

    it('should promote integers to float for mathematical functions', () => {
      const int8 = getDType('int8');
      const int32 = getDType('int32');

      expect(computeUnaryResultType(int8, 'sqrt').name).toBe('float32');
      expect(computeUnaryResultType(int8, 'exp').name).toBe('float32');
      expect(computeUnaryResultType(int8, 'log').name).toBe('float32');
      expect(computeUnaryResultType(int8, 'sin').name).toBe('float32');
      expect(computeUnaryResultType(int8, 'cos').name).toBe('float32');
      expect(computeUnaryResultType(int8, 'tan').name).toBe('float32');

      // Larger integers promote to float64
      expect(computeUnaryResultType(int32, 'sqrt').name).toBe('float64');
    });

    it('should preserve float types for mathematical functions', () => {
      const float32 = getDType('float32');
      const float64 = getDType('float64');

      expect(computeUnaryResultType(float32, 'sqrt')).toBe(float32);
      expect(computeUnaryResultType(float64, 'sqrt')).toBe(float64);
    });

    it('should handle rounding functions', () => {
      const float32 = getDType('float32');
      const float64 = getDType('float64');
      const int32 = getDType('int32');

      // Floats to integers
      expect(computeUnaryResultType(float32, 'floor').name).toBe('int32');
      expect(computeUnaryResultType(float32, 'ceil').name).toBe('int32');
      expect(computeUnaryResultType(float32, 'round').name).toBe('int32');
      expect(computeUnaryResultType(float64, 'floor').name).toBe('int64');

      // Integers preserve type
      expect(computeUnaryResultType(int32, 'floor')).toBe(int32);
      expect(computeUnaryResultType(int32, 'ceil')).toBe(int32);
      expect(computeUnaryResultType(int32, 'round')).toBe(int32);
    });

    it('should handle unknown operations', () => {
      const float32 = getDType('float32');
      expect(computeUnaryResultType(float32, 'unknown')).toBe(float32);
    });
  });

  describe('validatePromotionMatrix', () => {
    it('should not throw for valid promotion matrix', () => {
      expect(() => {
        validatePromotionMatrix();
      }).not.toThrow();
    });

    // Note: We can't easily test invalid matrices without modifying the source
  });

  describe('analyzePromotion', () => {
    it('should provide detailed promotion analysis', () => {
      const int32 = getDType('int32');
      const float32 = getDType('float32');

      const analysis = analyzePromotion(int32, float32);
      expect(analysis.inputTypes).toEqual(['int32', 'float32']);
      expect(analysis.resultType).toBe('float64');
      expect(analysis.isWidening).toBe(true);
      expect(analysis.reason).toContain('floating-point');
    });

    it('should identify same type promotions', () => {
      const float64 = getDType('float64');
      const analysis = analyzePromotion(float64, float64);
      expect(analysis.resultType).toBe('float64');
      expect(analysis.reason).toContain('Same type');
      expect(analysis.isWidening).toBe(false);
    });

    it('should identify bool promotions', () => {
      const bool = getDType('bool');
      const int32 = getDType('int32');
      const analysis = analyzePromotion(bool, int32);
      expect(analysis.reason).toContain('Boolean promotes');
      expect(analysis.isWidening).toBe(true);
    });

    it('should identify mixed signedness', () => {
      const int8 = getDType('int8');
      const uint8 = getDType('uint8');
      const analysis = analyzePromotion(int8, uint8);
      expect(analysis.reason).toContain('Mixed signedness');
      expect(analysis.isWidening).toBe(true);
    });

    it('should track precision preservation', () => {
      const int32 = getDType('int32');
      const float32 = getDType('float32');
      const analysis = analyzePromotion(int32, float32);
      expect(analysis.isPrecisionPreserving).toBe(false); // int32 + float32 → float64

      const int8 = getDType('int8');
      const int16 = getDType('int16');
      const analysis2 = analyzePromotion(int8, int16);
      expect(analysis2.isPrecisionPreserving).toBe(true); // int8 + int16 → int16
    });
  });

  describe('getPromotionRules', () => {
    it('should return promotion rules documentation', () => {
      const rules = getPromotionRules();
      expect(rules).toContain('NumPy-Compatible');
      expect(rules).toContain('Hierarchy');
      expect(rules).toContain('Bool < Int8');
      expect(rules).toContain('Mixed signedness');
      expect(rules).toContain('Integer + Float');
    });
  });

  describe('toPromotedDType', () => {
    it('should return compile-time dtype constants with __typedArray property', () => {
      // Test that toPromotedDType returns constants (with __typedArray) not RuntimeDType (with typedArrayConstructor)
      const result = toPromotedDType(int32, float32);
      
      // Should have __typedArray property (compile-time dtype interface)
      expect(result).toHaveProperty('__typedArray');
      expect(result).not.toHaveProperty('typedArrayConstructor');
      
      // Should be the correct promoted type (int32 + float32 = float64)
      expect(result.__dtype).toBe('float64');
      expect(result.__typedArray).toBe(Float64Array);
    });

    it('should follow same promotion logic as promoteTypes', () => {
      // Test a few key promotions to ensure logic is consistent
      const promotion1 = toPromotedDType(int8, uint8);
      expect(promotion1.__dtype).toBe('int16');
      expect(promotion1.__typedArray).toBe(Int16Array);

      const promotion2 = toPromotedDType(int32, int32);
      expect(promotion2.__dtype).toBe('int32');
      expect(promotion2.__typedArray).toBe(Int32Array);
    });

    it('should work with CPU backend createTypedArray expectations', () => {
      // This is the key test - ensure the returned dtype works with CPU backend
      const promotedDtype = toPromotedDType(int32, float32);
      
      // CPU backend expects __typedArray property
      const TypedArrayConstructor = promotedDtype.__typedArray;
      expect(TypedArrayConstructor).toBe(Float64Array);
      
      // Should be able to create typed array (this is what was failing before)
      const buffer = new ArrayBuffer(8);
      const typedArray = new TypedArrayConstructor(buffer);
      expect(typedArray).toBeInstanceOf(Float64Array);
      expect(typedArray.length).toBe(1);
    });
  });

  describe('Edge Cases', () => {
    it('should handle all edge case promotions correctly', () => {
      // Bool with everything
      expect(promoteTypes(getDType('bool'), getDType('bool')).name).toBe('bool');
      expect(promoteTypes(getDType('bool'), getDType('uint64')).name).toBe('uint64');

      // Large integer edge cases
      expect(promoteTypes(getDType('int64'), getDType('uint64')).name).toBe('float64');
      expect(promoteTypes(getDType('uint64'), getDType('int8')).name).toBe('float64');

      // Float precision edge cases
      expect(promoteTypes(getDType('int32'), getDType('float32')).name).toBe('float64');
      expect(promoteTypes(getDType('uint32'), getDType('float32')).name).toBe('float64');
    });
  });
});
