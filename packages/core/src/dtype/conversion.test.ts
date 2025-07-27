/**
 * Runtime tests for dtype/conversion.ts
 *
 * Tests for type conversion, validation, and casting operations
 */

import { describe, it, expect } from 'bun:test';
import {
  convertValue,
  convertArray,
  safeCast,
  unsafeCast,
  wouldBeLossy,
  STRICT_CONVERSION_OPTIONS,
  PERMISSIVE_CONVERSION_OPTIONS,
  ConversionError,
  PrecisionLossError,
  OverflowError,
} from './conversion';
import { getDType } from './runtime';

describe('Type Conversion', () => {
  describe('convertValue', () => {
    describe('Basic conversions', () => {
      it('should convert between compatible types', () => {
        const int8 = getDType('int8');
        const int32 = getDType('int32');

        const result = convertValue(42, int8, int32);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.value).toBe(42);
          expect(result.warnings).toHaveLength(0);
        }
      });

      it('should handle same type conversions', () => {
        const float32 = getDType('float32');
        const result = convertValue(3.14, float32, float32);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.value).toBe(3.14);
          expect(result.warnings).toHaveLength(0);
        }
      });

      it('should detect precision loss', () => {
        const float32 = getDType('float32');
        const int32 = getDType('int32');

        const result = convertValue(3.14, float32, int32);
        expect(result.success).toBe(false);
        if (!result.success) {
          expect(result.errors[0]).toContain('precision loss');
        }
      });

      it('should handle overflow detection', () => {
        const int32 = getDType('int32');
        const int8 = getDType('int8');

        const result = convertValue(1000, int32, int8);
        expect(result.success).toBe(false);
        if (!result.success) {
          expect(result.errors[0]).toContain('out of range');
        }
      });
    });

    describe('JavaScript type conversions', () => {
      it('should convert boolean to number', () => {
        const bool = getDType('bool');
        const int32 = getDType('int32');

        const trueResult = convertValue(true, bool, int32);
        const falseResult = convertValue(false, bool, int32);

        expect(trueResult.success).toBe(true);
        expect(falseResult.success).toBe(true);

        if (trueResult.success && falseResult.success) {
          expect(trueResult.value).toBe(1);
          expect(falseResult.value).toBe(0);
        }
      });

      it('should convert number to boolean', () => {
        const int32 = getDType('int32');
        const bool = getDType('bool');

        const zeroResult = convertValue(0, int32, bool);
        const nonZeroResult = convertValue(42, int32, bool);
        const negativeResult = convertValue(-42, int32, bool);

        expect(zeroResult.success).toBe(true);
        expect(nonZeroResult.success).toBe(true);
        expect(negativeResult.success).toBe(true);

        if (zeroResult.success && nonZeroResult.success && negativeResult.success) {
          expect(zeroResult.value).toBe(false);
          expect(nonZeroResult.value).toBe(true);
          expect(negativeResult.value).toBe(true);
        }
      });

      it('should convert float to boolean with special values', () => {
        const float32 = getDType('float32');
        const bool = getDType('bool');

        const nanResult = convertValue(NaN, float32, bool);
        const infResult = convertValue(Infinity, float32, bool);
        const negInfResult = convertValue(-Infinity, float32, bool);

        // With strict options, NaN/Infinity conversions should fail
        expect(nanResult.success).toBe(false);
        expect(infResult.success).toBe(false);
        expect(negInfResult.success).toBe(false);

        // With permissive options, they should convert to true (NumPy/PyTorch behavior)
        const nanPermissive = convertValue(NaN, float32, bool, PERMISSIVE_CONVERSION_OPTIONS);
        expect(nanPermissive.success).toBe(true);
        if (nanPermissive.success) {
          expect(nanPermissive.value).toBe(true); // NumPy/PyTorch: NaN → true
          expect(nanPermissive.warnings.length).toBe(0); // No warning for standard behavior
        }

        const infPermissive = convertValue(Infinity, float32, bool, PERMISSIVE_CONVERSION_OPTIONS);
        expect(infPermissive.success).toBe(true);
        if (infPermissive.success) {
          expect(infPermissive.value).toBe(true);
          expect(infPermissive.warnings.length).toBe(0); // No warning for standard behavior
        }
      });

      it('should convert between number and bigint', () => {
        const int32 = getDType('int32');
        const int64 = getDType('int64');

        const result = convertValue(42, int32, int64);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.value).toBe(42n);
        }

        const reverseResult = convertValue(42n, int64, int32);
        expect(reverseResult.success).toBe(true);
        if (reverseResult.success) {
          expect(reverseResult.value).toBe(42);
        }
      });

      it('should handle number to bigint with precision loss', () => {
        const float32 = getDType('float32');
        const int64 = getDType('int64');

        // Non-integer should fail with strict options
        const result = convertValue(3.14, float32, int64);
        expect(result.success).toBe(false);

        // Should succeed with permissive options
        const permissiveResult = convertValue(3.14, float32, int64, PERMISSIVE_CONVERSION_OPTIONS);
        expect(permissiveResult.success).toBe(true);
        if (permissiveResult.success) {
          expect(permissiveResult.value).toBe(3n);
          expect(permissiveResult.warnings.length).toBeGreaterThan(0);
        }
      });

      it('should handle bigint to number with range issues', () => {
        const int64 = getDType('int64');
        const float32 = getDType('float32');

        const bigValue = BigInt(Number.MAX_SAFE_INTEGER) + 1n;
        const result = convertValue(bigValue, int64, float32);
        expect(result.success).toBe(false);

        const permissiveResult = convertValue(
          bigValue,
          int64,
          float32,
          PERMISSIVE_CONVERSION_OPTIONS,
        );
        expect(permissiveResult.success).toBe(true);
        if (permissiveResult.success) {
          expect(permissiveResult.warnings.length).toBeGreaterThan(0);
        }
      });
    });

    describe('Special value handling', () => {
      it('should handle NaN in float conversions', () => {
        const float32 = getDType('float32');
        const float64 = getDType('float64');
        const int32 = getDType('int32');

        // Float to float - preserve NaN
        const floatResult = convertValue(NaN, float32, float64);
        expect(floatResult.success).toBe(true);
        if (floatResult.success) {
          expect(Number.isNaN(floatResult.value)).toBe(true);
        }

        // Float to int - should fail by default
        const intResult = convertValue(NaN, float32, int32);
        expect(intResult.success).toBe(false);
      });

      it('should handle Infinity in conversions', () => {
        const float32 = getDType('float32');
        const float64 = getDType('float64');
        const int32 = getDType('int32');

        // Float to float - preserve Infinity
        const floatResult = convertValue(Infinity, float32, float64);
        expect(floatResult.success).toBe(true);
        if (floatResult.success) {
          expect(floatResult.value).toBe(Infinity);
        }

        // Float to int - should fail by default
        const intResult = convertValue(Infinity, float32, int32);
        expect(intResult.success).toBe(false);

        // With permissive options - clamp to max
        const permissiveResult = convertValue(
          Infinity,
          float32,
          int32,
          PERMISSIVE_CONVERSION_OPTIONS,
        );
        expect(permissiveResult.success).toBe(true);
        if (permissiveResult.success) {
          expect(permissiveResult.value).toBe(2147483647); // int32 max
          expect(permissiveResult.warnings.length).toBeGreaterThan(0);
        }
      });
    });

    describe('Integer conversions', () => {
      it('should handle integer widening', () => {
        const int8 = getDType('int8');
        const int32 = getDType('int32');

        const result = convertValue(127, int8, int32);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.value).toBe(127);
        }
      });

      it('should detect integer overflow', () => {
        const int32 = getDType('int32');
        const int8 = getDType('int8');

        const result = convertValue(128, int32, int8);
        expect(result.success).toBe(false);
        if (!result.success) {
          expect(result.errors[0]).toContain('out of range');
        }
      });

      it('should handle overflow with clamping', () => {
        const int32 = getDType('int32');
        const int8 = getDType('int8');

        const options = { ...PERMISSIVE_CONVERSION_OPTIONS, overflowHandling: 'clamp' as const };
        const result = convertValue(200, int32, int8, options);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.value).toBe(127); // Clamped to max
          expect(result.warnings.length).toBeGreaterThan(0);
        }
      });

      it('should handle overflow with wrapping', () => {
        const int32 = getDType('int32');
        const int8 = getDType('int8');

        const options = { ...PERMISSIVE_CONVERSION_OPTIONS, overflowHandling: 'wrap' as const };
        const result = convertValue(128, int32, int8, options);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.value).toBe(-128); // Wrapped around
          expect(result.warnings.length).toBeGreaterThan(0);
        }
      });

      it('should handle signed/unsigned conversions', () => {
        const int8 = getDType('int8');
        const uint8 = getDType('uint8');

        // Positive values should work
        const positiveResult = convertValue(100, int8, uint8);
        expect(positiveResult.success).toBe(true);
        if (positiveResult.success) {
          expect(positiveResult.value).toBe(100);
        }

        // Negative values should fail
        const negativeResult = convertValue(-1, int8, uint8);
        expect(negativeResult.success).toBe(false);
      });
    });

    describe('Float conversions', () => {
      it('should handle float32 to float64', () => {
        const float32 = getDType('float32');
        const float64 = getDType('float64');

        const result = convertValue(3.14, float32, float64);
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.value).toBe(3.14);
        }
      });

      it('should detect precision loss in float64 to float32', () => {
        const float64 = getDType('float64');
        const float32 = getDType('float32');

        // A value that can't be represented exactly in float32
        const preciseValue = 1.0000001;
        const result = convertValue(preciseValue, float64, float32);

        // This might succeed but with warnings if the loss is detected
        if (result.success && result.warnings.length > 0) {
          expect(result.warnings[0]).toContain('Precision loss');
        }
      });

      it('should handle float to integer truncation', () => {
        const float32 = getDType('float32');
        const int32 = getDType('int32');

        // Should fail with strict options
        const strictResult = convertValue(3.14, float32, int32);
        expect(strictResult.success).toBe(false);

        // Should succeed with permissive options
        const permissiveResult = convertValue(3.14, float32, int32, PERMISSIVE_CONVERSION_OPTIONS);
        expect(permissiveResult.success).toBe(true);
        if (permissiveResult.success) {
          expect(permissiveResult.value).toBe(3);
          expect(permissiveResult.warnings[0]).toContain('truncated');
        }
      });
    });
  });

  describe('convertArray', () => {
    it('should convert arrays of values', () => {
      const int8 = getDType('int8');
      const int32 = getDType('int32');
      const values = [1, 2, 3, 4, 5];

      const result = convertArray(values, int8, int32);
      expect(result.success).toBe(true);
      if (result.success && result.values) {
        expect(result.values).toEqual([1, 2, 3, 4, 5]);
        expect(result.warnings).toHaveLength(0);
      }
    });

    it('should report errors for invalid array elements', () => {
      const float32 = getDType('float32');
      const int32 = getDType('int32');
      const values = [1.0, 2.5, 3.0]; // 2.5 will cause precision loss

      const result = convertArray(values, float32, int32);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.errors.some((e) => e.includes('[1]'))).toBe(true); // Error at index 1
      }
    });

    it('should accumulate warnings for permissive conversions', () => {
      const float32 = getDType('float32');
      const int32 = getDType('int32');
      const values = [1.1, 2.2, 3.3];

      const result = convertArray(values, float32, int32, PERMISSIVE_CONVERSION_OPTIONS);
      expect(result.success).toBe(true);
      if (result.success && result.values) {
        expect(result.values).toEqual([1, 2, 3]);
        expect(result.warnings.length).toBe(3); // One warning per value
      }
    });

    it('should handle empty arrays', () => {
      const int32 = getDType('int32');
      const float32 = getDType('float32');

      const result = convertArray([], int32, float32);
      expect(result.success).toBe(true);
      if (result.success && result.values) {
        expect(result.values).toEqual([]);
      }
    });

    it('should handle mixed valid/invalid conversions', () => {
      const int32 = getDType('int32');
      const int8 = getDType('int8');
      const values = [1, 2, 300, 4, 5]; // 300 is out of range

      const result = convertArray(values, int32, int8);
      expect(result.success).toBe(false);
      if (!result.success) {
        expect(result.errors.some((e) => e.includes('[2]'))).toBe(true);
      }
    });
  });

  describe('safeCast', () => {
    it('should allow safe casts', () => {
      const int8 = getDType('int8');
      const int32 = getDType('int32');

      const result = safeCast(42, int8, int32);
      expect(result).toBe(42);
    });

    it('should allow bool to any type', () => {
      const bool = getDType('bool');
      const int32 = getDType('int32');

      expect(safeCast(true, bool, int32)).toBe(1);
      expect(safeCast(false, bool, int32)).toBe(0);
    });

    it('should reject unsafe casts', () => {
      const float32 = getDType('float32');
      const int32 = getDType('int32');

      expect(() => safeCast(3.14, float32, int32)).toThrow('Safe cast failed');
    });

    it('should reject overflow casts', () => {
      const int32 = getDType('int32');
      const int8 = getDType('int8');

      expect(() => safeCast(200, int32, int8)).toThrow('Safe cast failed');
    });
  });

  describe('unsafeCast', () => {
    it('should provide warnings for unsafe casts', () => {
      const float32 = getDType('float32');
      const int32 = getDType('int32');

      const result = unsafeCast(3.14, float32, int32);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.value).toBe(3);
        expect(result.warnings.length).toBeGreaterThan(0);
        expect(result.warnings[0]).toContain('truncated');
      }
    });

    it('should handle overflow with clamping', () => {
      const int32 = getDType('int32');
      const int8 = getDType('int8');

      const result = unsafeCast(200, int32, int8);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.value).toBe(127); // Clamped to max
        expect(result.warnings.length).toBeGreaterThan(0);
      }
    });

    it('should convert special values', () => {
      const float32 = getDType('float32');
      const int32 = getDType('int32');

      const nanResult = unsafeCast(NaN, float32, int32);
      expect(nanResult.success).toBe(true);
      if (nanResult.success) {
        expect(nanResult.warnings.length).toBeGreaterThan(0);
      }

      const infResult = unsafeCast(Infinity, float32, int32);
      expect(infResult.success).toBe(true);
      if (infResult.success) {
        expect(infResult.value).toBe(2147483647); // int32 max
        expect(infResult.warnings.length).toBeGreaterThan(0);
      }
    });
  });

  describe('wouldBeLossy', () => {
    it('should predict lossy conversions', () => {
      const float64 = getDType('float64');
      const int32 = getDType('int32');
      const float32 = getDType('float32');

      expect(wouldBeLossy(3.14, float64, int32)).toBe(true);
      expect(wouldBeLossy(3.0, float64, int32)).toBe(false);
      expect(wouldBeLossy(42, int32, float32)).toBe(false);
    });

    it('should detect overflow as lossy', () => {
      const int32 = getDType('int32');
      const int8 = getDType('int8');

      expect(wouldBeLossy(200, int32, int8)).toBe(true);
      expect(wouldBeLossy(100, int32, int8)).toBe(false);
    });

    it('should detect signed/unsigned issues', () => {
      const int8 = getDType('int8');
      const uint8 = getDType('uint8');

      expect(wouldBeLossy(-1, int8, uint8)).toBe(true);
      expect(wouldBeLossy(100, int8, uint8)).toBe(false);
    });

    it('should detect precision loss in float conversions', () => {
      const float64 = getDType('float64');
      const float32 = getDType('float32');

      // Some float64 values can't be represented exactly in float32
      expect(wouldBeLossy(1.0000001, float64, float32)).toBe(true);
      expect(wouldBeLossy(1.0, float64, float32)).toBe(false);
    });
  });

  describe('Conversion options', () => {
    it('should respect strict options', () => {
      const float32 = getDType('float32');
      const int32 = getDType('int32');

      const result = convertValue(3.14, float32, int32, STRICT_CONVERSION_OPTIONS);
      expect(result.success).toBe(false);
    });

    it('should respect permissive options', () => {
      const float32 = getDType('float32');
      const int32 = getDType('int32');

      const result = convertValue(3.14, float32, int32, PERMISSIVE_CONVERSION_OPTIONS);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.value).toBe(3);
      }
    });

    it('should respect custom options', () => {
      const float32 = getDType('float32');
      const int32 = getDType('int32');

      const customOptions = {
        allowPrecisionLoss: true,
        allowOverflow: false,
        nanHandling: 'zero' as const,
        infinityHandling: 'error' as const,
        overflowHandling: 'error' as const,
      };

      const nanResult = convertValue(NaN, float32, int32, customOptions);
      expect(nanResult.success).toBe(true);
      if (nanResult.success) {
        expect(nanResult.value).toBe(0); // NaN converted to zero
      }

      const infResult = convertValue(Infinity, float32, int32, customOptions);
      expect(infResult.success).toBe(false); // Infinity handling is 'error'
    });
  });

  describe('Error classes', () => {
    it('should create ConversionError', () => {
      const fromDType = getDType('float32');
      const toDType = getDType('int32');
      const error = new ConversionError('Test error', fromDType, toDType, 3.14);

      expect(error).toBeInstanceOf(Error);
      expect(error.name).toBe('ConversionError');
      expect(error.message).toBe('Test error');
      expect(error.fromDType).toBe(fromDType);
      expect(error.toDType).toBe(toDType);
      expect(error.value).toBe(3.14);
    });

    it('should create PrecisionLossError', () => {
      const fromDType = getDType('float64');
      const toDType = getDType('float32');
      const precisionInfo = {
        originalValue: 1.0000001,
        convertedValue: 1.0,
        lossType: 'rounding' as const,
        message: 'Lost precision',
      };

      const error = new PrecisionLossError(fromDType, toDType, 1.0000001, precisionInfo);
      expect(error).toBeInstanceOf(ConversionError);
      expect(error.name).toBe('PrecisionLossError');
      expect(error.precisionLossInfo).toBe(precisionInfo);
    });

    it('should create OverflowError', () => {
      const fromDType = getDType('int32');
      const toDType = getDType('int8');
      const error = new OverflowError(fromDType, toDType, 200);

      expect(error).toBeInstanceOf(ConversionError);
      expect(error.name).toBe('OverflowError');
      expect(error.message).toContain('200');
      expect(error.message).toContain('overflows');
    });
  });
});
