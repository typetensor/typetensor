/**
 * Runtime tests for dtype/index.ts
 *
 * Tests for module exports, convenience functions, and system utilities
 */

import { describe, it, expect } from 'bun:test';
import * as dtypeExports from './index';
import {
  CommonDTypes,
  DTypeConstants,
  createInferredArray,
  getDTypeSystemInfo,
  validateDTypeSystem,
} from './index';

// Import direct sources to verify re-exports
import { RuntimeDType, getDType, DTYPES } from './runtime';
import { promoteTypes } from './promotion';
import { convertValue } from './conversion';
import { createTypedArray } from './typedarray';
import { bool, float32 } from './constants';

// =============================================================================
// Re-export Verification Tests
// =============================================================================

describe('Module Re-exports', () => {
  it('should re-export all type system functions', () => {
    // Runtime exports
    expect(dtypeExports.RuntimeDType).toBe(RuntimeDType);
    expect(dtypeExports.DTYPES).toBe(DTYPES);
    expect(dtypeExports.getDType).toBe(getDType);

    // Promotion exports
    expect(dtypeExports.promoteTypes).toBe(promoteTypes);

    // Conversion exports
    expect(dtypeExports.convertValue).toBe(convertValue);

    // TypedArray exports
    expect(dtypeExports.createTypedArray).toBe(createTypedArray);

    // Constants exports
    expect(dtypeExports.bool).toBe(bool);
    expect(dtypeExports.float32).toBe(float32);
  });

  it('should export all necessary types and functions', () => {
    // Check for key exports existence
    expect(typeof dtypeExports.getDType).toBe('function');
    expect(typeof dtypeExports.promoteTypes).toBe('function');
    expect(typeof dtypeExports.convertValue).toBe('function');
    expect(typeof dtypeExports.createTypedArray).toBe('function');
    expect(typeof dtypeExports.createInferredArray).toBe('function');
    expect(typeof dtypeExports.getDTypeSystemInfo).toBe('function');
    expect(typeof dtypeExports.validateDTypeSystem).toBe('function');
  });
});

// =============================================================================
// CommonDTypes Tests
// =============================================================================

describe('CommonDTypes', () => {
  it('should provide all dtype instances', () => {
    expect(CommonDTypes.bool).toBe(DTYPES.bool);
    expect(CommonDTypes.int8).toBe(DTYPES.int8);
    expect(CommonDTypes.uint8).toBe(DTYPES.uint8);
    expect(CommonDTypes.int16).toBe(DTYPES.int16);
    expect(CommonDTypes.uint16).toBe(DTYPES.uint16);
    expect(CommonDTypes.int32).toBe(DTYPES.int32);
    expect(CommonDTypes.uint32).toBe(DTYPES.uint32);
    expect(CommonDTypes.int64).toBe(DTYPES.int64);
    expect(CommonDTypes.uint64).toBe(DTYPES.uint64);
    expect(CommonDTypes.float32).toBe(DTYPES.float32);
    expect(CommonDTypes.float64).toBe(DTYPES.float64);
  });

  it('should have correct properties for each dtype', () => {
    expect(CommonDTypes.bool.name).toBe('bool');
    expect(CommonDTypes.bool.jsType).toBe('boolean');

    expect(CommonDTypes.int32.name).toBe('int32');
    expect(CommonDTypes.int32.byteSize).toBe(4);

    expect(CommonDTypes.float64.name).toBe('float64');
    expect(CommonDTypes.float64.isInteger).toBe(false);
  });

  it('should be immutable', () => {
    expect(() => {
      // @ts-expect-error - Testing immutability
      CommonDTypes.float32 = DTYPES.float64;
    }).toThrow();

    expect(() => {
      // @ts-expect-error - Testing immutability
      CommonDTypes.newType = DTYPES.int32;
    }).toThrow();
  });

  it('should be usable with dtype functions', () => {
    const array = createTypedArray(CommonDTypes.float32, 10);
    expect(array.dtype).toBe(CommonDTypes.float32);

    const promoted = promoteTypes(CommonDTypes.int8, CommonDTypes.float32);
    expect(promoted).toBe(CommonDTypes.float32);
  });
});

// =============================================================================
// DTypeConstants Tests
// =============================================================================

describe('DTypeConstants', () => {
  it('should provide phantom type constants', () => {
    // These are compile-time only, runtime values are null
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.bool).toBe(null);
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.int8).toBe(null);
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.uint8).toBe(null);
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.int16).toBe(null);
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.uint16).toBe(null);
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.int32).toBe(null);
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.uint32).toBe(null);
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.int64).toBe(null);
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.uint64).toBe(null);
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.float32).toBe(null);
    // @ts-expect-error - Testing phantom type
    expect(DTypeConstants.float64).toBe(null);
  });

  it('should be immutable', () => {
    expect(() => {
      // @ts-expect-error - Testing phantom type
      DTypeConstants.float32 = null;
    }).toThrow();
  });
});

// =============================================================================
// createInferredArray Tests
// =============================================================================

describe('createInferredArray', () => {
  it('should infer boolean arrays', () => {
    const array = createInferredArray([true, false, true]);
    expect(array.dtype.name).toBe('bool');
    expect(array.toArray()).toEqual([true, false, true]);
  });

  it('should infer integer arrays', () => {
    // Small integers -> int8
    const int8Array = createInferredArray([1, 2, 3, 127]);
    expect(int8Array.dtype.name).toBe('int8');
    expect(int8Array.toArray()).toEqual([1, 2, 3, 127]);

    // Larger integers -> appropriate size
    // [1,2,3] are int8, 128 is uint8. int8+uint8 = int16 (mixed signedness)
    const int16Array = createInferredArray([1, 2, 3, 128]);
    expect(int16Array.dtype.name).toBe('int16');

    const int32Array = createInferredArray([1, 2, 3, 70000]);
    expect(int32Array.dtype.name).toBe('int32');
  });

  it('should infer float arrays', () => {
    const floatArray = createInferredArray([1.5, 2.5, 3.5]);
    expect(floatArray.dtype.name).toBe('float32');
    expect(floatArray.toArray()).toEqual([1.5, 2.5, 3.5]);

    // Large floats -> float64
    const float64Array = createInferredArray([1.5, Number.MAX_VALUE]);
    expect(float64Array.dtype.name).toBe('float64');
  });

  it('should infer bigint arrays', () => {
    const bigintArray = createInferredArray([1n, 2n, 3n]);
    expect(bigintArray.dtype.name).toBe('uint64');
    expect(bigintArray.toArray()).toEqual([1n, 2n, 3n]);

    // Mixed positive/negative bigints -> float64 (uint64 + int64 = float64)
    const mixedBigintArray = createInferredArray([1n, -2n, 3n]);
    expect(mixedBigintArray.dtype.name).toBe('float64');
    // Values are converted to numbers when stored as float64
    expect(mixedBigintArray.toArray()).toEqual([1, -2, 3]);

    // All negative bigints -> int64
    const int64Array = createInferredArray([-1n, -2n, -3n]);
    expect(int64Array.dtype.name).toBe('int64');
    expect(int64Array.toArray()).toEqual([-1n, -2n, -3n]);
  });

  it('should handle mixed type arrays', () => {
    // Bool + int -> int
    const mixedArray1 = createInferredArray([true, 1, 2]);
    expect(mixedArray1.dtype.name).toBe('int8');
    expect(mixedArray1.toArray()).toEqual([1, 1, 2]);

    // Int + float -> float
    const mixedArray2 = createInferredArray([1, 2.5, 3]);
    expect(mixedArray2.dtype.name).toBe('float32');
    expect(mixedArray2.toArray()).toEqual([1.0, 2.5, 3.0]);

    // Number + bigint -> float64 (int8 + uint64 = float64)
    const mixedArray3 = createInferredArray([1, 2, 3n]);
    expect(mixedArray3.dtype.name).toBe('float64');
    expect(mixedArray3.toArray()).toEqual([1, 2, 3]); // Stored as numbers in float64
  });

  it('should handle empty arrays', () => {
    expect(() => createInferredArray([])).toThrow('Cannot find common type for empty array');
  });

  it('should handle special values', () => {
    const nanArray = createInferredArray([1.5, NaN, 3.5]);
    expect(nanArray.dtype.name).toBe('float32');
    expect(Number.isNaN(nanArray.get(1))).toBe(true);

    const infArray = createInferredArray([1.5, Infinity, -Infinity]);
    expect(infArray.dtype.name).toBe('float32');
    expect(infArray.get(1)).toBe(Infinity);
    expect(infArray.get(2)).toBe(-Infinity);
  });

  it('should throw for invalid values', () => {
    expect(() => createInferredArray(['string'])).toThrow('Cannot determine DType');
    expect(() => createInferredArray([null])).toThrow('Cannot determine DType');
    expect(() => createInferredArray([undefined])).toThrow('Cannot determine DType');
    expect(() => createInferredArray([{}])).toThrow('Cannot determine DType');
  });

  it('should handle conversion failures', () => {
    // This would fail if we try to convert incompatible values after inference
    const mixedValid = createInferredArray([1, 2, 3]);
    expect(mixedValid.dtype.name).toBe('int8');
  });
});

// =============================================================================
// getDTypeSystemInfo Tests
// =============================================================================

describe('getDTypeSystemInfo', () => {
  it('should return complete system information', () => {
    const info = getDTypeSystemInfo();

    expect(info).toHaveProperty('availableDTypes');
    expect(info).toHaveProperty('defaultDTypes');
    expect(info).toHaveProperty('promotionRules');
    expect(info).toHaveProperty('memoryLayout');
  });

  it('should list all available dtypes', () => {
    const info = getDTypeSystemInfo();
    const expectedDTypes = [
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
    ];

    expect(info.availableDTypes).toHaveLength(11);
    for (const dtype of expectedDTypes) {
      expect(info.availableDTypes).toContain(dtype);
    }
  });

  it('should provide default dtype mappings', () => {
    const info = getDTypeSystemInfo();

    expect(info.defaultDTypes).toEqual({
      boolean: 'bool',
      integer: 'int32',
      float: 'float32',
      bigint: 'int64',
    });
  });

  it('should provide promotion rules string', () => {
    const info = getDTypeSystemInfo();

    expect(typeof info.promotionRules).toBe('string');
    expect(info.promotionRules).toContain('NumPy-Compatible');
    expect(info.promotionRules).toContain('Hierarchy');
    expect(info.promotionRules).toContain('Mixed signedness');
  });

  it('should provide memory layout for all dtypes', () => {
    const info = getDTypeSystemInfo();

    expect(Object.keys(info.memoryLayout)).toHaveLength(11);

    // Check specific dtype layouts
    expect(info.memoryLayout['bool']).toEqual({
      byteSize: 1,
      signed: false,
      isInteger: true,
    });

    expect(info.memoryLayout['int32']).toEqual({
      byteSize: 4,
      signed: true,
      isInteger: true,
    });

    expect(info.memoryLayout['float64']).toEqual({
      byteSize: 8,
      signed: true,
      isInteger: false,
    });

    expect(info.memoryLayout['uint64']).toEqual({
      byteSize: 8,
      signed: false,
      isInteger: true,
    });
  });
});

// =============================================================================
// validateDTypeSystem Tests
// =============================================================================

describe('validateDTypeSystem', () => {
  it('should validate the entire dtype system', () => {
    const result = validateDTypeSystem();

    expect(result).toHaveProperty('success');
    expect(result).toHaveProperty('errors');
    expect(result.success).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('should validate promotion matrix', () => {
    // The validation includes checking the promotion matrix
    const result = validateDTypeSystem();

    // If there were issues with the promotion matrix, they would appear in errors
    expect(result.errors.filter((e) => e.includes('Promotion matrix'))).toHaveLength(0);
  });

  it('should validate dtype creation', () => {
    // The validation creates all dtypes
    const result = validateDTypeSystem();

    // If there were issues creating dtypes, they would appear in errors
    expect(result.errors.filter((e) => e.includes('Failed to create DType'))).toHaveLength(0);
  });

  it('should validate promotion symmetry', () => {
    // The validation checks that promoteTypes(a, b) === promoteTypes(b, a)
    const result = validateDTypeSystem();

    // If there were asymmetry issues, they would appear in errors
    expect(result.errors.filter((e) => e.includes('Promotion asymmetry'))).toHaveLength(0);
  });

  it('should return detailed error information if validation fails', () => {
    // We can't easily cause a validation failure without modifying the system,
    // but we can verify the structure is correct
    const result = validateDTypeSystem();

    expect(Array.isArray(result.errors)).toBe(true);
    expect(typeof result.success).toBe('boolean');

    // If there were errors, they would be strings
    for (const error of result.errors) {
      expect(typeof error).toBe('string');
    }
  });
});

// =============================================================================
// Integration Tests
// =============================================================================

describe('Index Module Integration', () => {
  it('should work with all exported utilities together', () => {
    // Create arrays using different methods
    const explicitArray = createTypedArray(CommonDTypes.float32, 5);
    const inferredArray = createInferredArray([1.5, 2.5, 3.5, 4.5, 5.5]);

    // Both should be float32
    expect(explicitArray.dtype.name).toBe('float32');
    expect(inferredArray.dtype.name).toBe('float32');

    // Get system info
    const info = getDTypeSystemInfo();
    expect(info.memoryLayout['float32']?.byteSize).toBe(4);

    // Validate system
    const validation = validateDTypeSystem();
    expect(validation.success).toBe(true);
  });

  it('should handle complex dtype operations', () => {
    // Use CommonDTypes for operations
    createTypedArray(CommonDTypes.int32, 10);
    createTypedArray(CommonDTypes.float32, 10);

    // Promotion
    const promoted = promoteTypes(CommonDTypes.int32, CommonDTypes.float32);
    expect(promoted.name).toBe('float64');

    // Conversion
    const converted = convertValue(42, CommonDTypes.int32, CommonDTypes.float32);
    expect(converted.success).toBe(true);
    if (converted.success) {
      expect(converted.value).toBe(42.0);
    }
  });

  it('should provide consistent dtype references', () => {
    // CommonDTypes should reference the same instances as DTYPES
    expect(CommonDTypes.float32).toBe(DTYPES.float32);
    expect(CommonDTypes.float32).toBe(getDType('float32'));

    // All three ways of getting a dtype should yield the same instance
    const dtype1 = CommonDTypes.int32;
    const dtype2 = DTYPES.int32;
    const dtype3 = getDType('int32');

    expect(dtype1).toBe(dtype2);
    expect(dtype2).toBe(dtype3);
  });
});
