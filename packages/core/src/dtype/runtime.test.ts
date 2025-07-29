/**
 * Runtime tests for dtype/runtime.ts
 *
 * Tests for RuntimeDType class, dtype registry, and runtime validation
 */

import { describe, it, expect } from 'bun:test';
import {
  RuntimeDType,
  DTYPES,
  getDType,
  getDTypeNames,
  isValidDTypeName,
  getDefaultDType,
  isRuntimeDType,
  validateTypedArrayDType,
  getTypedArrayDType,
  validateArrayData,
  calculateByteSize,
  validateBufferSize,
  isAligned,
  getAlignedOffset,
  DTypeError,
  DTypeValidationError,
  DTypeBufferError,
} from './runtime';
import type { DTypeName } from './types';

// =============================================================================
// RuntimeDType Class Tests
// =============================================================================

describe('RuntimeDType Class', () => {
  describe('Constructor and Properties', () => {
    it('should create RuntimeDType with correct properties', () => {
      const dtype = new RuntimeDType(
        'test' as unknown as DTypeName,
        'number',
        Float32Array,
        4,
        true,
        false,
        -3.4e38,
        3.4e38,
      );

      // @ts-expect-error - test is not a valid DTypeName
      expect(dtype.name).toBe('test');
      expect(dtype.jsType).toBe('number');
      expect(dtype.typedArrayConstructor).toBe(Float32Array);
      expect(dtype.byteSize).toBe(4);
      expect(dtype.signed).toBe(true);
      expect(dtype.isInteger).toBe(false);
      expect(dtype.minValue).toBe(-3.4e38);
      expect(dtype.maxValue).toBe(3.4e38);
    });
  });

  describe('Value Validation', () => {
    it('should validate boolean values', () => {
      const bool = DTYPES.bool;
      expect(bool.isValidValue(true)).toBe(true);
      expect(bool.isValidValue(false)).toBe(true);
      expect(bool.isValidValue(0)).toBe(false);
      expect(bool.isValidValue('true')).toBe(false);
    });

    it('should validate integer values with range checking', () => {
      const int8 = DTYPES.int8;
      expect(int8.isValidValue(0)).toBe(true);
      expect(int8.isValidValue(127)).toBe(true);
      expect(int8.isValidValue(-128)).toBe(true);
      expect(int8.isValidValue(128)).toBe(false); // Overflow
      expect(int8.isValidValue(-129)).toBe(false); // Underflow
      expect(int8.isValidValue(3.14)).toBe(false); // Non-integer
      expect(int8.isValidValue('42')).toBe(false); // Wrong type
    });

    it('should validate float values correctly', () => {
      const float32 = DTYPES.float32;
      expect(float32.isValidValue(3.14)).toBe(true);
      expect(float32.isValidValue(0)).toBe(true);
      expect(float32.isValidValue(Number.POSITIVE_INFINITY)).toBe(true);
      expect(float32.isValidValue(Number.NaN)).toBe(true);
      expect(float32.isValidValue('3.14')).toBe(false);
    });

    it('should validate bigint values correctly', () => {
      const int64 = DTYPES.int64;
      expect(int64.isValidValue(0n)).toBe(true);
      expect(int64.isValidValue(9223372036854775807n)).toBe(true); // Max int64
      expect(int64.isValidValue(-9223372036854775808n)).toBe(true); // Min int64
      expect(int64.isValidValue(0)).toBe(false); // Wrong JS type
    });

    it('should throw on invalid value validation', () => {
      const int32 = DTYPES.int32;
      expect(() => int32.validateValue('invalid')).toThrow('Invalid value for int32');
    });
  });

  describe('TypedArray Creation', () => {
    it('should create typed arrays', () => {
      const float32 = DTYPES.float32;
      const array = float32.createTypedArray(10);
      expect(array).toBeInstanceOf(Float32Array);
      expect(array.length).toBe(10);
    });

    it('should create typed arrays from buffer', () => {
      const int32 = DTYPES.int32;
      const buffer = new ArrayBuffer(40);
      const array = int32.createTypedArrayFromBuffer(buffer);
      expect(array).toBeInstanceOf(Int32Array);
      expect(array.length).toBe(10);
      expect(array.buffer).toBe(buffer);
    });

    it('should create typed arrays from data', () => {
      const int32 = DTYPES.int32;
      const data = [1, 2, 3, 4, 5];
      const array = int32.createTypedArrayFromData(data);
      expect(array).toBeInstanceOf(Int32Array);
      expect(Array.from(array)).toEqual(data);
    });

    it('should validate data when creating arrays', () => {
      const int8 = DTYPES.int8;
      const invalidData = [1, 2, 300]; // 300 is out of range
      expect(() => int8.createTypedArrayFromData(invalidData)).toThrow('Invalid value at index 2');
    });

    it('should handle bigint arrays correctly', () => {
      const int64 = DTYPES.int64;
      const data = [1n, 2n, 3n];
      const array = int64.createTypedArrayFromData(data);
      expect(array).toBeInstanceOf(BigInt64Array);
      expect(Array.from(array)).toEqual(data);
    });
  });

  describe('Compatibility Checking', () => {
    it('should check dtype compatibility', () => {
      const int32 = DTYPES.int32;
      const float32 = DTYPES.float32;
      const bool = DTYPES.bool;
      const int64 = DTYPES.int64;

      expect(int32.isCompatibleWith(int32)).toBe(true); // Same type
      expect(int32.isCompatibleWith(float32)).toBe(true); // Same JS type
      expect(bool.isCompatibleWith(int32)).toBe(true); // Bool is compatible
      expect(int32.isCompatibleWith(int64)).toBe(false); // Different JS types
    });
  });

  describe('Info and Serialization', () => {
    it('should provide dtype info', () => {
      const float32 = DTYPES.float32;
      const info = float32.getInfo();

      expect(info.name).toBe('float32');
      expect(info.jsType).toBe('number');
      expect(info.byteSize).toBe(4);
      expect(info.signed).toBe(true);
      expect(info.isInteger).toBe(false);
      expect(info.typedArrayName).toBe('Float32Array');
    });

    it('should have proper string representation', () => {
      const int32 = DTYPES.int32;
      expect(int32.toString()).toBe('RuntimeDType(int32)');
    });

    it('should have JSON representation', () => {
      const float64 = DTYPES.float64;
      const json = float64.toJSON();
      expect(json).toEqual({
        name: 'float64',
        byteSize: 8,
        signed: true,
        isInteger: false,
      });
    });
  });
});

// =============================================================================
// DType Registry Tests
// =============================================================================

describe('DType Registry', () => {
  it('should provide access to all standard dtypes', () => {
    const expectedNames = [
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

    for (const name of expectedNames) {
      expect(DTYPES[name as keyof typeof DTYPES]).toBeDefined();
      // @ts-expect-error - name is a string, not a DTypeName, the type is not narrowed here but we are testing runtime behavior
      expect(DTYPES[name as keyof typeof DTYPES].name).toBe(name);
    }
  });

  it('should have correct properties for each dtype', () => {
    // Bool
    expect(DTYPES.bool.jsType).toBe('boolean');
    expect(DTYPES.bool.byteSize).toBe(1);
    expect(DTYPES.bool.signed).toBe(false);
    expect(DTYPES.bool.isInteger).toBe(true);

    // Int32
    expect(DTYPES.int32.jsType).toBe('number');
    expect(DTYPES.int32.byteSize).toBe(4);
    expect(DTYPES.int32.signed).toBe(true);
    expect(DTYPES.int32.isInteger).toBe(true);
    expect(DTYPES.int32.minValue).toBe(-2147483648);
    expect(DTYPES.int32.maxValue).toBe(2147483647);

    // Float32
    expect(DTYPES.float32.jsType).toBe('number');
    expect(DTYPES.float32.byteSize).toBe(4);
    expect(DTYPES.float32.signed).toBe(true);
    expect(DTYPES.float32.isInteger).toBe(false);

    // Int64
    expect(DTYPES.int64.jsType).toBe('bigint');
    expect(DTYPES.int64.byteSize).toBe(8);
    expect(DTYPES.int64.signed).toBe(true);
    expect(DTYPES.int64.isInteger).toBe(true);
  });
});

// =============================================================================
// Factory Function Tests
// =============================================================================

describe('Factory Functions', () => {
  describe('getDType', () => {
    it('should return correct dtype instances', () => {
      const float32 = getDType('float32');
      expect(float32).toBe(DTYPES.float32);
      expect(float32.name).toBe('float32');
    });

    it('should provide consistent instances', () => {
      const float32_1 = getDType('float32');
      const float32_2 = getDType('float32');
      expect(float32_1).toBe(float32_2); // Same instance
    });

    it('should throw for invalid dtype names', () => {
      // NOTE: Explicitly testing type unsafe behavior to ensure proper runtime guards
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      expect(() => getDType('invalid' as any)).toThrow('Unknown DType: invalid');
    });
  });

  describe('getDTypeNames', () => {
    it('should return all dtype names', () => {
      const names = getDTypeNames();
      expect(names).toContain('bool');
      expect(names).toContain('int32');
      expect(names).toContain('float64');
      expect(names.length).toBe(11);
    });
  });

  describe('isValidDTypeName', () => {
    it('should validate dtype names', () => {
      expect(isValidDTypeName('float32')).toBe(true);
      expect(isValidDTypeName('int64')).toBe(true);
      expect(isValidDTypeName('invalid')).toBe(false);
      expect(isValidDTypeName('')).toBe(false);
    });
  });

  describe('getDefaultDType', () => {
    it('should infer default dtypes correctly', () => {
      expect(getDefaultDType(true).name).toBe('bool');
      expect(getDefaultDType(false).name).toBe('bool');
      expect(getDefaultDType(42).name).toBe('int32');
      expect(getDefaultDType(-42).name).toBe('int32');
      expect(getDefaultDType(3.14).name).toBe('float32');
      expect(getDefaultDType(42n).name).toBe('uint64');
      expect(getDefaultDType(-42n).name).toBe('int64');
    });

    it('should handle edge cases', () => {
      expect(getDefaultDType(Number.MAX_SAFE_INTEGER + 1).name).toBe('float64');
      expect(getDefaultDType(0).name).toBe('int32');
      expect(getDefaultDType(0.0).name).toBe('int32'); // Integer check
      expect(getDefaultDType(0.5).name).toBe('float32');
    });

    it('should throw for invalid values', () => {
      expect(() => getDefaultDType('string')).toThrow('Cannot determine DType');
      expect(() => getDefaultDType(null)).toThrow('Cannot determine DType');
      expect(() => getDefaultDType(undefined)).toThrow('Cannot determine DType');
    });
  });
});

// =============================================================================
// Type Guard Tests
// =============================================================================

describe('Type Guards', () => {
  it('should identify RuntimeDType instances', () => {
    const dtype = DTYPES.float32;
    const notDtype = { name: 'float32' };

    expect(isRuntimeDType(dtype)).toBe(true);
    expect(isRuntimeDType(notDtype)).toBe(false);
    expect(isRuntimeDType(null)).toBe(false);
    expect(isRuntimeDType('float32')).toBe(false);
  });
});

// =============================================================================
// TypedArray Validation Tests
// =============================================================================

describe('TypedArray Validation', () => {
  describe('validateTypedArrayDType', () => {
    it('should validate typed arrays match dtype', () => {
      const float32 = DTYPES.float32;
      const array = new Float32Array(10);
      const wrongArray = new Int32Array(10);

      expect(validateTypedArrayDType(array, float32)).toBe(true);
      expect(validateTypedArrayDType(wrongArray, float32)).toBe(false);
    });

    it('should reject non-typed arrays', () => {
      const float32 = DTYPES.float32;
      const regularArray = [1, 2, 3];

      expect(validateTypedArrayDType(regularArray, float32)).toBe(false);
    });
  });

  describe('getTypedArrayDType', () => {
    it('should identify typed array dtypes', () => {
      expect(getTypedArrayDType(new Float32Array(1))?.name).toBe('float32');
      expect(getTypedArrayDType(new Int32Array(1))?.name).toBe('int32');
      expect(getTypedArrayDType(new Uint8Array(1))?.name).toBe('uint8');
      expect(getTypedArrayDType(new BigInt64Array(1))?.name).toBe('int64');
    });

    it('should handle bool dtype (uses Uint8Array)', () => {
      const uint8Array = new Uint8Array(1);
      const dtype = getTypedArrayDType(uint8Array);
      // Note: Can't distinguish between bool and uint8 from TypedArray alone
      // We default to uint8 as it's the more common interpretation
      expect(dtype?.name).toBe('uint8');
    });

    it('should return null for unknown typed arrays', () => {
      const dataView = new DataView(new ArrayBuffer(10));
      expect(getTypedArrayDType(dataView)).toBeNull();
    });
  });

  describe('validateArrayData', () => {
    it('should validate and return array data', () => {
      const int32 = DTYPES.int32;
      const data = [1, 2, 3, 4, 5];
      const validated = validateArrayData(data, int32);

      expect(validated).toEqual(data);
      expect(validated).toHaveLength(5);
    });

    it('should throw for invalid data', () => {
      const int8 = DTYPES.int8;
      const invalidData = [1, 2, 300]; // 300 out of range

      expect(() => validateArrayData(invalidData, int8)).toThrow('Invalid value at index 2');
    });

    it('should validate mixed valid/invalid data correctly', () => {
      const float32 = DTYPES.float32;
      const data = [1, 2.5, 3];
      const validated = validateArrayData(data, float32);

      expect(validated).toEqual(data);
    });
  });
});

// =============================================================================
// Memory Utility Tests
// =============================================================================

describe('Memory Utilities', () => {
  describe('calculateByteSize', () => {
    it('should calculate correct byte sizes', () => {
      expect(calculateByteSize(10, DTYPES.int8)).toBe(10);
      expect(calculateByteSize(10, DTYPES.int16)).toBe(20);
      expect(calculateByteSize(10, DTYPES.int32)).toBe(40);
      expect(calculateByteSize(10, DTYPES.float64)).toBe(80);
    });

    it('should handle edge cases', () => {
      expect(calculateByteSize(0, DTYPES.float32)).toBe(0);
      expect(calculateByteSize(1, DTYPES.bool)).toBe(1);
    });
  });

  describe('validateBufferSize', () => {
    it('should validate buffer has sufficient size', () => {
      const buffer = new ArrayBuffer(100);
      const int32 = DTYPES.int32;

      expect(validateBufferSize(buffer, 25, int32)).toBe(true); // 25 * 4 = 100
      expect(validateBufferSize(buffer, 26, int32)).toBe(false); // 26 * 4 = 104 > 100
      expect(validateBufferSize(buffer, 10, int32, 60)).toBe(true); // 10 * 4 + 60 = 100
      expect(validateBufferSize(buffer, 11, int32, 60)).toBe(false); // 11 * 4 + 60 = 104 > 100
    });
  });

  describe('isAligned', () => {
    it('should check alignment correctly', () => {
      const int32 = DTYPES.int32;
      const float64 = DTYPES.float64;

      expect(isAligned(0, int32)).toBe(true);
      expect(isAligned(4, int32)).toBe(true);
      expect(isAligned(8, int32)).toBe(true);
      expect(isAligned(1, int32)).toBe(false);
      expect(isAligned(2, int32)).toBe(false);
      expect(isAligned(3, int32)).toBe(false);

      expect(isAligned(0, float64)).toBe(true);
      expect(isAligned(8, float64)).toBe(true);
      expect(isAligned(4, float64)).toBe(false);
    });
  });

  describe('getAlignedOffset', () => {
    it('should calculate aligned offsets', () => {
      const int32 = DTYPES.int32;
      const float64 = DTYPES.float64;

      expect(getAlignedOffset(0, int32)).toBe(0);
      expect(getAlignedOffset(1, int32)).toBe(4);
      expect(getAlignedOffset(3, int32)).toBe(4);
      expect(getAlignedOffset(4, int32)).toBe(4);
      expect(getAlignedOffset(5, int32)).toBe(8);

      expect(getAlignedOffset(0, float64)).toBe(0);
      expect(getAlignedOffset(1, float64)).toBe(8);
      expect(getAlignedOffset(7, float64)).toBe(8);
      expect(getAlignedOffset(8, float64)).toBe(8);
      expect(getAlignedOffset(9, float64)).toBe(16);
    });
  });
});

// =============================================================================
// Error Class Tests
// =============================================================================

describe('Error Classes', () => {
  describe('DTypeError', () => {
    it('should create basic dtype errors', () => {
      const error = new DTypeError('Test error');
      expect(error.message).toBe('Test error');
      expect(error.name).toBe('DTypeError');
      expect(error.dtype).toBeUndefined();
      expect(error.value).toBeUndefined();
    });

    it('should include dtype and value context', () => {
      const dtype = DTYPES.int32;
      const error = new DTypeError('Test error', dtype, 'invalid');
      expect(error.dtype).toBe(dtype);
      expect(error.value).toBe('invalid');
    });
  });

  describe('DTypeValidationError', () => {
    it('should create validation errors', () => {
      const dtype = DTYPES.int8;
      const error = new DTypeValidationError(300, dtype);
      expect(error.name).toBe('DTypeValidationError');
      expect(error.message).toContain('300');
      expect(error.message).toContain('int8');
      expect(error.dtype).toBe(dtype);
      expect(error.value).toBe(300);
    });
  });

  describe('DTypeBufferError', () => {
    it('should create buffer errors', () => {
      const dtype = DTYPES.float32;
      const error = new DTypeBufferError('Buffer too small', dtype);
      expect(error.name).toBe('DTypeBufferError');
      expect(error.message).toBe('Buffer too small');
      expect(error.dtype).toBe(dtype);
    });
  });
});
