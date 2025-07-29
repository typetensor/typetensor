/**
 * Runtime tests for dtype/typedarray.ts
 *
 * Tests for TypedArray integration, zero-copy operations, and memory alignment
 */

import { describe, it, expect } from 'bun:test';
import {
  createTypedArray,
  createTypedArrayFromBuffer,
  createTypedArrayFromData,
  createReadonlyTypedArray,
  convertTypedArray,
  createTypedArrayView,
  wrapTypedArray,
  sharesSameBuffer,
  hasMemoryOverlap,
  calculateMemoryUsage,
  validateTypedArray,
  copyTypedArrayData,
  createAlignedBuffer,
  TypedArrayError,
  AlignmentError,
  BoundsError,
  type DTypedArray,
} from './typedarray';
import { getDType } from './runtime';
import { PERMISSIVE_CONVERSION_OPTIONS } from './conversion';
import type { AnyDType } from './types';

// =============================================================================
// TypedArrayWrapper Implementation Tests
// =============================================================================

describe('TypedArrayWrapper Implementation', () => {
  describe('Constructor and Properties', () => {
    it('should create wrapper with correct properties', () => {
      const dtype = getDType('float32');
      const array = createTypedArray(dtype, 10);

      expect(array.dtype).toBe(dtype);
      expect(array.length).toBe(10);
      expect(array.byteLength).toBe(40); // 10 * 4 bytes
      expect(array.byteOffset).toBe(0);
      expect(array.readonly).toBe(false);
      expect(array.buffer).toBeInstanceOf(ArrayBuffer);
      expect(array.array).toBeInstanceOf(Float32Array);
    });

    it('should create readonly wrapper', () => {
      const dtype = getDType('int32');
      const mutable = createTypedArray(dtype, 5);
      const readonly = createReadonlyTypedArray(mutable);

      expect(readonly.readonly).toBe(true);
      expect(readonly.dtype).toBe(dtype);
      expect(readonly.length).toBe(5);
    });
  });

  describe('Accessors', () => {
    it('should get values with bounds checking', () => {
      const dtype = getDType('int32');
      const data = [10, 20, 30, 40, 50];
      const array = createTypedArrayFromData(dtype, data);

      expect(array.get(0)).toBe(10);
      expect(array.get(2)).toBe(30);
      expect(array.get(4)).toBe(50);
    });

    it('should throw on out of bounds get', () => {
      const array = createTypedArray(getDType('float32'), 5);

      expect(() => array.get(-1)).toThrow('Index -1 out of bounds');
      expect(() => array.get(5)).toThrow('Index 5 out of bounds');
      expect(() => array.get(10)).toThrow('Index 10 out of bounds');
    });

    it('should set values with validation', () => {
      const dtype = getDType('int8');
      const array = createTypedArray(dtype, 3);

      array.set(0, 100);
      array.set(1, -100);
      array.set(2, 0);

      expect(array.get(0)).toBe(100);
      expect(array.get(1)).toBe(-100);
      expect(array.get(2)).toBe(0);
    });

    it('should throw on readonly set', () => {
      const mutable = createTypedArray(getDType('float32'), 5);
      const readonly = createReadonlyTypedArray(mutable);

      expect(() => {
        readonly.set(0, 1.0);
      }).toThrow('Cannot modify readonly TypedArray');
    });

    it('should throw on out of bounds set', () => {
      const array = createTypedArray(getDType('int32'), 3);

      expect(() => {
        array.set(-1, 10);
      }).toThrow('Index -1 out of bounds');
      expect(() => {
        array.set(3, 10);
      }).toThrow('Index 3 out of bounds');
    });

    it('should validate value on set', () => {
      const dtype = getDType('int8');
      const array = createTypedArray(dtype, 1);

      expect(() => {
        array.set(0, 200);
      }).toThrow('Invalid value for int8: 200');
      expect(() => {
        array.set(0, 'invalid' as unknown as number);
      }).toThrow('Invalid value for int8: invalid');
    });
  });

  describe('Subarray Operations', () => {
    it('should create subarrays', () => {
      const dtype = getDType('float32');
      const data = [1.0, 2.0, 3.0, 4.0, 5.0];
      const array = createTypedArrayFromData(dtype, data);

      const sub1 = array.subarray(1, 4);
      expect(sub1.length).toBe(3);
      expect(sub1.get(0)).toBe(2.0);
      expect(sub1.get(2)).toBe(4.0);

      const sub2 = array.subarray(2);
      expect(sub2.length).toBe(3);
      expect(sub2.get(0)).toBe(3.0);
    });

    it('should share buffer with subarrays', () => {
      const array = createTypedArray(getDType('int32'), 10);
      const sub = array.subarray(2, 5);

      expect(sharesSameBuffer(array, sub)).toBe(true);

      // Modification in subarray affects original
      sub.set(0, 100);
      expect(array.get(2)).toBe(100);
    });

    it('should preserve readonly status in subarrays', () => {
      const mutable = createTypedArray(getDType('float64'), 10);
      const readonly = createReadonlyTypedArray(mutable);
      const sub = readonly.subarray(2, 5);

      expect(sub.readonly).toBe(true);
      expect(() => {
        sub.set(0, 1.0);
      }).toThrow('Cannot modify readonly TypedArray');
    });

    it('should create slices (copies)', () => {
      const dtype = getDType('int16');
      const data = [10, 20, 30, 40, 50];
      const array = createTypedArrayFromData(dtype, data);

      const slice = array.slice(1, 4);
      expect(slice.length).toBe(3);
      expect(slice.get(0)).toBe(20);

      // Slice is a copy, not sharing buffer
      expect(sharesSameBuffer(array, slice)).toBe(false);

      // Modification in slice doesn't affect original
      slice.set(0, 100);
      expect(array.get(1)).toBe(20);
    });
  });

  describe('Mutation Operations', () => {
    it('should copy within array', () => {
      const dtype = getDType('int32');
      const data = [1, 2, 3, 4, 5];
      const array = createTypedArrayFromData(dtype, data);

      array.copyWithin(0, 3, 5);
      expect(array.toArray()).toEqual([4, 5, 3, 4, 5]);
    });

    it('should throw on readonly copyWithin', () => {
      const mutable = createTypedArray(getDType('float32'), 5);
      const readonly = createReadonlyTypedArray(mutable);

      expect(() => readonly.copyWithin(0, 1)).toThrow('Cannot modify readonly TypedArray');
    });

    it('should fill array with value', () => {
      const dtype = getDType('uint8');
      const array = createTypedArray(dtype, 5);

      array.fill(42);
      expect(array.toArray()).toEqual([42, 42, 42, 42, 42]);

      array.fill(100, 1, 4);
      expect(array.toArray()).toEqual([42, 100, 100, 100, 42]);
    });

    it('should validate fill value', () => {
      const dtype = getDType('int8');
      const array = createTypedArray(dtype, 3);

      expect(() => array.fill(200)).toThrow('Invalid fill value for int8: 200');
      expect(() => array.fill('invalid' as unknown as number)).toThrow(
        'Invalid fill value for int8: invalid',
      );
    });

    it('should throw on readonly fill', () => {
      const mutable = createTypedArray(getDType('int32'), 5);
      const readonly = createReadonlyTypedArray(mutable);

      expect(() => readonly.fill(0)).toThrow('Cannot modify readonly TypedArray');
    });
  });

  describe('Array Conversion', () => {
    it('should convert to regular array for number types', () => {
      const dtype = getDType('float32');
      const data = [1.5, 2.5, 3.5];
      const array = createTypedArrayFromData(dtype, data);

      const regular = array.toArray();
      expect(regular).toEqual(data);
      expect(Array.isArray(regular)).toBe(true);
    });

    it('should convert to regular array for bigint types', () => {
      const dtype = getDType('int64');
      const data = [100n, 200n, 300n];
      const array = createTypedArrayFromData(dtype, data);

      const regular = array.toArray();
      expect(regular).toEqual(data);
      expect(Array.isArray(regular)).toBe(true);
    });

    it('should convert boolean dtype arrays', () => {
      const dtype = getDType('bool');
      const data = [true, false, true];
      const array = createTypedArrayFromData(dtype, data);

      const regular = array.toArray();
      expect(regular).toEqual([true, false, true]);
    });
  });

  describe('Iteration', () => {
    it('should iterate with forEach', () => {
      const dtype = getDType('int32');
      const data = [10, 20, 30];
      const array = createTypedArrayFromData(dtype, data);

      const values: number[] = [];
      const indices: number[] = [];
      const arrays: DTypedArray<AnyDType>[] = [];

      array.forEach((value, index, arr) => {
        values.push(value);
        indices.push(index);
        arrays.push(arr);
      });

      expect(values).toEqual([10, 20, 30]);
      expect(indices).toEqual([0, 1, 2]);
      expect(arrays.every((arr) => arr === array)).toBe(true);
    });

    it('should map to new array with different dtype', () => {
      const int32 = getDType('int32');
      const float32 = getDType('float32');
      const data = [1, 2, 3, 4];
      const array = createTypedArrayFromData(int32, data);

      const mapped = array.map((value) => value * 1.5, float32);

      expect(mapped.dtype).toBe(float32);
      expect(mapped.toArray()).toEqual([1.5, 3.0, 4.5, 6.0]);
      expect(sharesSameBuffer(array, mapped)).toBe(false);
    });

    it('should map with index and array arguments', () => {
      const dtype = getDType('int16');
      const data = [10, 20, 30];
      const array = createTypedArrayFromData(dtype, data);

      const mapped = array.map((value, index, arr) => {
        expect(arr).toBe(array);
        return value + index;
      }, dtype);

      expect(mapped.toArray()).toEqual([10, 21, 32]);
    });
  });

  describe('View Operations', () => {
    it('should create views with different dtype', () => {
      const float32 = getDType('float32');
      const uint32 = getDType('uint32');
      const array = createTypedArray(float32, 10);

      const view = array.createView(uint32);
      expect(view.dtype).toBe(uint32);
      expect(view.length).toBe(10);
      expect(sharesSameBuffer(array, view)).toBe(true);
    });

    it('should validate alignment for views', () => {
      const int32 = getDType('int32');
      const int16 = getDType('int16');
      const array = createTypedArray(int32, 10);

      // Valid: 4-byte aligned offset for int16
      const view1 = array.createView(int16, 4, 10);
      expect(view1.length).toBe(10);

      // Invalid: 2-byte offset not aligned for int32
      const int16Array = createTypedArray(int16, 10);
      expect(() => int16Array.createView(int32, 2)).toThrow('Buffer alignment error');
    });

    it('should calculate view length correctly', () => {
      const float64 = getDType('float64');
      const int32 = getDType('int32');
      const array = createTypedArray(float64, 10); // 80 bytes

      // Full view
      const view1 = array.createView(int32);
      expect(view1.length).toBe(20); // 80 bytes / 4 bytes per int32

      // Partial view with offset
      const view2 = array.createView(int32, 8, 5);
      expect(view2.length).toBe(5);

      // Auto-calculated length with offset
      const view3 = array.createView(int32, 40);
      expect(view3.length).toBe(10); // (80 - 40) / 4
    });

    it('should preserve readonly status in views', () => {
      const mutable = createTypedArray(getDType('float32'), 10);
      const readonly = createReadonlyTypedArray(mutable);
      const view = readonly.createView(getDType('int32'));

      expect(view.readonly).toBe(true);
    });

    it('should throw on invalid view length', () => {
      const array = createTypedArray(getDType('int32'), 10);
      const float64 = getDType('float64');

      // Buffer too small for requested length
      expect(() => array.createView(float64, 0, 10)).toThrow('Invalid length');
    });
  });
});

// =============================================================================
// Factory Function Tests
// =============================================================================

describe('Factory Functions', () => {
  describe('createTypedArray', () => {
    it('should create arrays of specified length', () => {
      const dtype = getDType('float64');
      const array = createTypedArray(dtype, 100);

      expect(array.length).toBe(100);
      expect(array.dtype).toBe(dtype);
      expect(array.byteLength).toBe(800); // 100 * 8
    });

    it('should validate length parameter', () => {
      const dtype = getDType('int32');

      expect(() => createTypedArray(dtype, -1)).toThrow('Invalid array length: -1');
      expect(() => createTypedArray(dtype, 3.5)).toThrow('Invalid array length: 3.5');
      expect(() => createTypedArray(dtype, NaN)).toThrow('Invalid array length: NaN');
    });

    it('should initialize with zeros', () => {
      const dtype = getDType('int16');
      const array = createTypedArray(dtype, 5);

      expect(array.toArray()).toEqual([0, 0, 0, 0, 0]);
    });
  });

  describe('createTypedArrayFromBuffer', () => {
    it('should create array from buffer', () => {
      const buffer = new ArrayBuffer(40);
      const dtype = getDType('int32');
      const array = createTypedArrayFromBuffer(dtype, buffer);

      expect(array.length).toBe(10); // 40 bytes / 4 bytes per int32
      expect(array.buffer).toBe(buffer);
      expect(array.byteOffset).toBe(0);
    });

    it('should handle byte offset', () => {
      const buffer = new ArrayBuffer(100);
      const dtype = getDType('float32');
      const array = createTypedArrayFromBuffer(dtype, buffer, 20, 10);

      expect(array.length).toBe(10);
      expect(array.byteOffset).toBe(20);
      expect(array.byteLength).toBe(40); // 10 * 4
    });

    it('should validate alignment', () => {
      const buffer = new ArrayBuffer(100);
      const dtype = getDType('int32');

      expect(() => createTypedArrayFromBuffer(dtype, buffer, 2)).toThrow('Buffer alignment error');
      expect(() => createTypedArrayFromBuffer(dtype, buffer, 3)).toThrow('Buffer alignment error');
    });

    it('should validate buffer size', () => {
      const buffer = new ArrayBuffer(10);
      const dtype = getDType('float64');

      expect(() => createTypedArrayFromBuffer(dtype, buffer, 0, 2)).toThrow('Invalid length');
    });

    it('should auto-calculate length', () => {
      const buffer = new ArrayBuffer(100);
      const dtype = getDType('int16');

      const array1 = createTypedArrayFromBuffer(dtype, buffer);
      expect(array1.length).toBe(50); // 100 / 2

      const array2 = createTypedArrayFromBuffer(dtype, buffer, 20);
      expect(array2.length).toBe(40); // (100 - 20) / 2
    });

    it('should handle edge cases', () => {
      const buffer = new ArrayBuffer(100);
      const dtype = getDType('int32');

      // Offset exceeds buffer size
      expect(() => createTypedArrayFromBuffer(dtype, buffer, 200)).toThrow('exceeds buffer size');

      // Zero-length array
      const empty = createTypedArrayFromBuffer(dtype, buffer, 100);
      expect(empty.length).toBe(0);
    });
  });

  describe('createTypedArrayFromData', () => {
    it('should create array from data', () => {
      const dtype = getDType('float32');
      const data = [1.5, 2.5, 3.5, 4.5];
      const array = createTypedArrayFromData(dtype, data);

      expect(array.length).toBe(4);
      expect(array.toArray()).toEqual(data);
    });

    it('should validate data values', () => {
      const dtype = getDType('int8');
      const invalidData = [100, 200, 300]; // Values out of range

      expect(() => createTypedArrayFromData(dtype, invalidData)).toThrow();
    });

    it('should handle bigint data', () => {
      const dtype = getDType('uint64');
      const data = [100n, 200n, 300n];
      const array = createTypedArrayFromData(dtype, data);

      expect(array.toArray()).toEqual(data);
    });

    it('should handle boolean data', () => {
      const dtype = getDType('bool');
      const data = [true, false, true, false];
      const array = createTypedArrayFromData(dtype, data);

      expect(array.get(0)).toBe(true);
      expect(array.get(1)).toBe(false);
    });
  });

  describe('createReadonlyTypedArray', () => {
    it('should create readonly wrapper', () => {
      const mutable = createTypedArray(getDType('float32'), 10);
      const readonly = createReadonlyTypedArray(mutable);

      expect(readonly.readonly).toBe(true);
      expect(readonly.length).toBe(mutable.length);
      expect(readonly.dtype).toBe(mutable.dtype);
      expect(sharesSameBuffer(mutable, readonly)).toBe(true);
    });

    it('should prevent modifications', () => {
      const mutable = createTypedArray(getDType('int32'), 5);
      mutable.set(0, 100);

      const readonly = createReadonlyTypedArray(mutable);
      expect(readonly.get(0)).toBe(100);
      expect(() => {
        readonly.set(0, 200);
      }).toThrow('Cannot modify readonly TypedArray');
      expect(() => {
        readonly.fill(0);
      }).toThrow('Cannot modify readonly TypedArray');
      expect(() => {
        readonly.copyWithin(0, 1);
      }).toThrow('Cannot modify readonly TypedArray');
    });
  });
});

// =============================================================================
// Conversion and Casting Tests
// =============================================================================

describe('Conversion and Casting', () => {
  describe('convertTypedArray', () => {
    it('should convert between compatible types', () => {
      const int32 = getDType('int32');
      const float32 = getDType('float32');
      const data = [1, 2, 3, 4];
      const source = createTypedArrayFromData(int32, data);

      const converted = convertTypedArray(source, float32);
      expect(converted.dtype).toBe(float32);
      expect(converted.toArray()).toEqual([1.0, 2.0, 3.0, 4.0]);
    });

    it('should handle conversion with precision loss', () => {
      const float32 = getDType('float32');
      const int32 = getDType('int32');
      const data = [1.5, 2.7, 3.9];
      const source = createTypedArrayFromData(float32, data);

      // Should fail with strict options
      expect(() => convertTypedArray(source, int32)).toThrow('Array conversion failed');

      // Should succeed with permissive options
      const converted = convertTypedArray(source, int32, PERMISSIVE_CONVERSION_OPTIONS);
      expect(converted.toArray()).toEqual([1, 2, 3]);
    });

    it('should convert between different JS types', () => {
      const int32 = getDType('int32');
      const int64 = getDType('int64');
      const data = [100, 200, 300];
      const source = createTypedArrayFromData(int32, data);

      const converted = convertTypedArray(source, int64);
      expect(converted.dtype).toBe(int64);
      expect(converted.toArray()).toEqual([100n, 200n, 300n]);
    });

    it('should handle empty arrays', () => {
      const source = createTypedArray(getDType('float32'), 0);
      const converted = convertTypedArray(source, getDType('int32'));

      expect(converted.length).toBe(0);
      expect(converted.toArray()).toEqual([]);
    });
  });

  describe('createTypedArrayView', () => {
    it('should create zero-copy views', () => {
      const float32 = getDType('float32');
      const uint32 = getDType('uint32');
      const source = createTypedArray(float32, 10);

      const view = createTypedArrayView(source, uint32);
      expect(view.dtype).toBe(uint32);
      expect(sharesSameBuffer(source, view)).toBe(true);
    });

    it('should handle offset and length', () => {
      const int32 = getDType('int32');
      const int16 = getDType('int16');
      const source = createTypedArray(int32, 10); // 40 bytes

      const view = createTypedArrayView(source, int16, 8, 10);
      expect(view.length).toBe(10);
      expect(view.byteOffset).toBe(8);
    });

    it('should validate alignment', () => {
      const int16 = getDType('int16');
      const int32 = getDType('int32');
      const source = createTypedArray(int16, 10);

      expect(() => createTypedArrayView(source, int32, 2)).toThrow('Buffer alignment error');
    });
  });
});

// =============================================================================
// Utility Function Tests
// =============================================================================

describe('Utility Functions', () => {
  describe('wrapTypedArray', () => {
    it('should wrap native typed arrays', () => {
      const jsArray = new Float32Array([1.5, 2.5, 3.5]);
      const wrapped = wrapTypedArray(jsArray);

      expect(wrapped).not.toBeNull();
      expect(wrapped?.dtype.name).toBe('float32');
      expect(wrapped?.length).toBe(3);
      expect(wrapped?.toArray()).toEqual([1.5, 2.5, 3.5]);
    });

    it('should handle all typed array types', () => {
      expect(wrapTypedArray(new Int8Array(1))?.dtype.name).toBe('int8');
      expect(wrapTypedArray(new Uint8Array(1))?.dtype.name).toBe('uint8');
      expect(wrapTypedArray(new Int16Array(1))?.dtype.name).toBe('int16');
      expect(wrapTypedArray(new Uint16Array(1))?.dtype.name).toBe('uint16');
      expect(wrapTypedArray(new Int32Array(1))?.dtype.name).toBe('int32');
      expect(wrapTypedArray(new Uint32Array(1))?.dtype.name).toBe('uint32');
      expect(wrapTypedArray(new Float32Array(1))?.dtype.name).toBe('float32');
      expect(wrapTypedArray(new Float64Array(1))?.dtype.name).toBe('float64');
      expect(wrapTypedArray(new BigInt64Array(1))?.dtype.name).toBe('int64');
      expect(wrapTypedArray(new BigUint64Array(1))?.dtype.name).toBe('uint64');
    });

    it('should return null for non-typed arrays', () => {
      expect(wrapTypedArray(new DataView(new ArrayBuffer(10)))).toBeNull();
      // @ts-expect-error - Wrong dtype is intentional for testing
      expect(wrapTypedArray({})).toBeNull();
    });
  });

  describe('sharesSameBuffer', () => {
    it('should detect shared buffers', () => {
      const array1 = createTypedArray(getDType('float32'), 10);
      const array2 = array1.subarray(2, 5);
      const array3 = createTypedArray(getDType('float32'), 10);

      expect(sharesSameBuffer(array1, array2)).toBe(true);
      expect(sharesSameBuffer(array1, array3)).toBe(false);
      expect(sharesSameBuffer(array2, array3)).toBe(false);
    });

    it('should work with views', () => {
      const array = createTypedArray(getDType('int32'), 10);
      const view = array.createView(getDType('float32'));

      expect(sharesSameBuffer(array, view)).toBe(true);
    });
  });

  describe('hasMemoryOverlap', () => {
    it('should detect overlapping arrays', () => {
      const array = createTypedArray(getDType('int32'), 20);
      const sub1 = array.subarray(0, 10);
      const sub2 = array.subarray(5, 15);
      const sub3 = array.subarray(10, 20);

      expect(hasMemoryOverlap(sub1, sub2)).toBe(true); // Overlaps at indices 5-9
      expect(hasMemoryOverlap(sub1, sub3)).toBe(false); // Adjacent, no overlap
      expect(hasMemoryOverlap(sub2, sub3)).toBe(true); // Overlaps at indices 10-14
    });

    it('should return false for different buffers', () => {
      const array1 = createTypedArray(getDType('float32'), 10);
      const array2 = createTypedArray(getDType('float32'), 10);

      expect(hasMemoryOverlap(array1, array2)).toBe(false);
    });

    it('should handle edge cases', () => {
      const array = createTypedArray(getDType('int16'), 10);
      const fullView = array.subarray();

      expect(hasMemoryOverlap(array, fullView)).toBe(true); // Complete overlap
    });
  });

  describe('calculateMemoryUsage', () => {
    it('should calculate memory usage correctly', () => {
      const array = createTypedArray(getDType('float64'), 1000);
      const usage = calculateMemoryUsage(array);

      expect(usage.dataBytes).toBe(8000); // 1000 * 8
      expect(usage.metadataBytes).toBeGreaterThan(0);
      expect(usage.totalBytes).toBe(usage.dataBytes + usage.metadataBytes);
      expect(usage.elementsPerMB).toBe(131072); // (1024 * 1024) / 8
    });

    it('should handle different dtypes', () => {
      const int8Usage = calculateMemoryUsage(createTypedArray(getDType('int8'), 100));
      const float64Usage = calculateMemoryUsage(createTypedArray(getDType('float64'), 100));

      expect(int8Usage.dataBytes).toBe(100); // 100 * 1
      expect(float64Usage.dataBytes).toBe(800); // 100 * 8
      expect(int8Usage.elementsPerMB).toBe(1048576); // 1MB / 1 byte
      expect(float64Usage.elementsPerMB).toBe(131072); // 1MB / 8 bytes
    });
  });

  describe('validateTypedArray', () => {
    it('should validate array constraints', () => {
      const dtype = getDType('int32');
      const array = createTypedArray(dtype, 100);

      expect(validateTypedArray(array, { minLength: 50 })).toBe(true);
      expect(validateTypedArray(array, { minLength: 200 })).toBe(false);
      expect(validateTypedArray(array, { maxLength: 200 })).toBe(true);
      expect(validateTypedArray(array, { maxLength: 50 })).toBe(false);
      expect(validateTypedArray(array, { dtype })).toBe(true);
      // @ts-expect-error - Wrong dtype is intentional for testing
      expect(validateTypedArray(array, { dtype: getDType('float32') })).toBe(false);
    });

    it('should validate alignment', () => {
      const buffer = new ArrayBuffer(100);
      const aligned = createTypedArrayFromBuffer(getDType('int32'), buffer, 8);
      const unaligned = createTypedArrayFromBuffer(getDType('int8'), buffer, 3);

      expect(validateTypedArray(aligned, { alignment: 4 })).toBe(true);
      expect(validateTypedArray(aligned, { alignment: 8 })).toBe(true);
      expect(validateTypedArray(unaligned, { alignment: 4 })).toBe(false);
    });

    it('should validate multiple constraints', () => {
      const dtype = getDType('float32');
      const array = createTypedArray(dtype, 100);

      expect(
        validateTypedArray(array, {
          minLength: 50,
          maxLength: 200,
          dtype: dtype,
          alignment: 4,
        }),
      ).toBe(true);

      expect(
        validateTypedArray(array, {
          minLength: 50,
          maxLength: 200,
          // @ts-expect-error - Wrong dtype is intentional for testing
          dtype: getDType('int32'), // Wrong dtype
          alignment: 4,
        }),
      ).toBe(false);
    });
  });
});

// =============================================================================
// Performance Utility Tests
// =============================================================================

describe('Performance Utilities', () => {
  describe('copyTypedArrayData', () => {
    it('should copy between arrays of same dtype', () => {
      const dtype = getDType('int32');
      const source = createTypedArrayFromData(dtype, [1, 2, 3, 4, 5]);
      const target = createTypedArray(dtype, 10);

      copyTypedArrayData(source, target);
      expect(target.toArray().slice(0, 5)).toEqual([1, 2, 3, 4, 5]);
      expect(target.toArray().slice(5)).toEqual([0, 0, 0, 0, 0]);
    });

    it('should handle offsets and lengths', () => {
      const dtype = getDType('float32');
      const source = createTypedArrayFromData(dtype, [1, 2, 3, 4, 5]);
      const target = createTypedArray(dtype, 10);

      copyTypedArrayData(source, target, 1, 5, 3);
      expect(target.get(5)).toBe(2); // source[1]
      expect(target.get(6)).toBe(3); // source[2]
      expect(target.get(7)).toBe(4); // source[3]
    });

    it('should convert between different dtypes', () => {
      const int32 = getDType('int32');
      const float32 = getDType('float32');
      const source = createTypedArrayFromData(int32, [10, 20, 30]);
      const target = createTypedArray(float32, 5);

      copyTypedArrayData(source, target);
      expect(target.toArray().slice(0, 3)).toEqual([10.0, 20.0, 30.0]);
    });

    it('should handle bigint conversions', () => {
      const int32 = getDType('int32');
      const int64 = getDType('int64');
      const source = createTypedArrayFromData(int32, [100, 200, 300]);
      const target = createTypedArray(int64, 5);

      copyTypedArrayData(source, target);
      expect(target.toArray().slice(0, 3)).toEqual([100n, 200n, 300n]);
    });

    it('should validate bounds', () => {
      const source = createTypedArray(getDType('int32'), 5);
      const target = createTypedArray(getDType('int32'), 5);

      expect(() => {
        copyTypedArrayData(source, target, 3, 0, 5);
      }).toThrow('Source copy bounds exceed');
      expect(() => {
        copyTypedArrayData(source, target, 0, 3, 5);
      }).toThrow('Target copy bounds exceed');
    });

    it('should handle conversion errors', () => {
      const float32 = getDType('float32');
      const int32 = getDType('int32');
      const source = createTypedArrayFromData(float32, [1.5, 2.5, 3.5]);
      const target = createTypedArray(int32, 5);

      expect(() => {
        copyTypedArrayData(source, target);
      }).toThrow('Conversion failed');
    });

    it('should handle empty copy', () => {
      const source = createTypedArray(getDType('int32'), 5);
      const target = createTypedArray(getDType('int32'), 5);

      // Zero length copy should succeed without modification
      copyTypedArrayData(source, target, 0, 0, 0);
      expect(target.toArray()).toEqual([0, 0, 0, 0, 0]);
    });
  });

  describe('createAlignedBuffer', () => {
    it('should create aligned buffers', () => {
      const buffer = createAlignedBuffer(1024, 64);
      expect(buffer).toBeInstanceOf(ArrayBuffer);
      expect(buffer.byteLength).toBeGreaterThanOrEqual(1024);
    });

    it('should validate alignment parameter', () => {
      expect(() => createAlignedBuffer(1024, 0)).toThrow('positive power of 2');
      expect(() => createAlignedBuffer(1024, -32)).toThrow('positive power of 2');
      expect(() => createAlignedBuffer(1024, 33)).toThrow('positive power of 2'); // Not power of 2
    });

    it('should handle various alignments', () => {
      const alignments = [1, 2, 4, 8, 16, 32, 64, 128];
      for (const alignment of alignments) {
        const buffer = createAlignedBuffer(1000, alignment);
        expect(buffer.byteLength).toBeGreaterThanOrEqual(1000);
      }
    });
  });
});

// =============================================================================
// Error Class Tests
// =============================================================================

describe('Error Classes', () => {
  describe('TypedArrayError', () => {
    it('should create basic typed array errors', () => {
      const error = new TypedArrayError('Test error');
      expect(error.message).toBe('Test error');
      expect(error.name).toBe('TypedArrayError');
      expect(error.dtype).toBeUndefined();
      expect(error.arrayLength).toBeUndefined();
    });

    it('should include context information', () => {
      const dtype = getDType('float32');
      const error = new TypedArrayError('Test error', dtype, 100);
      expect(error.dtype).toBe(dtype);
      expect(error.arrayLength).toBe(100);
    });
  });

  describe('AlignmentError', () => {
    it('should create alignment errors', () => {
      const error = new AlignmentError('Misaligned access', 4, 2);
      expect(error.name).toBe('AlignmentError');
      expect(error.message).toBe('Misaligned access');
      expect(error.requiredAlignment).toBe(4);
      expect(error.actualOffset).toBe(2);
    });

    it('should inherit from TypedArrayError', () => {
      const error = new AlignmentError('Test', 8, 3);
      expect(error).toBeInstanceOf(TypedArrayError);
      expect(error).toBeInstanceOf(Error);
    });
  });

  describe('BoundsError', () => {
    it('should create bounds errors', () => {
      const error = new BoundsError('Index out of bounds', 10, 5);
      expect(error.name).toBe('BoundsError');
      expect(error.message).toBe('Index out of bounds');
      expect(error.index).toBe(10);
      expect(error.arrayLength).toBe(5);
    });

    it('should inherit from TypedArrayError', () => {
      const error = new BoundsError('Test', 5, 3);
      expect(error).toBeInstanceOf(TypedArrayError);
      expect(error).toBeInstanceOf(Error);
    });
  });
});

// =============================================================================
// Integration Tests
// =============================================================================

describe('Integration Tests', () => {
  it('should handle complex array operations', () => {
    const int32 = getDType('int32');
    const float32 = getDType('float32');

    // Create source array
    const source = createTypedArrayFromData(int32, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    // Create subarray
    const sub = source.subarray(2, 8);
    expect(sub.toArray()).toEqual([3, 4, 5, 6, 7, 8]);

    // Convert to float
    const floatArray = convertTypedArray(sub, float32);
    expect(floatArray.toArray()).toEqual([3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Map operation
    const doubled = floatArray.map((v) => v * 2, float32);
    expect(doubled.toArray()).toEqual([6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

    // Create view
    const view = doubled.createView(getDType('uint32'));
    expect(view.length).toBe(6);
  });

  it('should handle zero-copy workflow', () => {
    // Create large buffer
    const buffer = new ArrayBuffer(1024);

    // Create multiple views
    const float32View = createTypedArrayFromBuffer(getDType('float32'), buffer, 0, 100);
    const int32View = createTypedArrayFromBuffer(getDType('int32'), buffer, 400, 100);

    // Verify no overlap
    expect(hasMemoryOverlap(float32View, int32View)).toBe(false);

    // Modify through views
    float32View.fill(3.14);
    int32View.fill(42);

    // Create overlapping view
    const uint8View = createTypedArrayFromBuffer(getDType('uint8'), buffer);
    expect(sharesSameBuffer(float32View, uint8View)).toBe(true);
  });

  it('should handle readonly protection consistently', () => {
    const mutable = createTypedArrayFromData(getDType('int16'), [10, 20, 30]);
    const readonly = createReadonlyTypedArray(mutable);

    // All mutation operations should throw
    expect(() => {
      readonly.set(0, 100);
    }).toThrow('readonly');
    expect(() => {
      readonly.fill(0);
    }).toThrow('readonly');
    expect(() => {
      readonly.copyWithin(0, 1);
    }).toThrow('readonly');

    // Non-mutating operations should work
    expect(readonly.get(0)).toBe(10);
    expect(readonly.toArray()).toEqual([10, 20, 30]);

    const sub = readonly.subarray(1);
    expect(sub.readonly).toBe(true);
    expect(() => {
      sub.set(0, 100);
    }).toThrow('readonly');
  });
});
