/**
 * Runtime tests for dtype/constants.ts
 *
 * Tests for dtype constants and their consistency
 */

import { describe, it, expect } from 'bun:test';
import {
  bool,
  int8,
  uint8,
  int16,
  uint16,
  int32,
  uint32,
  int64,
  uint64,
  float32,
  float64,
  DTYPE_CONSTANTS_MAP,
  getDTypeConstant,
} from './constants';
import { DTYPES } from './runtime';

describe('DType Constants', () => {
  describe('Individual Constants', () => {
    it('should define bool constant correctly', () => {
      expect(bool.__dtype).toBe('bool');
      expect(bool.__jsType).toBe(false);
      expect(bool.__typedArray).toBe(Uint8Array);
      expect(bool.__byteSize).toBe(1);
      expect(bool.__signed).toBe(false);
      expect(bool.__isInteger).toBe(true);
    });

    it('should define int8 constant correctly', () => {
      expect(int8.__dtype).toBe('int8');
      expect(int8.__jsType).toBe(0);
      expect(int8.__typedArray).toBe(Int8Array);
      expect(int8.__byteSize).toBe(1);
      expect(int8.__signed).toBe(true);
      expect(int8.__isInteger).toBe(true);
    });

    it('should define uint8 constant correctly', () => {
      expect(uint8.__dtype).toBe('uint8');
      expect(uint8.__jsType).toBe(0);
      expect(uint8.__typedArray).toBe(Uint8Array);
      expect(uint8.__byteSize).toBe(1);
      expect(uint8.__signed).toBe(false);
      expect(uint8.__isInteger).toBe(true);
    });

    it('should define int16 constant correctly', () => {
      expect(int16.__dtype).toBe('int16');
      expect(int16.__jsType).toBe(0);
      expect(int16.__typedArray).toBe(Int16Array);
      expect(int16.__byteSize).toBe(2);
      expect(int16.__signed).toBe(true);
      expect(int16.__isInteger).toBe(true);
    });

    it('should define uint16 constant correctly', () => {
      expect(uint16.__dtype).toBe('uint16');
      expect(uint16.__jsType).toBe(0);
      expect(uint16.__typedArray).toBe(Uint16Array);
      expect(uint16.__byteSize).toBe(2);
      expect(uint16.__signed).toBe(false);
      expect(uint16.__isInteger).toBe(true);
    });

    it('should define int32 constant correctly', () => {
      expect(int32.__dtype).toBe('int32');
      expect(int32.__jsType).toBe(0);
      expect(int32.__typedArray).toBe(Int32Array);
      expect(int32.__byteSize).toBe(4);
      expect(int32.__signed).toBe(true);
      expect(int32.__isInteger).toBe(true);
    });

    it('should define uint32 constant correctly', () => {
      expect(uint32.__dtype).toBe('uint32');
      expect(uint32.__jsType).toBe(0);
      expect(uint32.__typedArray).toBe(Uint32Array);
      expect(uint32.__byteSize).toBe(4);
      expect(uint32.__signed).toBe(false);
      expect(uint32.__isInteger).toBe(true);
    });

    it('should define int64 constant correctly', () => {
      expect(int64.__dtype).toBe('int64');
      expect(int64.__jsType).toBe(0n);
      expect(int64.__typedArray).toBe(BigInt64Array);
      expect(int64.__byteSize).toBe(8);
      expect(int64.__signed).toBe(true);
      expect(int64.__isInteger).toBe(true);
    });

    it('should define uint64 constant correctly', () => {
      expect(uint64.__dtype).toBe('uint64');
      expect(uint64.__jsType).toBe(0n);
      expect(uint64.__typedArray).toBe(BigUint64Array);
      expect(uint64.__byteSize).toBe(8);
      expect(uint64.__signed).toBe(false);
      expect(uint64.__isInteger).toBe(true);
    });

    it('should define float32 constant correctly', () => {
      expect(float32.__dtype).toBe('float32');
      expect(float32.__jsType).toBe(0);
      expect(float32.__typedArray).toBe(Float32Array);
      expect(float32.__byteSize).toBe(4);
      expect(float32.__signed).toBe(true);
      expect(float32.__isInteger).toBe(false);
    });

    it('should define float64 constant correctly', () => {
      expect(float64.__dtype).toBe('float64');
      expect(float64.__jsType).toBe(0);
      expect(float64.__typedArray).toBe(Float64Array);
      expect(float64.__byteSize).toBe(8);
      expect(float64.__signed).toBe(true);
      expect(float64.__isInteger).toBe(false);
    });
  });

  describe('Constants Map', () => {
    it('should contain all dtype constants', () => {
      expect(DTYPE_CONSTANTS_MAP.bool).toBe(bool);
      expect(DTYPE_CONSTANTS_MAP.int8).toBe(int8);
      expect(DTYPE_CONSTANTS_MAP.uint8).toBe(uint8);
      expect(DTYPE_CONSTANTS_MAP.int16).toBe(int16);
      expect(DTYPE_CONSTANTS_MAP.uint16).toBe(uint16);
      expect(DTYPE_CONSTANTS_MAP.int32).toBe(int32);
      expect(DTYPE_CONSTANTS_MAP.uint32).toBe(uint32);
      expect(DTYPE_CONSTANTS_MAP.int64).toBe(int64);
      expect(DTYPE_CONSTANTS_MAP.uint64).toBe(uint64);
      expect(DTYPE_CONSTANTS_MAP.float32).toBe(float32);
      expect(DTYPE_CONSTANTS_MAP.float64).toBe(float64);
    });

    it('should have correct number of entries', () => {
      const keys = Object.keys(DTYPE_CONSTANTS_MAP);
      expect(keys.length).toBe(11);
    });
  });

  describe('getDTypeConstant', () => {
    it('should retrieve constants by name', () => {
      expect(getDTypeConstant('bool')).toBe(bool);
      expect(getDTypeConstant('int32')).toBe(int32);
      expect(getDTypeConstant('float64')).toBe(float64);
    });

    it('should return the exact same reference', () => {
      const ref1 = getDTypeConstant('float32');
      const ref2 = getDTypeConstant('float32');
      expect(ref1).toBe(ref2);
      expect(ref1).toBe(float32);
    });
  });

  describe('Runtime Consistency', () => {
    it('should match runtime dtype properties', () => {
      // Check that constants properties align with runtime dtypes
      const runtimeFloat32 = DTYPES.float32;

      expect(float32.__dtype).toBe(runtimeFloat32.name);
      expect(typeof float32.__jsType).toBe(runtimeFloat32.jsType);
      expect(float32.__typedArray).toBe(runtimeFloat32.typedArrayConstructor as never);
      expect(float32.__byteSize).toBe(runtimeFloat32.byteSize as never);
      expect(float32.__signed).toBe(runtimeFloat32.signed as never);
      expect(float32.__isInteger).toBe(runtimeFloat32.isInteger as never);
    });

    it('should have matching properties for all dtypes', () => {
      const dtypeNames = Object.keys(DTYPE_CONSTANTS_MAP) as (keyof typeof DTYPE_CONSTANTS_MAP)[];

      for (const name of dtypeNames) {
        const constant = DTYPE_CONSTANTS_MAP[name];
        const runtime = DTYPES[name];

        expect(constant.__dtype).toBe(runtime.name);
        // using never to bypass type check
        expect(constant.__byteSize).toBe(runtime.byteSize as never);
        expect(constant.__signed).toBe(runtime.signed);
        expect(constant.__isInteger).toBe(runtime.isInteger);

        // Check JS type category matches
        if (runtime.jsType === 'boolean') {
          expect(typeof constant.__jsType).toBe('boolean');
        } else if (runtime.jsType === 'number') {
          expect(typeof constant.__jsType).toBe('number');
          // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        } else if (runtime.jsType === 'bigint') {
          expect(typeof constant.__jsType).toBe('bigint');
        }
      }
    });
  });

  describe('Type Safety', () => {
    it('should use satisfies operator correctly', () => {
      // These tests verify that the constants satisfy their expected types
      // TypeScript would catch these at compile time, but we can verify runtime

      // Bool constant should have boolean jsType
      expect(typeof bool.__jsType).toBe('boolean');

      // Number dtypes should have number jsType
      expect(typeof int32.__jsType).toBe('number');
      expect(typeof float32.__jsType).toBe('number');

      // BigInt dtypes should have bigint jsType
      expect(typeof int64.__jsType).toBe('bigint');
      expect(typeof uint64.__jsType).toBe('bigint');
    });

    it('should have immutable constants', () => {
      // Test that constants are truly constant (frozen)
      expect(() => {
        // eslint-disable-next-line
        (bool as any).__dtype = 'changed';
      }).toThrow();

      expect(() => {
        // eslint-disable-next-line
        (float32 as any).__byteSize = 8;
      }).toThrow();
    });
  });
});
