// Exactly mimic the test suite imports and setup
import { describe, it, expect } from 'bun:test';
import {
  generateTensorCreationTests,
  generateTensorPropertyTests,
  generateDevUtilsTests,
  generateViewOperationTests,
} from '@typetensor/test-utils';
import { cpu } from './src/index';
import { tensor, float32 } from '@typetensor/core';

// Test framework adapter exactly like the real test
const testFramework = {
  describe,
  it,
  expect: (actual: unknown) => ({
    toBe: (expected: unknown) => expect(actual).toBe(expected),
    toEqual: (expected: unknown) => expect(actual).toEqual(expected),
    toThrow: (error?: string | RegExp) => expect(() => actual).toThrow(error),
    toMatch: (pattern: RegExp) => expect(actual).toMatch(pattern),
    toContain: (substring: string) => expect(actual).toContain(substring),
    toBeGreaterThan: (expected: number) => expect(actual).toBeGreaterThan(expected),
    toBeCloseTo: (expected: number, precision?: number) => expect(actual).toBeCloseTo(expected, precision),
    toBeTruthy: () => expect(actual).toBeTruthy(),
    toBeFalsy: () => expect(actual).toBeFalsy(),
    toHaveLength: (length: number) => expect(actual).toHaveLength(length),
    toBeInstanceOf: (constructor: any) => expect(actual).toBeInstanceOf(constructor),
    rejects: {
      toThrow: async (error?: string | RegExp) => await expect(actual).rejects.toThrow(error),
    },
    not: {
      toThrow: () => expect(() => actual).not.toThrow(),
    },
  }),
};

describe('Mimic Test Suite Environment', () => {
  it('should throw on invalid reshape dimensions (mimicking real test)', async () => {
    console.log('=== MIMICKING EXACT TEST ENVIRONMENT ===');
    
    const tensor12 = await tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] as const, {
      device: cpu,
      dtype: float32,
    });

    console.log('Tensor created via real imports');
    console.log('Shape:', tensor12.shape);
    console.log('Size:', tensor12.size);

    // Test 1: Exact copy of the failing test
    console.log('\nTest 1: Exact test copy...');
    try {
      expect(() => {
        // @ts-expect-error - this is expected to throw at compile time but just testing that runtime also guards
        tensor12.reshape([3, 5] as const); // 15 ≠ 12 elements
      }).toThrow(/different number of elements/);
      console.log('✅ Test 1 passed');
    } catch (error) {
      console.log('❌ Test 1 failed:', (error as Error).message);
    }

    // Test 2: Using the test framework adapter
    console.log('\nTest 2: Using test framework adapter...');
    try {
      testFramework.expect(() => {
        tensor12.reshape([3, 5] as const); // 15 ≠ 12 elements
      }).toThrow(/different number of elements/);
      console.log('✅ Test 2 passed');
    } catch (error) {
      console.log('❌ Test 2 failed:', (error as Error).message);
    }

    // Test 3: Direct function call without adapter
    console.log('\nTest 3: Direct function check...');
    let didThrow = false;
    try {
      const fn = () => tensor12.reshape([3, 5] as const);
      fn();
    } catch (error) {
      didThrow = true;
      console.log('✅ Direct call threw:', (error as Error).message);
    }
    if (!didThrow) {
      console.log('❌ Direct call did not throw');
    }
  });
});