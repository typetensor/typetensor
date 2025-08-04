/**
 * Integration tests for CPU backend using shared test generators
 *
 * This test suite uses the standardized test generators from @typetensor/test-utils
 * to validate the CPU device implementation against the core tensor operations.
 */

import { describe, it, expect } from 'bun:test';
import {
  generateTensorCreationTests,
  generateTensorPropertyTests,
  generateDevUtilsTests,
  generateViewOperationTests,
  generateUnaryOperationTests,
  generateBinaryOperationTests,
  generateReductionOperationTests,
  generateSoftmaxOperationTests,
  generateMatmulOperationTests,
  generateEinopsOperationTests,
  generateDeviceOperationTests,
  generateUtilityOperationTests,
  generateDataAccessOperationTests,
} from '@typetensor/test-utils';
import { cpu } from './index';

// Test framework adapter for bun:test
const testFramework = {
  describe,
  it,
  expect: (actual: unknown) => ({
    toBe: (expected: unknown) => expect(actual).toBe(expected),
    toEqual: (expected: unknown) => expect(actual).toEqual(expected),
    toThrow: (error?: string | RegExp) => expect(actual as () => void).toThrow(error),
    toMatch: (pattern: RegExp) => expect(actual).toMatch(pattern),
    toContain: (substring: string) => expect(actual).toContain(substring),
    toBeGreaterThan: (expected: number) => expect(actual).toBeGreaterThan(expected),
    toBeCloseTo: (expected: number, precision?: number) =>
      expect(actual).toBeCloseTo(expected, precision),
    toBeTruthy: () => expect(actual).toBeTruthy(),
    toBeFalsy: () => expect(actual).toBeFalsy(),
    toBeLessThan: (expected: number) => expect(actual).toBeLessThan(expected),
    toHaveLength: (length: number) => expect(actual).toHaveLength(length),
    toBeInstanceOf: (constructor: any) => expect(actual).toBeInstanceOf(constructor),
    rejects: {
      toThrow: async (error?: string | RegExp) => {
        if (error === undefined) {
          return await expect(actual).rejects.toThrow();
        }
        return await expect(actual).rejects.toThrow(error);
      },
    },
    not: {
      toThrow: () => expect(() => actual).not.toThrow(),
      toBe: (expected: unknown) => expect(actual).not.toBe(expected),
      toContain: (substring: string) => expect(actual).not.toContain(substring),
    },
  }),
};

describe('CPU Backend Integration Tests', () => {
  describe('Device Information', () => {
    it('should provide correct device metadata', () => {
      expect(cpu.type).toBe('cpu');
      expect(cpu.id).toContain('cpu');
      expect(typeof cpu.id).toBe('string');
    });
  });

  // Run all standard test suites against CPU device
  generateTensorCreationTests(cpu, testFramework);
  generateTensorPropertyTests(cpu, testFramework);
  generateDevUtilsTests(cpu, testFramework);
  generateViewOperationTests(cpu, testFramework);
  generateUnaryOperationTests(cpu, testFramework);
  generateBinaryOperationTests(cpu, testFramework);
  generateReductionOperationTests(cpu, testFramework);
  generateSoftmaxOperationTests(cpu, testFramework);
  generateMatmulOperationTests(cpu, testFramework);
  generateEinopsOperationTests(cpu, testFramework);
  
  // New test suites for missing operations
  generateDeviceOperationTests(cpu, testFramework);
  generateUtilityOperationTests(cpu, testFramework);
  generateDataAccessOperationTests(cpu, testFramework);
});
