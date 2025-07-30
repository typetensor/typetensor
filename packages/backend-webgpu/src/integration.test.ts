/**
 * Integration tests for WebGPU backend using shared test generators
 *
 * This test suite uses the standardized test generators from @typetensor/test-utils
 * to validate the WebGPU device implementation against the core tensor operations.
 */

import { describe, it, expect, beforeAll } from 'bun:test';
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
import { webgpu, isWebGPUAvailable } from './index';

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

describe('WebGPU Backend Integration Tests', () => {
  // Check if WebGPU is available
  if (!isWebGPUAvailable()) {
    it('should skip tests when WebGPU is not available', () => {
      console.warn('WebGPU is not available in this environment. Skipping integration tests.');
      expect(true).toBe(true);
    });
    return;
  }

  // Create WebGPU device for tests
  let device: Awaited<ReturnType<typeof webgpu>>;
  
  beforeAll(async () => {
    device = await webgpu();
  });

  describe('Device Information', () => {
    it('should provide correct device metadata', () => {
      expect(device.type).toBe('webgpu');
      expect(device.id).toContain('webgpu');
      expect(typeof device.id).toBe('string');
    });
  });

  // Run all standard test suites against WebGPU device
  // These will fail for unimplemented operations, showing us what needs to be done
  generateTensorCreationTests(device, testFramework);
  generateTensorPropertyTests(device, testFramework);
  generateDevUtilsTests(device, testFramework);
  generateViewOperationTests(device, testFramework);
  generateUnaryOperationTests(device, testFramework);
  generateBinaryOperationTests(device, testFramework);
  generateReductionOperationTests(device, testFramework);
  generateSoftmaxOperationTests(device, testFramework);
  generateMatmulOperationTests(device, testFramework);
  generateEinopsOperationTests(device, testFramework);
  
  // New test suites for missing operations
  generateDeviceOperationTests(device, testFramework);
  generateUtilityOperationTests(device, testFramework);
  generateDataAccessOperationTests(device, testFramework);
});