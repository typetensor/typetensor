/**
 * Integration tests for WASM backend using shared test generators
 *
 * This test suite uses the standardized test generators from @typetensor/test-utils
 * to validate the WASM device implementation against the core tensor operations.
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
import { WASMDevice } from './device';

// Create device instance for all tests
let wasmDevice: WASMDevice;

beforeAll(async () => {
  console.log('Note: Run "bun run build:wasm" before running tests');
  wasmDevice = await WASMDevice.create({ debug: false });
});

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

describe('WASM Backend Integration Tests', () => {
  describe('Device Information', () => {
    it('should provide correct device metadata', () => {
      expect(wasmDevice.type).toBe('wasm');
      expect(wasmDevice.id).toContain('wasm');
      expect(typeof wasmDevice.id).toBe('string');
      expect(wasmDevice.isInitialized()).toBe(true);
    });

    it('should provide device capabilities', () => {
      const caps = wasmDevice.getCapabilities();
      expect(caps).toBeDefined();
      expect(typeof caps.simd).toBe('boolean');
      expect(typeof caps.sharedMemory).toBe('boolean');
      expect(caps.optimalThreadCount).toBeGreaterThan(0);
      expect(caps.version).toBeDefined();
    });

    it('should provide memory statistics', () => {
      const stats = wasmDevice.getMemoryStats();
      expect(stats).toBeDefined();
      expect(typeof stats.totalAllocated).toBe('number');
      expect(typeof stats.activeBuffers).toBe('number');
      expect(typeof stats.poolSummary).toBe('string');
    });

    it('should handle data allocation and disposal', () => {
      const data = wasmDevice.createData(1024);
      expect(data).toBeDefined();
      expect(data.byteLength).toBe(1024);
      expect(data.device.id).toBe(wasmDevice.id);
      
      // Dispose should not throw
      expect(() => wasmDevice.disposeData(data)).not.toThrow();
    });
  });

  // Run all standard test suites against WASM device
  generateTensorCreationTests(wasmDevice, testFramework);
  generateTensorPropertyTests(wasmDevice, testFramework);
  generateDevUtilsTests(wasmDevice, testFramework);
  generateViewOperationTests(wasmDevice, testFramework);
  generateUnaryOperationTests(wasmDevice, testFramework);
  generateBinaryOperationTests(wasmDevice, testFramework);
  generateReductionOperationTests(wasmDevice, testFramework);
  generateSoftmaxOperationTests(wasmDevice, testFramework);
  generateMatmulOperationTests(wasmDevice, testFramework);
  generateEinopsOperationTests(wasmDevice, testFramework);
  
  // New test suites for missing operations
  generateDeviceOperationTests(wasmDevice, testFramework);
  generateUtilityOperationTests(wasmDevice, testFramework);
  generateDataAccessOperationTests(wasmDevice, testFramework);
});