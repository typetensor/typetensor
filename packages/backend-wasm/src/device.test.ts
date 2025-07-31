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
import { float32, int32, uint8 } from '@typetensor/core';

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

  describe('Use After Free Prevention', () => {
    it('should prevent access to views after buffer disposal', async () => {
      // Create buffer and view
      const data = wasmDevice.createData(1024); // 256 float32s
      const view = wasmDevice.readDataView(data, float32);

      // View should work initially
      view[0] = 42;
      expect(view[0]).toBe(42);

      // Dispose the buffer
      wasmDevice.disposeData(data);

      // Accessing the view should now throw
      expect(() => (view[0] = 123)).toThrow(/[Vv]iew.*no longer valid|buffer.*disposed/i);
      expect(() => view[0]).toThrow(/[Vv]iew.*no longer valid|buffer.*disposed/i);
    });

    it('should prevent views from seeing reused buffer data', async () => {
      // Create first buffer
      const data1 = wasmDevice.createData(1024);
      const view1 = wasmDevice.readDataView(data1, float32);
      view1[0] = 111;
      view1[1] = 222;

      // Dispose first buffer
      wasmDevice.disposeData(data1);

      // Create second buffer - might reuse same memory
      const data2 = wasmDevice.createData(1024);
      const view2 = wasmDevice.readDataView(data2, float32);
      view2[0] = 999;
      view2[1] = 888;

      // view1 should not see data2's values
      expect(() => view1[0]).toThrow(/[Vv]iew.*no longer valid|buffer.*disposed/i);
      expect(() => view1[1]).toThrow(/[Vv]iew.*no longer valid|buffer.*disposed/i);
    });

    it('should invalidate multiple views of the same buffer', async () => {
      const data = wasmDevice.createData(1024);

      // Create multiple views
      const view1 = wasmDevice.readDataView(data, float32);
      const view2 = wasmDevice.readDataView(data, int32);
      const view3 = wasmDevice.readDataView(data, uint8);

      // All views should work
      view1[0] = 3.14;
      view2[0] = 42;
      view3[0] = 255;

      // Dispose buffer
      wasmDevice.disposeData(data);

      // All views should be invalidated
      expect(() => view1[0]).toThrow(/[Vv]iew.*no longer valid|buffer.*disposed/i);
      expect(() => view2[0]).toThrow(/[Vv]iew.*no longer valid|buffer.*disposed/i);
      expect(() => view3[0]).toThrow(/[Vv]iew.*no longer valid|buffer.*disposed/i);
    });

    it('should handle view invalidation after writeData', async () => {
      const data = wasmDevice.createData(1024);
      const view = wasmDevice.readDataView(data, float32);

      view[0] = 42;
      expect(view[0]).toBe(42);

      // Write new data - should invalidate old views
      const newBuffer = new ArrayBuffer(1024);
      new Float32Array(newBuffer)[0] = 999;
      await wasmDevice.writeData(data, newBuffer);

      // Old view should be invalid
      expect(() => view[0]).toThrow(/[Vv]iew.*no longer valid|buffer.*replaced/i);

      // New view should work
      const newView = wasmDevice.readDataView(data, float32);
      expect(newView[0]).toBe(999);
    });
  });

  describe('View Validity Tracking', () => {
    it('should track view validity correctly', async () => {
      const data = wasmDevice.createData(1024);
      const view = wasmDevice.readDataView(data, float32);

      // View should be valid initially
      expect(wasmDevice.isViewValid(view)).toBe(true);

      // After disposal, view should be invalid
      wasmDevice.disposeData(data);
      expect(wasmDevice.isViewValid(view)).toBe(false);
    });

    it('should handle subarray creation safely', async () => {
      const data = wasmDevice.createData(1024);
      const view = wasmDevice.readDataView(data, float32);

      // Create subarray
      const subarray = view.subarray(10, 20);

      // Both should work
      view[15] = 42;
      expect(subarray[5]).toBe(42); // Index 15 in view = index 5 in subarray

      // Dispose buffer
      wasmDevice.disposeData(data);

      // Both should be invalid
      expect(() => view[15]).toThrow(/[Vv]iew.*no longer valid|buffer.*disposed/i);
      expect(() => subarray[5]).toThrow(/[Vv]iew.*no longer valid|buffer.*disposed/i);
    });

    it('should handle slice() method safely', async () => {
      const data = wasmDevice.createData(1024);
      const view = wasmDevice.readDataView(data, float32) as Float32Array;

      view[0] = 11;
      view[1] = 22;
      view[2] = 33;

      // slice() creates a copy, should work even after disposal
      const sliced = view.slice(0, 3);

      wasmDevice.disposeData(data);

      // Original view should be invalid
      expect(() => view[0]).toThrow(/[Vv]iew.*no longer valid|buffer.*disposed/i);

      // But sliced copy should still work (it's a copy, not a view)
      expect(sliced[0]).toBe(11);
      expect(sliced[1]).toBe(22);
      expect(sliced[2]).toBe(33);
    });
  });

  describe('Memory Growth Handling', () => {
    it('should invalidate views when WASM memory grows', async () => {
      const data1 = wasmDevice.createData(1024);
      const view1 = wasmDevice.readDataView(data1, float32);

      view1[0] = 42;
      expect(view1[0]).toBe(42);

      // Try to force memory growth by allocating a large buffer
      // This is hypothetical - actual memory growth depends on WASM implementation
      let largeData: DeviceData | null = null;
      try {
        largeData = wasmDevice.createData(100 * 1024 * 1024); // 100MB
        
        // Check if memory grew (view.buffer would change)
        if (!wasmDevice.isViewValid(view1)) {
          // View should throw if memory grew
          expect(() => view1[0]).toThrow(/[Vv]iew.*no longer valid|memory.*grew/i);
        } else {
          // If memory didn't grow, view should still work
          expect(view1[0]).toBe(42);
        }
      } catch (error: any) {
        // If allocation failed due to memory limits, that's OK for this test
        if (error.message.includes('memory limit exceeded') || 
            error.message.includes('Out of bounds memory access')) {
          // Skip the rest of the test - we can't force memory growth
          expect(error.message).toMatch(/memory limit|Out of bounds/i);
        } else {
          throw error; // Re-throw unexpected errors
        }
      }

      // Cleanup
      if (largeData) {
        wasmDevice.disposeData(largeData);
      }
    });
  });

  describe('Error Messages', () => {
    it('should provide clear error messages', async () => {
      const data = wasmDevice.createData(1024);
      const view = wasmDevice.readDataView(data, float32);

      wasmDevice.disposeData(data);

      // Check for descriptive error message
      let error: Error | null = null;
      try {
        view[0] = 123;
      } catch (e) {
        error = e as Error;
      }

      expect(error).not.toBeNull();
      expect(error!.message).toMatch(/view.*no longer valid|buffer.*disposed/i);

      // Error should mention the buffer was disposed
      expect(error!.message.toLowerCase()).toContain('disposed');
    });
  });

  describe('Performance Considerations', () => {
    it('should have minimal overhead for valid views', async () => {
      const data = wasmDevice.createData(1024 * 1024); // 1MB
      const view = wasmDevice.readDataView(data, float32) as Float32Array;

      // Measure access time
      const iterations = 100000;
      const start = performance.now();

      for (let i = 0; i < iterations; i++) {
        view[i % view.length] = i;
      }

      const end = performance.now();
      const timePerAccess = (end - start) / iterations;

      // Should be very fast (< 1 microsecond per access)
      // This is a rough check - exact threshold depends on hardware
      expect(timePerAccess).toBeLessThan(0.001); // 1 microsecond

      // Cleanup
      wasmDevice.disposeData(data);
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
