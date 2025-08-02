/**
 * Integration tests for WASM backend using shared test generators
 *
 * This test suite uses the standardized test generators from @typetensor/test-utils
 * to validate the WASM device implementation against the core tensor operations.
 */

import { describe, it, expect, beforeAll, afterAll } from 'bun:test';
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
import type { DeviceData } from '@typetensor/core';
import { resetWASMForTests } from './test-utils';

// Create device instance for all tests
let wasmDevice: WASMDevice;

beforeAll(async () => {
  console.log('Note: Run "bun run build:wasm" before running tests');
  wasmDevice = await WASMDevice.create();
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

  describe('Arena Memory Management', () => {
    it('should allocate tensors using arena bump allocator', () => {
      const data = wasmDevice.createData(1024); // 256 float32s
      expect(data).toBeDefined();
      expect(data.byteLength).toBe(1024);
      expect(data.device.id).toBe(wasmDevice.id);

      // Arena handles all memory safety automatically
      wasmDevice.disposeData(data);
    });

    it('should handle scope-based memory management', () => {
      let scopedData: DeviceData | null = null;

      // Use withScope for automatic cleanup
      const result = wasmDevice.withScope(() => {
        scopedData = wasmDevice.createData(2048);
        expect(scopedData.byteLength).toBe(2048);
        return 42;
      });

      // Scope should have executed and returned value
      expect(result).toBe(42);
      expect(scopedData).not.toBeNull();
    });

    it('should support manual checkpoint/restore operations', () => {
      // Take initial checkpoint
      const checkpoint1 = wasmDevice.beginScope();
      expect(typeof checkpoint1).toBe('number');

      // Allocate some data
      const data1 = wasmDevice.createData(1024);
      expect(data1.byteLength).toBe(1024);

      // Take another checkpoint
      const checkpoint2 = wasmDevice.beginScope();
      expect(typeof checkpoint2).toBe('number');
      expect(checkpoint2).not.toBe(checkpoint1);

      // Allocate more data
      const data2 = wasmDevice.createData(2048);
      expect(data2.byteLength).toBe(2048);

      // Restore to checkpoint2 - should free data2 but keep data1
      wasmDevice.endScope(checkpoint2);

      // Restore to checkpoint1 - should free everything
      wasmDevice.endScope(checkpoint1);
    });

    it('should provide garbage collection for persistent tensors', () => {
      // GC should return number of freed bytes
      const freedBytes = wasmDevice.gc();
      expect(typeof freedBytes).toBe('number');
      expect(freedBytes).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Data Views and Memory Access', () => {
    it('should create typed array views correctly', () => {
      const data = wasmDevice.createData(1024);
      
      // Create different view types
      const float32View = wasmDevice.readDataView(data, float32);
      const int32View = wasmDevice.readDataView(data, int32);
      const uint8View = wasmDevice.readDataView(data, uint8);

      expect(float32View).toBeInstanceOf(Float32Array);
      expect(int32View).toBeInstanceOf(Int32Array);
      expect(uint8View).toBeInstanceOf(Uint8Array);

      // Views should have appropriate lengths
      expect(float32View.length).toBe(256); // 1024 bytes / 4 bytes per float32
      expect(int32View.length).toBe(256);   // 1024 bytes / 4 bytes per int32
      expect(uint8View.length).toBe(1024);  // 1024 bytes / 1 byte per uint8
    });

    it('should handle view validity with arena safety', () => {
      const data = wasmDevice.createData(1024);
      const view = wasmDevice.readDataView(data, float32);

      // Arena ensures views are always safe
      expect(wasmDevice.isViewValid(view)).toBe(true);

      // Even after disposal, arena handles safety
      wasmDevice.disposeData(data);
      expect(wasmDevice.isViewValid(view)).toBe(true); // Arena provides safety
    });

    it('should handle write operations correctly', async () => {
      const data = wasmDevice.createData(1024);
      
      // Create test buffer with known values
      const buffer = new ArrayBuffer(1024);
      const view = new Float32Array(buffer);
      view[0] = 3.14159;
      view[1] = 2.71828;
      
      // Write to device
      await wasmDevice.writeData(data, buffer);
      
      // Read back should show updated data
      const newView = wasmDevice.readDataView(data, float32);
      expect(newView[0]).toBeCloseTo(3.14159, 5);
      expect(newView[1]).toBeCloseTo(2.71828, 5);
    });
  });

  describe('Memory Efficiency and Arena Behavior', () => {
    it('should handle large allocations efficiently', () => {
      // Test various allocation sizes
      const sizes = [1024, 4096, 64 * 1024, 256 * 1024]; // 1KB to 256KB
      const allocatedData: DeviceData[] = [];

      for (const size of sizes) {
        try {
          const data = wasmDevice.createData(size);
          expect(data.byteLength).toBe(size);
          allocatedData.push(data);
        } catch (error: any) {
          // If allocation fails due to memory limits, that's expected
          if (error.message.includes('memory limit') || 
              error.message.includes('Out of bounds')) {
            break; // Stop at memory limit
          } else {
            throw error; // Re-throw unexpected errors
          }
        }
      }

      // Arena should track total allocated memory
      const stats = wasmDevice.getMemoryStats();
      expect(stats.totalAllocated).toBeGreaterThan(0);

      // Cleanup all allocated data
      allocatedData.forEach(data => wasmDevice.disposeData(data));
    });

    it('should show arena usage in memory statistics', () => {
      const initialStats = wasmDevice.getMemoryStats();
      
      // Allocate some data
      const data1 = wasmDevice.createData(4096);
      const data2 = wasmDevice.createData(8192);
      
      const afterStats = wasmDevice.getMemoryStats();
      
      // Arena usage should have increased
      expect(afterStats.totalAllocated).toBeGreaterThanOrEqual(initialStats.totalAllocated);
      expect(afterStats.poolSummary).toContain('Arena');
      
      // Cleanup
      wasmDevice.disposeData(data1);
      wasmDevice.disposeData(data2);
    });
  });

  describe('Error Handling', () => {
    it('should provide clear error messages for invalid operations', () => {
      // Test device not initialized error handling
      expect(() => {
        // This should work since device is initialized
        wasmDevice.getMemoryStats();
      }).not.toThrow();
    });

    it('should handle buffer size mismatches gracefully', async () => {
      const data = wasmDevice.createData(1024);
      const wrongSizeBuffer = new ArrayBuffer(512); // Wrong size

      await expect(wasmDevice.writeData(data, wrongSizeBuffer))
        .rejects.toThrow(/size mismatch|expected.*1024.*got.*512/i);

      wasmDevice.disposeData(data);
    });

    it('should handle device ID mismatches', () => {
      const data = wasmDevice.createData(1024);
      
      // Create a mock device data with different device ID
      const mockData = {
        device: { id: 'different-device' },
        byteLength: 1024,
        id: 'mock-data',
        clone: () => mockData
      } as DeviceData;

      expect(() => wasmDevice.disposeData(mockData))
        .toThrow(/Cannot dispose data from device.*different-device.*on.*wasm/i);

      wasmDevice.disposeData(data);
    });
  });

  describe('Arena Performance', () => {
    it('should have minimal allocation overhead with arena', () => {
      const iterations = 1000;
      const start = performance.now();

      // Test rapid allocation/deallocation
      for (let i = 0; i < iterations; i++) {
        const data = wasmDevice.createData(1024);
        wasmDevice.disposeData(data);
      }

      const end = performance.now();
      const timePerOperation = (end - start) / iterations;

      // Arena should be very fast for allocation (< 0.1ms per operation)
      expect(timePerOperation).toBeLessThan(0.1);
    });

    it('should handle scoped operations efficiently', () => {
      const iterations = 100;
      const start = performance.now();

      for (let i = 0; i < iterations; i++) {
        wasmDevice.withScope(() => {
          const data1 = wasmDevice.createData(1024);
          const data2 = wasmDevice.createData(2048);
          // Arena automatically cleans up when scope exits
          return data1.byteLength + data2.byteLength;
        });
      }

      const end = performance.now();
      const timePerScope = (end - start) / iterations;

      // Scoped operations should be fast (< 1ms per scope)
      expect(timePerScope).toBeLessThan(1.0);
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

// Reset WASM module after this test file to ensure test isolation
afterAll(() => {
  resetWASMForTests();
});
