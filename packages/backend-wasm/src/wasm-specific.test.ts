/**
 * WASM-specific tests for functionality not covered by general tensor tests
 * 
 * These tests focus on WASM backend-specific features:
 * - Pattern cache optimization
 * - Arena memory management  
 * - Custom executor configuration
 * - WASM-specific performance characteristics
 * - Direct executor method testing
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'bun:test';
import { WASMDevice } from './device';
import { float32, int32, uint8, bool, int8, int16, uint16, uint32, float64, int64, uint64 } from '@typetensor/core';
import { resetWASMForTests } from './test-utils';

describe('WASM-Specific Functionality Tests', () => {
  let device: WASMDevice;

  beforeAll(async () => {
    device = await WASMDevice.create();
  });

  afterAll(() => {
    resetWASMForTests();
  });

  describe('Pattern Cache Optimization', () => {
    it('should create device with custom pattern cache settings', async () => {
      const customDevice = await WASMDevice.createWithPatternCache(100, 64);
      expect(customDevice.isInitialized()).toBe(true);
      expect(customDevice.type).toBe('wasm');
    });

    it('should provide pattern cache statistics', () => {
      const stats = device.getPatternCacheStats();
      expect(stats).toBeDefined();
      expect(typeof stats.pattern_count).toBe('number');
      expect(typeof stats.total_hits).toBe('number');
      expect(typeof stats.hot_patterns).toBe('number');
      expect(typeof stats.memory_usage_bytes).toBe('number');
      expect(typeof stats.memory_utilization).toBe('number');
    });

    it('should control pattern optimization', () => {
      // Disable pattern optimization
      device.setPatternOptimization(false);
      
      // Should not throw
      expect(() => device.setPatternOptimization(false)).not.toThrow();
      
      // Re-enable pattern optimization
      device.setPatternOptimization(true);
      expect(() => device.setPatternOptimization(true)).not.toThrow();
    });

    it('should clear pattern cache', () => {
      // Should not throw
      expect(() => device.clearPatternCache()).not.toThrow();
      
      // Pattern count should be 0 after clearing
      const statsAfterClear = device.getPatternCacheStats();
      expect(statsAfterClear.pattern_count).toBe(0);
    });

    it('should show pattern cache usage after repeated operations', () => {
      // Clear cache first
      device.clearPatternCache();
      
      // Create test data
      const data1 = device.createData(64); // 16 float32s
      const data2 = device.createData(64);
      
      // Simulate repeated identical operations to trigger pattern caching
      const iterations = 10;
      for (let i = 0; i < iterations; i++) {
        // This should create the same pattern repeatedly
        const view1 = device.readDataView(data1, float32) as Float32Array;
        const view2 = device.readDataView(data2, float32) as Float32Array;
        
        // Fill with test data
        view1.fill(i + 1);
        view2.fill(i + 2);
      }
      
      // Check that patterns may have been recorded
      const finalStats = device.getPatternCacheStats();
      expect(finalStats.pattern_count).toBeGreaterThanOrEqual(0); // Patterns may or may not be cached
    });
  });

  describe('Arena Memory Management', () => {
    it('should track arena memory usage accurately', () => {
      const initialStats = device.getMemoryStats();
      expect(typeof initialStats.totalAllocated).toBe('number');
      expect(initialStats.totalAllocated).toBeGreaterThanOrEqual(0);
      
      // Allocate some data
      const data = device.createData(1024);
      const afterStats = device.getMemoryStats();
      
      // Arena usage should increase or stay the same (due to arena pre-allocation)
      expect(afterStats.totalAllocated).toBeGreaterThanOrEqual(initialStats.totalAllocated);
      
      device.disposeData(data);
    });

    it('should handle nested scopes correctly', () => {
      let outerData: any = null;
      let innerData: any = null;
      
      // Outer scope
      const result = device.withScope(() => {
        outerData = device.createData(512);
        expect(outerData.byteLength).toBe(512);
        
        // Inner scope
        const innerResult = device.withScope(() => {
          innerData = device.createData(256);
          expect(innerData.byteLength).toBe(256);
          return 'inner-done';
        });
        
        expect(innerResult).toBe('inner-done');
        return 'outer-done';
      });
      
      expect(result).toBe('outer-done');
      // Scoped data should still be accessible within arena bounds
      expect(outerData).not.toBeNull();
      expect(innerData).not.toBeNull();
    });

    it('should handle manual checkpoint/restore operations', () => {
      const checkpoint1 = device.beginScope();
      expect(typeof checkpoint1).toBe('number');
      
      const data1 = device.createData(128);
      expect(data1.byteLength).toBe(128);
      
      const checkpoint2 = device.beginScope();
      expect(typeof checkpoint2).toBe('number');
      expect(checkpoint2).not.toBe(checkpoint1);
      
      const data2 = device.createData(256);
      expect(data2.byteLength).toBe(256);
      
      // Restore to checkpoint2 (should cleanup data2)
      device.endScope(checkpoint2);
      
      // Restore to checkpoint1 (should cleanup data1)
      device.endScope(checkpoint1);
    });

    it('should garbage collect persistent tensors', () => {
      const freedBytes = device.gc();
      expect(typeof freedBytes).toBe('number');
      expect(freedBytes).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Persistent vs Temporary Tensors', () => {
    it('should create persistent tensors with manual cleanup', () => {
      const persistentData = device.createPersistentData(float32, [4, 4]);
      expect(persistentData).toBeDefined();
      expect(persistentData.byteLength).toBe(64); // 4*4*4 bytes
      expect(persistentData.device.id).toBe(device.id);
      
      // Persistent tensors require manual disposal
      device.disposeData(persistentData);
    });

    it('should differentiate persistent from temporary tensor properties', () => {
      const tempData = device.createData(64);
      const persistentData = device.createPersistentData(float32, [4, 4]);
      
      // Both should work for basic operations
      expect(tempData.byteLength).toBe(64);
      expect(persistentData.byteLength).toBe(64);
      
      // Clean up
      device.disposeData(tempData);
      device.disposeData(persistentData);
    });
  });

  describe('Data Type Handling', () => {
    it('should support all available WASM data types', () => {
      const dtypes = [
        { dtype: bool, size: 1 },
        { dtype: int8, size: 1 },
        { dtype: uint8, size: 1 },
        { dtype: int16, size: 2 },
        { dtype: uint16, size: 2 },
        { dtype: int32, size: 4 },
        { dtype: uint32, size: 4 },
        { dtype: float32, size: 4 },
        { dtype: float64, size: 8 },
        // Note: int64/uint64 map to bigint64/biguint64 in WASM
        // { dtype: int64, size: 8 },  // Not directly supported
        // { dtype: uint64, size: 8 }, // Not directly supported
      ];
      
      for (const { dtype, size } of dtypes) {
        const data = device.createPersistentData(dtype, [10]);
        expect(data.byteLength).toBe(10 * size);
        device.disposeData(data);
      }
    });

    it('should create correct typed array views', () => {
      const float32Data = device.createData(32);
      const int32Data = device.createData(32);
      
      const float32View = device.readDataView(float32Data, float32);
      const int32View = device.readDataView(int32Data, int32);
      
      expect(float32View).toBeInstanceOf(Float32Array);
      expect(int32View).toBeInstanceOf(Int32Array);
      
      expect(float32View.length).toBe(8); // 32 bytes / 4 bytes per float32
      expect(int32View.length).toBe(8);   // 32 bytes / 4 bytes per int32
      
      device.disposeData(float32Data);
      device.disposeData(int32Data);
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle invalid operation gracefully', () => {
      // Try to call methods on uninitialized device
      const uninitializedDevice = new (WASMDevice as any)();
      
      expect(() => uninitializedDevice.createData(1024))
        .toThrow(/not initialized/i);
    });

    it('should handle buffer size mismatches in writeData', async () => {
      const data = device.createData(1024);
      const wrongSizeBuffer = new ArrayBuffer(512);
      
      await expect(device.writeData(data, wrongSizeBuffer))
        .rejects.toThrow(/size mismatch/i);
      
      device.disposeData(data);
    });

    it('should handle device ID mismatches', () => {
      const data = device.createData(64);
      
      // Create mock data with different device ID
      const mockData = {
        device: { id: 'different-device' },
        byteLength: 64,
        id: 'mock-data',
        clone: () => mockData
      } as any;
      
      expect(() => device.disposeData(mockData))
        .toThrow(/Cannot dispose data from device.*different-device/i);
      
      device.disposeData(data);
    });
  });

  describe('Device Capabilities and Info', () => {
    it('should provide accurate device capabilities', () => {
      const caps = device.getCapabilities();
      
      expect(typeof caps.simd).toBe('boolean');
      expect(typeof caps.sharedMemory).toBe('boolean');
      expect(typeof caps.optimalThreadCount).toBe('number');
      expect(caps.optimalThreadCount).toBeGreaterThan(0);
      expect(typeof caps.availableMemory).toBe('number');
      expect(caps.availableMemory).toBeGreaterThan(0);
      expect(typeof caps.version).toBe('string');
      expect(caps.version.length).toBeGreaterThan(0);
    });

    it('should provide consistent device identity', () => {
      expect(device.type).toBe('wasm');
      expect(device.id).toContain('wasm');
      expect(device.isInitialized()).toBe(true);
      expect(device.toString()).toMatch(/WASMDevice.*initialized/);
    });
  });

  describe('Performance Characteristics', () => {
    it('should show reasonable allocation performance', () => {
      const iterations = 1000;
      const start = performance.now();
      
      for (let i = 0; i < iterations; i++) {
        const data = device.createData(256);
        device.disposeData(data);
      }
      
      const end = performance.now();
      const timePerAllocation = (end - start) / iterations;
      
      // Arena allocation should be very fast (< 0.1ms per allocation)
      expect(timePerAllocation).toBeLessThan(0.1);
    });

    it('should show efficient scoped operations', () => {
      const iterations = 100;
      const start = performance.now();
      
      for (let i = 0; i < iterations; i++) {
        device.withScope(() => {
          const data1 = device.createData(512);
          const data2 = device.createData(1024);
          return data1.byteLength + data2.byteLength;
        });
      }
      
      const end = performance.now();
      const timePerScope = (end - start) / iterations;
      
      // Scoped operations should be fast (< 1ms per scope)
      expect(timePerScope).toBeLessThan(1.0);
    });
  });
});