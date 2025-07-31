import { describe, it, expect, beforeEach, afterAll } from 'bun:test';
import { WASMDevice } from './device';
import { float32, int32, uint8 } from '@typetensor/core';
import * as tt from '@typetensor/core';
import { resetWASMForTests } from './test-utils';

describe('View Lifetime Management - Memory Safety', () => {
  let device: WASMDevice;

  beforeEach(async () => {
    device = await WASMDevice.create();
  });

  describe('Buffer disposal scenarios', () => {
    it('should prevent use-after-free when buffer is disposed', async () => {
      const data = device.createData(1024);
      const view = device.readDataView(data, float32);
      
      // This should work
      view[0] = 42;
      expect(view[0]).toBe(42);
      
      // Dispose the buffer
      device.disposeData(data);
      
      // This should throw, NOT access freed memory
      expect(() => view[0] = 123).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => view[0]).toThrow('View is no longer valid: buffer has been disposed');
    });

    it('should invalidate all views when buffer is released to pool', async () => {
      const data1 = device.createData(1024);
      const view1 = device.readDataView(data1, float32);
      
      view1[0] = 42;
      expect(view1[0]).toBe(42);
      
      device.disposeData(data1);
      
      // Create new buffer - might reuse same memory from pool!
      const data2 = device.createData(1024);
      const view2 = device.readDataView(data2, float32);
      
      // view1 should not see data2's values
      view2[0] = 999;
      expect(() => view1[0]).toThrow('View is no longer valid: buffer has been disposed');
      
      // view2 should work fine
      expect(view2[0]).toBe(999);
    });

    it('should invalidate multiple views of the same buffer', async () => {
      const data = device.createData(2048);
      
      // Create multiple views with different dtypes
      const floatView = device.readDataView(data, float32);
      const intView = device.readDataView(data, int32);
      const byteView = device.readDataView(data, uint8);
      
      // All views should work initially
      floatView[0] = 3.14;
      intView[0] = 42;
      byteView[0] = 255;
      
      // Dispose the buffer
      device.disposeData(data);
      
      // All views should be invalid
      expect(() => floatView[0]).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => intView[0]).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => byteView[0]).toThrow('View is no longer valid: buffer has been disposed');
    });
  });

  describe('View method and property access', () => {
    it('should validate all property access on disposed views', async () => {
      const data = device.createData(1024);
      const view = device.readDataView(data, float32);
      
      // Properties should work before disposal
      expect(view.length).toBe(256); // 1024 bytes / 4 bytes per float32
      expect(view.byteLength).toBe(1024);
      expect(view.byteOffset).toBeGreaterThanOrEqual(0); // Offset in WASM memory
      expect(view.buffer).toBeDefined();
      
      device.disposeData(data);
      
      // All property access should throw
      expect(() => view.length).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => view.byteLength).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => view.byteOffset).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => view.buffer).toThrow('View is no longer valid: buffer has been disposed');
    });

    it('should validate array methods on disposed views', async () => {
      const data = device.createData(1024);
      const view = device.readDataView(data, float32) as Float32Array;
      
      // Fill with test data
      for (let i = 0; i < view.length; i++) {
        view[i] = i;
      }
      
      device.disposeData(data);
      
      // Array methods should throw
      expect(() => view.subarray(0, 10)).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => view.slice(0, 10)).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => view.fill(0)).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => view.set([1, 2, 3])).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => view.reverse()).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => view.sort()).toThrow('View is no longer valid: buffer has been disposed');
    });

    it('should handle subarray lifetime correctly', async () => {
      const data = device.createData(1024);
      const view = device.readDataView(data, float32) as Float32Array;
      
      // Create a subarray
      const subarray = view.subarray(10, 20);
      
      // Subarray should work initially
      subarray[0] = 42;
      expect(subarray[0]).toBe(42);
      expect(subarray.length).toBe(10);
      
      // Dispose the original buffer
      device.disposeData(data);
      
      // Both view and subarray should be invalid
      expect(() => view[0]).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => subarray[0]).toThrow('View is no longer valid: buffer has been disposed');
    });

    it('should allow slice() on valid views (creates copy)', async () => {
      const data = device.createData(1024);
      const view = device.readDataView(data, float32) as Float32Array;
      
      // Fill with test data
      for (let i = 0; i < 10; i++) {
        view[i] = i;
      }
      
      // slice() creates a copy, not a view
      const sliced = view.slice(0, 10);
      
      // Dispose the original buffer
      device.disposeData(data);
      
      // Original view should be invalid
      expect(() => view[0]).toThrow('View is no longer valid: buffer has been disposed');
      
      // But sliced copy should still work (it's a separate ArrayBuffer)
      expect(sliced[0]).toBe(0);
      expect(sliced[9]).toBe(9);
      expect(sliced.length).toBe(10);
    });
  });

  describe('Buffer rewrite scenarios', () => {
    it('should invalidate views when buffer is rewritten', async () => {
      const data = device.createData(1024);
      const view = device.readDataView(data, float32);
      
      view[0] = 42;
      expect(view[0]).toBe(42);
      
      // Write new data to the buffer
      const newBuffer = new ArrayBuffer(1024);
      const newView = new Float32Array(newBuffer);
      newView[0] = 999;
      
      await device.writeData(data, newBuffer);
      
      // Old view should be invalid
      expect(() => view[0]).toThrow('View is no longer valid: buffer has been disposed');
      
      // Create new view to see the updated data
      const freshView = device.readDataView(data, float32);
      expect(freshView[0]).toBe(999);
    });
  });

  describe('Tensor operation integration', () => {
    it('should handle view invalidation with tensor reshaping', async () => {
      const tensor = await tt.tensor([1, 2, 3, 4, 5, 6], { device, dtype: float32 });
      const reshaped = await tensor.reshape([2, 3] as const);
      
      // Get views of both tensors
      const originalView = device.readDataView(tensor.data, float32);
      const reshapedView = device.readDataView(reshaped.data, float32);
      
      // Both should work and share data
      expect(originalView[0]).toBe(1);
      expect(reshapedView[0]).toBe(1);
      
      // Modify through one view
      originalView[0] = 999;
      expect(reshapedView[0]).toBe(999); // Should see the change
      
      // Dispose the shared data
      device.disposeData(tensor.data);
      
      // Both views should be invalid
      expect(() => originalView[0]).toThrow('View is no longer valid: buffer has been disposed');
      expect(() => reshapedView[0]).toThrow('View is no longer valid: buffer has been disposed');
    });
  });

  describe('Edge cases and error conditions', () => {
    it('should handle double disposal gracefully', async () => {
      const data = device.createData(1024);
      const view = device.readDataView(data, float32);
      
      view[0] = 42;
      
      // First disposal
      device.disposeData(data);
      expect(() => view[0]).toThrow('View is no longer valid: buffer has been disposed');
      
      // Second disposal should not crash (data's dispose method handles it)
      expect(() => (data as any).dispose()).not.toThrow();
      
      // View should still be invalid
      expect(() => view[0]).toThrow('View is no longer valid: buffer has been disposed');
    });

    it('should validate view even after garbage collection hints', async () => {
      const data = device.createData(1024);
      const view = device.readDataView(data, float32);
      
      view[0] = 42;
      
      // Try to trigger GC (if available)
      if (typeof global !== 'undefined' && global.gc) {
        global.gc();
      }
      
      // View should still work
      expect(view[0]).toBe(42);
      
      device.disposeData(data);
      
      // Should be invalid after disposal
      expect(() => view[0]).toThrow('View is no longer valid: buffer has been disposed');
    });

    it('should handle views of zero-sized buffers', async () => {
      const data = device.createData(0);
      const view = device.readDataView(data, float32);
      
      // Zero-length view should work
      expect(view.length).toBe(0);
      
      device.disposeData(data);
      
      // Should still validate even for zero-sized
      expect(() => view.length).toThrow('View is no longer valid: buffer has been disposed');
    });
  });

  describe('Performance characteristics', () => {
    it('should have minimal overhead for view validation', async () => {
      const data = device.createData(4 * 1024 * 1024); // 4MB
      const view = device.readDataView(data, float32) as Float32Array;
      
      // Measure access time with validation
      const iterations = 100000;
      const start = performance.now();
      
      for (let i = 0; i < iterations; i++) {
        view[i % view.length] = i;
      }
      
      const end = performance.now();
      const timePerAccess = (end - start) / iterations;
      
      // Should be very fast (< 1 microsecond per access)
      expect(timePerAccess).toBeLessThan(0.001);
      
      // Cleanup
      device.disposeData(data);
    });
  });
});

// Reset WASM module after this test file to ensure test isolation
afterAll(() => {
  resetWASMForTests();
});