import { describe, it, expect, beforeAll, afterEach, afterAll } from 'bun:test';
import { WASMDevice } from './device';
import { resetWASMForTests } from './test-utils';

describe('Memory Efficiency', () => {
  let device: WASMDevice;

  beforeAll(async () => {
    device = await WASMDevice.create();
  });

  it('should not allocate unnecessary JavaScript buffers in createData', async () => {
    // Track ArrayBuffer allocations
    let allocations = 0;
    const originalArrayBuffer = globalThis.ArrayBuffer;
    const ArrayBufferProxy = new Proxy(originalArrayBuffer, {
      construct(target, args) {
        allocations++;
        return Reflect.construct(target, args);
      }
    });
    
    // Replace global ArrayBuffer temporarily
    (globalThis as any).ArrayBuffer = ArrayBufferProxy;
    
    try {
      // This should not allocate any JavaScript ArrayBuffer
      const data = device.createData(1024);
      expect(allocations).toBe(0);
      
      // Cleanup
      device.disposeData(data);
    } finally {
      // Restore original ArrayBuffer
      (globalThis as any).ArrayBuffer = originalArrayBuffer;
    }
  });

  it('should read data without unnecessary copies when aligned', async () => {
    const data = device.createData(1024);
    
    // Write some test data
    const testData = new ArrayBuffer(1024);
    const view = new Uint8Array(testData);
    for (let i = 0; i < 1024; i++) {
      view[i] = i % 256;
    }
    await device.writeData(data, testData);
    
    // Track slice calls
    let sliceCalls = 0;
    const originalSlice = ArrayBuffer.prototype.slice;
    ArrayBuffer.prototype.slice = function(...args: any[]) {
      sliceCalls++;
      return originalSlice.apply(this, args as any);
    };
    
    try {
      // Read the data - should not call slice if aligned
      const result = await device.readData(data);
      
      // Note: copy_buffer_to_js creates a new buffer, so we expect 0 slices
      // The optimization avoids an additional unnecessary slice
      expect(sliceCalls).toBe(0);
      
      // Verify data integrity
      const resultView = new Uint8Array(result);
      expect(resultView.length).toBe(1024);
      for (let i = 0; i < 1024; i++) {
        expect(resultView[i]).toBe(i % 256);
      }
    } finally {
      ArrayBuffer.prototype.slice = originalSlice;
      device.disposeData(data);
    }
  });

  it('should not make unnecessary copies in createDataWithBuffer', async () => {
    const originalBuffer = new ArrayBuffer(2048);
    const originalView = new Uint8Array(originalBuffer);
    for (let i = 0; i < 2048; i++) {
      originalView[i] = i % 256;
    }
    
    // Track slice calls
    let sliceCalls = 0;
    const originalSlice = ArrayBuffer.prototype.slice;
    ArrayBuffer.prototype.slice = function(...args: any[]) {
      sliceCalls++;
      return originalSlice.apply(this, args as any);
    };
    
    try {
      // Create data with buffer - should not slice the input
      const data = device.createDataWithBuffer(originalBuffer);
      expect(sliceCalls).toBe(0);
      
      // Verify data was copied correctly
      const result = await device.readData(data);
      const resultView = new Uint8Array(result);
      expect(resultView.length).toBe(2048);
      for (let i = 0; i < 2048; i++) {
        expect(resultView[i]).toBe(i % 256);
      }
      
      device.disposeData(data);
    } finally {
      ArrayBuffer.prototype.slice = originalSlice;
    }
  });

  it('should handle large allocations efficiently', async () => {
    // Allocate 10MB without creating intermediate JS buffers
    const size = 10 * 1024 * 1024;
    
    let allocations = 0;
    const originalArrayBuffer = globalThis.ArrayBuffer;
    const ArrayBufferProxy = new Proxy(originalArrayBuffer, {
      construct(target, args) {
        // Only count large allocations
        if (args[0] >= size) {
          allocations++;
        }
        return Reflect.construct(target, args);
      }
    });
    
    (globalThis as any).ArrayBuffer = ArrayBufferProxy;
    
    try {
      const data = device.createData(size);
      expect(allocations).toBe(0); // No JS allocation needed
      
      device.disposeData(data);
    } finally {
      (globalThis as any).ArrayBuffer = originalArrayBuffer;
    }
  });

  it('should provide memory stats without allocations', () => {
    // Getting stats should not allocate buffers
    let allocations = 0;
    const originalArrayBuffer = globalThis.ArrayBuffer;
    const ArrayBufferProxy = new Proxy(originalArrayBuffer, {
      construct(target, args) {
        allocations++;
        return Reflect.construct(target, args);
      }
    });
    
    (globalThis as any).ArrayBuffer = ArrayBufferProxy;
    
    try {
      const stats = device.getMemoryStats();
      expect(allocations).toBe(0);
      expect(stats.totalAllocated).toBeGreaterThanOrEqual(0);
    } finally {
      (globalThis as any).ArrayBuffer = originalArrayBuffer;
    }
  });
});

// Reset WASM module after this test file to ensure test isolation
afterAll(() => {
  resetWASMForTests();
});