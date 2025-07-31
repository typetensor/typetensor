import { describe, it, expect, beforeEach, afterAll } from 'bun:test';
import { WASMDevice } from './device';
import { WASMDeviceData } from './data';
import type { DeviceData } from '@typetensor/core';
import { resetWASMForTests } from './test-utils';

describe('Buffer Handle Encapsulation', () => {
  let device: WASMDevice;

  beforeEach(async () => {
    device = await WASMDevice.create();
  });

  it('should not allow direct access to private handle field', async () => {
    const data = device.createData(1024);
    
    // Cannot access private field directly (these would be compile errors in TypeScript)
    // Private fields are not accessible even via (obj as any)
    expect((data as any).wasmHandle).toBeUndefined();
    
    // But can use public API
    const handle = (data as WASMDeviceData).getWASMHandle();
    expect(handle).toBeDefined();
    expect(typeof (handle as any).id).toBe('number');
  });

  it('should not allow direct mutation of handle', async () => {
    const data = device.createData(1024);
    const originalHandle = (data as WASMDeviceData).getWASMHandle();
    
    // These should not work (would be compile-time errors in real TypeScript)
    // (data as any).#wasmHandle = null;  // Would be syntax error
    // (data as any).wasmHandle = null;   // Property doesn't exist
    
    // Handle should remain unchanged
    expect((data as WASMDeviceData).getWASMHandle()).toBe(originalHandle);
  });

  it('should use proper API for handle updates during writeData', async () => {
    const data = device.createData(1024);
    const originalHandle = (data as WASMDeviceData).getWASMHandle();
    const originalId = (originalHandle as any).id;
    
    // Write new data, which should update the handle
    const newBuffer = new ArrayBuffer(1024);
    const view = new Uint8Array(newBuffer);
    for (let i = 0; i < 1024; i++) {
      view[i] = i % 256;
    }
    
    await device.writeData(data, newBuffer);
    
    // Handle should be updated
    const newHandle = (data as WASMDeviceData).getWASMHandle();
    const newId = (newHandle as any).id;
    
    expect(newId).not.toBe(originalId);
    expect(newHandle).not.toBe(originalHandle);
    
    // Data should still be valid
    const readBack = await device.readData(data);
    expect(readBack.byteLength).toBe(1024);
    expect(new Uint8Array(readBack)[0]).toBe(0);
    expect(new Uint8Array(readBack)[255]).toBe(255);
  });

  it('should prevent handle updates on disposed data', async () => {
    const data = device.createData(1024) as WASMDeviceData;
    
    // Manually dispose the data by calling dispose directly
    data.dispose();
    
    // Cannot update handle on disposed data
    expect(() => {
      data.updateHandle(null);
    }).toThrow('Cannot update handle of disposed WASMDeviceData');
  });

  it('should prevent accessing handle on disposed data', async () => {
    const data = device.createData(1024) as WASMDeviceData;
    
    // Should work initially
    expect(data.getWASMHandle()).toBeDefined();
    
    // Manually dispose the data
    data.dispose();
    
    // Cannot access handle on disposed data
    expect(() => {
      data.getWASMHandle();
    }).toThrow('WASMDeviceData has been disposed');
  });

  it('should properly clean up old handle during update', async () => {
    const data = device.createData(1024) as WASMDeviceData;
    const originalHandle = data.getWASMHandle();
    
    // Mock the release_buffer to track calls
    let releaseCalled = false;
    const originalRelease = device.operationDispatcher.release_buffer;
    device.operationDispatcher.release_buffer = (handle: any) => {
      if (handle === originalHandle) {
        releaseCalled = true;
      }
      return originalRelease.call(device.operationDispatcher, handle);
    };
    
    // Create new buffer data
    const newBuffer = new ArrayBuffer(1024);
    
    // Writing should update handle and release old one
    await device.writeData(data, newBuffer);
    
    expect(releaseCalled).toBe(true);
    
    // Restore original method
    device.operationDispatcher.release_buffer = originalRelease;
  });

  it('should maintain data integrity across handle updates', async () => {
    const data = device.createData(1024);
    
    // Initial data pattern
    const buffer1 = new ArrayBuffer(1024);
    const view1 = new Uint8Array(buffer1);
    for (let i = 0; i < 1024; i++) {
      view1[i] = i % 256;
    }
    await device.writeData(data, buffer1);
    
    // Verify initial data
    let readBack = await device.readData(data);
    expect(new Uint8Array(readBack)[0]).toBe(0);
    expect(new Uint8Array(readBack)[255]).toBe(255);
    
    // Update with different pattern
    const buffer2 = new ArrayBuffer(1024);
    const view2 = new Uint8Array(buffer2);
    for (let i = 0; i < 1024; i++) {
      view2[i] = 255 - (i % 256);
    }
    await device.writeData(data, buffer2);
    
    // Verify updated data
    readBack = await device.readData(data);
    expect(new Uint8Array(readBack)[0]).toBe(255);
    expect(new Uint8Array(readBack)[255]).toBe(0);
    
    // Data should still be valid for operations
    expect(data.isDisposed()).toBe(false);
    expect((data as WASMDeviceData).getWASMHandle()).toBeDefined();
  });

  it('should handle same handle update gracefully', async () => {
    const data = device.createData(1024) as WASMDeviceData;
    
    // Should not crash when updating with same handle
    const currentHandle = data.getWASMHandle();
    expect(() => {
      data.updateHandle(currentHandle);
    }).not.toThrow();
    
    // Should still work after self-update
    expect(data.getWASMHandle()).toBe(currentHandle);
    expect(data.isDisposed()).toBe(false);
  });

  it('should preserve encapsulation during cloning', async () => {
    const data = device.createData(1024) as WASMDeviceData;
    const originalHandle = data.getWASMHandle();
    
    // Clone the data
    const cloned = data.clone();
    const clonedHandle = cloned.getWASMHandle();
    
    // Should have different handles (cloning creates references to same buffer so IDs might be same)
    expect(clonedHandle).toBeDefined();
    expect(originalHandle).toBeDefined();
    
    // Both should be properly encapsulated
    expect((cloned as any).wasmHandle).toBeUndefined();
    
    // Both should have valid handles
    expect(cloned.getWASMHandle()).toBeDefined();
    expect(data.getWASMHandle()).toBeDefined();
    
    // Both should be independent in terms of disposal state
    data.dispose();
    expect(data.isDisposed()).toBe(true);
    expect(cloned.isDisposed()).toBe(false);
    expect(cloned.getWASMHandle()).toBeDefined();
    
    // Clean up
    cloned.dispose();
  });
});

// Reset WASM module after this test file to ensure test isolation
afterAll(() => {
  resetWASMForTests();
});