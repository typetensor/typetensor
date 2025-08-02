/**
 * Tests for Issue #8: Reference Counting Coordination
 * 
 * Verifies that buffer cloning uses proper reference counting to prevent
 * double-free errors and memory corruption.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { WASMDevice } from './device';
import { WASMDeviceData } from './data';
import { resetWASMForTests } from './test-utils';

describe('Reference Counting Coordination', () => {
  let device: WASMDevice;

  beforeAll(async () => {
    device = await WASMDevice.create();
  });

  afterAll(() => {
    resetWASMForTests();
  });

  beforeEach(() => {
    // Ensure clean state between tests
    device.performIntensiveCleanup();
  });

  it('should coordinate reference counting between original and clones', async () => {
    // Create original buffer
    const original = device.createData(1024);
    const originalId = (original as WASMDeviceData).id;
    
    // Reference count should be 1
    expect((original as WASMDeviceData).getRefCount()).toBe(1);
    
    // Create clone
    const clone1 = (original as WASMDeviceData).clone();
    
    // Both should have same ID (sharing same WASM buffer)
    expect(clone1.id).toBe(originalId);
    
    // Reference count should be 2 (original + clone)
    expect((original as WASMDeviceData).getRefCount()).toBe(2);
    expect(clone1.getRefCount()).toBe(2);
    
    // Create second clone
    const clone2 = clone1.clone();
    
    // All should have same ID
    expect(clone2.id).toBe(originalId);
    
    // Reference count should be 3 (original + 2 clones)
    expect((original as WASMDeviceData).getRefCount()).toBe(3);
    expect(clone1.getRefCount()).toBe(3);
    expect(clone2.getRefCount()).toBe(3);
    
    // Dispose one clone
    clone1.dispose();
    
    // Reference count should decrease to 2
    expect((original as WASMDeviceData).getRefCount()).toBe(2);
    expect(clone2.getRefCount()).toBe(2);
    expect(clone1.getRefCount()).toBe(0); // Disposed clone shows 0
    
    // Dispose original
    device.disposeData(original);
    
    // Clone2 should still be valid with refCount 1
    expect(clone2.getRefCount()).toBe(1);
    expect((original as WASMDeviceData).getRefCount()).toBe(0); // Disposed original shows 0
    
    // Dispose last clone
    clone2.dispose();
    
    // All should be 0 now
    expect(clone2.getRefCount()).toBe(0);
  });

  it('should prevent double-free when disposing clones in different orders', async () => {
    const original = device.createData(512);
    const clone1 = (original as WASMDeviceData).clone();
    const clone2 = (original as WASMDeviceData).clone();
    
    // All should share same buffer
    expect(clone1.id).toBe((original as WASMDeviceData).id);
    expect(clone2.id).toBe((original as WASMDeviceData).id);
    
    // Reference count should be 3
    expect((original as WASMDeviceData).getRefCount()).toBe(3);
    
    // Dispose in reverse order: clone2, original, clone1
    clone2.dispose();
    expect((original as WASMDeviceData).getRefCount()).toBe(2);
    
    device.disposeData(original);
    expect(clone1.getRefCount()).toBe(1);
    
    // This should not cause double-free
    clone1.dispose();
    expect(clone1.getRefCount()).toBe(0);
  });

  it('should handle cloning of already cloned buffers', async () => {
    const original = device.createData(256);
    const clone1 = (original as WASMDeviceData).clone();
    const clone2 = clone1.clone(); // Clone of clone
    const clone3 = clone2.clone(); // Clone of clone of clone
    
    // All should share same ID
    const originalId = (original as WASMDeviceData).id;
    expect(clone1.id).toBe(originalId);
    expect(clone2.id).toBe(originalId);
    expect(clone3.id).toBe(originalId);
    
    // Reference count should be 4
    expect((original as WASMDeviceData).getRefCount()).toBe(4);
    expect(clone1.getRefCount()).toBe(4);
    expect(clone2.getRefCount()).toBe(4);
    expect(clone3.getRefCount()).toBe(4);
    
    // Dispose middle clone
    clone2.dispose();
    
    // Remaining should have refCount 3
    expect((original as WASMDeviceData).getRefCount()).toBe(3);
    expect(clone1.getRefCount()).toBe(3);
    expect(clone3.getRefCount()).toBe(3);
    
    // Clean up remaining
    device.disposeData(original);
    clone1.dispose();
    clone3.dispose();
  });

  it('should handle data read/write operations correctly with clones (copy-on-write)', async () => {
    // Create buffer with initial data
    const testData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const buffer = testData.buffer.slice();
    const original = device.createDataWithBuffer(buffer);
    
    // Clone the buffer
    const clone = (original as WASMDeviceData).clone();
    
    // Both should be able to read the same initial data
    const originalReadData = await device.readData(original);
    const cloneReadData = await device.readData(clone);
    
    expect(new Float32Array(originalReadData)).toEqual(testData);
    expect(new Float32Array(cloneReadData)).toEqual(testData);
    
    // Write new data through original - this breaks sharing (copy-on-write)
    const newData = new Float32Array([5.0, 6.0, 7.0, 8.0]);
    await device.writeData(original, newData.buffer);
    
    // Original should have new data
    const originalUpdatedData = await device.readData(original);
    expect(new Float32Array(originalUpdatedData)).toEqual(newData);
    
    // Clone should still have original data (copy-on-write semantics)
    const cloneStillOriginalData = await device.readData(clone);
    expect(new Float32Array(cloneStillOriginalData)).toEqual(testData);
    
    // Clean up
    device.disposeData(original);
    clone.dispose();
  });

  it('should handle memory pressure with multiple clones', async () => {
    const buffers: WASMDeviceData[] = [];
    const clones: WASMDeviceData[] = [];
    
    // Create multiple buffers and their clones
    for (let i = 0; i < 10; i++) {
      const buffer = device.createData(1024 * 1024); // 1MB each
      const clone = (buffer as WASMDeviceData).clone();
      
      buffers.push(buffer as WASMDeviceData);
      clones.push(clone);
      
      // Each pair should have refCount 2
      expect(buffer.getRefCount()).toBe(2);
      expect(clone.getRefCount()).toBe(2);
    }
    
    // Dispose all originals first
    buffers.forEach(buffer => device.disposeData(buffer));
    
    // Clones should still be valid with refCount 1
    clones.forEach(clone => {
      expect(clone.getRefCount()).toBe(1);
      expect(clone.isDisposed()).toBe(false);
    });
    
    // Dispose all clones
    clones.forEach(clone => clone.dispose());
    
    // All should be disposed now
    clones.forEach(clone => {
      expect(clone.getRefCount()).toBe(0);
      expect(clone.isDisposed()).toBe(true);
    });
  });

  it('should handle clone attempts on disposed buffers', async () => {
    const original = device.createData(512);
    
    // Dispose original
    device.disposeData(original);
    
    // Attempting to clone disposed buffer should throw
    expect(() => {
      (original as WASMDeviceData).clone();
    }).toThrow('Cannot clone disposed WASMDeviceData');
  });

  it('should handle rapid clone and dispose cycles', async () => {
    const original = device.createData(1024);
    
    // Rapid clone/dispose cycles
    for (let i = 0; i < 100; i++) {
      const clone = (original as WASMDeviceData).clone();
      expect(clone.getRefCount()).toBe(2);
      clone.dispose();
      expect((original as WASMDeviceData).getRefCount()).toBe(1);
    }
    
    // Original should still be valid
    expect((original as WASMDeviceData).getRefCount()).toBe(1);
    expect((original as WASMDeviceData).isDisposed()).toBe(false);
    
    device.disposeData(original);
  });
});