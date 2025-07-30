/**
 * Basic tests for WebGPU backend
 */

import { describe, it, expect } from 'bun:test';
import { webgpu, isWebGPUAvailable } from './index';

describe('WebGPU Backend', () => {
  it('should check if WebGPU is available', () => {
    const available = isWebGPUAvailable();
    expect(typeof available).toBe('boolean');
    
    if (!available) {
      console.log('WebGPU is not available in this environment');
    }
  });

  it('should create a WebGPU device if available', async () => {
    if (!isWebGPUAvailable()) {
      console.log('Skipping WebGPU device creation test - WebGPU not available');
      return;
    }

    try {
      const device = await webgpu();
      expect(device).toBeDefined();
      expect(device.type).toBe('webgpu');
      expect(device.id).toContain('webgpu');
      
      // Test basic device functionality
      const data = device.createData(16); // 16 bytes
      expect(data).toBeDefined();
      expect(data.byteLength).toBe(16);
      
      // Test data read/write
      const testData = new Float32Array([1, 2, 3, 4]);
      await device.writeData(data, testData.buffer);
      
      const readData = await device.readData(data);
      const readArray = new Float32Array(readData);
      expect(readArray).toEqual(testData);
      
      // Clean up
      device.disposeData(data);
    } catch (error) {
      console.error('Failed to create WebGPU device:', error);
      throw error;
    }
  });

  it('should support basic tensor operations if WebGPU is available', async () => {
    if (!isWebGPUAvailable()) {
      console.log('Skipping tensor operations test - WebGPU not available');
      return;
    }

    try {
      const { tensor } = await import('@typetensor/core');
      const device = await webgpu();
      
      // Create a simple tensor
      const a = await tensor([1, 2, 3, 4], { device });
      expect(a.shape).toEqual([4]);
      
      // Test unary operation (neg)
      const b = await a.neg();
      const bData = await b.toArray();
      expect(bData).toEqual([-1, -2, -3, -4]);
      
      // Test binary operation (add)
      const c = await a.add(b);
      const cData = await c.toArray();
      expect(cData).toEqual([0, 0, 0, 0]);
      
      // Clean up
      a.dispose();
      b.dispose();
      c.dispose();
    } catch (error) {
      console.error('Failed tensor operations:', error);
      throw error;
    }
  });
});