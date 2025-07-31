import { describe, it, expect, afterAll } from 'bun:test';
import { WASMDevice } from './device';
import type { DeviceData } from '@typetensor/core';
import { resetWASMForTests } from './test-utils';

describe('Memory Pressure - Simple Tests', () => {
  it('should respect configured memory limit', async () => {
    // Create device with small memory limit
    const device = await WASMDevice.create({
      memoryConfig: {
        maxMemory: 50 * 1024 * 1024, // 50MB limit
        autoCompact: true,
        compactThreshold: 0.8
      }
    });

    const allocations: DeviceData[] = [];
    let totalAllocated = 0;
    let allocationFailed = false;

    try {
      // Try to allocate 100MB (should fail)
      for (let i = 0; i < 10; i++) {
        allocations.push(device.createData(10 * 1024 * 1024)); // 10MB each
        totalAllocated += 10 * 1024 * 1024;
      }
    } catch (error: any) {
      allocationFailed = true;
      expect(error.message).toMatch(/Memory limit exceeded|WASM memory limit exceeded/);
    }

    expect(allocationFailed).toBe(true);
    expect(totalAllocated).toBeLessThanOrEqual(50 * 1024 * 1024);

    // Clean up
    allocations.forEach(d => device.disposeData(d));
  });

  it('should handle memory config', async () => {
    const config = {
      maxMemory: 256 * 1024 * 1024,
      autoCompact: true,
      compactThreshold: 0.75
    };

    const device = await WASMDevice.create({
      memoryConfig: config
    });

    const deviceConfig = device.getMemoryConfig();
    expect(deviceConfig.maxMemory).toBe(config.maxMemory);
    expect(deviceConfig.autoCompact).toBe(config.autoCompact);
    expect(deviceConfig.compactThreshold).toBe(config.compactThreshold);
  });

  it('should enforce memory limits with configured device', async () => {
    const device = await WASMDevice.create({
      memoryConfig: {
        maxMemory: 512 * 1024 * 1024, // 512MB limit
        autoCompact: true
      }
    });
    const allocations: DeviceData[] = [];

    try {
      // Try to allocate 600MB (should fail)
      for (let i = 0; i < 60; i++) {
        allocations.push(device.createData(10 * 1024 * 1024)); // 10MB each
      }
    } catch (error: any) {
      // Expected to fail
      expect(error.message).toMatch(/Memory limit exceeded|WASM memory limit exceeded/);
    }

    // Should have allocated less than 512MB
    expect(allocations.length).toBeLessThan(52);

    // Clean up
    allocations.forEach(d => device.disposeData(d));
  });

  it('should compact memory when requested', async () => {
    const device = await WASMDevice.create({
      memoryConfig: {
        maxMemory: 100 * 1024 * 1024, // 100MB
        autoCompact: false // Manual compaction
      }
    });

    // Allocate and free to create pooled buffers
    const temp: DeviceData[] = [];
    for (let i = 0; i < 5; i++) {
      temp.push(device.createData(10 * 1024 * 1024)); // 10MB each
    }
    temp.forEach(d => device.disposeData(d));

    const statsBeforeCompact = device.getMemoryStats();
    expect(statsBeforeCompact.totalAllocated).toBeGreaterThan(40 * 1024 * 1024);

    // Manually compact
    device.performIntensiveCleanup();

    const statsAfterCompact = device.getMemoryStats();
    expect(statsAfterCompact.totalAllocated).toBeLessThan(statsBeforeCompact.totalAllocated);
  });

  // Reset WASM module after this test file to ensure test isolation
  afterAll(() => {
    resetWASMForTests();
  });
});