#!/usr/bin/env bun
/**
 * Stress test to reproduce WASM memory corruption
 */

import { WASMDevice } from './src/device';

async function stressTest() {
  console.log('🔥 Starting WASM stress test...');
  
  const device = await WASMDevice.create();
  console.log('✅ Device created');
  
  try {
    console.log('\n1. Testing rapid buffer creation/disposal...');
    for (let i = 0; i < 100; i++) {
      const buffer = device.createData(1024 * 1024); // 1MB each
      if (i % 20 === 0) {
        console.log(`  Created ${i} buffers...`);
      }
      device.disposeData(buffer);
    }
    console.log('✅ Rapid buffer creation/disposal successful');
    
    console.log('\n2. Testing memory pressure scenario...');
    const buffers = [];
    try {
      for (let i = 0; i < 50; i++) {
        const buffer = device.createData(5 * 1024 * 1024); // 5MB each = 250MB total
        buffers.push(buffer);
        if (i % 10 === 0) {
          console.log(`  Allocated ${i * 5}MB so far...`);
        }
      }
      console.log('✅ Memory pressure allocation successful');
    } finally {
      // Cleanup
      for (const buffer of buffers) {
        device.disposeData(buffer);
      }
      console.log('✅ Memory pressure cleanup successful');
    }
    
    console.log('\n3. Testing write operations after pressure...');
    const testBuffer = device.createData(1024);
    const writeData = new ArrayBuffer(1024);
    await device.writeData(testBuffer, writeData);
    console.log('✅ Write after pressure successful');
    
    console.log('\n4. Testing read operations after pressure...');
    const readData = await device.readData(testBuffer);
    console.log(`✅ Read after pressure successful: ${readData.byteLength} bytes`);
    
    console.log('\n5. Testing memory stats after pressure...');
    const stats = device.getMemoryStats();
    console.log(`✅ Memory stats: ${stats.totalAllocated} bytes, ${stats.activeBuffers} buffers`);
    
    device.disposeData(testBuffer);
    
    console.log('\n🎉 Stress test completed successfully!');
    
  } catch (error) {
    console.error('❌ Stress test failed:', error);
    console.error('Stack:', error.stack);
    
    // Get current memory state
    try {
      const stats = device.getMemoryStats();
      console.log('Final memory stats:', stats);
    } catch (e) {
      console.error('Cannot get final memory stats:', e);
    }
    
    throw error;
  }
}

async function main() {
  try {
    await stressTest();
  } catch (error) {
    console.error('Test failed:', error);
    process.exit(1);
  }
}

if (import.meta.main) {
  main();
}