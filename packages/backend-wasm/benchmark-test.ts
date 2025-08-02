#!/usr/bin/env bun
/**
 * Benchmark stress test to reproduce potential crashes
 */

import { WASMDevice } from './src/device';

async function createStressTest() {
  console.log('Creating WASM device...');
  const device = await WASMDevice.create();
  
  console.log('Starting benchmark stress test...');
  
  try {
    // Create many buffers rapidly
    const buffers = [];
    for (let i = 0; i < 1000; i++) {
      const data = device.createData(1024); // 1KB each
      buffers.push(data);
      
      if (i % 100 === 0) {
        console.log(`Created ${i} buffers...`);
      }
    }
    
    // Perform operations on the buffers
    console.log('Performing operations...');
    for (let i = 0; i < buffers.length - 1; i++) {
      try {
        // Clone some buffers
        if (i % 10 === 0) {
          const cloned = (buffers[i] as any).clone();
          cloned.dispose();
        }
        
        // Read some data
        if (i % 20 === 0) {
          await device.readData(buffers[i]);
        }
      } catch (error) {
        console.error(`Error on buffer ${i}:`, error);
        throw error;
      }
    }
    
    // Cleanup all buffers
    console.log('Cleaning up buffers...');
    for (let i = 0; i < buffers.length; i++) {
      try {
        device.disposeData(buffers[i]);
      } catch (error) {
        console.error(`Error disposing buffer ${i}:`, error);
      }
    }
    
    console.log('Stress test completed successfully!');
    
  } catch (error) {
    console.error('Benchmark stress test crashed:', error);
    throw error;
  }
}

async function main() {
  try {
    await createStressTest();
    console.log('All tests passed!');
  } catch (error) {
    console.error('Test failed:', error);
    process.exit(1);
  }
}

if (import.meta.main) {
  main();
}