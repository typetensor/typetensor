#!/usr/bin/env bun
/**
 * Debug script to investigate WASM Unreachable code errors
 */

import { WASMDevice } from './src/device';

async function debugWasmIssue() {
  console.log('🔍 Debugging WASM Unreachable code error...');
  
  try {
    console.log('1. Creating WASM device...');
    const device = await WASMDevice.create();
    console.log('✅ Device created successfully');
    
    console.log('\n2. Testing basic buffer creation...');
    const smallBuffer = device.createData(1024); // 1KB
    console.log('✅ Small buffer created successfully');
    
    console.log('\n3. Testing buffer read operation...');
    const readResult = await device.readData(smallBuffer);
    console.log(`✅ Buffer read successful: ${readResult.byteLength} bytes`);
    
    console.log('\n4. Testing buffer write operation...');
    const writeBuffer = new ArrayBuffer(1024);
    await device.writeData(smallBuffer, writeBuffer);
    console.log('✅ Buffer write successful');
    
    console.log('\n5. Testing memory stats...');
    const memStats = device.getMemoryStats();
    console.log(`✅ Memory stats: ${memStats.totalAllocated} bytes, ${memStats.activeBuffers} buffers`);
    
    console.log('\n6. Testing larger buffer creation...');
    const largeBuffer = device.createData(10 * 1024 * 1024); // 10MB
    console.log('✅ Large buffer created successfully');
    
    console.log('\n7. Testing large buffer operations...');
    const largeWriteBuffer = new ArrayBuffer(10 * 1024 * 1024);
    await device.writeData(largeBuffer, largeWriteBuffer);
    console.log('✅ Large buffer write successful');
    
    const largeReadResult = await device.readData(largeBuffer);
    console.log(`✅ Large buffer read successful: ${largeReadResult.byteLength} bytes`);
    
    console.log('\n8. Cleanup...');
    device.disposeData(smallBuffer);
    device.disposeData(largeBuffer);
    console.log('✅ Cleanup successful');
    
    console.log('\n🎉 All tests passed! No WASM errors detected.');
    
  } catch (error) {
    console.error('❌ Error detected:', error);
    console.error('Stack:', error.stack);
    
    // Try to get more info about the WASM state
    try {
      console.log('\n🔍 Attempting to get more diagnostics...');
      const device = await WASMDevice.create();
      const stats = device.getMemoryStats();
      console.log('Memory stats after error:', stats);
    } catch (diagError) {
      console.error('❌ Cannot get diagnostics:', diagError);
    }
    
    throw error;
  }
}

async function main() {
  try {
    await debugWasmIssue();
    console.log('\n✅ Debug completed successfully');
  } catch (error) {
    console.error('\n❌ Debug failed:', error);
    process.exit(1);
  }
}

if (import.meta.main) {
  main();
}