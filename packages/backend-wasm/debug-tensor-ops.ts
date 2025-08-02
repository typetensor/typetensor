#!/usr/bin/env bun
/**
 * Debug tensor operations to reproduce benchmark failures
 */

import { tensor, zeros, ones, float32 } from '@typetensor/core';
import { WASMDevice } from './src/device';

function generateRandomData(shape: number[]): number[] {
  const size = shape.reduce((acc, dim) => acc * dim, 1);
  return Array.from({ length: size }, () => Math.random());
}

async function debugTensorOperations() {
  console.log('🧮 Testing tensor operations that fail in benchmarks...');
  
  const device = await WASMDevice.create();
  console.log('✅ Device created');
  
  try {
    console.log('\n1. Testing basic tensor creation...');
    const data = generateRandomData([1000]);
    const testTensor = await tensor(data, { device, dtype: float32 });
    console.log('✅ Basic tensor creation successful');
    
    console.log('\n2. Testing unary operations...');
    const negResult = await (testTensor as any).neg();
    console.log('✅ Neg operation successful');
    
    const absResult = await (testTensor as any).abs();
    console.log('✅ Abs operation successful');
    
    console.log('\n3. Testing binary operations (this is where failures start)...');
    const data2 = generateRandomData([1000]);
    const testTensor2 = await tensor(data2, { device, dtype: float32 });
    console.log('✅ Second tensor created');
    
    try {
      const addResult = await (testTensor as any).add(testTensor2);
      console.log('✅ Add operation successful');
    } catch (error) {
      console.error('❌ Add operation failed:', error);
      throw error;
    }
    
    console.log('\n4. Testing memory transfer operations...');
    const transferData = generateRandomData([256]); // 1KB
    const transferTensor = await tensor(transferData, { device, dtype: float32 });
    
    try {
      const readResult = await device.readData(transferTensor.data);
      console.log(`✅ Direct read successful: ${readResult.byteLength} bytes`);
    } catch (error) {
      console.error('❌ Direct read failed:', error);
      throw error;
    }
    
    try {
      const writeBuffer = new ArrayBuffer(256 * 4); // 1KB
      const writeData = device.createData(writeBuffer.byteLength);
      await device.writeData(writeData, writeBuffer);
      console.log('✅ Direct write successful');
      device.disposeData(writeData);
    } catch (error) {
      console.error('❌ Direct write failed:', error);
      throw error;
    }
    
    console.log('\n5. Testing memory stats...');
    const stats = device.getMemoryStats();
    console.log(`✅ Memory stats: ${stats.totalAllocated} bytes, ${stats.activeBuffers} buffers`);
    
    console.log('\n🎉 All tensor operations successful!');
    
  } catch (error) {
    console.error('❌ Tensor operations failed:', error);
    console.error('Stack:', error.stack);
    
    // Additional debugging
    try {
      console.log('\n🔍 Getting debug info...');
      const stats = device.getMemoryStats();
      console.log('Current memory stats:', stats);
    } catch (e) {
      console.error('Cannot get debug info:', e);
    }
    
    throw error;
  }
}

async function main() {
  try {
    await debugTensorOperations();
  } catch (error) {
    console.error('Test failed:', error);
    process.exit(1);
  }
}

if (import.meta.main) {
  main();
}