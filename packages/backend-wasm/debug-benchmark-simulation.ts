#!/usr/bin/env bun
/**
 * Simulate the exact benchmark scenario that causes failures
 */

import { tensor, zeros, ones, float32 } from '@typetensor/core';
import { WASMDevice } from './src/device';

function generateRandomData(shape: number[]): number[] {
  const size = shape.reduce((acc, dim) => acc * dim, 1);
  return Array.from({ length: size }, () => Math.random());
}

async function simulateBenchmarkScenario() {
  console.log('🎯 Simulating exact benchmark scenario...');
  
  const device = await WASMDevice.create();
  console.log('✅ Device created');
  
  let operationCount = 0;
  
  try {
    // Simulate the exact benchmark sequence
    console.log('\n📊 Phase 1: Tensor creation benchmarks (similar to what works)...');
    
    const creationSizes = [
      { name: 'tiny', elements: 10 },
      { name: 'small', elements: 100 },
      { name: 'medium', elements: 1000 },
      { name: 'large', elements: 10000 }
    ];
    
    for (const size of creationSizes) {
      for (let i = 0; i < 20; i++) { // Simulate multiple runs
        const data = generateRandomData([size.elements]);
        const testTensor = await tensor(data, { device, dtype: float32 });
        operationCount++;
        
        if (operationCount % 50 === 0) {
          console.log(`  Completed ${operationCount} operations...`);
        }
      }
    }
    console.log(`✅ Phase 1 completed: ${operationCount} tensor creations`);
    
    console.log('\n⚡ Phase 2: Unary operations (these work in benchmarks)...');
    
    const unaryTestSizes = [
      { name: 'small', shape: [1000] },
      { name: 'medium', shape: [100, 100] }
    ];
    
    for (const size of unaryTestSizes) {
      const data = generateRandomData(size.shape);
      const testTensor = await tensor(data, { device, dtype: float32 });
      
      const unaryOps = ['neg', 'abs', 'sin', 'cos', 'exp', 'log', 'sqrt', 'square'];
      
      for (const op of unaryOps) {
        for (let i = 0; i < 10; i++) { // Multiple runs
          try {
            await (testTensor as any)[op]();
            operationCount++;
            
            if (operationCount % 50 === 0) {
              console.log(`  Completed ${operationCount} operations...`);
            }
          } catch (error) {
            console.error(`❌ ${op} operation failed after ${operationCount} operations:`, error);
            throw error;
          }
        }
      }
    }
    console.log(`✅ Phase 2 completed: ${operationCount} total operations`);
    
    console.log('\n🔥 Phase 3: Binary operations (these fail in benchmarks)...');
    
    const binaryTestSizes = [
      { name: 'small', shape: [1000] },
      { name: 'medium', shape: [100, 100] }
    ];
    
    for (const size of binaryTestSizes) {
      const data1 = generateRandomData(size.shape);
      const data2 = generateRandomData(size.shape);
      const tensor1 = await tensor(data1, { device, dtype: float32 });
      const tensor2 = await tensor(data2, { device, dtype: float32 });
      
      const binaryOps = [
        { name: 'add', fn: (a: any, b: any) => a.add(b) },
        { name: 'sub', fn: (a: any, b: any) => a.sub(b) },
        { name: 'mul', fn: (a: any, b: any) => a.mul(b) },
        { name: 'div', fn: (a: any, b: any) => a.div(b) }
      ];
      
      for (const op of binaryOps) {
        for (let i = 0; i < 10; i++) { // Multiple runs
          try {
            await op.fn(tensor1, tensor2);
            operationCount++;
            
            if (operationCount % 50 === 0) {
              console.log(`  Completed ${operationCount} operations...`);
            }
          } catch (error) {
            console.error(`❌ ${op.name} operation failed after ${operationCount} operations:`, error);
            console.error('Error details:', error.message);
            
            // Try to get memory stats
            try {
              const stats = device.getMemoryStats();
              console.log('Memory stats at failure:', stats);
            } catch (statsError) {
              console.error('Cannot get memory stats:', statsError);
            }
            
            throw error;
          }
        }
      }
    }
    console.log(`✅ Phase 3 completed: ${operationCount} total operations`);
    
    console.log('\n💾 Phase 4: Memory transfer operations (these fail in benchmarks)...');
    
    const transferSizes = [
      { name: '1KB', elements: 256 },
      { name: '1MB', elements: 262144 }
    ];
    
    for (const size of transferSizes) {
      for (let i = 0; i < 10; i++) {
        try {
          // Read test
          const data = generateRandomData([size.elements]);
          const testTensor = await tensor(data, { device, dtype: float32 });
          const readResult = await device.readData(testTensor.data);
          operationCount++;
          
          // Write test
          const writeBuffer = new ArrayBuffer(size.elements * 4);
          const writeData = device.createData(writeBuffer.byteLength);
          await device.writeData(writeData, writeBuffer);
          device.disposeData(writeData);
          operationCount++;
          
          if (operationCount % 50 === 0) {
            console.log(`  Completed ${operationCount} operations...`);
          }
        } catch (error) {
          console.error(`❌ Memory transfer failed after ${operationCount} operations:`, error);
          throw error;
        }
      }
    }
    console.log(`✅ Phase 4 completed: ${operationCount} total operations`);
    
    console.log(`\n🎉 Benchmark simulation completed successfully! Total operations: ${operationCount}`);
    
    // Final memory stats
    const finalStats = device.getMemoryStats();
    console.log(`Final memory: ${finalStats.totalAllocated} bytes, ${finalStats.activeBuffers} buffers`);
    
  } catch (error) {
    console.error(`❌ Simulation failed after ${operationCount} operations:`, error);
    throw error;
  }
}

async function main() {
  try {
    await simulateBenchmarkScenario();
  } catch (error) {
    console.error('Simulation failed:', error);
    process.exit(1);
  }
}

if (import.meta.main) {
  main();
}