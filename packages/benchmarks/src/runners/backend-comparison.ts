/**
 * Backend comparison benchmark runner
 *
 * Runs identical operations on both CPU and WASM backends to compare
 * performance characteristics and identify the best backend for different
 * operation types and data sizes.
 */

import { Bench } from 'tinybench';
import { tensor, zeros, ones, float32 } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';
import { WASMDevice } from '@typetensor/backend-wasm';
import { generateRandomData } from '../utils/data';
import { getBenchmarkConfig, getBenchmarkRecommendations } from '../utils/config';

interface BackendComparison {
  operation: string;
  size: string;
  cpuOpsPerSec: number;
  wasmOpsPerSec: number;
  speedup: number; // WASM/CPU ratio
  winner: 'CPU' | 'WASM' | 'Tie';
}

async function runBackendComparison() {
  console.log('🏁 Running Backend Comparison Benchmarks\n');

  // Display benchmark recommendations
  const recommendations = getBenchmarkRecommendations();
  console.log('💡 Benchmark Reliability Tips:');
  recommendations.forEach((tip) => {
    console.log(`   ${tip}`);
  });
  console.log('');

  // Initialize WASM device
  console.log('⚡ Initializing WASM device...');
  const wasmDevice = await WASMDevice.create();
  console.log(`✅ WASM device initialized\n`);

  const config = getBenchmarkConfig();
  const results: BackendComparison[] = [];

  // Test configurations for different operation categories
  const testConfigs = [
    // Small data - overhead might dominate
    { name: 'small', shape: [1000] as const, category: 'Small Data' },
    // Medium data - sweet spot for many operations
    { name: 'medium', shape: [100, 100] as const, category: 'Medium Data' },
    // Large data - where WASM should shine
    { name: 'large', shape: [500, 500] as const, category: 'Large Data' },
  ];

  console.log('🔄 Running comparative benchmarks...\n');

  // === Tensor Creation Comparison ===
  console.log('📊 Comparing tensor creation performance...');
  
  for (const testConfig of testConfigs) {
    const data = generateRandomData(testConfig.shape);
    
    const bench = new Bench({ ...config, name: `Creation ${testConfig.name}` });
    
    bench.add(`CPU: create ${testConfig.name}`, async () => {
      await tensor(data, { device: cpu, dtype: float32 });
    });

    bench.add(`WASM: create ${testConfig.name}`, async () => {
      await tensor(data, { device: wasmDevice, dtype: float32 });
    });

    await bench.run();
    
    const cpuResult = bench.tasks.find(t => t.name.includes('CPU'))?.result;
    const wasmResult = bench.tasks.find(t => t.name.includes('WASM'))?.result;
    
    if (cpuResult && wasmResult) {
      const comparison: BackendComparison = {
        operation: `create ${testConfig.name}`,
        size: testConfig.shape.join('x'),
        cpuOpsPerSec: cpuResult.hz,
        wasmOpsPerSec: wasmResult.hz,
        speedup: wasmResult.hz / cpuResult.hz,
        winner: wasmResult.hz > cpuResult.hz * 1.05 ? 'WASM' : 
                cpuResult.hz > wasmResult.hz * 1.05 ? 'CPU' : 'Tie'
      };
      results.push(comparison);
      console.log(`   ${comparison.operation}: ${comparison.winner} wins (${comparison.speedup.toFixed(2)}x)`);
    }
  }

  // === Unary Operations Comparison ===
  console.log('\n📊 Comparing unary operations performance...');
  
  const unaryOps = ['neg', 'abs', 'sin', 'exp', 'sqrt'] as const;
  
  for (const testConfig of testConfigs.slice(0, 2)) { // Skip large for unary to save time
    const data = generateRandomData(testConfig.shape);
    const cpuTensor = await tensor(data, { device: cpu, dtype: float32 });
    const wasmTensor = await tensor(data, { device: wasmDevice, dtype: float32 });

    for (const op of unaryOps) {
      const bench = new Bench({ ...config, name: `${op} ${testConfig.name}` });
      
      bench.add(`CPU: ${op} ${testConfig.name}`, async () => {
        await (cpuTensor as any)[op]();
      });

      bench.add(`WASM: ${op} ${testConfig.name}`, async () => {
        await (wasmTensor as any)[op]();
      });

      await bench.run();
      
      const cpuResult = bench.tasks.find(t => t.name.includes('CPU'))?.result;
      const wasmResult = bench.tasks.find(t => t.name.includes('WASM'))?.result;
      
      if (cpuResult && wasmResult) {
        const comparison: BackendComparison = {
          operation: `${op} ${testConfig.name}`,
          size: testConfig.shape.join('x'),
          cpuOpsPerSec: cpuResult.hz,
          wasmOpsPerSec: wasmResult.hz,
          speedup: wasmResult.hz / cpuResult.hz,
          winner: wasmResult.hz > cpuResult.hz * 1.05 ? 'WASM' : 
                  cpuResult.hz > wasmResult.hz * 1.05 ? 'CPU' : 'Tie'
        };
        results.push(comparison);
        console.log(`   ${comparison.operation}: ${comparison.winner} wins (${comparison.speedup.toFixed(2)}x)`);
      }
    }
  }

  // === Binary Operations Comparison ===
  console.log('\n📊 Comparing binary operations performance...');
  
  const binaryOps = [
    { name: 'add', fn: (a: any, b: any) => a.add(b) },
    { name: 'mul', fn: (a: any, b: any) => a.mul(b) },
    { name: 'div', fn: (a: any, b: any) => a.div(b) },
  ];

  for (const testConfig of testConfigs.slice(0, 2)) { // Skip large for binary to save time
    const data1 = generateRandomData(testConfig.shape);
    const data2 = generateRandomData(testConfig.shape);
    const cpuTensor1 = await tensor(data1, { device: cpu, dtype: float32 });
    const cpuTensor2 = await tensor(data2, { device: cpu, dtype: float32 });
    const wasmTensor1 = await tensor(data1, { device: wasmDevice, dtype: float32 });
    const wasmTensor2 = await tensor(data2, { device: wasmDevice, dtype: float32 });

    for (const op of binaryOps) {
      const bench = new Bench({ ...config, name: `${op.name} ${testConfig.name}` });
      
      bench.add(`CPU: ${op.name} ${testConfig.name}`, async () => {
        await op.fn(cpuTensor1, cpuTensor2);
      });

      bench.add(`WASM: ${op.name} ${testConfig.name}`, async () => {
        await op.fn(wasmTensor1, wasmTensor2);
      });

      await bench.run();
      
      const cpuResult = bench.tasks.find(t => t.name.includes('CPU'))?.result;
      const wasmResult = bench.tasks.find(t => t.name.includes('WASM'))?.result;
      
      if (cpuResult && wasmResult) {
        const comparison: BackendComparison = {
          operation: `${op.name} ${testConfig.name}`,
          size: testConfig.shape.join('x'),
          cpuOpsPerSec: cpuResult.hz,
          wasmOpsPerSec: wasmResult.hz,
          speedup: wasmResult.hz / cpuResult.hz,
          winner: wasmResult.hz > cpuResult.hz * 1.05 ? 'WASM' : 
                  cpuResult.hz > wasmResult.hz * 1.05 ? 'CPU' : 'Tie'
        };
        results.push(comparison);
        console.log(`   ${comparison.operation}: ${comparison.winner} wins (${comparison.speedup.toFixed(2)}x)`);
      }
    }
  }

  // === Matrix Multiplication Comparison ===
  console.log('\n📊 Comparing matrix multiplication performance...');
  
  const matmulSizes = [
    { name: 'small', a: [64, 64] as const, b: [64, 64] as const },
    { name: 'medium', a: [128, 128] as const, b: [128, 128] as const },
    { name: 'large', a: [256, 256] as const, b: [256, 256] as const },
  ];

  for (const size of matmulSizes) {
    const dataA = generateRandomData(size.a);
    const dataB = generateRandomData(size.b);
    const cpuTensorA = await tensor(dataA, { device: cpu, dtype: float32 });
    const cpuTensorB = await tensor(dataB, { device: cpu, dtype: float32 });
    const wasmTensorA = await tensor(dataA, { device: wasmDevice, dtype: float32 });
    const wasmTensorB = await tensor(dataB, { device: wasmDevice, dtype: float32 });

    const bench = new Bench({ ...config, name: `matmul ${size.name}` });
    
    bench.add(`CPU: matmul ${size.name}`, async () => {
      await (cpuTensorA as any).matmul(cpuTensorB);
    });

    bench.add(`WASM: matmul ${size.name}`, async () => {
      await (wasmTensorA as any).matmul(wasmTensorB);
    });

    await bench.run();
    
    const cpuResult = bench.tasks.find(t => t.name.includes('CPU'))?.result;
    const wasmResult = bench.tasks.find(t => t.name.includes('WASM'))?.result;
    
    if (cpuResult && wasmResult) {
      const comparison: BackendComparison = {
        operation: `matmul ${size.name}`,
        size: `${size.a.join('x')} × ${size.b.join('x')}`,
        cpuOpsPerSec: cpuResult.hz,
        wasmOpsPerSec: wasmResult.hz,
        speedup: wasmResult.hz / cpuResult.hz,
        winner: wasmResult.hz > cpuResult.hz * 1.05 ? 'WASM' : 
                cpuResult.hz > wasmResult.hz * 1.05 ? 'CPU' : 'Tie'
      };
      results.push(comparison);
      console.log(`   ${comparison.operation}: ${comparison.winner} wins (${comparison.speedup.toFixed(2)}x)`);
    }
  }

  // === Results Analysis ===
  console.log('\n📈 Backend Comparison Results\n');
  console.log('='.repeat(120));
  
  // Create detailed results table
  const tableData = results.map(r => ({
    'Operation': r.operation,
    'Size': r.size,
    'CPU (ops/sec)': r.cpuOpsPerSec.toLocaleString(),
    'WASM (ops/sec)': r.wasmOpsPerSec.toLocaleString(),
    'Speedup': `${r.speedup.toFixed(2)}x`,
    'Winner': r.winner === 'WASM' ? '🚀 WASM' : r.winner === 'CPU' ? '💻 CPU' : '🤝 Tie'
  }));
  
  console.table(tableData);

  // Summary statistics
  const wasmWins = results.filter(r => r.winner === 'WASM').length;
  const cpuWins = results.filter(r => r.winner === 'CPU').length;
  const ties = results.filter(r => r.winner === 'Tie').length;
  const avgSpeedup = results.reduce((sum, r) => sum + r.speedup, 0) / results.length;
  const maxSpeedup = Math.max(...results.map(r => r.speedup));
  const minSpeedup = Math.min(...results.map(r => r.speedup));

  console.log('\n📊 Summary Statistics:');
  console.log(`   Total Comparisons: ${results.length}`);
  console.log(`   WASM Wins: ${wasmWins} (${(wasmWins/results.length*100).toFixed(1)}%)`);
  console.log(`   CPU Wins: ${cpuWins} (${(cpuWins/results.length*100).toFixed(1)}%)`);
  console.log(`   Ties: ${ties} (${(ties/results.length*100).toFixed(1)}%)`);
  console.log(`   Average WASM Speedup: ${avgSpeedup.toFixed(2)}x`);
  console.log(`   Best WASM Speedup: ${maxSpeedup.toFixed(2)}x`);
  console.log(`   Worst WASM Speedup: ${minSpeedup.toFixed(2)}x`);

  // Operation category analysis
  console.log('\n🎯 Performance by Operation Category:');
  
  const categories = {
    'Creation': results.filter(r => r.operation.includes('create')),
    'Unary': results.filter(r => ['neg', 'abs', 'sin', 'exp', 'sqrt'].some(op => r.operation.includes(op))),
    'Binary': results.filter(r => ['add', 'mul', 'div'].some(op => r.operation.includes(op))),
    'MatMul': results.filter(r => r.operation.includes('matmul'))
  };

  for (const [category, categoryResults] of Object.entries(categories)) {
    if (categoryResults.length === 0) continue;
    
    const categoryWasmWins = categoryResults.filter(r => r.winner === 'WASM').length;
    const categoryAvgSpeedup = categoryResults.reduce((sum, r) => sum + r.speedup, 0) / categoryResults.length;
    
    console.log(`   ${category}: ${categoryWasmWins}/${categoryResults.length} WASM wins, ${categoryAvgSpeedup.toFixed(2)}x avg speedup`);
  }

  // Recommendations
  console.log('\n💡 Backend Selection Recommendations:');
  
  if (avgSpeedup > 1.1) {
    console.log('   🚀 WASM backend shows overall better performance');
    console.log('   📝 Consider WASM as default for production workloads');
  } else if (avgSpeedup < 0.9) {
    console.log('   💻 CPU backend shows overall better performance');
    console.log('   📝 Consider CPU as default, WASM for specific operations');
  } else {
    console.log('   🤝 Both backends show similar performance');
    console.log('   📝 Choose based on deployment constraints and specific use cases');
  }

  // Best use cases for each backend
  const bestWasmOps = results
    .filter(r => r.winner === 'WASM' && r.speedup > 1.2)
    .sort((a, b) => b.speedup - a.speedup)
    .slice(0, 3);
    
  const bestCpuOps = results
    .filter(r => r.winner === 'CPU' && r.speedup < 0.8)
    .sort((a, b) => a.speedup - b.speedup)
    .slice(0, 3);

  if (bestWasmOps.length > 0) {
    console.log('\n🚀 Best WASM Use Cases:');
    bestWasmOps.forEach(op => {
      console.log(`   ${op.operation} (${op.size}): ${op.speedup.toFixed(2)}x faster`);
    });
  }

  if (bestCpuOps.length > 0) {
    console.log('\n💻 Best CPU Use Cases:');
    bestCpuOps.forEach(op => {
      console.log(`   ${op.operation} (${op.size}): ${(1/op.speedup).toFixed(2)}x faster`);
    });
  }

  return results;
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runBackendComparison().catch(console.error);
}

export { runBackendComparison, type BackendComparison };