#!/usr/bin/env bun

/**
 * Matrix Multiplication Operations Benchmark Runner
 * 
 * Benchmarks performance of matrix multiplication operations (dot product, matmul)
 */

import { Bench } from 'tinybench';
import { tensor } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';
import { float32 } from '@typetensor/core';

console.log('🚀 Running Matrix Multiplication Benchmarks\n');

// Print benchmark tips
console.log('💡 Benchmark Reliability Tips:');
console.log('   🔧 For best results, run benchmarks on a dedicated machine');
console.log('   🔋 Ensure stable power supply (avoid battery mode)');
console.log('   🌡️  Allow system to warm up and reach thermal equilibrium');
console.log('   🔇 Close unnecessary applications to reduce system noise');
console.log('   ⚡ Consider using Node.js with --expose-gc flag for garbage collection control');
console.log('   📊 Run multiple benchmark sessions and compare results');
console.log('   ⏰ Be aware that results may vary between different times of day\n');

console.log('📊 Using profile: standard');
console.log('⏱️  Runtime: 1000ms per benchmark, min undefined iterations\n');

/**
 * Generate random data for a given shape
 */
function generateRandomData(shape: readonly number[]): any {
  if (shape.length === 0) {
    return Math.random();
  }
  
  if (shape.length === 1) {
    return Array.from({ length: shape[0]! }, () => Math.random());
  }
  
  const result = [];
  for (let i = 0; i < shape[0]!; i++) {
    result.push(generateRandomData(shape.slice(1)));
  }
  return result;
}

const bench = new Bench({
  time: 1000,
  warmup: true,
});

// Matrix multiplication cases to benchmark
const matmulCases = [
  // Vector dot products (1D × 1D → scalar)
  { name: 'dot tiny vectors', shapeA: [10] as const, shapeB: [10] as const },
  { name: 'dot small vectors', shapeA: [100] as const, shapeB: [100] as const },
  { name: 'dot medium vectors', shapeA: [1000] as const, shapeB: [1000] as const },
  { name: 'dot large vectors', shapeA: [10000] as const, shapeB: [10000] as const },
  
  // Vector-matrix multiplication (1D × 2D → 1D)
  { name: 'vec-mat tiny', shapeA: [10] as const, shapeB: [10, 10] as const },
  { name: 'vec-mat small', shapeA: [32] as const, shapeB: [32, 32] as const },
  { name: 'vec-mat medium', shapeA: [100] as const, shapeB: [100, 100] as const },
  { name: 'vec-mat large', shapeA: [256] as const, shapeB: [256, 256] as const },
  
  // Matrix-vector multiplication (2D × 1D → 1D)
  { name: 'mat-vec tiny', shapeA: [10, 10] as const, shapeB: [10] as const },
  { name: 'mat-vec small', shapeA: [32, 32] as const, shapeB: [32] as const },
  { name: 'mat-vec medium', shapeA: [100, 100] as const, shapeB: [100] as const },
  { name: 'mat-vec large', shapeA: [256, 256] as const, shapeB: [256] as const },
  
  // Matrix-matrix multiplication (2D × 2D → 2D)
  { name: 'mat-mat tiny', shapeA: [10, 10] as const, shapeB: [10, 10] as const },
  { name: 'mat-mat small', shapeA: [32, 32] as const, shapeB: [32, 32] as const },
  { name: 'mat-mat medium', shapeA: [64, 64] as const, shapeB: [64, 64] as const },
  { name: 'mat-mat large', shapeA: [128, 128] as const, shapeB: [128, 128] as const },
];

// Pre-create tensor pairs for benchmarking
console.log('Setting up tensor pairs for matrix multiplication...');
const tensorPairs = new Map();

for (const testCase of matmulCases) {
  const dataA = generateRandomData(testCase.shapeA);
  const dataB = generateRandomData(testCase.shapeB);
  const tensorA = await tensor(dataA, { device: cpu, dtype: float32 });
  const tensorB = await tensor(dataB, { device: cpu, dtype: float32 });
  tensorPairs.set(testCase.name, { a: tensorA, b: tensorB });
  console.log(`✓ Created ${testCase.name}: ${testCase.shapeA.join('×')} × ${testCase.shapeB.join('×')}`);
}

console.log('\nSetting up benchmarks...');

// Add benchmarks for each matrix multiplication case
for (const testCase of matmulCases) {
  const { a, b } = tensorPairs.get(testCase.name);
  
  bench.add(`${testCase.name} (${testCase.shapeA.join('×')} × ${testCase.shapeB.join('×')})`, async () => {
    await a.matmul(b);
  });
}

console.log(`\nRunning ${bench.tasks.length} benchmarks...\n`);

// Run benchmarks with progress tracking
let completed = 0;
const total = bench.tasks.length;

bench.addEventListener('cycle', (e) => {
  completed++;
  console.log(`[${completed}/${total}] Completed: ${e.task.name}`);
});

await bench.run();

console.log('\n📊 Benchmark Results\n');
console.log('=' .repeat(80));
console.table(bench.table());

// Also show individual task details
console.log('\n📋 Detailed Results:\n');
bench.tasks.forEach((task) => {
  const result = task.result;
  if (result) {
    console.log(`\n### ${task.name}`);
    console.log(`  Operations/sec: ${result.hz.toLocaleString()}`);
    console.log(`  Mean time: ${(result.mean * 1000).toFixed(4)}ms`);
    console.log(`  Standard deviation: ${(result.sd * 1000).toFixed(4)}ms`);
    console.log(`  Margin of error: ±${result.rme.toFixed(2)}%`);
    console.log(`  P99: ${(result.p99 * 1000).toFixed(4)}ms`);
    console.log(`  Samples: ${result.samples.length.toLocaleString()}`);
  }
});