#!/usr/bin/env bun

/**
 * Binary Operations Benchmark Runner
 *
 * Benchmarks performance of binary tensor operations like add, sub, mul, div
 */

import { Bench } from 'tinybench';
import { tensor } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';
import { float32 } from '@typetensor/core';

console.log('🚀 Running Binary Operations Benchmarks\n');

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

// Binary operations to benchmark
const binaryOps = [
  { name: 'add', op: (a: any, b: any) => a.add(b) },
  { name: 'sub', op: (a: any, b: any) => a.sub(b) },
  { name: 'mul', op: (a: any, b: any) => a.mul(b) },
  { name: 'div', op: (a: any, b: any) => a.div(b) },
];

// Size presets for benchmarking
const sizes = [
  { name: 'tiny vector', shape: [10] as const },
  { name: 'small vector', shape: [100] as const },
  { name: 'medium vector', shape: [1000] as const },
  { name: 'large vector', shape: [10000] as const },
  { name: 'tiny matrix', shape: [10, 10] as const },
  { name: 'small matrix', shape: [32, 32] as const },
  { name: 'medium matrix', shape: [100, 100] as const },
  { name: 'large matrix', shape: [256, 256] as const },
];

// Pre-create tensor pairs for benchmarking
console.log('Setting up tensor pairs for binary operations...');
const tensorPairs = new Map();

for (const size of sizes) {
  const dataA = generateRandomData(size.shape);
  const dataB = generateRandomData(size.shape);
  const tensorA = await tensor(dataA, { device: cpu, dtype: float32 });
  const tensorB = await tensor(dataB, { device: cpu, dtype: float32 });
  tensorPairs.set(size.name, { a: tensorA, b: tensorB });
  console.log(`✓ Created ${size.name} tensor pair: ${size.shape.join('×')}`);
}

console.log('\nSetting up benchmarks...');

// Add benchmarks for each operation and size combination
for (const op of binaryOps) {
  for (const size of sizes) {
    const { a, b } = tensorPairs.get(size.name);

    bench.add(`${op.name} ${size.name} (${size.shape.join('×')})`, async () => {
      await op.op(a, b);
    });
  }
}

console.log(`\nRunning ${bench.tasks.length} benchmarks...\n`);

// Run benchmarks with progress tracking
let completed = 0;
const total = bench.tasks.length;

bench.addEventListener('cycle', (e) => {
  completed++;
  console.log(`[${completed}/${total}] Completed: ${e.task?.name}`);
});

await bench.run();

console.log('\n📊 Benchmark Results\n');
console.log('='.repeat(80));
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
