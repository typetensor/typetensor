#!/usr/bin/env bun

/**
 * Unary Operations Benchmark Runner
 * 
 * Benchmarks performance of unary tensor operations like neg, abs, sin, cos, exp, log, sqrt, square
 */

import { Bench } from 'tinybench';
import { tensor } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';
import { float32 } from '@typetensor/core';

console.log('🚀 Running Unary Operations Benchmarks\n');

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

// Unary operations to benchmark
const unaryOps = [
  { name: 'neg', op: (t: any) => t.neg() },
  { name: 'abs', op: (t: any) => t.abs() },
  { name: 'sin', op: (t: any) => t.sin() },
  { name: 'cos', op: (t: any) => t.cos() },
  { name: 'exp', op: (t: any) => t.exp() },
  { name: 'log', op: (t: any) => t.log() },
  { name: 'sqrt', op: (t: any) => t.sqrt() },
  { name: 'square', op: (t: any) => t.square() },
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

// Pre-create tensors for benchmarking
console.log('Setting up tensors for unary operations...');
const tensors = new Map();

for (const size of sizes) {
  const data = generateRandomData(size.shape);
  const t = await tensor(data, { device: cpu, dtype: float32 });
  tensors.set(size.name, t);
  console.log(`✓ Created ${size.name} tensor: ${size.shape.join('×')}`);
}

console.log('\nSetting up benchmarks...');

// Add benchmarks for each operation and size combination
for (const op of unaryOps) {
  for (const size of sizes) {
    const t = tensors.get(size.name);
    
    bench.add(`${op.name} ${size.name} (${size.shape.join('×')})`, async () => {
      await op.op(t);
    });
  }
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