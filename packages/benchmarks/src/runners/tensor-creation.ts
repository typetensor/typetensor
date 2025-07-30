/**
 * Standalone tensor creation benchmark runner using tinybench directly
 * 
 * This provides more control over the benchmarking process and result formatting
 */

import { Bench } from 'tinybench';
import { tensor, zeros, ones, float32 } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';
import { VECTOR_SIZES, MATRIX_SIZES } from '../utils/sizes';
import { generateRandomData } from '../utils/data';
import { getBenchmarkConfig, getBenchmarkRecommendations } from '../utils/config';

async function runTensorCreationBenchmarks() {
  console.log('🚀 Running Tensor Creation Benchmarks\n');
  
  // Display benchmark recommendations
  const recommendations = getBenchmarkRecommendations();
  console.log('💡 Benchmark Reliability Tips:');
  recommendations.forEach(tip => console.log(`   ${tip}`));
  console.log('');
  
  // Get appropriate benchmark configuration
  const config = getBenchmarkConfig();
  const bench = new Bench({ 
    name: 'Tensor Creation',
    ...config,
  });
  
  console.log(`📊 Using profile: ${process.env.BENCHMARK_PROFILE || 'standard'}`);
  console.log(`⏱️  Runtime: ${config.time}ms per benchmark, min ${config.iterations} iterations\n`);

  // Add vector creation benchmarks
  console.log('Setting up vector benchmarks...');
  for (const size of VECTOR_SIZES) {
    const data = generateRandomData(size.shape);
    
    bench.add(`create vector ${size.name} (${size.elements} elements)`, async () => {
      await tensor(data, { device: cpu, dtype: float32 });
    });
  }

  // Add matrix creation benchmarks
  console.log('Setting up matrix benchmarks...');
  for (const size of MATRIX_SIZES) {
    const data = generateRandomData(size.shape);
    
    bench.add(`create matrix ${size.name} ${size.shape.join('x')}`, async () => {
      await tensor(data, { device: cpu, dtype: float32 });
    });
  }

  // Add zeros/ones benchmarks
  console.log('Setting up zeros/ones benchmarks...');
  for (const size of MATRIX_SIZES.slice(0, 3)) { // Only first 3 sizes for zeros/ones
    bench.add(`zeros ${size.name} ${size.shape.join('x')}`, async () => {
      await zeros(size.shape, { device: cpu, dtype: float32 });
    });
    
    bench.add(`ones ${size.name} ${size.shape.join('x')}`, async () => {
      await ones(size.shape, { device: cpu, dtype: float32 });
    });
  }

  // Get final count after all tasks are added
  const totalTasks = bench.tasks.size;
  
  // Add event listeners for progress tracking
  let completed = 0;
  
  bench.addEventListener('cycle', (evt) => {
    const task = evt.task!;
    completed++;
    console.log(`[${completed}/${totalTasks}] Completed: ${task.name}`);
  });
  
  // Run benchmarks
  console.log(`\nRunning ${totalTasks} benchmarks...\n`);
  await bench.run();

  // Display results using tinybench's built-in table
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
  
  return bench;
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runTensorCreationBenchmarks().catch(console.error);
}

export { runTensorCreationBenchmarks };