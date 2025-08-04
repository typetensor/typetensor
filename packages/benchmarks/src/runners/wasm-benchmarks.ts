/**
 * Comprehensive WASM backend benchmark runner
 *
 * Tests all major operations on the WASM backend to evaluate performance
 * characteristics and identify optimization opportunities.
 */

import { Bench } from 'tinybench';
import { tensor, zeros, ones, float32, int32 } from '@typetensor/core';
import { WASMDevice } from '@typetensor/backend-wasm';
import { VECTOR_SIZES, MATRIX_SIZES, TENSOR_3D_SIZES } from '../utils/sizes';
import { generateRandomData } from '../utils/data';
import { getBenchmarkConfig, getBenchmarkRecommendations } from '../utils/config';

async function runWASMBenchmarks() {
  console.log('🚀 Running WASM Backend Benchmarks\n');

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
  console.log(`✅ WASM device initialized: ${wasmDevice.toString()}\n`);

  // Display WASM capabilities
  const capabilities = wasmDevice.getCapabilities();
  console.log('🔧 WASM Capabilities:');
  console.log(`   SIMD Support: ${capabilities.simd ? '✅' : '❌'}`);
  console.log(`   Shared Memory: ${capabilities.sharedMemory ? '✅' : '❌'}`);
  console.log(`   Optimal Threads: ${capabilities.optimalThreadCount}`);
  console.log(`   Available Memory: ${(capabilities.availableMemory / 1024 / 1024).toFixed(1)}MB`);
  console.log(`   Version: ${capabilities.version}\n`);

  const config = getBenchmarkConfig();
  const bench = new Bench({
    name: 'WASM Backend Performance',
    ...config,
  });

  console.log(`📊 Using profile: ${process.env.BENCHMARK_PROFILE || 'standard'}`);
  console.log(`⏱️  Runtime: ${config.time}ms per benchmark\n`);

  // === Tensor Creation Benchmarks ===
  console.log('Setting up tensor creation benchmarks...');

  // Vector creation
  for (const size of VECTOR_SIZES) {
    const data = generateRandomData(size.shape);

    bench.add(`WASM: create vector ${size.name} (${size.elements} elements)`, async () => {
      await tensor(data, { device: wasmDevice, dtype: float32 });
    });
  }

  // Matrix creation
  for (const size of MATRIX_SIZES) {
    const data = generateRandomData(size.shape);

    bench.add(`WASM: create matrix ${size.name} ${size.shape.join('x')}`, async () => {
      await tensor(data, { device: wasmDevice, dtype: float32 });
    });
  }

  // Zeros and ones
  for (const size of MATRIX_SIZES.slice(0, 3)) {
    bench.add(`WASM: zeros ${size.name} ${size.shape.join('x')}`, async () => {
      await zeros(size.shape, { device: wasmDevice, dtype: float32 });
    });

    bench.add(`WASM: ones ${size.name} ${size.shape.join('x')}`, async () => {
      await ones(size.shape, { device: wasmDevice, dtype: float32 });
    });
  }

  // === Unary Operations Benchmarks ===
  console.log('Setting up unary operations benchmarks...');

  // Prepare test tensors for unary ops
  const unaryTestSizes = [
    { name: 'small', shape: [1000] },
    { name: 'medium', shape: [100, 100] },
    { name: 'large', shape: [500, 500] },
  ] as const;

  for (const size of unaryTestSizes) {
    const unaryOps = ['neg', 'abs', 'sin', 'cos', 'exp', 'log', 'sqrt', 'square'] as const;

    for (const op of unaryOps) {
      bench.add(`WASM: ${op} ${size.name} ${size.shape.join('x')}`, async () => {
        const data = generateRandomData(size.shape);
        const testTensor = await tensor(data, { device: wasmDevice, dtype: float32 });
        await (testTensor as any)[op]();
      });
    }
  }

  // === Binary Operations Benchmarks ===
  console.log('Setting up binary operations benchmarks...');

  const binaryTestSizes = [
    { name: 'small', shape: [1000] },
    { name: 'medium', shape: [100, 100] },
    { name: 'large', shape: [250, 250] },
  ] as const;

  for (const size of binaryTestSizes) {
    const data1 = generateRandomData(size.shape);
    const data2 = generateRandomData(size.shape);
    const tensor1 = await tensor(data1, { device: wasmDevice, dtype: float32 });
    const tensor2 = await tensor(data2, { device: wasmDevice, dtype: float32 });

    const binaryOps = [
      { name: 'add', fn: (a: any, b: any) => a.add(b) },
      { name: 'sub', fn: (a: any, b: any) => a.sub(b) },
      { name: 'mul', fn: (a: any, b: any) => a.mul(b) },
      { name: 'div', fn: (a: any, b: any) => a.div(b) },
    ];

    for (const op of binaryOps) {
      bench.add(`WASM: ${op.name} ${size.name} ${size.shape.join('x')}`, async () => {
        await op.fn(tensor1, tensor2);
      });
    }
  }

  // === Matrix Multiplication Benchmarks ===
  console.log('Setting up matrix multiplication benchmarks...');

  const matmulSizes = [
    { name: 'tiny', a: [32, 32], b: [32, 32] },
    { name: 'small', a: [64, 64], b: [64, 64] },
    { name: 'medium', a: [128, 128], b: [128, 128] },
    { name: 'large', a: [256, 256], b: [256, 256] },
  ] as const;

  for (const size of matmulSizes) {
    const dataA = generateRandomData(size.a);
    const dataB = generateRandomData(size.b);
    const tensorA = await tensor(dataA, { device: wasmDevice, dtype: float32 });
    const tensorB = await tensor(dataB, { device: wasmDevice, dtype: float32 });

    bench.add(`WASM: matmul ${size.name} ${size.a.join('x')} × ${size.b.join('x')}`, async () => {
      await (tensorA as any).matmul(tensorB);
    });
  }

  // === Reduction Operations Benchmarks ===
  console.log('Setting up reduction operations benchmarks...');

  const reductionTestSizes = [
    { name: 'vector', shape: [10000] },
    { name: 'matrix', shape: [100, 100] },
    { name: 'large', shape: [316, 316] }, // ~100k elements
  ] as const;

  for (const size of reductionTestSizes) {
    const data = generateRandomData(size.shape);
    const testTensor = await tensor(data, { device: wasmDevice, dtype: float32 });

    const reductionOps = ['sum', 'mean', 'max', 'min'] as const;

    for (const op of reductionOps) {
      bench.add(`WASM: ${op} ${size.name} ${size.shape.join('x')}`, async () => {
        await (testTensor as any)[op]();
      });
    }
  }

  // === Memory Transfer Benchmarks ===
  console.log('Setting up memory transfer benchmarks...');

  const transferSizes = [
    { name: '1KB', elements: 256 }, // 256 * 4 bytes = 1KB
    { name: '1MB', elements: 262144 }, // 262144 * 4 bytes = 1MB
    { name: '10MB', elements: 2621440 }, // ~10MB
  ] as const;

  for (const size of transferSizes) {
    const data = generateRandomData([size.elements]);
    const wasmTensor = await tensor(data, { device: wasmDevice, dtype: float32 });

    bench.add(`WASM: read data ${size.name}`, async () => {
      await wasmDevice.readData(wasmTensor.data);
    });

    const writeBuffer = new ArrayBuffer(size.elements * 4);
    const writeData = wasmDevice.createData(writeBuffer.byteLength);

    bench.add(`WASM: write data ${size.name}`, async () => {
      await wasmDevice.writeData(writeData, writeBuffer);
    });
  }

  // Get final count after all tasks are added
  const totalTasks = bench.tasks.size;

  // Add progress tracking
  let completed = 0;
  bench.addEventListener('cycle', (evt) => {
    const task = evt.task!;
    completed++;
    const result = task.result;
    const opsPerSec = result?.hz ? result.hz.toLocaleString() : 'Failed';
    console.log(`[${completed}/${totalTasks}] ${task.name}: ${opsPerSec} ops/sec`);
  });

  // Run benchmarks
  console.log(`\n🏃‍♀️ Running ${totalTasks} WASM benchmarks...\n`);
  await bench.run();

  // Display results
  console.log('\n📊 WASM Benchmark Results\n');
  console.log('='.repeat(100));
  console.table(bench.table());

  // Show memory stats after benchmarks
  const memStats = wasmDevice.getMemoryStats();
  console.log('\n🧠 WASM Memory Statistics:');
  console.log(`   Total Allocated: ${(memStats.totalAllocated / 1024).toFixed(1)}KB`);
  console.log(`   Active Buffers: ${memStats.activeBuffers}`);
  if (memStats.poolSummary) {
    console.log('\n📋 Buffer Pool Details:');
    console.log(memStats.poolSummary);
  }

  // Performance analysis
  console.log('\n📈 Performance Analysis:');

  const creationTasks = bench.tasks.filter((task) => task.name.includes('create'));
  const unaryTasks = bench.tasks.filter(
    (task) => task.name.includes('neg') || task.name.includes('abs') || task.name.includes('sin'),
  );
  const binaryTasks = bench.tasks.filter(
    (task) => task.name.includes('add') || task.name.includes('mul'),
  );
  const matmulTasks = bench.tasks.filter((task) => task.name.includes('matmul'));

  const getAveragePerformance = (tasks: any[]) => {
    const validResults = tasks.filter((task) => task.result && task.result.hz > 0);
    if (validResults.length === 0) {
      return 0;
    }
    return validResults.reduce((sum, task) => sum + task.result.hz, 0) / validResults.length;
  };

  console.log(
    `   Avg Creation Performance: ${getAveragePerformance(creationTasks).toLocaleString()} ops/sec`,
  );
  console.log(
    `   Avg Unary Performance: ${getAveragePerformance(unaryTasks).toLocaleString()} ops/sec`,
  );
  console.log(
    `   Avg Binary Performance: ${getAveragePerformance(binaryTasks).toLocaleString()} ops/sec`,
  );
  console.log(
    `   Avg MatMul Performance: ${getAveragePerformance(matmulTasks).toLocaleString()} ops/sec`,
  );

  return bench;
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runWASMBenchmarks().catch(console.error);
}

export { runWASMBenchmarks };
