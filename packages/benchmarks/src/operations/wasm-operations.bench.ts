/**
 * WASM backend operations benchmarks
 *
 * Tests the performance of unary, binary, and matrix operations
 * using the WASM backend.
 */

import { bench, describe } from 'vitest';
import { tensor, float32 } from '@typetensor/core';
import { WASMDevice } from '@typetensor/backend-wasm';
import { generateRandomData } from '../utils/data';

// Initialize WASM device once for all tests
let wasmDevice: WASMDevice;

beforeAll(async () => {
  wasmDevice = await WASMDevice.create();
});

describe('WASM unary operations', () => {
  const testSizes = [
    { name: 'small', shape: [1000] as const },
    { name: 'medium', shape: [100, 100] as const },
    { name: 'large', shape: [316, 316] as const }, // ~100k elements
  ];

  const unaryOps = [
    'neg', 'abs', 'sin', 'cos', 'exp', 'log', 'sqrt', 'square'
  ] as const;

  for (const size of testSizes) {
    const data = generateRandomData(size.shape);
    let testTensor: any;

    beforeAll(async () => {
      testTensor = await tensor(data, { device: wasmDevice, dtype: float32 });
    });

    for (const op of unaryOps) {
      bench(`WASM: ${op} ${size.name} ${size.shape.join('x')}`, async () => {
        await testTensor[op]();
      });
    }
  }
});

describe('WASM binary operations', () => {
  const testSizes = [
    { name: 'small', shape: [1000] as const },
    { name: 'medium', shape: [100, 100] as const },
    { name: 'large', shape: [200, 200] as const },
  ];

  const binaryOps = [
    { name: 'add', fn: (a: any, b: any) => a.add(b) },
    { name: 'sub', fn: (a: any, b: any) => a.sub(b) },
    { name: 'mul', fn: (a: any, b: any) => a.mul(b) },
    { name: 'div', fn: (a: any, b: any) => a.div(b) },
  ];

  for (const size of testSizes) {
    const data1 = generateRandomData(size.shape);
    const data2 = generateRandomData(size.shape);
    let tensor1: any, tensor2: any;

    beforeAll(async () => {
      tensor1 = await tensor(data1, { device: wasmDevice, dtype: float32 });
      tensor2 = await tensor(data2, { device: wasmDevice, dtype: float32 });
    });

    for (const op of binaryOps) {
      bench(`WASM: ${op.name} ${size.name} ${size.shape.join('x')}`, async () => {
        await op.fn(tensor1, tensor2);
      });
    }
  }
});

describe('WASM matrix multiplication', () => {
  const matmulSizes = [
    { name: 'tiny', a: [32, 32] as const, b: [32, 32] as const },
    { name: 'small', a: [64, 64] as const, b: [64, 64] as const },
    { name: 'medium', a: [128, 128] as const, b: [128, 128] as const },
    { name: 'large', a: [256, 256] as const, b: [256, 256] as const },
  ];

  for (const size of matmulSizes) {
    const dataA = generateRandomData(size.a);
    const dataB = generateRandomData(size.b);
    let tensorA: any, tensorB: any;

    beforeAll(async () => {
      tensorA = await tensor(dataA, { device: wasmDevice, dtype: float32 });
      tensorB = await tensor(dataB, { device: wasmDevice, dtype: float32 });
    });

    bench(`WASM: matmul ${size.name} ${size.a.join('x')} × ${size.b.join('x')}`, async () => {
      await tensorA.matmul(tensorB);
    });
  }
});

describe('WASM reduction operations', () => {
  const testSizes = [
    { name: 'vector', shape: [10000] as const },
    { name: 'matrix', shape: [100, 100] as const },
    { name: 'large', shape: [316, 316] as const }, // ~100k elements
  ];

  const reductionOps = ['sum', 'mean', 'max', 'min'] as const;

  for (const size of testSizes) {
    const data = generateRandomData(size.shape);
    let testTensor: any;

    beforeAll(async () => {
      testTensor = await tensor(data, { device: wasmDevice, dtype: float32 });
    });

    for (const op of reductionOps) {
      bench(`WASM: ${op} ${size.name} ${size.shape.join('x')}`, async () => {
        await testTensor[op]();
      });
    }
  }
});

describe('WASM activation functions', () => {
  const testSizes = [
    { name: 'small', shape: [1000] as const },
    { name: 'medium', shape: [100, 100] as const },
  ];

  for (const size of testSizes) {
    const data = generateRandomData(size.shape);
    let testTensor: any;

    beforeAll(async () => {
      testTensor = await tensor(data, { device: wasmDevice, dtype: float32 });
    });

    bench(`WASM: softmax ${size.name} ${size.shape.join('x')}`, async () => {
      await testTensor.softmax();
    });
  }
});

describe('WASM memory management', () => {
  const sizes = [
    { name: '1KB', elements: 256 },
    { name: '1MB', elements: 262144 },
  ];

  for (const size of sizes) {
    bench(`WASM: allocate ${size.name}`, async () => {
      const data = wasmDevice.createData(size.elements * 4);
      wasmDevice.disposeData(data);
    });

    bench(`WASM: buffer lifecycle ${size.name}`, async () => {
      const buffer = new ArrayBuffer(size.elements * 4);
      const data = wasmDevice.createDataWithBuffer(buffer);
      await wasmDevice.readData(data);
      wasmDevice.disposeData(data);
    });
  }
});