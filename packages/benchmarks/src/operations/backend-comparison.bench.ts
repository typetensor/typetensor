/**
 * Backend comparison benchmarks
 *
 * Direct comparison between CPU and WASM backends for identical operations
 * to identify performance characteristics and optimal use cases.
 */

import { bench, describe } from 'vitest';
import { tensor, zeros, ones, float32 } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';
import { WASMDevice } from '@typetensor/backend-wasm';
import { generateRandomData } from '../utils/data';

// Initialize WASM device once for all tests
let wasmDevice: WASMDevice;

beforeAll(async () => {
  wasmDevice = await WASMDevice.create();
});

describe('CPU vs WASM: tensor creation', () => {
  const testSizes = [
    { name: 'small', shape: [1000] as const },
    { name: 'medium', shape: [100, 100] as const },
    { name: 'large', shape: [500, 500] as const },
  ];

  for (const size of testSizes) {
    const data = generateRandomData(size.shape);

    bench(`CPU: create ${size.name} ${size.shape.join('x')}`, async () => {
      await tensor(data, { device: cpu, dtype: float32 });
    });

    bench(`WASM: create ${size.name} ${size.shape.join('x')}`, async () => {
      await tensor(data, { device: wasmDevice, dtype: float32 });
    });
  }

  // Zeros/ones comparison
  for (const size of testSizes.slice(0, 2)) { // Only small and medium
    bench(`CPU: zeros ${size.name} ${size.shape.join('x')}`, async () => {
      await zeros(size.shape, { device: cpu, dtype: float32 });
    });

    bench(`WASM: zeros ${size.name} ${size.shape.join('x')}`, async () => {
      await zeros(size.shape, { device: wasmDevice, dtype: float32 });
    });
  }
});

describe('CPU vs WASM: unary operations', () => {
  const testSizes = [
    { name: 'small', shape: [1000] as const },
    { name: 'medium', shape: [100, 100] as const },
  ];

  const unaryOps = ['neg', 'abs', 'sin', 'exp', 'sqrt'] as const;

  for (const size of testSizes) {
    const data = generateRandomData(size.shape);
    let cpuTensor: any, wasmTensor: any;

    beforeAll(async () => {
      cpuTensor = await tensor(data, { device: cpu, dtype: float32 });
      wasmTensor = await tensor(data, { device: wasmDevice, dtype: float32 });
    });

    for (const op of unaryOps) {
      bench(`CPU: ${op} ${size.name} ${size.shape.join('x')}`, async () => {
        await cpuTensor[op]();
      });

      bench(`WASM: ${op} ${size.name} ${size.shape.join('x')}`, async () => {
        await wasmTensor[op]();
      });
    }
  }
});

describe('CPU vs WASM: binary operations', () => {
  const testSizes = [
    { name: 'small', shape: [1000] as const },
    { name: 'medium', shape: [100, 100] as const },
  ];

  const binaryOps = [
    { name: 'add', fn: (a: any, b: any) => a.add(b) },
    { name: 'mul', fn: (a: any, b: any) => a.mul(b) },
    { name: 'div', fn: (a: any, b: any) => a.div(b) },
  ];

  for (const size of testSizes) {
    const data1 = generateRandomData(size.shape);
    const data2 = generateRandomData(size.shape);
    let cpuTensor1: any, cpuTensor2: any;
    let wasmTensor1: any, wasmTensor2: any;

    beforeAll(async () => {
      cpuTensor1 = await tensor(data1, { device: cpu, dtype: float32 });
      cpuTensor2 = await tensor(data2, { device: cpu, dtype: float32 });
      wasmTensor1 = await tensor(data1, { device: wasmDevice, dtype: float32 });
      wasmTensor2 = await tensor(data2, { device: wasmDevice, dtype: float32 });
    });

    for (const op of binaryOps) {
      bench(`CPU: ${op.name} ${size.name} ${size.shape.join('x')}`, async () => {
        await op.fn(cpuTensor1, cpuTensor2);
      });

      bench(`WASM: ${op.name} ${size.name} ${size.shape.join('x')}`, async () => {
        await op.fn(wasmTensor1, wasmTensor2);
      });
    }
  }
});

describe('CPU vs WASM: matrix multiplication', () => {
  const matmulSizes = [
    { name: 'small', a: [64, 64] as const, b: [64, 64] as const },
    { name: 'medium', a: [128, 128] as const, b: [128, 128] as const },
    { name: 'large', a: [256, 256] as const, b: [256, 256] as const },
  ];

  for (const size of matmulSizes) {
    const dataA = generateRandomData(size.a);
    const dataB = generateRandomData(size.b);
    let cpuTensorA: any, cpuTensorB: any;
    let wasmTensorA: any, wasmTensorB: any;

    beforeAll(async () => {
      cpuTensorA = await tensor(dataA, { device: cpu, dtype: float32 });
      cpuTensorB = await tensor(dataB, { device: cpu, dtype: float32 });
      wasmTensorA = await tensor(dataA, { device: wasmDevice, dtype: float32 });
      wasmTensorB = await tensor(dataB, { device: wasmDevice, dtype: float32 });
    });

    bench(`CPU: matmul ${size.name} ${size.a.join('x')} × ${size.b.join('x')}`, async () => {
      await cpuTensorA.matmul(cpuTensorB);
    });

    bench(`WASM: matmul ${size.name} ${size.a.join('x')} × ${size.b.join('x')}`, async () => {
      await wasmTensorA.matmul(wasmTensorB);
    });
  }
});

describe('CPU vs WASM: reduction operations', () => {
  const testSizes = [
    { name: 'vector', shape: [10000] as const },
    { name: 'matrix', shape: [100, 100] as const },
  ];

  const reductionOps = ['sum', 'mean', 'max', 'min'] as const;

  for (const size of testSizes) {
    const data = generateRandomData(size.shape);
    let cpuTensor: any, wasmTensor: any;

    beforeAll(async () => {
      cpuTensor = await tensor(data, { device: cpu, dtype: float32 });
      wasmTensor = await tensor(data, { device: wasmDevice, dtype: float32 });
    });

    for (const op of reductionOps) {
      bench(`CPU: ${op} ${size.name} ${size.shape.join('x')}`, async () => {
        await cpuTensor[op]();
      });

      bench(`WASM: ${op} ${size.name} ${size.shape.join('x')}`, async () => {
        await wasmTensor[op]();
      });
    }
  }
});

describe('CPU vs WASM: memory operations', () => {
  const sizes = [
    { name: '1KB', elements: 256 },
    { name: '100KB', elements: 25600 },
    { name: '1MB', elements: 262144 },
  ];

  for (const size of sizes) {
    const buffer = new ArrayBuffer(size.elements * 4);

    bench(`CPU: data lifecycle ${size.name}`, async () => {
      const data = cpu.createDataWithBuffer(buffer);
      await cpu.readData(data);
      cpu.disposeData(data);
    });

    bench(`WASM: data lifecycle ${size.name}`, async () => {
      const data = wasmDevice.createDataWithBuffer(buffer);
      await wasmDevice.readData(data);
      wasmDevice.disposeData(data);
    });
  }
});