/**
 * WASM backend tensor creation benchmarks
 *
 * Tests the performance of creating tensors using the WASM backend
 * with various sizes, shapes, and data types.
 */

import { bench, describe } from 'vitest';
import { tensor, zeros, ones, float32, int32 } from '@typetensor/core';
import { WASMDevice } from '@typetensor/backend-wasm';
import { VECTOR_SIZES, MATRIX_SIZES, TENSOR_3D_SIZES } from '../utils/sizes';
import { generateRandomData, generateSequentialData, generateConstantData } from '../utils/data';

// Initialize WASM device once for all tests
let wasmDevice: WASMDevice;

beforeAll(async () => {
  wasmDevice = await WASMDevice.create();
});

describe('WASM tensor creation from data', () => {
  // Benchmark vector creation
  for (const size of VECTOR_SIZES) {
    const data = generateRandomData(size.shape);

    bench(`WASM: create vector ${size.name} (${size.elements} elements) - float32`, async () => {
      await tensor(data, { device: wasmDevice, dtype: float32 });
    });

    bench(`WASM: create vector ${size.name} (${size.elements} elements) - int32`, async () => {
      await tensor(data, { device: wasmDevice, dtype: int32 });
    });
  }

  // Benchmark matrix creation
  for (const size of MATRIX_SIZES) {
    const data = generateRandomData(size.shape);

    bench(`WASM: create matrix ${size.name} ${size.shape.join('x')} - float32`, async () => {
      await tensor(data, { device: wasmDevice, dtype: float32 });
    });
  }

  // Benchmark 3D tensor creation
  for (const size of TENSOR_3D_SIZES) {
    const data = generateRandomData(size.shape);

    bench(`WASM: create 3D tensor ${size.name} ${size.shape.join('x')} - float32`, async () => {
      await tensor(data, { device: wasmDevice, dtype: float32 });
    });
  }
});

describe('WASM tensor creation with zeros/ones', () => {
  // Benchmark zeros creation
  for (const size of MATRIX_SIZES) {
    bench(`WASM: zeros ${size.name} ${size.shape.join('x')} - float32`, async () => {
      await zeros(size.shape, { device: wasmDevice, dtype: float32 });
    });
  }

  // Benchmark ones creation
  for (const size of MATRIX_SIZES) {
    bench(`WASM: ones ${size.name} ${size.shape.join('x')} - float32`, async () => {
      await ones(size.shape, { device: wasmDevice, dtype: float32 });
    });
  }
});

describe('WASM tensor creation patterns', () => {
  const shape = [100, 100] as const;

  bench('WASM: create from random data', async () => {
    const data = generateRandomData(shape);
    await tensor(data, { device: wasmDevice, dtype: float32 });
  });

  bench('WASM: create from sequential data', async () => {
    const { data } = generateSequentialData(shape);
    await tensor(data, { device: wasmDevice, dtype: float32 });
  });

  bench('WASM: create from constant data', async () => {
    const data = generateConstantData(shape, 42);
    await tensor(data, { device: wasmDevice, dtype: float32 });
  });

  bench('WASM: create using zeros', async () => {
    await zeros(shape, { device: wasmDevice, dtype: float32 });
  });

  bench('WASM: create using ones', async () => {
    await ones(shape, { device: wasmDevice, dtype: float32 });
  });
});

describe('WASM memory transfer benchmarks', () => {
  const transferSizes = [
    { name: '1KB', shape: [256] as const }, // 256 * 4 bytes = 1KB
    { name: '100KB', shape: [25600] as const }, // ~100KB
    { name: '1MB', shape: [262144] as const }, // ~1MB
  ];

  for (const size of transferSizes) {
    const data = generateRandomData(size.shape);

    bench(`WASM: create and read ${size.name}`, async () => {
      const t = await tensor(data, { device: wasmDevice, dtype: float32 });
      await wasmDevice.readData(t.data);
    });

    bench(`WASM: write data ${size.name}`, async () => {
      const buffer = new ArrayBuffer(size.shape[0] * 4);
      const deviceData = wasmDevice.createData(buffer.byteLength);
      await wasmDevice.writeData(deviceData, buffer);
    });
  }
});