/**
 * Tensor creation benchmarks
 *
 * Tests the performance of creating tensors of various sizes and shapes
 * using the CPU backend.
 */

import { bench, describe } from 'vitest';
import { tensor, zeros, ones, float32, int32 } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';
import { VECTOR_SIZES, MATRIX_SIZES, TENSOR_3D_SIZES } from '../utils/sizes';
import { generateRandomData, generateSequentialData, generateConstantData } from '../utils/data';

describe('tensor creation from data', () => {
  // Benchmark vector creation
  for (const size of VECTOR_SIZES) {
    const data = generateRandomData(size.shape);

    bench(`create vector ${size.name} (${size.elements} elements) - float32`, async () => {
      await tensor(data, { device: cpu, dtype: float32 });
    });

    bench(`create vector ${size.name} (${size.elements} elements) - int32`, async () => {
      await tensor(data, { device: cpu, dtype: int32 });
    });
  }

  // Benchmark matrix creation
  for (const size of MATRIX_SIZES) {
    const data = generateRandomData(size.shape);

    bench(`create matrix ${size.name} ${size.shape.join('x')} - float32`, async () => {
      await tensor(data, { device: cpu, dtype: float32 });
    });
  }

  // Benchmark 3D tensor creation
  for (const size of TENSOR_3D_SIZES) {
    const data = generateRandomData(size.shape);

    bench(`create 3D tensor ${size.name} ${size.shape.join('x')} - float32`, async () => {
      await tensor(data, { device: cpu, dtype: float32 });
    });
  }
});

describe('tensor creation with zeros/ones', () => {
  // Benchmark zeros creation
  for (const size of MATRIX_SIZES) {
    bench(`zeros ${size.name} ${size.shape.join('x')} - float32`, async () => {
      await zeros(size.shape, { device: cpu, dtype: float32 });
    });
  }

  // Benchmark ones creation
  for (const size of MATRIX_SIZES) {
    bench(`ones ${size.name} ${size.shape.join('x')} - float32`, async () => {
      await ones(size.shape, { device: cpu, dtype: float32 });
    });
  }
});

describe('tensor creation patterns', () => {
  const shape = [100, 100] as const;

  bench('create from random data', async () => {
    const data = generateRandomData(shape);
    await tensor(data, { device: cpu, dtype: float32 });
  });

  bench('create from sequential data', async () => {
    const { data } = generateSequentialData(shape);
    await tensor(data, { device: cpu, dtype: float32 });
  });

  bench('create from constant data', async () => {
    const data = generateConstantData(shape, 42);
    await tensor(data, { device: cpu, dtype: float32 });
  });

  bench('create using zeros', async () => {
    await zeros(shape, { device: cpu, dtype: float32 });
  });

  bench('create using ones', async () => {
    await ones(shape, { device: cpu, dtype: float32 });
  });
});
