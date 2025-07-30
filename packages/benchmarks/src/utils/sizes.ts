/**
 * Common tensor sizes for benchmarking
 */

export interface BenchmarkSize {
  name: string;
  shape: readonly number[];
  elements: number;
}

export const VECTOR_SIZES: BenchmarkSize[] = [
  { name: 'tiny', shape: [10] as const, elements: 10 },
  { name: 'small', shape: [100] as const, elements: 100 },
  { name: 'medium', shape: [1000] as const, elements: 1000 },
  { name: 'large', shape: [10000] as const, elements: 10000 },
  { name: 'xlarge', shape: [100000] as const, elements: 100000 },
];

export const MATRIX_SIZES: BenchmarkSize[] = [
  { name: 'tiny', shape: [10, 10] as const, elements: 100 },
  { name: 'small', shape: [32, 32] as const, elements: 1024 },
  { name: 'medium', shape: [100, 100] as const, elements: 10000 },
  { name: 'large', shape: [512, 512] as const, elements: 262144 },
  { name: 'xlarge', shape: [1024, 1024] as const, elements: 1048576 },
];

export const TENSOR_3D_SIZES: BenchmarkSize[] = [
  { name: 'tiny', shape: [4, 4, 4] as const, elements: 64 },
  { name: 'small', shape: [16, 16, 16] as const, elements: 4096 },
  { name: 'medium', shape: [32, 32, 32] as const, elements: 32768 },
  { name: 'large', shape: [64, 64, 64] as const, elements: 262144 },
];

export const BATCH_SIZES: BenchmarkSize[] = [
  { name: 'small_batch', shape: [8, 224, 224, 3] as const, elements: 1204224 },
  { name: 'medium_batch', shape: [32, 224, 224, 3] as const, elements: 4816896 },
  { name: 'large_batch', shape: [128, 224, 224, 3] as const, elements: 19267584 },
];

export function formatSize(size: BenchmarkSize): string {
  return `${size.name} ${size.shape.join('x')} (${size.elements.toLocaleString()} elements)`;
}
