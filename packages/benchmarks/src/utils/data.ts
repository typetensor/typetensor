/**
 * Data generation utilities for benchmarks
 */

export type NestedArray = number | readonly NestedArray[];

/**
 * Generate random data for a given shape
 * OPTIMIZED: Generate flat data directly instead of nested arrays
 */
export function generateRandomData(shape: readonly number[]): NestedArray {
  const totalElements = shape.reduce((a, b) => a * b, 1);

  // For single element, return scalar
  if (totalElements === 1) {
    return Math.random();
  }

  // Generate flat array first (much faster)
  const flatData = new Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    flatData[i] = Math.random();
  }

  // Convert to nested structure (only if needed for compatibility)
  return reshapeFlat(flatData, shape);
}

/**
 * Convert flat array to nested array based on shape
 */
function reshapeFlat(flatData: number[], shape: readonly number[]): NestedArray {
  if (shape.length === 0) {
    return flatData[0];
  }

  if (shape.length === 1) {
    return flatData;
  }

  const [first, ...rest] = shape;
  if (first === undefined) {
    throw new Error('Invalid shape');
  }

  const elementsPerSlice = rest.reduce((a, b) => a * b, 1);
  const result: NestedArray[] = [];

  for (let i = 0; i < first; i++) {
    const startIdx = i * elementsPerSlice;
    const slice = flatData.slice(startIdx, startIdx + elementsPerSlice);
    result.push(reshapeFlat(slice, rest));
  }

  return result;
}

/**
 * Generate sequential data for a given shape
 */
export function generateSequentialData(
  shape: readonly number[],
  start = 0,
): { data: NestedArray; nextValue: number } {
  let counter = start;

  function generate(dims: readonly number[]): NestedArray {
    if (dims.length === 0) {
      return counter++;
    }

    const [first, ...rest] = dims;
    if (first === undefined) {
      throw new Error('Invalid shape');
    }

    return Array.from({ length: first }, () => generate(rest)) as NestedArray;
  }

  return { data: generate(shape), nextValue: counter };
}

/**
 * Generate data filled with a constant value
 */
export function generateConstantData(shape: readonly number[], value: number): NestedArray {
  if (shape.length === 0) {
    return value;
  }

  const [first, ...rest] = shape;
  if (first === undefined) {
    throw new Error('Invalid shape');
  }

  return Array.from({ length: first }, () => generateConstantData(rest, value)) as NestedArray;
}
