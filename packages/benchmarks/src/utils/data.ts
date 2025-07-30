/**
 * Data generation utilities for benchmarks
 */

export type NestedArray = number | readonly NestedArray[];

/**
 * Generate random data for a given shape
 */
export function generateRandomData(shape: readonly number[]): NestedArray {
  if (shape.length === 0) {
    return Math.random();
  }
  
  const [first, ...rest] = shape;
  if (first === undefined) {
    throw new Error('Invalid shape');
  }
  
  return Array.from({ length: first }, () => generateRandomData(rest)) as NestedArray;
}

/**
 * Generate sequential data for a given shape
 */
export function generateSequentialData(shape: readonly number[], start = 0): { data: NestedArray; nextValue: number } {
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