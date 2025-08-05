/**
 * Integration test for reduce caching system - Phase 3
 * 
 * This test verifies that the cache is working correctly with real reduce operations.
 */

import { describe, it, expect, beforeEach } from 'bun:test';
import { reduce } from './reduce';
import { getReduceCacheStats, clearReduceCache } from './reduce-cache';
import { tensor, float32, ones } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';

// =============================================================================
// Reduce Cache Integration Test
// =============================================================================

describe('Reduce Cache Integration', () => {
  beforeEach(() => {
    // Clear cache before each test for clean state
    clearReduceCache();
  });

  it('should cache and reuse reduce recipes', async () => {
    // Create a test tensor
    const testTensor = await ones([3, 4, 5] as const, { device: cpu, dtype: float32 });
    
    // First operation - should create cache entry
    const stats1 = getReduceCacheStats();
    expect(stats1.size).toBe(0); // Cache starts empty
    
    const result1 = await reduce(testTensor, 'h w c -> h w', 'mean');
    expect(result1.shape).toEqual([3, 4]);
    
    const stats2 = getReduceCacheStats();
    expect(stats2.size).toBe(1); // One entry added
    expect(stats2.totalAccesses).toBe(1); // One access (the set)
    
    // Second operation with same pattern, operation, and shape - should use cache
    const testTensor2 = await ones([3, 4, 5] as const, { device: cpu, dtype: float32 });
    const result2 = await reduce(testTensor2, 'h w c -> h w', 'mean');
    expect(result2.shape).toEqual([3, 4]);
    
    const stats3 = getReduceCacheStats();
    expect(stats3.size).toBe(1); // Still one entry
    expect(stats3.totalAccesses).toBe(2); // Two accesses (set + get)
    expect(stats3.cacheHitRatio).toBeCloseTo(0.5); // 1 hit out of 2 total accesses
  });

  it('should create separate cache entries for different operations', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    
    // First operation
    await reduce(testTensor, 'a b c -> a b', 'mean');
    const stats1 = getReduceCacheStats();
    expect(stats1.size).toBe(1);
    
    // Different operation - should create new cache entry
    await reduce(testTensor, 'a b c -> a b', 'sum');
    const stats2 = getReduceCacheStats();
    expect(stats2.size).toBe(2);
    
    // Original operation again - should hit cache
    await reduce(testTensor, 'a b c -> a b', 'mean');
    const stats3 = getReduceCacheStats();
    expect(stats3.size).toBe(2); // No new entries
    expect(stats3.totalAccesses).toBe(3); // First entry now has 2 accesses, second has 1
  });

  it('should create separate cache entries for different keepDims', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    
    // First operation - keepDims false (default)
    await reduce(testTensor, 'a b c -> a b', 'mean');
    const stats1 = getReduceCacheStats();
    expect(stats1.size).toBe(1);
    
    // Same operation but keepDims true - should create new cache entry
    await reduce(testTensor, 'a b c -> a b', 'mean', true);
    const stats2 = getReduceCacheStats();
    expect(stats2.size).toBe(2);
  });

  it('should create separate cache entries for different shapes', async () => {
    // First shape
    const tensor1 = await ones([3, 4, 5] as const, { device: cpu, dtype: float32 });
    await reduce(tensor1, 'h w c -> h w', 'mean');
    const stats1 = getReduceCacheStats();
    expect(stats1.size).toBe(1);
    
    // Different shape, same pattern and operation - should create new cache entry
    const tensor2 = await ones([5, 6, 7] as const, { device: cpu, dtype: float32 });
    await reduce(tensor2, 'h w c -> h w', 'mean');
    const stats2 = getReduceCacheStats();
    expect(stats2.size).toBe(2);
  });

  it('should maintain correctness while using cache', async () => {
    // Create a tensor with specific values to verify correctness
    const inputArray = [
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]]
    ];
    const testTensor = await tensor(inputArray, { device: cpu, dtype: float32 });
    
    // First operation - mean reduction along last axis
    const result1 = await reduce(testTensor, 'a b c -> a b', 'mean');
    const expectedArray1 = [
      [1.5, 3.5], // (1+2)/2, (3+4)/2
      [5.5, 7.5]  // (5+6)/2, (7+8)/2
    ];
    expect(await result1.toArray()).toEqual(expectedArray1);
    
    // Second operation with same pattern - should use cache but produce same result
    const result2 = await reduce(testTensor, 'a b c -> a b', 'mean');
    expect(await result2.toArray()).toEqual(expectedArray1);
    
    // Verify cache was used
    const stats = getReduceCacheStats();
    expect(stats.cacheHitRatio).toBeGreaterThan(0);
  });

  it('should handle complex reduction patterns', async () => {
    const testTensor = await ones([2, 3, 4, 5] as const, { device: cpu, dtype: float32 });
    
    // Complex pattern - should cache correctly
    const result = await reduce(testTensor, 'a b c d -> a c', 'sum');
    expect(result.shape).toEqual([2, 4]);
    
    const stats1 = getReduceCacheStats();
    expect(stats1.size).toBe(1);
    
    // Same pattern again - should hit cache
    const result2 = await reduce(testTensor, 'a b c d -> a c', 'sum');
    expect(result2.shape).toEqual([2, 4]);
    
    const stats2 = getReduceCacheStats();
    expect(stats2.cacheHitRatio).toBeCloseTo(0.5);
  });
});