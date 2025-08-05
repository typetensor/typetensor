/**
 * Integration test for rearrange caching system - Phase 2
 * 
 * This test verifies that the cache is working correctly with real tensor operations.
 */

import { describe, it, expect, beforeEach } from 'bun:test';
import { rearrange } from './rearrange';
import { getRearrangeCacheStats, clearRearrangeCache } from './rearrange-cache';
import { tensor, float32, ones } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';

// =============================================================================
// Cache Integration Test
// =============================================================================

describe('Rearrange Cache Integration', () => {
  beforeEach(() => {
    // Clear cache before each test for clean state
    clearRearrangeCache();
  });

  it('should cache and reuse rearrange recipes', async () => {
    // Create a test tensor
    const testTensor = await ones([3, 4] as const, { device: cpu, dtype: float32 });
    
    // First operation - should create cache entry
    const stats1 = getRearrangeCacheStats();
    expect(stats1.size).toBe(0); // Cache starts empty
    
    const result1 = await rearrange(testTensor, 'h w -> w h');
    expect(result1.shape).toEqual([4, 3]);
    
    const stats2 = getRearrangeCacheStats();
    expect(stats2.size).toBe(1); // One entry added
    expect(stats2.totalAccesses).toBe(1); // One access (the set)
    
    // Second operation with same pattern and shape - should use cache
    const testTensor2 = await ones([3, 4] as const, { device: cpu, dtype: float32 });
    const result2 = await rearrange(testTensor2, 'h w -> w h');
    expect(result2.shape).toEqual([4, 3]);
    
    const stats3 = getRearrangeCacheStats();
    expect(stats3.size).toBe(1); // Still one entry
    expect(stats3.totalAccesses).toBe(2); // Two accesses (set + get)
    expect(stats3.cacheHitRatio).toBeCloseTo(0.5); // 1 hit out of 2 total accesses
    
    // Third operation - should hit cache again
    const testTensor3 = await ones([3, 4] as const, { device: cpu, dtype: float32 });
    const result3 = await rearrange(testTensor3, 'h w -> w h');
    expect(result3.shape).toEqual([4, 3]);
    
    const stats4 = getRearrangeCacheStats();
    expect(stats4.size).toBe(1); // Still one entry
    expect(stats4.totalAccesses).toBe(3); // Three accesses
    expect(stats4.cacheHitRatio).toBeCloseTo(0.67, 0.01); // 2 hits out of 3 total accesses
  });

  it('should create separate cache entries for different patterns', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    
    // First pattern
    await rearrange(testTensor, 'a b c -> c a b');
    const stats1 = getRearrangeCacheStats();
    expect(stats1.size).toBe(1);
    
    // Different pattern - should create new cache entry
    await rearrange(testTensor, 'a b c -> a c b');
    const stats2 = getRearrangeCacheStats();
    expect(stats2.size).toBe(2);
    
    // Original pattern again - should hit cache
    await rearrange(testTensor, 'a b c -> c a b');
    const stats3 = getRearrangeCacheStats();
    expect(stats3.size).toBe(2); // No new entries
    expect(stats3.totalAccesses).toBe(3); // First entry now has 2 accesses, second has 1
  });

  it('should create separate cache entries for different shapes', async () => {
    // First shape
    const tensor1 = await ones([3, 4] as const, { device: cpu, dtype: float32 });
    await rearrange(tensor1, 'h w -> w h');
    const stats1 = getRearrangeCacheStats();
    expect(stats1.size).toBe(1);
    
    // Different shape, same pattern - should create new cache entry
    const tensor2 = await ones([5, 6] as const, { device: cpu, dtype: float32 });
    await rearrange(tensor2, 'h w -> w h');
    const stats2 = getRearrangeCacheStats();
    expect(stats2.size).toBe(2);
  });

  it('should handle identity patterns correctly', async () => {
    const testTensor = await ones([2, 3, 4] as const, { device: cpu, dtype: float32 });
    
    // Identity pattern - should still cache
    const result = await rearrange(testTensor, 'a b c -> a b c');
    expect(result.shape).toEqual([2, 3, 4]);
    
    const stats = getRearrangeCacheStats();
    expect(stats.size).toBe(1);
    expect(stats.totalAccesses).toBe(1);
    
    // Second identity operation - should hit cache
    const result2 = await rearrange(testTensor, 'a b c -> a b c');
    expect(result2.shape).toEqual([2, 3, 4]);
    
    const stats2 = getRearrangeCacheStats();
    expect(stats2.size).toBe(1);
    expect(stats2.totalAccesses).toBe(2);
  });

  it('should maintain correctness while using cache', async () => {
    // Create a tensor with specific values to verify correctness
    const inputArray = [
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]]
    ];
    const testTensor = await tensor(inputArray, { device: cpu, dtype: float32 });
    
    // First operation
    const result1 = await rearrange(testTensor, 'a b c -> b a c');
    const expectedArray1 = [
      [[1, 2], [5, 6]],
      [[3, 4], [7, 8]]
    ];
    expect(await result1.toArray()).toEqual(expectedArray1);
    
    // Second operation with same pattern - should use cache but produce same result
    const result2 = await rearrange(testTensor, 'a b c -> b a c');
    expect(await result2.toArray()).toEqual(expectedArray1);
    
    // Verify cache was used
    const stats = getRearrangeCacheStats();
    expect(stats.cacheHitRatio).toBeGreaterThan(0);
  });
});