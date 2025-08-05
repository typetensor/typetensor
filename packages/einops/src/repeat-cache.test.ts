/**
 * Integration test for repeat caching system - Phase 4
 * 
 * This test verifies that the cache is working correctly with real repeat operations.
 */

import { describe, it, expect, beforeEach } from 'bun:test';
import { repeat } from './repeat';
import { getRepeatCacheStats, clearRepeatCache } from './repeat-cache';
import { tensor, float32, ones } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';

// =============================================================================
// Repeat Cache Integration Test
// =============================================================================

describe('Repeat Cache Integration', () => {
  beforeEach(() => {
    // Clear cache before each test for clean state
    clearRepeatCache();
  });

  it('should cache and reuse repeat recipes', async () => {
    // Create a test tensor
    const testTensor = await ones([3, 4] as const, { device: cpu, dtype: float32 });
    
    // First operation - should create cache entry
    const stats1 = getRepeatCacheStats();
    expect(stats1.size).toBe(0); // Cache starts empty
    
    const result1 = await repeat(testTensor, 'h w -> h w c', { c: 2 });
    expect(result1.shape).toEqual([3, 4, 2]);
    
    const stats2 = getRepeatCacheStats();
    expect(stats2.size).toBe(1); // One entry added
    expect(stats2.totalAccesses).toBe(1); // One access (the set)
    
    // Second operation with same pattern and shape - should use cache
    const testTensor2 = await ones([3, 4] as const, { device: cpu, dtype: float32 });
    const result2 = await repeat(testTensor2, 'h w -> h w c', { c: 2 });
    expect(result2.shape).toEqual([3, 4, 2]);
    
    const stats3 = getRepeatCacheStats();
    expect(stats3.size).toBe(1); // Still one entry
    expect(stats3.totalAccesses).toBe(2); // Two accesses (set + get)
    expect(stats3.cacheHitRatio).toBeCloseTo(0.5); // 1 hit out of 2 total accesses
  });

  it('should create separate cache entries for different axes', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    
    // First operation
    await repeat(testTensor, 'h w -> h w c', { c: 2 });
    const stats1 = getRepeatCacheStats();
    expect(stats1.size).toBe(1);
    
    // Different axes - should create new cache entry
    await repeat(testTensor, 'h w -> h w c', { c: 3 });
    const stats2 = getRepeatCacheStats();
    expect(stats2.size).toBe(2);
    
    // Original axes again - should hit cache
    await repeat(testTensor, 'h w -> h w c', { c: 2 });
    const stats3 = getRepeatCacheStats();
    expect(stats3.size).toBe(2); // No new entries
    expect(stats3.totalAccesses).toBe(3); // First entry now has 2 accesses, second has 1
  });

  it('should create separate cache entries for different patterns', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    
    // First pattern
    await repeat(testTensor, 'h w -> h w c', { c: 2 });
    const stats1 = getRepeatCacheStats();
    expect(stats1.size).toBe(1);
    
    // Different pattern - should create new cache entry
    await repeat(testTensor, 'h w -> batch h w', { batch: 4 });
    const stats2 = getRepeatCacheStats();
    expect(stats2.size).toBe(2);
    
    // Original pattern again - should hit cache
    await repeat(testTensor, 'h w -> h w c', { c: 2 });
    const stats3 = getRepeatCacheStats();
    expect(stats3.size).toBe(2); // No new entries
    expect(stats3.totalAccesses).toBe(3); // First entry now has 2 accesses, second has 1
  });

  it('should create separate cache entries for different shapes', async () => {
    // First shape
    const tensor1 = await ones([3, 4] as const, { device: cpu, dtype: float32 });
    await repeat(tensor1, 'h w -> h w c', { c: 2 });
    const stats1 = getRepeatCacheStats();
    expect(stats1.size).toBe(1);
    
    // Different shape, same pattern and axes - should create new cache entry
    const tensor2 = await ones([5, 6] as const, { device: cpu, dtype: float32 });
    await repeat(tensor2, 'h w -> h w c', { c: 2 });
    const stats2 = getRepeatCacheStats();
    expect(stats2.size).toBe(2);
  });

  it('should maintain correctness while using cache', async () => {
    // Create a tensor with specific values to verify correctness
    const inputArray = [[1, 2], [3, 4]];
    const testTensor = await tensor(inputArray, { device: cpu, dtype: float32 });
    
    // First operation - repeat along new axis
    const result1 = await repeat(testTensor, 'h w -> h w c', { c: 3 });
    const expectedArray1: number[][][] = [
      [[1, 1, 1], [2, 2, 2]],
      [[3, 3, 3], [4, 4, 4]]
    ];
    expect(await result1.toArray()).toEqual(expectedArray1 as any);
    
    // Second operation with same pattern - should use cache but produce same result
    const result2 = await repeat(testTensor, 'h w -> h w c', { c: 3 });
    expect(await result2.toArray()).toEqual(expectedArray1 as any);
    
    // Verify cache was used
    const stats = getRepeatCacheStats();
    expect(stats.cacheHitRatio).toBeGreaterThan(0);
  });

  it('should handle complex repetition patterns', async () => {
    const testTensor = await ones([2, 3] as const, { device: cpu, dtype: float32 });
    
    // Complex pattern with multiple new axes - should cache correctly
    const result = await repeat(testTensor, 'h w -> batch h w c', { batch: 2, c: 4 });
    expect(result.shape).toEqual([2, 2, 3, 4]);
    
    const stats1 = getRepeatCacheStats();
    expect(stats1.size).toBe(1);
    
    // Same pattern again - should hit cache
    const result2 = await repeat(testTensor, 'h w -> batch h w c', { batch: 2, c: 4 });
    expect(result2.shape).toEqual([2, 2, 3, 4]);
    
    const stats2 = getRepeatCacheStats();
    expect(stats2.cacheHitRatio).toBeCloseTo(0.5);
  });
});