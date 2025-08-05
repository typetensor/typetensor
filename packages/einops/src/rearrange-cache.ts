/**
 * Simple recipe caching for rearrange operations - Phase 2
 * 
 * This module provides a minimal caching system for rearrange operation plans
 * to improve performance for repeated operations with the same pattern and shape.
 */

import type { EinopsAST } from './ast';
import type { ResolvedPattern } from './axis-resolver';

// Reuse the existing TensorOperation interface from rearrange.ts
interface TensorOperation {
  readonly type: 'reshape' | 'permute' | 'transpose' | 'identity' | 'sum';
  readonly params?: {
    readonly shape?: readonly number[];
    readonly axes?: readonly number[];
    readonly keepdims?: boolean;
  };
}

// =============================================================================
// Simple Cache Types
// =============================================================================

/**
 * Cache key for rearrange operations
 */
interface RearrangeCacheKey {
  readonly pattern: string;
  readonly inputShape: readonly number[];
  readonly providedAxes?: Record<string, number> | undefined;
}

/**
 * Cached rearrange recipe
 */
interface RearrangeRecipe {
  readonly ast: EinopsAST;
  readonly resolved: ResolvedPattern;
  readonly operations: TensorOperation[];
  readonly isIdentity: boolean;
  readonly createdAt: number;
  accessCount: number; // mutable for analytics
}

// =============================================================================
// Simple Cache Implementation
// =============================================================================

/**
 * Simple Map-based cache for rearrange recipes
 */
class RearrangeCache {
  private readonly cache = new Map<string, RearrangeRecipe>();
  
  constructor(private readonly maxSize: number = 100) {}
  
  private createKey(key: RearrangeCacheKey): string {
    const parts = [
      key.pattern,
      key.inputShape.join(','),
      key.providedAxes ? JSON.stringify(key.providedAxes) : '',
    ];
    return parts.join('|');
  }
  
  get(key: RearrangeCacheKey): RearrangeRecipe | undefined {
    const cacheKey = this.createKey(key);
    const recipe = this.cache.get(cacheKey);
    
    if (recipe) {
      recipe.accessCount++;
      return recipe;
    }
    
    return undefined;
  }
  
  set(key: RearrangeCacheKey, recipe: Omit<RearrangeRecipe, 'createdAt' | 'accessCount'>): void {
    const cacheKey = this.createKey(key);
    
    // Simple eviction: remove oldest if at capacity
    if (this.cache.size >= this.maxSize && !this.cache.has(cacheKey)) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }
    
    const fullRecipe: RearrangeRecipe = {
      ...recipe,
      createdAt: Date.now(),
      accessCount: 1,
    };
    
    this.cache.set(cacheKey, fullRecipe);
  }
  
  clear(): void {
    this.cache.clear();
  }
  
  // Simple analytics
  getStats() {
    const recipes = Array.from(this.cache.values());
    return {
      size: this.cache.size,
      totalAccesses: recipes.reduce((sum, r) => sum + r.accessCount, 0),
      cacheHitRatio: recipes.length > 0 ? 
        recipes.reduce((sum, r) => sum + (r.accessCount - 1), 0) / recipes.reduce((sum, r) => sum + r.accessCount, 0) : 0,
    };
  }
}

// =============================================================================
// Global Cache Instance
// =============================================================================

/**
 * Global cache instance for rearrange operations
 */
const rearrangeCache = new RearrangeCache();

/**
 * Get or create a cached rearrange recipe
 */
export function getCachedRearrangeRecipe(
  pattern: string,
  inputShape: readonly number[],
  providedAxes?: Record<string, number>,
): RearrangeRecipe | null {
  const key: RearrangeCacheKey = { pattern, inputShape, providedAxes };
  return rearrangeCache.get(key) ?? null;
}

/**
 * Store a rearrange recipe in the cache
 */
export function setCachedRearrangeRecipe(
  pattern: string,
  inputShape: readonly number[],
  ast: EinopsAST,
  resolved: ResolvedPattern,
  operations: TensorOperation[],
  providedAxes?: Record<string, number>,
): void {
  const key: RearrangeCacheKey = { pattern, inputShape, providedAxes };
  const isIdentity = operations.length === 1 && operations[0]?.type === 'identity';
  
  rearrangeCache.set(key, {
    ast,
    resolved,
    operations,
    isIdentity,
  });
}

/**
 * Get cache statistics for debugging and performance analysis
 */
export function getRearrangeCacheStats() {
  return rearrangeCache.getStats();
}

/**
 * Clear the rearrange cache (useful for testing)
 */
export function clearRearrangeCache(): void {
  rearrangeCache.clear();
}