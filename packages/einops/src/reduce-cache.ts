/**
 * Simple recipe caching for reduce operations - Phase 3
 * 
 * This module provides a minimal caching system for reduce operation plans
 * to improve performance for repeated operations with the same pattern and parameters.
 */

import type { EinopsAST } from './ast';
import type { ResolvedPattern } from './axis-resolver';

// Import the actual ReduceOperation type from reduce.ts to avoid conflicts
import type { ReductionOp } from './reduce';

interface ReduceOperation {
  readonly type: 'reshape' | 'reduce';
  readonly params?: {
    readonly shape?: readonly number[];
    readonly axes?: readonly number[];
    readonly operation?: ReductionOp;
    readonly keepDims?: boolean;
  };
}

// =============================================================================
// Reduce Cache Types
// =============================================================================

/**
 * Cache key for reduce operations
 */
interface ReduceCacheKey {
  readonly pattern: string;
  readonly inputShape: readonly number[];
  readonly operation: string; // 'mean', 'sum', etc.
  readonly keepDims: boolean;
  readonly providedAxes?: Record<string, number> | undefined;
}

/**
 * Cached reduce recipe
 */
interface ReduceRecipe {
  readonly ast: EinopsAST;
  readonly resolved: ResolvedPattern;
  readonly operations: ReduceOperation[];
  readonly createdAt: number;
  accessCount: number; // mutable for analytics
}

// =============================================================================
// Reduce Cache Implementation
// =============================================================================

/**
 * Simple Map-based cache for reduce recipes
 */
class ReduceCache {
  private readonly cache = new Map<string, ReduceRecipe>();
  
  constructor(private readonly maxSize: number = 100) {}
  
  private createKey(key: ReduceCacheKey): string {
    const parts = [
      key.pattern,
      key.inputShape.join(','),
      key.operation,
      key.keepDims.toString(),
      key.providedAxes ? JSON.stringify(key.providedAxes) : '',
    ];
    return parts.join('|');
  }
  
  get(key: ReduceCacheKey): ReduceRecipe | undefined {
    const cacheKey = this.createKey(key);
    const recipe = this.cache.get(cacheKey);
    
    if (recipe) {
      recipe.accessCount++;
      return recipe;
    }
    
    return undefined;
  }
  
  set(key: ReduceCacheKey, recipe: Omit<ReduceRecipe, 'createdAt' | 'accessCount'>): void {
    const cacheKey = this.createKey(key);
    
    // Simple eviction: remove oldest if at capacity
    if (this.cache.size >= this.maxSize && !this.cache.has(cacheKey)) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }
    
    const fullRecipe: ReduceRecipe = {
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
 * Global cache instance for reduce operations
 */
const reduceCache = new ReduceCache();

/**
 * Get or create a cached reduce recipe
 */
export function getCachedReduceRecipe(
  pattern: string,
  inputShape: readonly number[],
  operation: string,
  keepDims: boolean,
  providedAxes?: Record<string, number>,
): ReduceRecipe | null {
  const key: ReduceCacheKey = { pattern, inputShape, operation, keepDims, providedAxes };
  return reduceCache.get(key) ?? null;
}

/**
 * Store a reduce recipe in the cache
 */
export function setCachedReduceRecipe(
  pattern: string,
  inputShape: readonly number[],
  operation: string,
  keepDims: boolean,
  ast: EinopsAST,
  resolved: ResolvedPattern,
  operations: ReduceOperation[],
  providedAxes?: Record<string, number>,
): void {
  const key: ReduceCacheKey = { pattern, inputShape, operation, keepDims, providedAxes };
  
  reduceCache.set(key, {
    ast,
    resolved,
    operations,
  });
}

/**
 * Get cache statistics for debugging and performance analysis
 */
export function getReduceCacheStats() {
  return reduceCache.getStats();
}

/**
 * Clear the reduce cache (useful for testing)
 */
export function clearReduceCache(): void {
  reduceCache.clear();
}