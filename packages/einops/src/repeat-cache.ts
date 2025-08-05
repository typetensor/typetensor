/**
 * Simple recipe caching for repeat operations - Phase 4
 * 
 * This module provides a minimal caching system for repeat operation plans
 * to improve performance for repeated operations with the same pattern and parameters.
 */

import type { EinopsAST } from './ast';
import type { ResolvedPattern } from './axis-resolver';

// Import from repeat.ts to avoid type conflicts
interface RepeatOperation {
  readonly type:
    | 'reshape'
    | 'expand'
    | 'tile'
    | 'identity'
    | 'repeat_with_new_axes'
    | 'composite_repetition'
    | 'create_with_coordinate_mapping';
  readonly params?: {
    readonly shape?: readonly number[];
    readonly reps?: readonly number[];
    readonly targetShape?: readonly number[];
    readonly newAxes?: readonly string[];
    readonly inputPatterns?: readonly import('./ast').AxisPattern[];
    readonly outputPatterns?: readonly import('./ast').AxisPattern[];
    readonly axisDimensions?: Map<string, number>;
  };
}

// =============================================================================
// Repeat Cache Types
// =============================================================================

/**
 * Cache key for repeat operations
 */
interface RepeatCacheKey {
  readonly pattern: string;
  readonly inputShape: readonly number[];
  readonly providedAxes?: Record<string, number> | undefined;
}

/**
 * Cached repeat recipe
 */
interface RepeatRecipe {
  readonly ast: EinopsAST;
  readonly resolved: ResolvedPattern;
  readonly operations: RepeatOperation[];
  readonly createdAt: number;
  accessCount: number; // mutable for analytics
}

// =============================================================================
// Repeat Cache Implementation
// =============================================================================

/**
 * Simple Map-based cache for repeat recipes
 */
class RepeatCache {
  private readonly cache = new Map<string, RepeatRecipe>();
  
  constructor(private readonly maxSize: number = 100) {}
  
  private createKey(key: RepeatCacheKey): string {
    const parts = [
      key.pattern,
      key.inputShape.join(','),
      key.providedAxes ? JSON.stringify(key.providedAxes) : '',
    ];
    return parts.join('|');
  }
  
  get(key: RepeatCacheKey): RepeatRecipe | undefined {
    const cacheKey = this.createKey(key);
    const recipe = this.cache.get(cacheKey);
    
    if (recipe) {
      recipe.accessCount++;
      return recipe;
    }
    
    return undefined;
  }
  
  set(key: RepeatCacheKey, recipe: Omit<RepeatRecipe, 'createdAt' | 'accessCount'>): void {
    const cacheKey = this.createKey(key);
    
    // Simple eviction: remove oldest if at capacity
    if (this.cache.size >= this.maxSize && !this.cache.has(cacheKey)) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey) {
        this.cache.delete(oldestKey);
      }
    }
    
    const fullRecipe: RepeatRecipe = {
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
 * Global cache instance for repeat operations
 */
const repeatCache = new RepeatCache();

/**
 * Get or create a cached repeat recipe
 */
export function getCachedRepeatRecipe(
  pattern: string,
  inputShape: readonly number[],
  providedAxes?: Record<string, number>,
): RepeatRecipe | null {
  const key: RepeatCacheKey = { pattern, inputShape, providedAxes };
  return repeatCache.get(key) ?? null;
}

/**
 * Store a repeat recipe in the cache
 */
export function setCachedRepeatRecipe(
  pattern: string,
  inputShape: readonly number[],
  ast: EinopsAST,
  resolved: ResolvedPattern,
  operations: RepeatOperation[],
  providedAxes?: Record<string, number>,
): void {
  const key: RepeatCacheKey = { pattern, inputShape, providedAxes };
  
  repeatCache.set(key, {
    ast,
    resolved,
    operations,
  });
}

/**
 * Get cache statistics for debugging and performance analysis
 */
export function getRepeatCacheStats() {
  return repeatCache.getStats();
}

/**
 * Clear the repeat cache (useful for testing)
 */
export function clearRepeatCache(): void {
  repeatCache.clear();
}