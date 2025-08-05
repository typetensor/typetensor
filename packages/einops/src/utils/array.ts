/**
 * Shared array utilities extracted from einops operations
 * These are exact duplicates that appear in multiple files
 */

/**
 * Check if two arrays are equal by value
 * Used in: rearrange.ts (line 592), reduce.ts (line 514), repeat.ts (line ~1600)
 */
export function arraysEqual(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((val, i) => val === b[i]);
}
