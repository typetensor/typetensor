/**
 * AST (Abstract Syntax Tree) types for einops pattern parsing
 *
 * This module defines the structural representation of parsed einops patterns,
 * providing the foundation for pattern validation and transformation planning.
 */

import type { Position } from './types';

// =============================================================================
// Core AST Pattern Types
// =============================================================================

/**
 * A simple axis representing a single tensor dimension
 * Examples: "batch", "height", "channels"
 */
export interface SimpleAxis {
  readonly type: 'simple';
  readonly name: string;
  readonly position: Position;
}

/**
 * A composite axis representing grouped dimensions
 * Examples: "(h w)", "(batch seq)", "((a b) c)"
 */
export interface CompositeAxis {
  readonly type: 'composite';
  readonly axes: readonly AxisPattern[];
  readonly position: Position;
}

/**
 * An ellipsis axis representing variable dimensions
 * Examples: "..." in "batch ... -> ..."
 */
export interface EllipsisAxis {
  readonly type: 'ellipsis';
  readonly position: Position;
}

/**
 * A singleton axis representing unit dimensions
 * Examples: "1" in "h w 1 -> h w"
 */
export interface SingletonAxis {
  readonly type: 'singleton';
  readonly position: Position;
}

/**
 * Union type representing all possible axis patterns in einops
 */
export type AxisPattern = SimpleAxis | CompositeAxis | EllipsisAxis | SingletonAxis;

// =============================================================================
// Main AST Structure
// =============================================================================

/**
 * Metadata about the parsed einops pattern
 */
export interface ASTMetadata {
  /** Original pattern string */
  readonly originalPattern: string;
  /** Position of the arrow operator */
  readonly arrowPosition: Position;
  /** Number of tokens in input side */
  readonly inputTokenCount: number;
  /** Number of tokens in output side */
  readonly outputTokenCount: number;
}

/**
 * Complete AST representation of an einops pattern
 *
 * Represents a fully parsed pattern like "batch (h w) -> batch h w"
 */
export interface EinopsAST {
  /** Input side axis patterns */
  readonly input: readonly AxisPattern[];
  /** Output side axis patterns */
  readonly output: readonly AxisPattern[];
  /** Additional metadata about the pattern */
  readonly metadata: ASTMetadata;
}

// =============================================================================
// Type Guards
// =============================================================================

/**
 * Type guard to check if a pattern is a SimpleAxis
 */
export function isSimpleAxis(pattern: AxisPattern): pattern is SimpleAxis {
  return pattern.type === 'simple';
}

/**
 * Type guard to check if a pattern is a CompositeAxis
 */
export function isCompositeAxis(pattern: AxisPattern): pattern is CompositeAxis {
  return pattern.type === 'composite';
}

/**
 * Type guard to check if a pattern is an EllipsisAxis
 */
export function isEllipsisAxis(pattern: AxisPattern): pattern is EllipsisAxis {
  return pattern.type === 'ellipsis';
}

/**
 * Type guard to check if a pattern is a SingletonAxis
 */
export function isSingletonAxis(pattern: AxisPattern): pattern is SingletonAxis {
  return pattern.type === 'singleton';
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Extract all axis names from a list of patterns
 * Only includes names from SimpleAxis patterns
 */
export function getAxisNames(patterns: readonly AxisPattern[]): string[] {
  const names: string[] = [];

  for (const pattern of patterns) {
    if (isSimpleAxis(pattern)) {
      names.push(pattern.name);
    } else if (isCompositeAxis(pattern)) {
      names.push(...getAxisNames(pattern.axes));
    }
  }

  return names;
}

/**
 * Check if any pattern in the list is an ellipsis
 */
export function hasEllipsis(patterns: readonly AxisPattern[]): boolean {
  return patterns.some((pattern) => {
    if (isEllipsisAxis(pattern)) {
      return true;
    }
    if (isCompositeAxis(pattern)) {
      return hasEllipsis(pattern.axes);
    }
    return false;
  });
}

/**
 * Get the maximum nesting depth of composite axes
 * Returns 1 for simple composite, 2+ for nested composites
 */
export function getCompositeDepth(pattern: CompositeAxis): number {
  let maxChildDepth = 0;

  for (const axis of pattern.axes) {
    if (isCompositeAxis(axis)) {
      maxChildDepth = Math.max(maxChildDepth, getCompositeDepth(axis));
    }
  }

  return maxChildDepth + 1;
}

/**
 * Count the total number of simple axes (recursively through composites)
 */
export function countSimpleAxes(patterns: readonly AxisPattern[]): number {
  let count = 0;

  for (const pattern of patterns) {
    if (isSimpleAxis(pattern)) {
      count++;
    } else if (isCompositeAxis(pattern)) {
      count += countSimpleAxes(pattern.axes);
    }
  }

  return count;
}

/**
 * Get all unique axis names from patterns (no duplicates)
 */
export function getUniqueAxisNames(patterns: readonly AxisPattern[]): string[] {
  const names = getAxisNames(patterns);
  return [...new Set(names)];
}

/**
 * Check if patterns contain any singleton axes
 */
export function hasSingleton(patterns: readonly AxisPattern[]): boolean {
  return patterns.some((pattern) => {
    if (isSingletonAxis(pattern)) {
      return true;
    }
    if (isCompositeAxis(pattern)) {
      return hasSingleton(pattern.axes);
    }
    return false;
  });
}
