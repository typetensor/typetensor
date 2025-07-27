/**
 * Axis resolution system for einops patterns
 *
 * This module handles the mapping of axis names to concrete dimensions,
 * resolving composite patterns, handling ellipsis, and computing output shapes.
 */

import type { EinopsAST, AxisPattern, SimpleAxis, CompositeAxis } from './ast';
import { isSimpleAxis, isCompositeAxis, isEllipsisAxis, isSingletonAxis } from './ast';

// =============================================================================
// Types
// =============================================================================

/**
 * Mapping from axis names to their dimension sizes
 */
export type AxisMapping = Map<string, number>;

/**
 * Result of resolving a pattern against a tensor shape
 */
export interface ResolvedPattern {
  /**
   * Mapping from axis names to dimension sizes
   */
  readonly axisDimensions: AxisMapping;

  /**
   * The computed output shape
   */
  readonly outputShape: readonly number[];

  /**
   * Dimensions consumed by ellipsis (if present)
   */
  readonly ellipsisDimensions?: readonly number[] | undefined;
}

/**
 * Error thrown during axis resolution
 */
export class AxisResolutionError extends Error {
  constructor(
    message: string,
    public readonly pattern: string,
    public readonly context?: {
      inputShape?: readonly number[];
      providedAxes?: Record<string, number>;
      axis?: string;
    },
  ) {
    super(message);
    this.name = 'AxisResolutionError';
  }
}

// =============================================================================
// Main Resolver Class
// =============================================================================

/**
 * Resolves einops patterns to concrete dimensions
 */
export class AxisResolver {
  private originalPattern: string = '';

  /**
   * Resolve a parsed einops pattern against a tensor shape
   */
  resolvePattern(
    ast: EinopsAST,
    inputShape: readonly number[],
    providedAxes?: Record<string, number>,
  ): ResolvedPattern {
    // Store original pattern for error messages
    this.originalPattern = ast.metadata?.originalPattern || '<unknown pattern>';

    // Step 1: Build axis dimension map from input pattern
    const { axisDimensions, ellipsisDimensions } = this.resolveInputPattern(
      ast.input,
      inputShape,
      providedAxes,
    );

    // Step 2: Validate all output axes are known
    this.validateOutputAxes(ast.output, axisDimensions);

    // Step 3: Compute output shape
    const outputShape = this.computeOutputShape(ast.output, axisDimensions, ellipsisDimensions);

    return {
      axisDimensions,
      outputShape,
      ellipsisDimensions,
    };
  }

  /**
   * Resolve input pattern to axis dimensions
   */
  private resolveInputPattern(
    patterns: readonly AxisPattern[],
    shape: readonly number[],
    providedAxes?: Record<string, number>,
  ): { axisDimensions: AxisMapping; ellipsisDimensions?: readonly number[] } {
    const axisDimensions = new Map<string, number>();
    let shapeIndex = 0;
    let ellipsisDimensions: number[] | undefined;

    // Handle empty pattern (scalar)
    if (patterns.length === 0) {
      if (shape.length !== 0) {
        throw new AxisResolutionError(
          `Empty pattern expects scalar (0-dimensional) tensor but got shape [${shape.join(', ')}]`,
          this.originalPattern,
          { inputShape: shape },
        );
      }
      return { axisDimensions };
    }

    for (let i = 0; i < patterns.length; i++) {
      const pattern = patterns[i];
      if (!pattern) continue; // Skip undefined patterns

      if (isSimpleAxis(pattern)) {
        // Simple axis: direct mapping
        if (shapeIndex >= shape.length) {
          throw new AxisResolutionError(
            `Pattern has more axes than tensor dimensions`,
            this.originalPattern,
            { inputShape: shape },
          );
        }

        const actualDim = shape[shapeIndex];
        shapeIndex++; // Always increment

        // If a value was provided, verify it matches
        if (providedAxes?.[pattern.name] !== undefined) {
          const expectedDim = providedAxes[pattern.name];
          if (actualDim !== expectedDim) {
            throw new AxisResolutionError(
              `Axis '${pattern.name}' expected ${expectedDim} but got ${actualDim}`,
              this.originalPattern,
              { inputShape: shape, providedAxes },
            );
          }
        }

        if (actualDim !== undefined) {
          axisDimensions.set(pattern.name, actualDim);
        }
      } else if (isCompositeAxis(pattern)) {
        // Composite axis: resolve nested axes
        if (shapeIndex >= shape.length) {
          throw new AxisResolutionError(
            `Pattern has more axes than tensor dimensions`,
            this.originalPattern,
            { inputShape: shape },
          );
        }
        const compositeDim = shape[shapeIndex];
        if (compositeDim !== undefined) {
          shapeIndex++;
          this.resolveCompositeAxis(pattern, compositeDim, axisDimensions, providedAxes);
        }
      } else if (isEllipsisAxis(pattern)) {
        // Ellipsis: consume remaining dimensions
        const remainingPatterns = patterns.length - i - 1;
        const remainingDims = shape.length - shapeIndex;

        if (remainingDims < remainingPatterns) {
          throw new AxisResolutionError(
            `Not enough dimensions for pattern after ellipsis`,
            this.originalPattern,
            { inputShape: shape },
          );
        }

        const ellipsisCount = remainingDims - remainingPatterns;
        ellipsisDimensions = [];
        for (let j = 0; j < ellipsisCount; j++) {
          const dim = shape[shapeIndex + j];
          if (dim !== undefined) {
            ellipsisDimensions.push(dim);
          }
        }
        shapeIndex += ellipsisCount;
      } else if (isSingletonAxis(pattern)) {
        // Singleton: verify dimension is 1
        if (shapeIndex >= shape.length) {
          throw new AxisResolutionError(
            `Pattern has more axes than tensor dimensions`,
            this.originalPattern,
            { inputShape: shape },
          );
        }
        const dim = shape[shapeIndex];
        if (dim !== 1) {
          throw new AxisResolutionError(
            `Expected singleton dimension but got ${dim}`,
            this.originalPattern,
            { inputShape: shape },
          );
        }
        shapeIndex++;
      }
    }

    // Verify all dimensions were consumed
    if (shapeIndex < shape.length) {
      throw new AxisResolutionError(
        `Pattern does not consume all tensor dimensions`,
        this.originalPattern,
        { inputShape: shape },
      );
    }

    if (ellipsisDimensions !== undefined) {
      return { axisDimensions, ellipsisDimensions };
    }
    return { axisDimensions };
  }

  /**
   * Flatten a composite axis to extract all simple axes
   */
  private flattenComposite(composite: CompositeAxis): SimpleAxis[] {
    const result: SimpleAxis[] = [];

    for (const axis of composite.axes) {
      if (isSimpleAxis(axis)) {
        result.push(axis);
      } else if (isCompositeAxis(axis)) {
        // Recursively flatten nested composites
        result.push(...this.flattenComposite(axis));
      } else {
        throw new AxisResolutionError(
          `Composite patterns can only contain simple axes or other composites`,
          this.originalPattern,
          {},
        );
      }
    }

    return result;
  }

  /**
   * Resolve a composite axis pattern
   */
  private resolveCompositeAxis(
    composite: CompositeAxis,
    totalDim: number,
    axisDimensions: AxisMapping,
    providedAxes?: Record<string, number>,
  ): void {
    // Flatten nested composites to get all simple axes
    const simpleAxes = this.flattenComposite(composite);
    const innerAxes: Array<{ axis: SimpleAxis; value?: number }> = [];
    let knownProduct = 1;
    let unknownCount = 0;

    // Collect known values for all simple axes
    for (const axis of simpleAxes) {
      const value = providedAxes?.[axis.name] ?? axisDimensions.get(axis.name);
      if (value !== undefined) {
        innerAxes.push({ axis, value });
        knownProduct *= value;
      } else {
        innerAxes.push({ axis });
        unknownCount++;
      }
    }

    // Handle different cases
    if (unknownCount === 0) {
      // All axes known: validate product equals total dimension
      if (knownProduct !== totalDim) {
        throw new AxisResolutionError(
          `Composite dimension mismatch: product of axes ${knownProduct} does not equal dimension ${totalDim}`,
          this.originalPattern,
          providedAxes ? { providedAxes } : {},
        );
      }

      // Set all provided values in axisDimensions
      for (const { axis, value } of innerAxes) {
        if (value !== undefined) {
          axisDimensions.set(axis.name, value);
        }
      }
    } else if (unknownCount === 1) {
      // One unknown: can infer its value
      if (totalDim % knownProduct !== 0) {
        throw new AxisResolutionError(
          `Cannot evenly split dimension ${totalDim} with known product ${knownProduct}`,
          this.originalPattern,
          providedAxes ? { providedAxes } : {},
        );
      }

      const inferredValue = totalDim / knownProduct;

      // Set the inferred value
      for (const { axis, value } of innerAxes) {
        if (value === undefined) {
          axisDimensions.set(axis.name, inferredValue);
        } else {
          axisDimensions.set(axis.name, value);
        }
      }
    } else {
      // Multiple unknowns: cannot infer
      throw new AxisResolutionError(
        `Cannot infer multiple unknown dimensions in composite pattern`,
        this.originalPattern,
        providedAxes ? { providedAxes } : {},
      );
    }
  }

  /**
   * Validate that all output axes are known
   */
  private validateOutputAxes(patterns: readonly AxisPattern[], axisDimensions: AxisMapping): void {
    for (const pattern of patterns) {
      if (isSimpleAxis(pattern)) {
        if (!axisDimensions.has(pattern.name)) {
          throw new AxisResolutionError(
            `Unknown axis '${pattern.name}' in output pattern`,
            this.originalPattern,
            { axis: pattern.name },
          );
        }
      } else if (isCompositeAxis(pattern)) {
        // Use flattening to handle nested composites properly
        const simpleAxes = this.flattenComposite(pattern);
        for (const innerAxis of simpleAxes) {
          if (!axisDimensions.has(innerAxis.name)) {
            throw new AxisResolutionError(
              `Unknown axis '${innerAxis.name}' in output pattern`,
              this.originalPattern,
              { axis: innerAxis.name },
            );
          }
        }
      }
      // Ellipsis and singleton are always valid in output
    }
  }

  /**
   * Compute the output shape from output patterns
   */
  private computeOutputShape(
    patterns: readonly AxisPattern[],
    axisDimensions: AxisMapping,
    ellipsisDimensions?: readonly number[],
  ): number[] {
    const outputShape: number[] = [];

    for (const pattern of patterns) {
      if (isSimpleAxis(pattern)) {
        const dim = axisDimensions.get(pattern.name);
        if (dim === undefined) {
          // Should not happen due to validation
          throw new Error(`Internal error: axis '${pattern.name}' not found`);
        }
        outputShape.push(dim);
      } else if (isCompositeAxis(pattern)) {
        // Composite in output: compute product of all simple axes (including nested)
        const simpleAxes = this.flattenComposite(pattern);
        let product = 1;
        for (const axis of simpleAxes) {
          const dim = axisDimensions.get(axis.name);
          if (dim === undefined) {
            throw new Error(`Internal error: axis '${axis.name}' not found`);
          }
          product *= dim;
        }
        outputShape.push(product);
      } else if (isEllipsisAxis(pattern)) {
        // Ellipsis: insert all ellipsis dimensions
        if (ellipsisDimensions) {
          outputShape.push(...ellipsisDimensions);
        }
      } else if (isSingletonAxis(pattern)) {
        // Singleton: always 1
        outputShape.push(1);
      }
    }

    return outputShape;
  }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/**
 * Resolve a pattern against a tensor shape
 */
export function resolvePattern(
  ast: EinopsAST,
  inputShape: readonly number[],
  providedAxes?: Record<string, number>,
): ResolvedPattern {
  const resolver = new AxisResolver();
  return resolver.resolvePattern(ast, inputShape, providedAxes);
}
