/**
 * Repeat operation for einops patterns
 *
 * This module provides the repeat functionality for replicating tensor
 * elements using einops pattern syntax. Unlike reduce, repeat can create
 * new axes and repeat elements along existing axes.
 */

import { parse } from './scanner';
import { AxisResolver, type ResolvedPattern } from './axis-resolver';
import type { EinopsAST, AxisPattern, CompositeAxis } from './ast';
import {
  isSimpleAxis,
  isCompositeAxis,
  isEllipsisAxis,
  isSingletonAxis,
  getAxisNames,
} from './ast';
import { Tensor, ChainablePromise } from '../tensor/tensor';
import type { AnyStorageTransformation, AnyTensorStorage } from '../storage/layout';
import type { Shape } from '../shape/types';
import type { RepeatOp } from '../storage/einops';
import type { ValidRepeatPattern } from './type-shape-resolver-repeat';

// =============================================================================
// Types
// =============================================================================

/**
 * Options for repeat operation
 */
export interface RepeatOptions {
  /**
   * Required axis dimensions for new axes and repetition factors
   * New axes in repeat output require explicit size specifications
   */
  readonly axes?: Record<string, number>;
}

/**
 * Planned tensor operation for repetition
 */
interface RepeatOperation {
  readonly type: 'reshape' | 'expand' | 'tile' | 'identity';
  readonly params?: {
    readonly shape?: readonly number[];
    readonly reps?: readonly number[];
    readonly targetShape?: readonly number[];
  };
}

/**
 * Error thrown during repeat operation
 */
export class RepeatError extends Error {
  constructor(
    message: string,
    public readonly pattern: string,
    public readonly context?: {
      inputShape?: readonly number[];
      outputShape?: readonly number[];
      newAxes?: string[];
      repeatedAxes?: string[];
    },
  ) {
    super(message);
    this.name = 'RepeatError';
  }
}

// =============================================================================
// Main Repeat Function
// =============================================================================

/**
 * Repeat tensor elements according to einops pattern
 *
 * @param tensor - Input tensor to repeat (can be Tensor or ChainablePromise)
 * @param pattern - Einops pattern like "h w -> h w c" or "h w -> (h h2) w"
 * @param axes - Required axis dimensions for new axes and repetition factors
 * @returns ChainablePromise with repeated tensor
 *
 * @example
 * ```typescript
 * // Add new axis (convert grayscale to RGB)
 * const rgb = await repeat(grayscale, 'h w -> h w c', { c: 3 });
 *
 * // Repeat along existing axis (2x upsampling)
 * const upsampled = await repeat(image, 'h w -> (h h2) (w w2)', { h2: 2, w2: 2 });
 *
 * // Add batch dimension
 * const batch = await repeat(image, 'h w c -> batch h w c', { batch: 8 });
 *
 * // Complex repetition pattern
 * const tiled = await repeat(patch, 'h w -> (h h2) (w w2) c', { h2: 3, w2: 3, c: 4 });
 * ```
 */
export function repeat<
  S extends AnyStorageTransformation & { __output: AnyTensorStorage },
  Pattern extends string,
  const Axes extends Record<string, number> | undefined = undefined,
>(
  tensor: Tensor<S> | ChainablePromise<S>,
  pattern: ValidRepeatPattern<Pattern, S['__output']['__shape'], Axes> extends Shape
    ? Pattern
    : ValidRepeatPattern<Pattern, S['__output']['__shape'], Axes>,
  axes?: Axes,
): ChainablePromise<RepeatOp<S['__output'], Pattern, Axes>> {
  return new ChainablePromise((resolve, reject) => {
    void (async () => {
      try {
        // Resolve tensor if it's a ChainablePromise
        const resolvedTensor = tensor instanceof ChainablePromise ? await tensor : tensor;

        if (!(resolvedTensor instanceof Tensor)) {
          throw new Error('Expected a Tensor instance');
        }

        // Step 1: Parse the pattern
        const ast = parse(pattern);

        // Step 2: Validate the pattern follows repeat rules
        validateRepeatPattern(ast, axes);

        // Step 3: Resolve axes against tensor shape
        const resolved = resolveAxes(ast, resolvedTensor.shape, axes);

        // Step 4: Plan operations
        const operations = planRepeatOperations(
          ast,
          resolvedTensor.shape,
          resolved,
        );

        // Step 5: Execute operations
        const result = await executeRepeatOperations(resolvedTensor, operations);

        // Return with correct type
        resolve(
          result as unknown as Tensor<RepeatOp<S['__output'], Pattern, Axes>>,
        );
      } catch (error) {
        if (error instanceof Error) {
          const inputShape = tensor instanceof ChainablePromise ? undefined : tensor.shape;
          reject(
            new RepeatError(`Failed to repeat tensor: ${error.message}`, pattern, {
              // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-explicit-any
              inputShape: inputShape as any,
            }),
          );
        } else {
          reject(error);
        }
      }
    })();
  });
}

// =============================================================================
// Validation Functions
// =============================================================================

/**
 * Validate that a pattern is valid for repeat operation
 * Rules:
 * 1. New axes in output require explicit sizes in axes parameter
 * 2. No duplicate axes in input or output
 * 3. Repetition factors must be positive integers
 */
function validateRepeatPattern(
  ast: EinopsAST, 
  providedAxes?: Record<string, number>
): void {
  // Check for multiple ellipsis
  const inputEllipsisCount = ast.input.filter((p) => isEllipsisAxis(p)).length;
  const outputEllipsisCount = ast.output.filter((p) => isEllipsisAxis(p)).length;

  if (inputEllipsisCount > 1) {
    throw new RepeatError(
      'Multiple ellipsis (...) in input pattern is not allowed',
      ast.metadata.originalPattern,
    );
  }

  if (outputEllipsisCount > 1) {
    throw new RepeatError(
      'Multiple ellipsis (...) in output pattern is not allowed',
      ast.metadata.originalPattern,
    );
  }

  // Collect all axis names (not singletons)
  const inputAxes = new Set(getAxisNames(ast.input));
  const outputAxes = new Set(getAxisNames(ast.output));

  // Check for duplicate axes in input
  const inputAxisList = getAxisNames(ast.input);
  const inputDuplicates = inputAxisList.filter((axis, index) => inputAxisList.indexOf(axis) !== index);
  if (inputDuplicates.length > 0) {
    throw new RepeatError(
      `Duplicate axes in input pattern: {${[...new Set(inputDuplicates)].join(', ')}}`,
      ast.metadata.originalPattern,
    );
  }

  // Check for duplicate axes in output
  const outputAxisList = getAxisNames(ast.output);
  const outputDuplicates = outputAxisList.filter((axis, index) => outputAxisList.indexOf(axis) !== index);
  if (outputDuplicates.length > 0) {
    throw new RepeatError(
      `Duplicate axes in output pattern: {${[...new Set(outputDuplicates)].join(', ')}}`,
      ast.metadata.originalPattern,
    );
  }

  // Find new axes (axes in output but not in input)
  const newAxes = [...outputAxes].filter((axis) => !inputAxes.has(axis));
  
  // Check that all new axes have provided sizes
  if (newAxes.length > 0) {
    if (!providedAxes) {
      throw new RepeatError(
        `New axes in output require explicit sizes: {${newAxes.join(', ')}}. ` +
        `Specify: repeat(tensor, pattern, {${newAxes.map(axis => `${axis}: number`).join(', ')}})`,
        ast.metadata.originalPattern,
      );
    }

    const missingAxes = newAxes.filter(axis => !(axis in providedAxes));
    if (missingAxes.length > 0) {
      throw new RepeatError(
        `Missing sizes for new axes: {${missingAxes.join(', ')}}. ` +
        `Specify: repeat(tensor, pattern, {${missingAxes.map(axis => `${axis}: number`).join(', ')}})`,
        ast.metadata.originalPattern,
      );
    }

    // Validate all new axis sizes are positive integers
    for (const axis of newAxes) {
      const size = providedAxes[axis];
      if (size === undefined || size <= 0 || !Number.isInteger(size)) {
        throw new RepeatError(
          `Invalid size for axis '${axis}': ${size}. Repeat sizes must be positive integers`,
          ast.metadata.originalPattern,
        );
      }
    }
  }

  // Validate provided axes that aren't new axes (repetition factors)
  if (providedAxes) {
    for (const [axis, size] of Object.entries(providedAxes)) {
      if (size <= 0 || !Number.isInteger(size)) {
        throw new RepeatError(
          `Invalid size for axis '${axis}': ${size}. Repeat sizes must be positive integers`,
          ast.metadata.originalPattern,
        );
      }
    }
  }
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Resolve axes using the axis resolver
 * This extends the input axis mapping with new axes from the provided axes
 */
function resolveAxes(
  ast: EinopsAST,
  inputShape: readonly number[],
  providedAxes?: Record<string, number>,
): ResolvedPattern {
  const resolver = new AxisResolver();
  
  // Get the base resolution from input pattern
  const baseResolved = resolver.resolvePattern(ast, inputShape, providedAxes);
  
  // Extend axis dimensions with new axes from output
  const inputAxes = new Set(getAxisNames(ast.input));
  const outputAxes = new Set(getAxisNames(ast.output));
  const newAxes = [...outputAxes].filter((axis) => !inputAxes.has(axis));
  
  // Add new axes to the axis dimensions map
  const extendedAxisDimensions = new Map(baseResolved.axisDimensions);
  if (providedAxes) {
    for (const axis of newAxes) {
      if (axis in providedAxes) {
        extendedAxisDimensions.set(axis, providedAxes[axis]);
      }
    }
  }
  
  return {
    ...baseResolved,
    axisDimensions: extendedAxisDimensions,
  };
}

/**
 * Compute the shape that includes all axes from the output pattern
 */
function computeExtendedShape(
  outputPatterns: readonly AxisPattern[], 
  axisDimensions: Map<string, number>,
  ellipsisDimensions?: readonly number[],
): number[] {
  const shape: number[] = [];

  for (const pattern of outputPatterns) {
    if (isSimpleAxis(pattern)) {
      const dim = axisDimensions.get(pattern.name);
      if (dim === undefined) {
        throw new Error(`Axis ${pattern.name} not found in dimensions map`);
      }
      shape.push(dim);
    } else if (isCompositeAxis(pattern)) {
      // Compute composite dimension
      const compositeDim = computeCompositeShape(pattern, axisDimensions);
      shape.push(compositeDim);
    } else if (isSingletonAxis(pattern)) {
      shape.push(1);
    } else if (isEllipsisAxis(pattern)) {
      if (ellipsisDimensions) {
        shape.push(...ellipsisDimensions);
      }
    }
  }

  return shape;
}

/**
 * Compute dimension for composite axis
 */
function computeCompositeShape(
  composite: CompositeAxis, 
  axisDimensions: Map<string, number>
): number {
  let product = 1;
  
  for (const pattern of composite.axes) {
    if (isSimpleAxis(pattern)) {
      const dim = axisDimensions.get(pattern.name);
      if (dim === undefined) {
        throw new Error(`Axis ${pattern.name} not found in dimensions map`);
      }
      product *= dim;
    } else if (isCompositeAxis(pattern)) {
      product *= computeCompositeShape(pattern, axisDimensions);
    }
    // Skip singletons in composites (they contribute factor of 1)
  }
  
  return product;
}

/**
 * Plan the sequence of operations for repetition
 * Strategy: Use expand() for new singleton dimensions, tile() for repetition
 */
function planRepeatOperations(
  ast: EinopsAST,
  inputShape: readonly number[],
  resolved: ResolvedPattern,
): RepeatOperation[] {
  const operations: RepeatOperation[] = [];

  // Compute final output shape
  const outputShape = computeExtendedShape(
    ast.output,
    resolved.axisDimensions,
    resolved.ellipsisDimensions,
  );

  // Check if this is just an identity operation
  if (arraysEqual(inputShape, outputShape)) {
    operations.push({ type: 'identity' });
    return operations;
  }

  // Strategy depends on the type of repetition needed
  const inputAxes = new Set(getAxisNames(ast.input));
  const outputAxes = new Set(getAxisNames(ast.output));
  const newAxes = [...outputAxes].filter(axis => !inputAxes.has(axis));
  
  if (newAxes.length > 0) {
    // Has new axes - need to use expand and potentially tile
    
    // First, compute intermediate shape by adding singleton dimensions for new axes
    const expandShape = computeExpandShapeWithNewAxes(
      ast.input,
      ast.output,
      inputShape,
      resolved.axisDimensions,
      resolved.ellipsisDimensions,
    );
    
    // Expand to add new singleton dimensions
    if (!arraysEqual(inputShape, expandShape)) {
      operations.push({
        type: 'expand',
        params: { targetShape: expandShape },
      });
    }
    
    // Then tile to reach final shape
    if (!arraysEqual(expandShape, outputShape)) {
      const reps = computeTilingReps(expandShape, outputShape);
      operations.push({
        type: 'tile',
        params: { reps },
      });
    }
  } else {
    // No new axes - pure repetition/reshaping
    
    // Check if we can use direct tiling
    const reps = computeTilingReps(inputShape, outputShape);
    if (reps.length === inputShape.length && reps.every(r => Number.isInteger(r) && r > 0)) {
      operations.push({
        type: 'tile',
        params: { reps },
      });
    } else {
      // Need reshaping - use reshape then tile
      const intermediateShape = computeIntermediateShapeForTiling(
        ast.input,
        ast.output,
        resolved.axisDimensions,
        resolved.ellipsisDimensions,
      );
      
      if (!arraysEqual(inputShape, intermediateShape)) {
        operations.push({
          type: 'reshape',
          params: { shape: intermediateShape },
        });
      }
      
      if (!arraysEqual(intermediateShape, outputShape)) {
        const reps = computeTilingReps(intermediateShape, outputShape);
        operations.push({
          type: 'tile',
          params: { reps },
        });
      }
    }
  }

  return operations;
}

/**
 * Compute shape for expansion that includes new singleton axes
 */
function computeExpandShapeWithNewAxes(
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  inputShape: readonly number[],
  axisDimensions: Map<string, number>,
  ellipsisDimensions?: readonly number[],
): number[] {
  // This is a simplified approach - in practice would need more sophisticated logic
  // For now, just compute the output shape and use expand to broadcast
  return computeExtendedShape(outputPatterns, axisDimensions, ellipsisDimensions);
}

/**
 * Compute intermediate shape for complex tiling operations
 */
function computeIntermediateShapeForTiling(
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
  ellipsisDimensions?: readonly number[],
): number[] {
  // Simplified - compute shape that can be easily tiled to output
  return computeExtendedShape(outputPatterns, axisDimensions, ellipsisDimensions);
}

/**
 * Compute repetition factors for tiling
 */
function computeTilingReps(fromShape: readonly number[], toShape: readonly number[]): number[] {
  if (fromShape.length !== toShape.length) {
    // Different ranks - need to handle dimension broadcasting
    const maxRank = Math.max(fromShape.length, toShape.length);
    const paddedFrom = [...Array(maxRank - fromShape.length).fill(1), ...fromShape];
    const paddedTo = [...Array(maxRank - toShape.length).fill(1), ...toShape];
    return paddedTo.map((toDim, i) => toDim / paddedFrom[i]);
  }
  
  return toShape.map((toDim, i) => toDim / fromShape[i]);
}

/**
 * Execute the planned operations on the tensor
 */
async function executeRepeatOperations<S extends AnyStorageTransformation>(
  tensor: Tensor<S>,
  operations: RepeatOperation[],
): Promise<Tensor<S>> {
  let result = tensor;

  for (const operation of operations) {
    switch (operation.type) {
      case 'identity':
        // No-op
        break;
        
      case 'reshape':
        if (operation.params?.shape) {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
          result = (await result.reshape(operation.params.shape as any)) as unknown as Tensor<S>;
        }
        break;

      case 'expand':
        if (operation.params?.targetShape) {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
          result = (await result.expand(operation.params.targetShape as any)) as unknown as Tensor<S>;
        }
        break;

      case 'tile':
        if (operation.params?.reps) {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
          result = (await result.tile(operation.params.reps as any)) as unknown as Tensor<S>;
        }
        break;

      default:
        throw new Error(`Unknown operation type: ${operation.type}`);
    }
  }

  return result;
}

/**
 * Check if two arrays are equal
 */
function arraysEqual(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((val, i) => val === b[i]);
}