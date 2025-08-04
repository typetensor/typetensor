/**
 * Reduce operation for einops patterns
 *
 * This module provides the reduce functionality for aggregating tensor
 * dimensions using einops pattern syntax.
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
import type { ReduceEinopsOp } from '../storage/einops';
import type { ValidReducePattern } from './type-shape-resolver-reduce';
import { arraysEqual } from './utils/array';

// =============================================================================
// Types
// =============================================================================

/**
 * Supported reduction operations
 */
export type ReductionOp = 'sum' | 'mean' | 'max' | 'min' | 'prod';

/**
 * Options for reduce operation
 */
export interface ReduceOptions {
  /**
   * Explicitly provided axis dimensions
   * Used to resolve composite patterns where dimensions cannot be inferred
   */
  readonly axes?: Record<string, number>;
}

/**
 * Planned tensor operation for reduction
 */
interface ReduceOperation {
  readonly type: 'reshape' | 'reduce';
  readonly params?: {
    readonly shape?: readonly number[];
    readonly axes?: readonly number[];
    readonly operation?: ReductionOp;
    readonly keepDims?: boolean;
  };
}

/**
 * Error thrown during reduce operation
 */
export class ReduceError extends Error {
  constructor(
    message: string,
    public readonly pattern: string,
    public readonly context?: {
      inputShape?: readonly number[];
      outputShape?: readonly number[];
      operation?: string;
      reducedAxes?: string[];
    },
  ) {
    super(message);
    this.name = 'ReduceError';
  }
}

// =============================================================================
// Main Reduce Function
// =============================================================================

/**
 * Reduce tensor dimensions according to einops pattern
 *
 * @param tensor - Input tensor to reduce (can be Tensor or ChainablePromise)
 * @param pattern - Einops pattern like "h w c -> c" or "(h 2) (w 2) c -> h w c"
 * @param operation - Reduction operation: 'sum', 'mean', 'max', 'min', 'prod'
 * @param keepDims - Whether to keep reduced dimensions as size 1
 * @param axes - Optional axis dimensions for composite patterns
 * @returns ChainablePromise with reduced tensor
 *
 * @example
 * ```typescript
 * // Average over spatial dimensions
 * const result = await reduce(tensor, 'batch h w c -> batch c', 'mean');
 *
 * // Max pooling with 2x2 windows
 * const pooled = await reduce(tensor, 'batch (h 2) (w 2) c -> batch h w c', 'max');
 *
 * // Global sum
 * const total = await reduce(tensor, 'batch h w c ->', 'sum');
 *
 * // Keep reduced dimensions
 * const reduced = await reduce(tensor, 'h w c -> c', 'mean', true);
 * ```
 */
export function reduce<
  S extends AnyStorageTransformation & { __output: AnyTensorStorage },
  Pattern extends string,
  Op extends ReductionOp = 'mean',
  KeepDims extends boolean = false,
  const Axes extends Record<string, number> | undefined = undefined,
>(
  tensor: Tensor<S> | ChainablePromise<S>,
  pattern: ValidReducePattern<Pattern, S['__output']['__shape'], KeepDims, Axes> extends string
    ? ValidReducePattern<Pattern, S['__output']['__shape'], KeepDims, Axes> // Show the actual error message
    : Pattern,
  operation: Op = 'mean' as Op,
  keepDims?: KeepDims,
  axes?: Axes,
): ChainablePromise<ReduceEinopsOp<S['__output'], Pattern, Op, KeepDims, Axes>> {
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

        // Step 2: Validate the pattern follows reduce rules
        validateReducePattern(ast);

        // Step 3: Resolve axes against tensor shape
        const resolved = resolveAxes(ast, resolvedTensor.shape, axes);

        // Step 4: Plan operations
        const operations = planReduceOperations(
          ast,
          resolvedTensor.shape,
          resolved,
          operation,
          keepDims ?? false,
        );

        // Step 5: Execute operations
        const result = await executeReduceOperations(resolvedTensor, operations);

        // Return with correct type
        resolve(
          result as unknown as Tensor<ReduceEinopsOp<S['__output'], Pattern, Op, KeepDims, Axes>>,
        );
      } catch (error) {
        if (error instanceof Error) {
          const inputShape = tensor instanceof ChainablePromise ? undefined : tensor.shape;
          reject(
            new ReduceError(`Failed to reduce tensor: ${error.message}`, pattern, {
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
 * Validate that a pattern is valid for reduce operation
 * Rules:
 * 1. All named axes in output must exist in input
 * 2. No duplicate axes in output
 * 3. Allow patterns where nothing is reduced (like einops)
 */
function validateReducePattern(ast: EinopsAST): void {
  // Check for multiple ellipsis
  const inputEllipsisCount = ast.input.filter((p) => isEllipsisAxis(p)).length;
  const outputEllipsisCount = ast.output.filter((p) => isEllipsisAxis(p)).length;

  if (inputEllipsisCount > 1) {
    throw new ReduceError(
      'Multiple ellipsis (...) in input pattern is not allowed',
      ast.metadata.originalPattern,
    );
  }

  if (outputEllipsisCount > 1) {
    throw new ReduceError(
      'Multiple ellipsis (...) in output pattern is not allowed',
      ast.metadata.originalPattern,
    );
  }

  // Collect all axis names (not singletons)
  const inputAxes = new Set(getAxisNames(ast.input));
  const outputAxes = new Set(getAxisNames(ast.output));

  // Check for axes only in output (not allowed in reduce)
  const newAxes = [...outputAxes].filter((axis) => !inputAxes.has(axis));
  if (newAxes.length > 0) {
    throw new ReduceError(
      `Unknown axes in output: {${newAxes.join(', ')}}. All output axes must exist in input.`,
      ast.metadata.originalPattern,
    );
  }

  // Check for duplicate axes in output
  const outputAxisList = getAxisNames(ast.output);
  const duplicates = outputAxisList.filter((axis, index) => outputAxisList.indexOf(axis) !== index);
  if (duplicates.length > 0) {
    throw new ReduceError(
      `Duplicate axes in output pattern: {${[...new Set(duplicates)].join(', ')}}`,
      ast.metadata.originalPattern,
    );
  }

  // Note: We allow patterns where nothing is reduced (like einops does)
  // The actual reduction will be determined by comparing positions
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Resolve axes using the axis resolver
 */
function resolveAxes(
  ast: EinopsAST,
  inputShape: readonly number[],
  providedAxes?: Record<string, number>,
): ResolvedPattern {
  const resolver = new AxisResolver();
  return resolver.resolvePattern(ast, inputShape, providedAxes);
}

/**
 * Compute which dimensions to reduce by comparing input and output patterns
 */
function computeReducedDimensions(
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  _axisDimensions: Map<string, number>,
  ellipsisDimensions?: readonly number[],
): number[] {
  // ALGORITHM: Determine which dimensions to reduce
  // 1. Flatten input pattern to get position -> axis name mapping
  // 2. Collect all axes that appear in output
  // 3. Dimensions whose axes don't appear in output are reduced
  //
  // Example: 'h w c -> h c' reduces dimension 1 (w)

  // First, flatten input patterns to get ordered list of axes
  const flattenedInput: string[] = [];
  const positionToAxis = new Map<number, string>();
  let position = 0;

  for (const pattern of inputPatterns) {
    if (isSimpleAxis(pattern)) {
      flattenedInput.push(pattern.name);
      positionToAxis.set(position, pattern.name);
      position++;
    } else if (isCompositeAxis(pattern)) {
      // Composite patterns in input are flattened before reduction.
      // Example: '(h w) c -> c' first reshapes to 'hw c' then reduces
      const innerAxes = flattenComposite(pattern);
      for (const axis of innerAxes) {
        flattenedInput.push(axis);
        positionToAxis.set(position, axis);
        position++;
      }
    } else if (isSingletonAxis(pattern)) {
      // Singletons get a special marker
      flattenedInput.push(`__singleton_${position}`);
      position++;
    } else if (isEllipsisAxis(pattern)) {
      if (ellipsisDimensions) {
        for (let i = 0; i < ellipsisDimensions.length; i++) {
          flattenedInput.push(`__ellipsis_${i}`);
          position++;
        }
      }
    }
  }

  // Collect what appears in output (including singletons)
  const outputIdentifiers = new Set<string>();
  for (const pattern of outputPatterns) {
    if (isSimpleAxis(pattern)) {
      outputIdentifiers.add(pattern.name);
    } else if (isCompositeAxis(pattern)) {
      const innerAxes = flattenComposite(pattern);
      for (const axis of innerAxes) {
        outputIdentifiers.add(axis);
      }
    } else if (isSingletonAxis(pattern)) {
      // Singletons in output don't reduce anything
      // They either match input singletons or are added
    } else if (isEllipsisAxis(pattern)) {
      // Ellipsis in output preserves those dimensions
      if (ellipsisDimensions) {
        for (let i = 0; i < ellipsisDimensions.length; i++) {
          outputIdentifiers.add(`__ellipsis_${i}`);
        }
      }
    }
  }

  // Find positions to reduce: axes in input but not in output
  const reducedPositions: number[] = [];
  for (let i = 0; i < flattenedInput.length; i++) {
    const axis = flattenedInput[i];
    if (axis !== undefined && !outputIdentifiers.has(axis)) {
      reducedPositions.push(i);
    }
  }

  return reducedPositions;
}

/**
 * Flatten a composite axis to get all inner axis names
 */
function flattenComposite(composite: CompositeAxis): string[] {
  const result: string[] = [];
  for (const pattern of composite.axes) {
    if (isSimpleAxis(pattern)) {
      result.push(pattern.name);
    } else if (isCompositeAxis(pattern)) {
      result.push(...flattenComposite(pattern));
    }
    // Skip singletons in composites
  }
  return result;
}

/**
 * Plan the sequence of operations for reduction
 */
function planReduceOperations(
  ast: EinopsAST,
  inputShape: readonly number[],
  resolved: ResolvedPattern,
  operation: ReductionOp,
  keepDims: boolean,
): ReduceOperation[] {
  const operations: ReduceOperation[] = [];

  // Check if we need to expand composites first
  const hasInputComposite = ast.input.some((p) => isCompositeAxis(p));
  const hasOutputComposite = ast.output.some((p) => isCompositeAxis(p));

  if (hasInputComposite) {
    // Need to reshape to expand composites
    const expandedShape = computeExpandedShape(
      ast.input,
      resolved.axisDimensions,
      resolved.ellipsisDimensions,
    );

    if (!arraysEqual(inputShape, expandedShape)) {
      operations.push({
        type: 'reshape',
        params: { shape: expandedShape },
      });
    }

    // Compute reduction axes on expanded shape
    const reducedDims = computeReducedDimensions(
      ast.input,
      ast.output,
      resolved.axisDimensions,
      resolved.ellipsisDimensions,
    );

    operations.push({
      type: 'reduce',
      params: {
        axes: reducedDims,
        operation,
        keepDims,
      },
    });

    // If output has composites, need final reshape
    if (hasOutputComposite && !keepDims) {
      const finalShape = Array.from(resolved.outputShape);
      operations.push({
        type: 'reshape',
        params: { shape: finalShape },
      });
    }
  } else {
    // Simple case: direct reduction
    const reducedDims = computeReducedDimensions(
      ast.input,
      ast.output,
      resolved.axisDimensions,
      resolved.ellipsisDimensions,
    );

    operations.push({
      type: 'reduce',
      params: {
        axes: reducedDims,
        operation,
        keepDims,
      },
    });

    // Check if we need a final reshape to match output shape
    // This happens when output has singletons or different structure
    const needsFinalReshape =
      !keepDims && (ast.output.some((p) => isSingletonAxis(p)) || reducedDims.length === 0); // No reduction case

    if (needsFinalReshape) {
      const finalShape = Array.from(resolved.outputShape);
      operations.push({
        type: 'reshape',
        params: { shape: finalShape },
      });
    }
  }

  return operations;
}

/**
 * Compute the expanded shape when composites are flattened
 */
function computeExpandedShape(
  patterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
  ellipsisDimensions?: readonly number[],
): number[] {
  const shape: number[] = [];

  for (const pattern of patterns) {
    if (isSimpleAxis(pattern)) {
      const dim = axisDimensions.get(pattern.name);
      if (dim === undefined) {
        throw new Error(`Axis ${pattern.name} not found in dimensions map`);
      }
      shape.push(dim);
    } else if (isCompositeAxis(pattern)) {
      // Expand composite to its constituent axes
      const expandedShape = computeExpandedShape(pattern.axes, axisDimensions, ellipsisDimensions);
      shape.push(...expandedShape);
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
 * Execute the planned operations on the tensor
 */
async function executeReduceOperations<S extends AnyStorageTransformation>(
  tensor: Tensor<S>,
  operations: ReduceOperation[],
): Promise<Tensor<S>> {
  let result = tensor;

  for (const operation of operations) {
    switch (operation.type) {
      case 'reshape':
        if (operation.params?.shape) {
          // TODO: it would be better to leverage type guards or type assertions here...well tested at runtime though...
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
          result = (await result.reshape(operation.params.shape as any)) as unknown as Tensor<S>;
        }
        break;

      case 'reduce':
        if (operation.params?.operation && operation.params.axes !== undefined) {
          const { operation: op, axes, keepDims } = operation.params;
          switch (op) {
            case 'sum':
              result = (await result.sum(axes, keepDims)) as unknown as Tensor<S>;
              break;
            case 'mean':
              result = (await result.mean(axes, keepDims)) as unknown as Tensor<S>;
              break;
            case 'max':
              result = (await result.max(axes, keepDims)) as unknown as Tensor<S>;
              break;
            case 'min':
              result = (await result.min(axes, keepDims)) as unknown as Tensor<S>;
              break;
            case 'prod':
              result = (await result.prod(axes, keepDims)) as unknown as Tensor<S>;
              break;
            default:
              throw new Error(`Unknown reduction operation: ${op}`);
          }
        }
        break;

      default:
        throw new Error(`Unknown operation type: ${operation.type}`);
    }
  }

  return result;
}
