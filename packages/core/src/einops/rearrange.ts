/**
 * Basic rearrange function for einops patterns
 *
 * This module provides the core rearrange functionality, implementing
 * simple tensor transformations using einops patterns.
 */

import { parse } from './scanner';
import { AxisResolver, type ResolvedPattern } from './axis-resolver';
import type { EinopsAST, AxisPattern } from './ast';
import { isSimpleAxis, isCompositeAxis } from './ast';
import { Tensor, ChainablePromise } from '../tensor/tensor';
import type { AnyStorageTransformation } from '../storage/layout';
import type { RearrangeOp } from '../storage/einops';

// =============================================================================
// Types
// =============================================================================

/**
 * Options for rearrange operation
 */
export interface RearrangeOptions {
  /**
   * Explicitly provided axis dimensions
   * Used to resolve composite patterns where dimensions cannot be inferred
   */
  readonly axes?: Record<string, number>;
}

/**
 * Planned tensor operation
 */
interface TensorOperation {
  readonly type: 'reshape' | 'permute' | 'transpose' | 'identity' | 'sum';
  readonly params?: {
    readonly shape?: readonly number[];
    readonly axes?: readonly number[];
    readonly keepdims?: boolean;
  };
}

/**
 * Error thrown during rearrange operation
 */
export class RearrangeError extends Error {
  constructor(
    message: string,
    public readonly pattern: string,
    public readonly context?: {
      inputShape?: readonly number[];
      outputShape?: readonly number[];
      operation?: string;
    },
  ) {
    super(message);
    this.name = 'RearrangeError';
  }
}

// =============================================================================
// Main Rearrange Function
// =============================================================================

/**
 * Rearrange tensor dimensions according to einops pattern
 *
 * @param tensor - Input tensor to rearrange (can be Tensor or ChainablePromise)
 * @param pattern - Einops pattern like "h w -> w h" or "(h w) c -> h w c"
 * @param axes - Optional axis dimensions for composite patterns
 * @returns ChainablePromise with rearranged tensor
 *
 * @example
 * ```typescript
 * // Simple transpose
 * const result = await rearrange(tensor, "h w -> w h");
 *
 * // Composite pattern with provided axis
 * const result = await rearrange(tensor, "(h w) c -> h w c", { h: 32 });
 *
 * // Works with chained operations
 * const result = await rearrange(tensor.add(other), "b h w c -> b c h w");
 *
 * // Can be chained further
 * const result = await rearrange(tensor, "h w -> w h").add(other);
 * ```
 */
export function rearrange<
  S extends AnyStorageTransformation,
  Pattern extends string,
  Axes extends Record<string, number> | undefined = undefined,
>(
  tensor: Tensor<S> | ChainablePromise<S>,
  pattern: Pattern,
  axes?: Axes,
): ChainablePromise<RearrangeOp<S['__output'], Pattern, Axes>> {
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

        // Step 2: Validate the pattern follows rearrange rules
        validateRearrangePattern(ast);

        // Step 3: Resolve axes against tensor shape
        const resolved = resolveAxes(ast, resolvedTensor.shape, axes);

        // Step 3: Plan operations
        const operations = planOperations(ast, resolvedTensor.shape, resolved);

        // Step 4: Execute operations
        const result = executeOperations(resolvedTensor, operations);

        // Return with correct type
        resolve(result as unknown as Tensor<RearrangeOp<S['__output'], Pattern, Axes>>);
      } catch (error) {
        if (error instanceof Error) {
          const inputShape = tensor instanceof ChainablePromise ? undefined : tensor.shape;
          reject(
            new RearrangeError(`Failed to rearrange tensor: ${error.message}`, pattern, {
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
 * Validate that a pattern is valid for rearrange operation
 * Following PyTorch einops rules
 */
function validateRearrangePattern(ast: EinopsAST): void {
  // Collect all axis names from input and output
  const inputAxes = new Set<string>();
  const outputAxes = new Set<string>();
  const outputAxisCounts = new Map<string, number>();

  // Process input axes
  function collectInputAxes(pattern: AxisPattern): void {
    if (isSimpleAxis(pattern)) {
      inputAxes.add(pattern.name);
    } else if (isCompositeAxis(pattern)) {
      pattern.axes.forEach(collectInputAxes);
    }
  }

  // Process output axes  
  function collectOutputAxes(pattern: AxisPattern): void {
    if (isSimpleAxis(pattern)) {
      outputAxes.add(pattern.name);
      outputAxisCounts.set(pattern.name, (outputAxisCounts.get(pattern.name) || 0) + 1);
    } else if (isCompositeAxis(pattern)) {
      pattern.axes.forEach(collectOutputAxes);
    }
  }

  ast.input.forEach(collectInputAxes);
  ast.output.forEach(collectOutputAxes);

  // Check for axes only on one side
  const onlyInInput = [...inputAxes].filter(axis => !outputAxes.has(axis));
  const onlyInOutput = [...outputAxes].filter(axis => !inputAxes.has(axis));
  
  if (onlyInInput.length > 0 || onlyInOutput.length > 0) {
    const problematicAxes = [...onlyInInput, ...onlyInOutput];
    throw new RearrangeError(
      `Identifiers only on one side of expression (should be on both): {${problematicAxes.join(', ')}}`,
      ast.metadata.originalPattern
    );
  }

  // Check for duplicate axes in output
  const duplicateAxes = [...outputAxisCounts.entries()]
    .filter(([_, count]) => count > 1)
    .map(([axis, _]) => axis);
    
  if (duplicateAxes.length > 0) {
    throw new RearrangeError(
      `Indexing expression contains duplicate dimension "${duplicateAxes[0]}"`,
      ast.metadata.originalPattern
    );
  }
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
 * Plan the sequence of tensor operations needed
 */
function planOperations(
  ast: EinopsAST,
  inputShape: readonly number[],
  resolved: ResolvedPattern,
): TensorOperation[] {
  const operations: TensorOperation[] = [];

  // Handle simple identity case
  if (isIdentityPattern(ast)) {
    return [{ type: 'identity' }];
  }

  // Handle simple transpose
  if (isSimpleTranspose(ast)) {
    const permutation = computeSimplePermutation(ast);
    if (permutation.length === 2 && permutation[0] === 1 && permutation[1] === 0) {
      // Special case: 2D transpose
      return [{ type: 'transpose' }];
    } else {
      return [{ type: 'permute', params: { axes: permutation } }];
    }
  }

  // General case: Always just reshape to final output shape
  // The axis resolver has already computed the correct output shape
  const outputShape = Array.from(resolved.outputShape);

  // If shapes are different, we need to reshape
  if (!arraysEqual(inputShape, outputShape)) {
    operations.push({
      type: 'reshape',
      params: { shape: outputShape },
    });
  }

  return operations;
}

/**
 * Execute the planned operations on the tensor
 */
function executeOperations<S extends AnyStorageTransformation>(
  tensor: Tensor<S>,
  operations: TensorOperation[],
): Tensor<S> {
  return operations.reduce((tensor, operation) => {
    switch (operation.type) {
      case 'identity':
        return tensor;

      case 'reshape':
        if (operation.params?.shape) {
          return tensor.reshape(operation.params.shape as any) as unknown as Tensor<S>;
        }
        return tensor;

      case 'permute':
        if (operation.params?.axes) {
          return tensor.permute(operation.params.axes as any) as unknown as Tensor<S>;
        }
        return tensor;

      case 'transpose':
        return tensor.transpose() as unknown as Tensor<S>;

      default:
        throw new Error(`Unknown operation type: ${operation.type}`);
    }
  }, tensor);
}

// =============================================================================
// Pattern Analysis Functions
// =============================================================================

/**
 * Check if pattern is identity (input same as output)
 */
function isIdentityPattern(ast: EinopsAST): boolean {
  if (ast.input.length !== ast.output.length) {
    return false;
  }

  for (let i = 0; i < ast.input.length; i++) {
    const inputPattern = ast.input[i];
    const outputPattern = ast.output[i];

    if (!inputPattern || !outputPattern) {
      return false;
    }

    if (inputPattern.type !== outputPattern.type) {
      return false;
    }

    if (isSimpleAxis(inputPattern) && isSimpleAxis(outputPattern)) {
      if (inputPattern.name !== outputPattern.name) {
        return false;
      }
    }
    // For composite, ellipsis, singleton, we'd need deeper comparison
    // For now, assume non-identity for these complex cases
    else if (!isSimpleAxis(inputPattern)) {
      return false;
    }
  }

  return true;
}

/**
 * Check if pattern is a simple transpose (all simple axes, different order)
 */
function isSimpleTranspose(ast: EinopsAST): boolean {
  // All patterns must be simple axes
  const allInputSimple = ast.input.every(isSimpleAxis);
  const allOutputSimple = ast.output.every(isSimpleAxis);

  if (!allInputSimple || !allOutputSimple) {
    return false;
  }

  // Same number of axes
  if (ast.input.length !== ast.output.length) {
    return false;
  }

  // All input axes must appear in output (different order is ok)
  const inputNames = ast.input.map((p) => (isSimpleAxis(p) ? p.name : ''));
  const outputNames = ast.output.map((p) => (isSimpleAxis(p) ? p.name : ''));

  const inputSet = new Set(inputNames);
  const outputSet = new Set(outputNames);

  return inputSet.size === outputSet.size && [...inputSet].every((name) => outputSet.has(name));
}

/**
 * Compute permutation for simple transpose
 */
function computeSimplePermutation(ast: EinopsAST): number[] {
  const inputNames = ast.input.map((p) => (isSimpleAxis(p) ? p.name : ''));
  const outputNames = ast.output.map((p) => (isSimpleAxis(p) ? p.name : ''));

  const permutation: number[] = [];

  for (const outputName of outputNames) {
    const inputIndex = inputNames.indexOf(outputName);
    if (inputIndex === -1) {
      throw new Error(`Axis ${outputName} not found in input`);
    }
    permutation.push(inputIndex);
  }

  return permutation;
}

/**
 * Check if two arrays are equal
 */
function arraysEqual(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((val, i) => val === b[i]);
}
