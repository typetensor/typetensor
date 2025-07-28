/**
 * Basic rearrange function for einops patterns
 * 
 * This module provides the core rearrange functionality, implementing
 * simple tensor transformations using einops patterns.
 */

import { parse } from './scanner';
import { AxisResolver, type ResolvedPattern } from './axis-resolver';
import type { EinopsAST } from './ast';
import { isSimpleAxis } from './ast';

// =============================================================================
// Types
// =============================================================================

/**
 * Basic tensor interface for rearrange operations
 */
export interface RearrangeTensor {
  readonly shape: readonly number[];
  reshape(newShape: readonly number[]): RearrangeTensor;
  permute(axes: readonly number[]): RearrangeTensor;
  transpose(): RearrangeTensor;
}

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
  readonly type: 'reshape' | 'permute' | 'transpose' | 'identity';
  readonly params?: {
    readonly shape?: readonly number[];
    readonly axes?: readonly number[];
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
 * @param tensor - Input tensor to rearrange
 * @param pattern - Einops pattern like "h w -> w h" or "(h w) c -> h w c"
 * @param options - Optional configuration including axis dimensions
 * @returns New tensor with rearranged dimensions
 * 
 * @example
 * ```typescript
 * // Simple transpose
 * const result = rearrange(tensor, "h w -> w h");
 * 
 * // Composite pattern with provided axis
 * const result = rearrange(tensor, "(h w) c -> h w c", { axes: { h: 32 } });
 * 
 * // Axis reordering
 * const result = rearrange(tensor, "b h w c -> b c h w");
 * ```
 */
export function rearrange(
  tensor: RearrangeTensor,
  pattern: string,
  options: RearrangeOptions = {},
): RearrangeTensor {
  try {
    // Step 1: Parse the pattern
    const ast = parse(pattern);
    
    // Step 2: Resolve axes against tensor shape
    const resolved = resolveAxes(ast, tensor.shape, options.axes);
    
    // Step 3: Plan operations
    const operations = planOperations(ast, tensor.shape, resolved);
    
    // Step 4: Execute operations
    return executeOperations(tensor, operations);
    
  } catch (error) {
    if (error instanceof Error) {
      throw new RearrangeError(
        `Failed to rearrange tensor: ${error.message}`,
        pattern,
        { inputShape: tensor.shape },
      );
    }
    throw error;
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
function executeOperations(
  tensor: RearrangeTensor,
  operations: TensorOperation[],
): RearrangeTensor {
  let result = tensor;
  
  for (const operation of operations) {
    switch (operation.type) {
      case 'identity':
        // No-op
        break;
        
      case 'reshape':
        if (operation.params?.shape) {
          result = result.reshape(operation.params.shape);
        }
        break;
        
      case 'permute':
        if (operation.params?.axes) {
          result = result.permute(operation.params.axes);
        }
        break;
        
      case 'transpose':
        result = result.transpose();
        break;
        
      default:
        throw new Error(`Unknown operation type: ${(operation as any).type}`);
    }
  }
  
  return result;
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
  const inputNames = ast.input.map(p => isSimpleAxis(p) ? p.name : '');
  const outputNames = ast.output.map(p => isSimpleAxis(p) ? p.name : '');
  
  const inputSet = new Set(inputNames);
  const outputSet = new Set(outputNames);
  
  return inputSet.size === outputSet.size && 
         [...inputSet].every(name => outputSet.has(name));
}

/**
 * Compute permutation for simple transpose
 */
function computeSimplePermutation(ast: EinopsAST): number[] {
  const inputNames = ast.input.map(p => isSimpleAxis(p) ? p.name : '');
  const outputNames = ast.output.map(p => isSimpleAxis(p) ? p.name : '');
  
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