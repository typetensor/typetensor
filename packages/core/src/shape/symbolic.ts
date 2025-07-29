/**
 * Symbolic dimensions system for advanced shape inference and constraint solving
 *
 * This module provides symbolic dimension management, constraint solving, and
 * dynamic shape resolution for complex tensor operations and neural network patterns.
 */

import type { Shape, SymbolicShape, SymbolicDim, ResolvedShape, PartialShape } from './types.js';
import { RuntimeShape } from './runtime.js';

// =============================================================================
// Core Symbolic Types and Interfaces
// =============================================================================

/**
 * Types of constraints between symbolic dimensions
 */
export type ConstraintType = 'eq' | 'gt' | 'lt' | 'gte' | 'lte' | 'ne';

/**
 * Constraint between two symbolic dimensions or values
 */
export interface SymbolicConstraint {
  readonly id: string;
  readonly left: SymbolicDim | number;
  readonly right: SymbolicDim | number;
  readonly type: ConstraintType;
  readonly description?: string;
}

/**
 * Binding of a symbolic dimension to a concrete value
 */
export interface DimensionBinding {
  readonly dimension: SymbolicDim;
  readonly value: number;
  readonly source: 'explicit' | 'inferred' | 'constraint';
}

/**
 * Resolution context for symbolic shape solving
 */
export interface ResolutionContext {
  readonly bindings: Map<string, number>;
  readonly constraints: SymbolicConstraint[];
  readonly strictMode: boolean;
}

/**
 * Result of symbolic shape resolution with proper type safety
 */
export type ResolutionResult =
  | {
      readonly success: true;
      readonly type: 'resolved';
      readonly shape: ResolvedShape;
      readonly bindings: Map<string, number>;
      readonly warnings?: string[];
    }
  | {
      readonly success: true;
      readonly type: 'partial';
      readonly shape: PartialShape;
      readonly bindings: Map<string, number>;
      readonly warnings: string[];
    }
  | {
      readonly success: false;
      readonly errors: string[];
    };

// =============================================================================
// Shape Type Utilities
// =============================================================================

/**
 * Create a fully resolved shape (branded type)
 */
function createResolvedShape(dims: readonly number[]): ResolvedShape {
  return dims as ResolvedShape;
}

/**
 * Create a partially resolved shape (branded type)
 */

function createPartialShape(dims: readonly (number | -1)[]): PartialShape {
  return dims as PartialShape;
}

/**
 * Check if a shape is fully resolved (no -1 dimensions)
 */

function isFullyResolved(dims: readonly (number | -1)[]): dims is readonly number[] {
  return dims.every((dim) => dim !== -1);
}

// =============================================================================
// Symbolic Dimension Factory and Management
// =============================================================================

/**
 * Internal counter for generating unique symbolic dimension IDs
 */
let nextSymbolicId = 0;

/**
 * Create a symbolic dimension with a given name
 *
 * @example
 * const batch = createSymbolicDim('batch');
 * const seqLen = createSymbolicDim('seq_len');
 * const shape: SymbolicShape = [batch, seqLen, 768];
 */
export function createSymbolicDim<N extends string>(name: N): SymbolicDim<N> {
  return {
    __symbolic: name,
  } as SymbolicDim<N>;
}

/**
 * Check if a value is a symbolic dimension
 */
export function isSymbolicDim(value: unknown): value is SymbolicDim {
  return typeof value === 'object' && value !== null && '__symbolic' in value;
}

/**
 * Get the name of a symbolic dimension
 */
export function getSymbolicName(dim: SymbolicDim): string {
  return dim.__symbolic;
}

/**
 * Check if two symbolic dimensions are the same
 */
export function isSameSymbolicDim(a: SymbolicDim, b: SymbolicDim): boolean {
  return a.__symbolic === b.__symbolic;
}

// =============================================================================
// Constraint Management
// =============================================================================

/**
 * Create a constraint between two dimensions or values
 */
export function createConstraint(
  left: SymbolicDim | number,
  type: ConstraintType,
  right: SymbolicDim | number,
  description?: string,
): SymbolicConstraint {
  const id = `constraint_${(nextSymbolicId++).toString()}`;
  if (description !== undefined) {
    return {
      id,
      left,
      right,
      type,
      description,
    };
  } else {
    return {
      id,
      left,
      right,
      type,
    };
  }
}

/**
 * Validate that a constraint is well-formed
 */
export function validateConstraint(constraint: SymbolicConstraint): boolean {
  const { left, right, type } = constraint;

  // Cannot constrain a number to itself unless it's equality
  if (typeof left === 'number' && typeof right === 'number') {
    return evaluateNumericConstraint(left, type, right);
  }

  return true; // Symbolic constraints are always valid to define
}

/**
 * Evaluate a constraint between two numeric values
 */
function evaluateNumericConstraint(left: number, type: ConstraintType, right: number): boolean {
  switch (type) {
    case 'eq':
      return left === right;
    case 'ne':
      return left !== right;
    case 'gt':
      return left > right;
    case 'lt':
      return left < right;
    case 'gte':
      return left >= right;
    case 'lte':
      return left <= right;
    default:
      return false;
  }
}

// =============================================================================
// Symbolic Environment
// =============================================================================

/**
 * Environment for managing symbolic dimensions and their constraints
 */
export class SymbolicEnvironment {
  private readonly bindings = new Map<string, number>();
  private readonly constraints: SymbolicConstraint[] = [];
  private readonly dimensions = new Set<string>();

  /**
   * Define a symbolic dimension in this environment
   */
  define<N extends string>(name: N, value?: number): SymbolicDim<N> {
    this.dimensions.add(name);

    if (value !== undefined) {
      this.bind(name, value);
    }

    return createSymbolicDim(name);
  }

  /**
   * Bind a symbolic dimension to a concrete value
   */
  bind(dimension: string | SymbolicDim, value: number): void {
    const name = typeof dimension === 'string' ? dimension : dimension.__symbolic;

    if (value < 0) {
      throw new Error(
        `Cannot bind symbolic dimension '${name}' to negative value ${value.toString()}`,
      );
    }

    if (!Number.isInteger(value)) {
      throw new Error(
        `Cannot bind symbolic dimension '${name}' to non-integer value ${value.toString()}`,
      );
    }

    this.bindings.set(name, value);
  }

  /**
   * Get the binding for a symbolic dimension
   */
  getBind(dimension: string | SymbolicDim): number | undefined {
    const name = typeof dimension === 'string' ? dimension : dimension.__symbolic;
    return this.bindings.get(name);
  }

  /**
   * Check if a dimension is bound
   */
  isBound(dimension: string | SymbolicDim): boolean {
    const name = typeof dimension === 'string' ? dimension : dimension.__symbolic;
    return this.bindings.has(name);
  }

  /**
   * Add a constraint between dimensions
   */
  constrain(
    left: SymbolicDim | number,
    type: ConstraintType,
    right: SymbolicDim | number,
    description?: string,
  ): void {
    const constraint = createConstraint(left, type, right, description);

    if (!validateConstraint(constraint)) {
      throw new Error(`Invalid constraint: ${this.formatConstraint(constraint)}`);
    }

    this.constraints.push(constraint);
  }

  /**
   * Add an equality constraint (convenience method)
   */
  equal(left: SymbolicDim | number, right: SymbolicDim | number): void {
    this.constrain(left, 'eq', right);
  }

  /**
   * Get all constraints in this environment
   */
  getConstraints(): readonly SymbolicConstraint[] {
    return [...this.constraints];
  }

  /**
   * Get all bindings in this environment
   */
  getBindings(): ReadonlyMap<string, number> {
    return new Map(this.bindings);
  }

  /**
   * Clear all bindings and constraints
   */
  clear(): void {
    this.bindings.clear();
    this.constraints.splice(0);
    this.dimensions.clear();
  }

  /**
   * Clone this environment
   */
  clone(): SymbolicEnvironment {
    const env = new SymbolicEnvironment();

    // Copy dimensions
    this.dimensions.forEach((dim) => {
      env.dimensions.add(dim);
    });

    // Copy bindings
    this.bindings.forEach((value, name) => {
      env.bindings.set(name, value);
    });

    // Copy constraints
    env.constraints.push(...this.constraints);

    return env;
  }

  /**
   * Format a constraint for display
   */
  private formatConstraint(constraint: SymbolicConstraint): string {
    const leftStr =
      typeof constraint.left === 'number' ? constraint.left.toString() : constraint.left.__symbolic;

    const rightStr =
      typeof constraint.right === 'number'
        ? constraint.right.toString()
        : constraint.right.__symbolic;

    const opStr = {
      eq: '==',
      ne: '!=',
      gt: '>',
      lt: '<',
      gte: '>=',
      lte: '<=',
    }[constraint.type];

    return `${leftStr} ${opStr} ${rightStr}`;
  }
}

// =============================================================================
// Symbolic Shape Resolution
// =============================================================================

/**
 * Resolve a symbolic shape to a concrete shape using an environment
 */
export function resolveSymbolicShape(
  shape: SymbolicShape,
  environment: SymbolicEnvironment,
  options: { strict?: boolean } = {},
): ResolutionResult {
  const { strict = true } = options; // Default to strict mode
  const errors: string[] = [];
  const warnings: string[] = [];

  // Initialize resolved dimensions array properly to avoid sparse arrays
  const resolvedDimensions: (number | undefined)[] = new Array<number | undefined>(
    shape.length,
  ).fill(undefined);

  // First pass: resolve directly bound dimensions
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i];

    if (typeof dim === 'number') {
      resolvedDimensions[i] = dim;
    } else if (isSymbolicDim(dim)) {
      const binding = environment.getBind(dim);
      if (binding !== undefined) {
        resolvedDimensions[i] = binding;
      }
      // Don't error for unbound dimensions yet - constraint inference may resolve them
    }
  }

  // Second pass: try to infer missing dimensions from constraints
  const bindings = new Map(environment.getBindings());
  const constraints = environment.getConstraints();

  let changed = true;
  let iterations = 0;
  const maxIterations = 10; // Prevent infinite loops

  while (changed && iterations < maxIterations) {
    changed = false;
    iterations++;

    for (const constraint of constraints) {
      const result = tryResolveConstraint(constraint, bindings);
      if (result.newBinding) {
        bindings.set(result.newBinding.name, result.newBinding.value);
        changed = true;
      }
    }
  }

  // Final pass: apply inferred bindings
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[i];

    if (isSymbolicDim(dim) && resolvedDimensions[i] === undefined) {
      const name = getSymbolicName(dim);
      const binding = bindings.get(name);

      if (binding !== undefined) {
        resolvedDimensions[i] = binding;
      }
    }
  }

  // Check if all dimensions are resolved and collect warnings for unresolved ones
  const hasUnresolved = resolvedDimensions.some((dim) => dim === undefined);

  if (hasUnresolved) {
    const unresolvedNames: string[] = [];
    for (let i = 0; i < shape.length; i++) {
      const dim = shape[i];
      if (isSymbolicDim(dim) && resolvedDimensions[i] === undefined) {
        unresolvedNames.push(getSymbolicName(dim));
      }
    }

    if (strict) {
      errors.push(`Cannot resolve symbolic dimensions: ${unresolvedNames.join(', ')}`);
      return { success: false, errors };
    } else {
      // In non-strict mode, add warnings for dimensions that remain unbound after constraint inference
      for (const name of unresolvedNames) {
        warnings.push(`Unbound symbolic dimension: ${name}`);
      }
    }
  }

  // Validate constraints with resolved values
  for (const constraint of constraints) {
    const validation = validateResolvedConstraint(constraint, bindings);
    if (!validation.valid) {
      if (validation.message !== undefined) {
        errors.push(`Constraint violation: ${validation.message}`);
      } else {
        errors.push(`Constraint violation: Internal error, reason unknown`);
      }
    }
  }

  if (errors.length > 0) {
    return { success: false, errors };
  }

  // Create final shape with proper type safety

  const finalDimensions: (number | -1)[] = resolvedDimensions.map((dim) => dim ?? -1);

  if (isFullyResolved(finalDimensions)) {
    // All dimensions resolved - return ResolvedShape
    if (warnings.length > 0) {
      return {
        success: true,
        type: 'resolved',
        shape: createResolvedShape(finalDimensions),
        bindings,
        warnings,
      };
    } else {
      return {
        success: true,
        type: 'resolved',
        shape: createResolvedShape(finalDimensions),
        bindings,
      };
    }
  } else {
    // Some dimensions unresolved - return PartialShape
    if (strict) {
      // This shouldn't happen in strict mode, but add safety check

      const dimensions = finalDimensions as (number | -1)[];
      const unresolvedIndices = dimensions
        .map((dim, i) => (dim === -1 ? i : null))
        .filter((i) => i !== null);
      errors.push(
        `Strict mode: cannot return partial shape with unresolved dimensions at indices: ${unresolvedIndices.join(', ')}`,
      );
      return { success: false, errors };
    }

    return {
      success: true,
      type: 'partial',
      shape: createPartialShape(finalDimensions),
      bindings,
      warnings,
    };
  }
}

/**
 * Try to resolve a single constraint and infer new bindings
 */
function tryResolveConstraint(
  constraint: SymbolicConstraint,
  bindings: Map<string, number>,
): { newBinding?: { name: string; value: number } } {
  const { left, right, type } = constraint;

  // Only handle equality constraints for inference
  if (type !== 'eq') {
    return {};
  }

  const leftValue = typeof left === 'number' ? left : bindings.get(left.__symbolic);
  const rightValue = typeof right === 'number' ? right : bindings.get(right.__symbolic);

  // If both are bound, nothing to infer
  if (leftValue !== undefined && rightValue !== undefined) {
    return {};
  }

  // If left is bound and right is symbolic, bind right to left's value
  if (leftValue !== undefined && isSymbolicDim(right)) {
    const rightName = right.__symbolic;
    if (!bindings.has(rightName)) {
      return { newBinding: { name: rightName, value: leftValue } };
    }
  }

  // If right is bound and left is symbolic, bind left to right's value
  if (rightValue !== undefined && isSymbolicDim(left)) {
    const leftName = left.__symbolic;
    if (!bindings.has(leftName)) {
      return { newBinding: { name: leftName, value: rightValue } };
    }
  }

  return {};
}

/**
 * Validate a constraint against resolved bindings
 */
function validateResolvedConstraint(
  constraint: SymbolicConstraint,
  bindings: Map<string, number>,
): { valid: boolean; message?: string } {
  const { left, right, type } = constraint;

  const leftValue = typeof left === 'number' ? left : bindings.get(left.__symbolic);
  const rightValue = typeof right === 'number' ? right : bindings.get(right.__symbolic);

  // Skip validation if either side is not bound
  if (leftValue === undefined || rightValue === undefined) {
    return { valid: true };
  }

  const valid = evaluateNumericConstraint(leftValue, type, rightValue);

  if (!valid) {
    const leftStr =
      typeof left === 'number' ? left.toString() : `${left.__symbolic}=${leftValue.toString()}`;
    const rightStr =
      typeof right === 'number' ? right.toString() : `${right.__symbolic}=${rightValue.toString()}`;
    const opStr = { eq: '==', ne: '!=', gt: '>', lt: '<', gte: '>=', lte: '<=' }[type];

    return {
      valid: false,
      message: `${leftStr} ${opStr} ${rightStr} is false`,
    };
  }

  return { valid: true };
}

// =============================================================================
// Utility Functions for Common Patterns
// =============================================================================

/**
 * Create common neural network symbolic dimensions
 */
export const CommonSymbols = {
  /**
   * Create a batch dimension symbol
   */
  batch: (name = 'batch') => createSymbolicDim(name),

  /**
   * Create a sequence length dimension symbol
   */
  seqLen: (name = 'seq_len') => createSymbolicDim(name),

  /**
   * Create feature dimension symbols
   */
  features: (name = 'features') => createSymbolicDim(name),

  /**
   * Create height/width dimension symbols for images
   */
  height: (name = 'height') => createSymbolicDim(name),
  width: (name = 'width') => createSymbolicDim(name),

  /**
   * Create channel dimension symbol
   */
  channels: (name = 'channels') => createSymbolicDim(name),
};

/**
 * Create a symbolic environment with common ML patterns pre-defined
 */
export function createMLEnvironment(): SymbolicEnvironment {
  const env = new SymbolicEnvironment();

  // Define common dimensions without binding them
  env.define('batch');
  env.define('seq_len');
  env.define('features');
  env.define('height');
  env.define('width');
  env.define('channels');

  return env;
}

/**
 * Resolve symbolic shapes for neural network layers
 */
export class LayerShapeResolver {
  private readonly environment: SymbolicEnvironment;

  constructor(environment?: SymbolicEnvironment) {
    this.environment = environment ?? createMLEnvironment();
  }

  /**
   * Resolve linear layer shapes: [batch, in_features] -> [batch, out_features]
   */
  linear(
    inputShape: SymbolicShape,
    outFeatures: number,
  ): { inputResolved: ResolutionResult; outputShape: SymbolicShape } {
    if (inputShape.length !== 2) {
      throw new Error(`Linear layer expects 2D input, got ${inputShape.length.toString()}D`);
    }

    const inputResolved = resolveSymbolicShape(inputShape, this.environment);
    const batchDim = inputShape[0];
    if (batchDim === undefined) {
      throw new Error('Input shape must have at least one dimension for linear layer');
    }
    const outputShape: SymbolicShape = [batchDim, outFeatures];

    return { inputResolved, outputShape };
  }

  /**
   * Resolve attention layer shapes
   */
  attention(
    inputShape: SymbolicShape, // [batch, seq_len, features]
  ): { inputResolved: ResolutionResult; outputShape: SymbolicShape } {
    if (inputShape.length !== 3) {
      throw new Error(`Attention layer expects 3D input, got ${inputShape.length.toString()}D`);
    }

    const inputResolved = resolveSymbolicShape(inputShape, this.environment);

    // Output has same shape as input for self-attention
    const outputShape: SymbolicShape = [...inputShape];

    return { inputResolved, outputShape };
  }

  /**
   * Get the underlying environment
   */
  getEnvironment(): SymbolicEnvironment {
    return this.environment;
  }
}

// =============================================================================
// Integration with Shape System
// =============================================================================

/**
 * Check if a symbolic shape can be broadcast with a concrete shape
 */
export function canBroadcastSymbolic(
  symbolicShape: SymbolicShape,
  concreteShape: Shape,
  environment: SymbolicEnvironment,
): boolean {
  const resolution = resolveSymbolicShape(symbolicShape, environment);

  if (!resolution.success) {
    return false; // Cannot determine compatibility without resolution
  }

  // Only proceed with fully resolved shapes for broadcasting compatibility
  if (resolution.type !== 'resolved') {
    return false; // Cannot determine compatibility with partial resolution
  }

  return RuntimeShape.canBroadcast(resolution.shape, concreteShape);
}

/**
 * Attempt to infer symbolic dimension values from a concrete shape
 */
export function inferFromConcrete(
  symbolicShape: SymbolicShape,
  concreteShape: Shape,
  environment: SymbolicEnvironment,
): { success: boolean; inferences?: Map<string, number>; errors?: string[] } {
  if (symbolicShape.length !== concreteShape.length) {
    return {
      success: false,
      errors: [
        `Shape rank mismatch: symbolic has ${symbolicShape.length.toString()}, concrete has ${concreteShape.length.toString()}`,
      ],
    };
  }

  const inferences = new Map<string, number>();
  const errors: string[] = [];

  for (let i = 0; i < symbolicShape.length; i++) {
    const symbolicDim = symbolicShape[i];
    const concreteDim = concreteShape[i];

    // Add runtime validation for array bounds safety
    if (concreteDim === undefined) {
      throw new Error(`Concrete dimension at index ${i.toString()} is undefined`);
    }

    if (typeof symbolicDim === 'number') {
      if (symbolicDim !== concreteDim) {
        errors.push(
          `Dimension ${i.toString()}: expected ${symbolicDim.toString()}, got ${concreteDim.toString()}`,
        );
      }
    } else if (isSymbolicDim(symbolicDim)) {
      const name = getSymbolicName(symbolicDim);
      const existingBinding = environment.getBind(symbolicDim);

      if (existingBinding !== undefined) {
        if (existingBinding !== concreteDim) {
          errors.push(
            `Dimension ${i.toString()}: symbolic '${name}' bound to ${existingBinding.toString()}, but concrete shape has ${concreteDim.toString()}`,
          );
        }
      } else {
        inferences.set(name, concreteDim);
      }
    }
  }

  if (errors.length > 0) {
    return { success: false, errors };
  }

  return { success: true, inferences };
}
