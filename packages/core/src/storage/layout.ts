/**
 * Layout and base storage types for tensor operations
 *
 * This module provides the fundamental types for tensor storage,
 * including layout flags, memory properties, and base storage interface.
 */

import type { Shape, Product } from '../shape/types';
import type { AnyDType } from '../dtype/types';

// =============================================================================
// Layout and Memory Flags
// =============================================================================

/**
 * Layout flags that describe memory properties
 * These are phantom types that don't exist at runtime
 */
export interface LayoutFlags {
  readonly c_contiguous: boolean | 'unknown';
  readonly f_contiguous: boolean | 'unknown';
  readonly is_view: boolean;
  readonly writeable: boolean;
  readonly aligned: boolean;
}

/**
 * Default layout flags for newly created tensors
 */
export interface DefaultLayoutFlags extends LayoutFlags {
  readonly c_contiguous: true;
  readonly f_contiguous: false;
  readonly is_view: false;
  readonly writeable: true;
  readonly aligned: true;
}

// =============================================================================
// Base Storage Type
// =============================================================================

/**
 * Base storage type that represents tensor metadata
 * This is implementation-agnostic and focuses on type propagation
 *
 * @template DT - Data type (dtype)
 * @template S - Shape tuple
 * @template St - Strides tuple
 * @template L - Layout flags for memory properties
 */
export interface TensorStorage<
  DT extends AnyDType,
  S extends Shape,
  St extends Shape = ComputeStrides<S>,
  L extends LayoutFlags = DefaultLayoutFlags,
> {
  readonly __dtype: DT;
  readonly __shape: S;
  readonly __strides: St;
  readonly __size: Product<S>;
  readonly __layout: L;
  readonly __offset: number;
}

/**
 * Compute C-order (row-major) strides from shape
 */
export type ComputeStrides<S extends Shape> = ComputeStridesHelper<S>;

type ComputeStridesHelper<S extends Shape, Acc extends Shape = readonly []> = S extends readonly []
  ? Acc
  : S extends readonly [infer Head, ...infer Tail]
    ? Head extends number
      ? Tail extends Shape
        ? ComputeStridesHelper<Tail, readonly [...Acc, Product<readonly [...Tail]>]>
        : never
      : never
    : never;

// =============================================================================
// Storage Transformations (Lazy Operations)
// =============================================================================

/**
 * Union of all supported tensor operation types
 *
 * This exhaustive list ensures that device implementations must handle
 * all operations. Adding a new operation here will cause TypeScript
 * errors in device implementations until they add support.
 */
export type AllOperationTypes =
  | 'create' // Tensor creation
  | 'neg' // Unary negation
  | 'abs' // Absolute value
  | 'sin' // Sine
  | 'cos' // Cosine
  | 'exp' // Exponential
  | 'log' // Natural logarithm
  | 'sqrt' // Square root
  | 'square' // Square
  | 'add' // Element-wise addition
  | 'sub' // Element-wise subtraction
  | 'mul' // Element-wise multiplication
  | 'div' // Element-wise division
  | 'reshape' // Reshape view
  | 'view' // View with dimension inference
  | 'slice' // Tensor slicing
  | 'flatten' // Flatten to 1D
  | 'permute' // Permute dimensions
  | 'matmul' // Matrix multiplication
  | 'transpose'; // Transpose dimensions

/**
 * Base interface for all storage transformations
 * Transformations are lazy and only define the operation type
 */
export interface StorageTransformation<
  OpType extends AllOperationTypes,
  Output extends TensorStorage<AnyDType, Shape, Shape, LayoutFlags>,
  Inputs extends readonly TensorStorage<AnyDType, Shape, Shape, LayoutFlags>[] = readonly [],
> {
  readonly __op: OpType;
  readonly __output: Output;
  readonly __inputs: Inputs;
}

/**
 * Forces TypeScript to check that all operation cases are handled
 * Use this in the default case of switch statements to ensure exhaustiveness
 *
 * @param op - Should be `never` if all cases are handled
 * @throws {Error} Always throws with operation details
 *
 * @example
 * switch (op.__op) {
 *   case 'add': return handleAdd(op);
 *   case 'sub': return handleSub(op);
 *   // ... all other cases
 *   default:
 *     return assertExhaustiveSwitch(op.__op); // TypeScript error if cases missing
 * }
 */
export function assertExhaustiveSwitch(op: never): never {
  throw new Error(`Unhandled operation: ${String(op)}`);
}

// =============================================================================
// Type utilities for working with storage
// =============================================================================

/**
 * Type constraint for any valid TensorStorage
 */
export type AnyTensorStorage = TensorStorage<AnyDType, Shape, Shape, LayoutFlags>;

/**
 * Type constraint for any valid StorageTransformation
 * Now constrained to only allow known operation types
 */
export type AnyStorageTransformation = StorageTransformation<
  AllOperationTypes,
  AnyTensorStorage,
  readonly AnyTensorStorage[]
>;

/**
 * Extract dtype from storage
 */
export type DTypeOf<T extends AnyTensorStorage> = T['__dtype'];

/**
 * Extract shape from storage
 */
export type ShapeOf<T extends AnyTensorStorage> = T['__shape'];

/**
 * Extract strides from storage
 */
export type StridesOf<T extends AnyTensorStorage> = T['__strides'];

/**
 * Extract layout flags from storage
 */
export type LayoutOf<T extends AnyTensorStorage> = T['__layout'];

/**
 * Extract the output storage from a transformation
 * This is useful for getting the resulting TensorStorage from operations
 *
 * @example
 * type NegOp = Neg<SomeTensor>;
 * type Result = OutputOf<NegOp>; // TensorStorage<...>
 */
export type OutputOf<T extends AnyStorageTransformation> = T['__output'];

/**
 * Create operation for initial tensor creation
 * This wraps a TensorStorage in a transformation with no inputs
 *
 * @example
 * type Initial = CreateOp<TensorStorage<Float32, [2, 3]>>;
 */
export type CreateOp<T extends AnyTensorStorage> = StorageTransformation<'create', T, readonly []>;
