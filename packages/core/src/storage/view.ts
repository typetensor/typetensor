/**
 * View operations for tensor storage with type-safe transformations
 *
 * This module provides operations that create different views of tensor data
 * (reshape, flatten, etc.) with compile-time shape validation and layout preservation.
 */

import type { Shape, Product, CanReshape, ShapeToString } from '../shape/types';
import type { Divide, Mod } from 'ts-arithmetic';
import type {
  TensorStorage,
  StorageTransformation,
  LayoutFlags,
  AnyTensorStorage,
  ComputeStrides,
} from './layout';

// =============================================================================
// Helper Types
// =============================================================================

/**
 * Check if A is evenly divisible by B (no remainder)
 * IsDivisible<12, 3> = true
 * IsDivisible<12, 5> = false
 */
type IsDivisible<A extends number, B extends number> = Mod<A, B> extends 0 ? true : false;

// =============================================================================
// View Operations
// =============================================================================

/**
 * Compute output layout for view operations
 * Views preserve the contiguity type of the input tensor
 */
interface ViewLayout<InputLayout extends LayoutFlags> extends LayoutFlags {
  // Preserve the contiguity type (C stays C, F stays F)
  readonly c_contiguous: InputLayout['c_contiguous'];
  readonly f_contiguous: InputLayout['f_contiguous'];
  // View operations create views, not copies
  readonly is_view: true;
  // Inherit writeability from input
  readonly writeable: InputLayout['writeable'];
  // Views of aligned data remain aligned
  readonly aligned: InputLayout['aligned'];
}

/**
 * Check if a tensor can be reshaped based on its layout
 * Only contiguous tensors (C or F order) can be reshaped as views
 */
type CanReshapeStorage<T extends AnyTensorStorage> = T['__layout']['c_contiguous'] extends true
  ? true
  : T['__layout']['f_contiguous'] extends true
    ? true
    : false;

/**
 * Compute F-order (column-major) strides from shape
 * For shape [2, 3, 4], F-order strides are [1, 2, 6]
 */
type ComputeFortranStrides<S extends Shape> = ComputeFortranStridesHelper<S, 1, readonly []>;

type ComputeFortranStridesHelper<
  S extends Shape,
  CurrentStride extends number,
  Acc extends Shape,
> = S extends readonly []
  ? Acc
  : S extends readonly [infer Head, ...infer Tail]
    ? Head extends number
      ? Tail extends Shape
        ? ComputeFortranStridesHelper<
            Tail,
            Product<readonly [CurrentStride, Head]>,
            readonly [...Acc, CurrentStride]
          >
        : never
      : never
    : never;

/**
 * Compute strides for reshape based on tensor's memory layout
 */
type ComputeReshapeStrides<
  Input extends AnyTensorStorage,
  NewShape extends Shape,
> = Input['__layout']['c_contiguous'] extends true
  ? ComputeStrides<NewShape> // C-contiguous: use C-order strides
  : Input['__layout']['f_contiguous'] extends true
    ? ComputeFortranStrides<NewShape> // F-contiguous: use F-order strides
    : never; // Non-contiguous: cannot reshape

/**
 * Reshape operation with compile-time shape validation
 *
 * Creates a new view of the tensor with a different shape.
 * The total number of elements must remain the same.
 *
 * Requirements:
 * - Input tensor must be contiguous (C or F order)
 * - Total number of elements must remain the same
 * - Result is a view of the original data
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3, 4]>; // 24 elements
 * type B = ReshapeOp<A, [6, 4]>; // Valid: 24 elements
 * type C = ReshapeOp<A, [5, 5]>; // Error: incompatible shapes
 */
export type ReshapeOp<Input extends AnyTensorStorage, NewShape extends Shape> =
  CanReshapeStorage<Input> extends false
    ? never & {
        __error: 'Cannot reshape non-contiguous tensor. Call contiguous() first.';
        __hint: 'Non-contiguous tensors must be made contiguous before reshaping.';
      }
    : CanReshape<Input['__shape'], NewShape> extends true
      ? StorageTransformation<
          'reshape',
          TensorStorage<
            Input['__dtype'],
            NewShape,
            ComputeReshapeStrides<Input, NewShape>,
            ViewLayout<Input['__layout']>
          >,
          readonly [Input]
        >
      : never & {
          __error: `Cannot reshape tensor of shape [${ShapeToString<Input['__shape']>}] to shape [${ShapeToString<NewShape>}]`;
          __cause: 'Total number of elements must be the same';
          __from_size: Product<Input['__shape']>;
          __to_size: Product<NewShape>;
        };

/**
 * Flatten operation - reshape tensor to 1D
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3, 4]>;
 * type B = Flatten<A>; // Shape: [24]
 */
export type Flatten<T extends AnyTensorStorage> = ReshapeOp<T, readonly [Product<T['__shape']>]>;

/**
 * View operation with dimension inference
 * Allows using -1 for one dimension to be automatically inferred
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3, 4]>; // 24 elements
 * type B = View<A, [6, -1]>; // Shape: [6, 4] (4 is inferred)
 * type C = View<A, [-1, 6]>; // Shape: [4, 6] (4 is inferred)
 */
export type View<Input extends AnyTensorStorage, NewShape extends readonly (number | -1)[]> =
  InferViewShape<Input['__shape'], NewShape> extends infer InferredShape
    ? InferredShape extends Shape
      ? ReshapeOp<Input, InferredShape>
      : never & {
          __error: 'Invalid view shape specification';
          __hint: 'Only one dimension can be -1';
        }
    : never;

/**
 * Helper to infer shape when -1 is used
 */
type InferViewShape<OldShape extends Shape, NewShape extends readonly (number | -1)[]> =
  CountNegativeOnes<NewShape> extends 0
    ? NewShape extends Shape
      ? NewShape // No -1, use as-is
      : never
    : CountNegativeOnes<NewShape> extends 1
      ? InferSingleDimension<OldShape, NewShape>
      : never; // More than one -1

/**
 * Count number of -1s in shape specification
 */
type CountNegativeOnes<S extends readonly (number | -1)[]> = CountNegativeOnesHelper<S, 0>;

type CountNegativeOnesHelper<
  S extends readonly (number | -1)[],
  Count extends number,
> = S extends readonly []
  ? Count
  : S extends readonly [infer Head, ...infer Tail]
    ? Head extends -1
      ? Tail extends readonly (number | -1)[]
        ? CountNegativeOnesHelper<Tail, Inc<Count>>
        : never
      : Tail extends readonly (number | -1)[]
        ? CountNegativeOnesHelper<Tail, Count>
        : never
    : never;

/**
 * Increment helper (limited but sufficient for our use case)
 */
type Inc<N extends number> = N extends 0 ? 1 : N extends 1 ? 2 : N extends 2 ? 3 : number;

/**
 * Infer the value of -1 dimension based on total size
 */
type InferSingleDimension<
  OldShape extends Shape,
  NewShape extends readonly (number | -1)[],
> = ComputeInferredShape<Product<OldShape>, NewShape, readonly []>;

/**
 * Helper to compute shape with inferred dimension
 */
type ComputeInferredShape<
  TotalSize extends number,
  RemainingShape extends readonly (number | -1)[],
  Acc extends readonly number[],
> = RemainingShape extends readonly []
  ? Acc
  : RemainingShape extends readonly [infer Head, ...infer Tail]
    ? Head extends -1
      ? Tail extends readonly (number | -1)[]
        ? ProductOfKnown<readonly [...Acc, ...Tail]> extends infer Divisor
          ? Divisor extends number
            ? IsDivisible<TotalSize, Divisor> extends true
              ? ComputeInferredShape<TotalSize, Tail, readonly [...Acc, Divide<TotalSize, Divisor>]>
              : never // Division would result in non-integer
            : never
          : never
        : never
      : Head extends number
        ? Tail extends readonly (number | -1)[]
          ? ComputeInferredShape<TotalSize, Tail, readonly [...Acc, Head]>
          : never
        : never
    : never;

/**
 * Product of known dimensions (skip -1)
 */
type ProductOfKnown<S extends readonly (number | -1)[]> = ProductOfKnownHelper<S, 1>;

type ProductOfKnownHelper<
  S extends readonly (number | -1)[],
  Acc extends number,
> = S extends readonly []
  ? Acc
  : S extends readonly [infer Head, ...infer Tail]
    ? Head extends -1
      ? Tail extends readonly (number | -1)[]
        ? ProductOfKnownHelper<Tail, Acc>
        : Acc
      : Head extends number
        ? Tail extends readonly (number | -1)[]
          ? ProductOfKnownHelper<Tail, Product<readonly [Acc, Head]>>
          : Product<readonly [Acc, Head]>
        : Acc
    : Acc;
