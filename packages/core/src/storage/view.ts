/**
 * View operations for tensor storage with type-safe transformations
 *
 * This module provides operations that create different views of tensor data
 * (reshape, flatten, etc.) with compile-time shape validation and layout preservation.
 */

import type {
  Shape,
  Product,
  CanReshape,
  ShapeToString,
  SlicedShape,
  SliceSpec,
  SliceIndex,
  Permute,
} from '../shape/types';
import type { Divide, Mod, Multiply, Compare } from 'ts-arithmetic';
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

// =============================================================================
// Slice Operations
// =============================================================================

/**
 * Compute new strides after slicing
 * Integer indices remove the corresponding stride, slices preserve it
 *
 * Note: The actual stride value may be multiplied by step at runtime
 */
export type ComputeSlicedStrides<
  InputStrides extends Shape,
  Indices extends readonly SliceIndex[],
> = ComputeSlicedStridesHelper<InputStrides, Indices, readonly []>;

type ComputeSlicedStridesHelper<
  RemainingStrides extends Shape,
  RemainingIndices extends readonly SliceIndex[],
  Acc extends Shape,
> = RemainingIndices extends readonly []
  ? readonly [...Acc, ...RemainingStrides] // No more indices
  : RemainingStrides extends readonly []
    ? Acc // No more strides
    : RemainingIndices extends readonly [infer FirstIndex, ...infer RestIndices]
      ? RemainingStrides extends readonly [infer FirstStride, ...infer RestStrides]
        ? FirstStride extends number
          ? RestStrides extends Shape
            ? RestIndices extends readonly SliceIndex[]
              ? FirstIndex extends number
                ? ComputeSlicedStridesHelper<RestStrides, RestIndices, Acc> // Integer index: remove stride
                : FirstIndex extends null
                  ? ComputeSlicedStridesHelper<
                      RestStrides,
                      RestIndices,
                      readonly [...Acc, FirstStride] // null: preserve stride
                    >
                  : FirstIndex extends SliceSpec
                    ? FirstIndex['step'] extends number
                      ? ComputeSlicedStridesHelper<
                          RestStrides,
                          RestIndices,
                          readonly [...Acc, Multiply<FirstStride, FirstIndex['step']>] // Multiply stride by step
                        >
                      : ComputeSlicedStridesHelper<
                          RestStrides,
                          RestIndices,
                          readonly [...Acc, FirstStride] // No step specified: default to 1, preserve stride
                        >
                    : never
              : never
            : never
          : never
        : never
      : never;

/**
 * Slice operation storage transformation
 *
 * Creates a view of the tensor with a subset of elements along each dimension.
 * The actual slicing computation is performed by the device at execution time.
 *
 * @example
 * type A = TensorStorage<Float32, [10, 20, 30]>;
 * type B = SliceOp<A, [SliceSpec, 5, null]>; // Shape: [5, 30]
 * type C = SliceOp<A, [null, null, SliceSpec]>; // Shape: [10, 20, number]
 *
 * The device implementation will use the indices to compute the actual offset
 * and shape at runtime.
 */
export type SliceOp<Input extends AnyTensorStorage, Indices extends readonly SliceIndex[]> =
  SlicedShape<Input['__shape'], Indices> extends never
    ? never // If shape computation fails (e.g., zero step), propagate never
    : StorageTransformation<
        'slice',
        TensorStorage<
          Input['__dtype'],
          SlicedShape<Input['__shape'], Indices>,
          ComputeSlicedStrides<Input['__strides'], Indices>,
          ViewLayout<Input['__layout']>
        > & {
          // Store the slice indices as metadata for device implementation
          readonly __sliceIndices: Indices;
        },
        readonly [Input]
      >;

// =============================================================================
// Transpose and Permute Operations
// =============================================================================

/**
 * Compute layout after transpose/permute operations
 * These operations generally break contiguity
 */
interface TransposeLayout<InputLayout extends LayoutFlags> extends LayoutFlags {
  readonly c_contiguous: false; // Generally breaks C-contiguity
  readonly f_contiguous: false; // Generally breaks F-contiguity
  readonly is_view: true; // Always a view
  readonly writeable: InputLayout['writeable'];
  readonly aligned: InputLayout['aligned'];
}

/**
 * Transpose operation storage transformation
 *
 * Default behavior: swaps the last two dimensions (matrix transpose)
 * For <2D tensors: returns the tensor unchanged
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3, 4]>; // Shape: [2, 3, 4]
 * type B = TransposeOp<A>; // Shape: [2, 4, 3]
 *
 * type C = TensorStorage<Float32, [5]>; // 1D tensor
 * type D = TransposeOp<C>; // Shape: [5] (unchanged)
 */
export type TransposeOp<Input extends AnyTensorStorage> = Input['__shape']['length'] extends 0 | 1
  ? StorageTransformation<'transpose', Input, readonly [Input]> // Return unchanged
  : Input['__shape'] extends readonly [...infer BatchDims, infer SecondLast, infer Last]
    ? SecondLast extends number
      ? Last extends number
        ? BatchDims extends Shape
          ? StorageTransformation<
              'transpose',
              TensorStorage<
                Input['__dtype'],
                readonly [...BatchDims, Last, SecondLast],
                ComputeTransposedStrides<Input['__strides'], Input['__shape']['length']>,
                TransposeLayout<Input['__layout']>
              >,
              readonly [Input]
            >
          : never
        : never
      : never
    : never;

/**
 * Helper: Compute strides after default transpose (swap last two dims)
 */
type ComputeTransposedStrides<Strides extends Shape, Rank extends number> = Rank extends 0 | 1
  ? Strides // No change for scalars or 1D
  : Strides extends readonly [...infer Init, infer SecondLast, infer Last]
    ? Init extends Shape
      ? SecondLast extends number
        ? Last extends number
          ? readonly [...Init, Last, SecondLast]
          : never
        : never
      : never
    : never;

/**
 * Permute operation storage transformation
 *
 * Rearranges tensor dimensions according to the specified axes order.
 * Each axis index must appear exactly once.
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3, 4]>; // Shape: [2, 3, 4]
 * type B = PermuteOp<A, [2, 0, 1]>; // Shape: [4, 2, 3]
 * type C = PermuteOp<A, [1, 0, 2]>; // Shape: [3, 2, 4]
 */
export type PermuteOp<Input extends AnyTensorStorage, Axes extends readonly number[]> =
  IsValidPermutation<Input['__shape'], Axes> extends true
    ? StorageTransformation<
        'permute',
        TensorStorage<
          Input['__dtype'],
          Permute<Input['__shape'], Axes>,
          Permute<Input['__strides'], Axes>, // Reuse Permute from shape module
          TransposeLayout<Input['__layout']>
        >,
        readonly [Input]
      >
    : never & {
        __error: `Invalid permutation axes for tensor of rank ${Input['__shape']['length']}`;
        __hint: 'Axes must contain each dimension index exactly once';
      };

/**
 * Helper: Validate permutation axes
 */
type IsValidPermutation<
  S extends Shape,
  Axes extends readonly number[],
> = Axes['length'] extends S['length']
  ? HasUniqueElements<Axes> extends true
    ? AreAxesInRange<Axes, S['length']> extends true
      ? true
      : false
    : false
  : false;

/**
 * Helper: Check if all elements are unique
 */
type HasUniqueElements<T extends readonly number[]> = T extends readonly []
  ? true
  : T extends readonly [infer Head, ...infer Tail]
    ? Head extends number
      ? Tail extends readonly number[]
        ? Contains<Tail, Head> extends true
          ? false
          : HasUniqueElements<Tail>
        : false
      : false
    : false;

/**
 * Helper: Check if array contains element
 */
type Contains<T extends readonly number[], E extends number> = T extends readonly []
  ? false
  : T extends readonly [infer Head, ...infer Tail]
    ? Head extends E
      ? true
      : Tail extends readonly number[]
        ? Contains<Tail, E>
        : false
    : false;

/**
 * Helper: Check if all axes are in valid range [0, rank)
 */
type AreAxesInRange<Axes extends readonly number[], Rank extends number> = Axes extends readonly []
  ? true
  : Axes extends readonly [infer Head, ...infer Tail]
    ? Head extends number
      ? IsInRange<Head, Rank> extends true
        ? Tail extends readonly number[]
          ? AreAxesInRange<Tail, Rank>
          : false
        : false
      : false
    : false;

/**
 * Helper: Check if number is in range [0, N)
 */
type IsInRange<X extends number, N extends number> = X extends number
  ? `${X}` extends `-${string}`
    ? false // Negative numbers not allowed
    : Compare<X, N> extends -1
      ? true
      : false
  : false;
