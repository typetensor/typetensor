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
  Squeeze,
  Unsqueeze,
  ExpandShape,
  CanExpand,
  TileShape,
  Length,
  Drop,
} from '../shape/types';
import type { Divide, Mod, Multiply, Compare, Subtract } from 'ts-arithmetic';
import type { Decrement } from '../arithmetic';
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
        > & {
          // Store the permutation axes as metadata for device implementation
          readonly __permuteAxes: Axes;
        },
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

// =============================================================================
// Squeeze and Unsqueeze Operations
// =============================================================================

/**
 * Squeeze operation storage transformation
 *
 * Removes dimensions of size 1 from the tensor shape.
 * If no axes are specified, removes all size-1 dimensions.
 * If axes are specified, removes only those dimensions (which must have size 1).
 *
 * @example
 * type A = TensorStorage<Float32, [2, 1, 3, 1]>;
 * type B = SqueezeOp<A, undefined>; // Global squeeze: Shape [2, 3]
 * type C = SqueezeOp<A, [1]>; // Axis-specific: Shape [2, 3, 1]
 * type D = SqueezeOp<A, [0]>; // Error: dimension 0 has size 2, not 1
 */
export type SqueezeOp<
  Input extends AnyTensorStorage,
  Axes extends readonly number[] | undefined = undefined,
> = StorageTransformation<
  'squeeze',
  TensorStorage<
    Input['__dtype'],
    Axes extends undefined
      ? Squeeze<Input['__shape'], -1> // Global squeeze
      : Axes extends readonly number[]
        ? SqueezeAxes<Input['__shape'], Axes> // Axis-specific squeeze
        : never,
    ComputeSqueezedStrides<Input['__strides'], Input['__shape'], Axes>,
    ViewLayout<Input['__layout']>
  > & {
    // Store the squeeze axes as metadata for device implementation
    readonly __squeezeAxes: Axes;
  },
  readonly [Input]
>;

/**
 * Helper: Squeeze specific axes from shape
 */
type SqueezeAxes<S extends Shape, Axes extends readonly number[]> = SqueezeAxesHelper<
  S,
  NormalizeAxes<Axes, S['length']>,
  0,
  readonly []
>;

type SqueezeAxesHelper<
  S extends Shape,
  NormalizedAxes extends readonly number[],
  CurrentIndex extends number,
  Acc extends Shape,
> = S extends readonly [infer Head, ...infer Tail]
  ? Head extends number
    ? Tail extends Shape
      ? Contains<NormalizedAxes, CurrentIndex> extends true
        ? SqueezeAxesHelper<Tail, NormalizedAxes, Inc<CurrentIndex>, Acc> // Skip this dimension
        : SqueezeAxesHelper<Tail, NormalizedAxes, Inc<CurrentIndex>, readonly [...Acc, Head]> // Keep this dimension
      : Acc
    : Acc
  : Acc;

/**
 * Helper: Normalize negative axes
 */
type NormalizeAxes<Axes extends readonly number[], Rank extends number> = {
  readonly [K in keyof Axes]: Axes[K] extends number
    ? Axes[K] extends -1
      ? Subtract<Rank, 1>
      : Axes[K] extends -2
        ? Subtract<Rank, 2>
        : Axes[K]
    : never;
};

/**
 * Helper: Compute strides after squeeze operation
 */
type ComputeSqueezedStrides<
  InputStrides extends Shape,
  InputShape extends Shape,
  Axes extends readonly number[] | undefined,
> = Axes extends undefined
  ? ComputeGlobalSqueezedStrides<InputStrides, InputShape, 0, readonly []>
  : Axes extends readonly number[]
    ? ComputeAxisSqueezedStrides<
        InputStrides,
        NormalizeAxes<Axes, InputShape['length']>,
        0,
        readonly []
      >
    : never;

type ComputeGlobalSqueezedStrides<
  InputStrides extends Shape,
  InputShape extends Shape,
  CurrentIndex extends number,
  Acc extends Shape,
> = InputShape extends readonly [infer HeadDim, ...infer TailShape]
  ? InputStrides extends readonly [infer HeadStride, ...infer TailStrides]
    ? HeadDim extends 1
      ? TailShape extends Shape
        ? TailStrides extends Shape
          ? ComputeGlobalSqueezedStrides<TailStrides, TailShape, Inc<CurrentIndex>, Acc> // Skip size-1 dimension
          : Acc
        : Acc
      : TailShape extends Shape
        ? TailStrides extends Shape
          ? HeadStride extends number
            ? ComputeGlobalSqueezedStrides<
                TailStrides,
                TailShape,
                Inc<CurrentIndex>,
                readonly [...Acc, HeadStride]
              >
            : Acc
          : Acc
        : Acc
    : Acc
  : Acc;

type ComputeAxisSqueezedStrides<
  InputStrides extends Shape,
  NormalizedAxes extends readonly number[],
  CurrentIndex extends number,
  Acc extends Shape,
> = InputStrides extends readonly [infer HeadStride, ...infer TailStrides]
  ? Contains<NormalizedAxes, CurrentIndex> extends true
    ? TailStrides extends Shape
      ? ComputeAxisSqueezedStrides<TailStrides, NormalizedAxes, Inc<CurrentIndex>, Acc> // Skip this stride
      : Acc
    : TailStrides extends Shape
      ? HeadStride extends number
        ? ComputeAxisSqueezedStrides<
            TailStrides,
            NormalizedAxes,
            Inc<CurrentIndex>,
            readonly [...Acc, HeadStride]
          >
        : Acc
      : Acc
  : Acc;

/**
 * Unsqueeze operation storage transformation
 *
 * Adds a dimension of size 1 at the specified axis position.
 * The axis can be negative to indicate position from the end.
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3]>;
 * type B = UnsqueezeOp<A, 0>; // Shape: [1, 2, 3]
 * type C = UnsqueezeOp<A, 1>; // Shape: [2, 1, 3]
 * type D = UnsqueezeOp<A, -1>; // Shape: [2, 3, 1]
 */
export type UnsqueezeOp<
  Input extends AnyTensorStorage,
  Axis extends number,
> = StorageTransformation<
  'unsqueeze',
  TensorStorage<
    Input['__dtype'],
    Unsqueeze<Input['__shape'], Axis>,
    ComputeUnsqueezedStrides<Input['__strides'], Input['__shape'], Axis>,
    ViewLayout<Input['__layout']>
  > & {
    // Store the unsqueeze axis as metadata for device implementation
    readonly __unsqueezeAxis: Axis;
  },
  readonly [Input]
>;

/**
 * Helper: Compute strides after unsqueeze operation
 */
type ComputeUnsqueezedStrides<
  InputStrides extends Shape,
  InputShape extends Shape,
  Axis extends number,
> = ComputeUnsqueezedStridesHelper<
  InputStrides,
  NormalizeUnsqueezeAxis<Axis, InputShape['length']>,
  0,
  readonly []
>;

type ComputeUnsqueezedStridesHelper<
  InputStrides extends Shape,
  NormalizedAxis extends number,
  CurrentIndex extends number,
  Acc extends Shape,
> = CurrentIndex extends NormalizedAxis
  ? readonly [...Acc, 1, ...InputStrides] // Insert stride of 1 at the target position
  : InputStrides extends readonly [infer HeadStride, ...infer TailStrides]
    ? HeadStride extends number
      ? TailStrides extends Shape
        ? ComputeUnsqueezedStridesHelper<
            TailStrides,
            NormalizedAxis,
            Inc<CurrentIndex>,
            readonly [...Acc, HeadStride]
          >
        : readonly [...Acc, HeadStride]
      : Acc
    : CurrentIndex extends NormalizedAxis
      ? readonly [...Acc, 1] // Insert at end if we've processed all input strides
      : Acc;

/**
 * Helper: Normalize unsqueeze axis (handle negative indices)
 */
type NormalizeUnsqueezeAxis<Axis extends number, Rank extends number> = Axis extends number
  ? Axis extends -1
    ? Rank
    : Axis extends -2
      ? Subtract<Rank, 1>
      : Axis
  : never;

// =============================================================================
// Expand and Tile Operations
// =============================================================================

/**
 * Expand operation storage transformation
 *
 * Broadcasts tensor to new shape by expanding singleton dimensions.
 * This is a view operation - no data is copied.
 *
 * Rules:
 * - Can only expand dimensions of size 1
 * - Use -1 to keep existing dimension size
 * - Can add new dimensions on the left
 *
 * @example
 * type A = TensorStorage<Float32, [3, 1, 5]>;
 * type B = ExpandOp<A, [3, 4, 5]>; // Expands middle dim from 1 to 4
 * type C = ExpandOp<A, [-1, 4, -1]>; // Same, using -1 to keep dims
 * type D = ExpandOp<A, [2, 3, 4, 5]>; // Adds new dimension
 */
export type ExpandOp<Input extends AnyTensorStorage, TargetShape extends readonly (number | -1)[]> =
  CanExpand<Input['__shape'], TargetShape> extends true
    ? StorageTransformation<
        'expand',
        TensorStorage<
          Input['__dtype'],
          ExpandShape<Input['__shape'], TargetShape>,
          ComputeExpandedStrides<Input['__strides'], Input['__shape'], TargetShape>,
          ViewLayout<Input['__layout']>
        > & {
          readonly __expandTargetShape: TargetShape;
        },
        readonly [Input]
      >
    : never & {
        __error: 'Cannot expand: incompatible shapes';
        __hint: 'Can only expand singleton dimensions (size 1)';
        __inputShape: Input['__shape'];
        __targetShape: TargetShape;
      };

/**
 * Compute strides for expanded tensor
 * Expanded dimensions get stride 0 (broadcasting)
 */
type ComputeExpandedStrides<
  InputStrides extends Shape,
  InputShape extends Shape,
  TargetShape extends readonly (number | -1)[],
> =
  Length<TargetShape> extends infer TargetLen
    ? TargetLen extends number
      ? Length<InputShape> extends infer InputLen
        ? InputLen extends number
          ? Compare<TargetLen, InputLen> extends 1 // More target dims
            ? ComputeExpandedStridesWithNewDims<
                InputStrides,
                InputShape,
                TargetShape,
                Subtract<TargetLen, InputLen>
              >
            : ComputeExpandedStridesSameDims<InputStrides, InputShape, TargetShape>
          : never
        : never
      : never
    : never;

/**
 * Compute strides when adding new dimensions (prepend 0s)
 */
type ComputeExpandedStridesWithNewDims<
  InputStrides extends Shape,
  InputShape extends Shape,
  TargetShape extends readonly (number | -1)[],
  NewDimsCount extends number,
> = NewDimsCount extends 0
  ? ComputeExpandedStridesSameDims<InputStrides, InputShape, TargetShape>
  : readonly [
      0,
      ...ComputeExpandedStridesWithNewDims<
        InputStrides,
        InputShape,
        Drop<TargetShape, 1>,
        Decrement<NewDimsCount>
      >,
    ];

/**
 * Compute strides for same number of dimensions
 * Expanded dims (from 1 to N) get stride 0
 */
type ComputeExpandedStridesSameDims<
  InputStrides extends Shape,
  InputShape extends Shape,
  TargetShape extends readonly (number | -1)[],
> = InputStrides extends readonly []
  ? readonly []
  : InputStrides extends readonly [infer FirstStride, ...infer RestStrides]
    ? InputShape extends readonly [infer FirstShape, ...infer RestShape]
      ? TargetShape extends readonly [infer FirstTarget, ...infer RestTarget]
        ? FirstStride extends number
          ? RestStrides extends Shape
            ? FirstShape extends number
              ? RestShape extends Shape
                ? RestTarget extends readonly (number | -1)[]
                  ? FirstShape extends 1
                    ? FirstTarget extends -1
                      ? readonly [
                          FirstStride,
                          ...ComputeExpandedStridesSameDims<RestStrides, RestShape, RestTarget>,
                        ]
                      : readonly [
                          0,
                          ...ComputeExpandedStridesSameDims<RestStrides, RestShape, RestTarget>,
                        ] // Expanded: stride 0
                    : readonly [
                        FirstStride,
                        ...ComputeExpandedStridesSameDims<RestStrides, RestShape, RestTarget>,
                      ] // Not expanded
                  : never
                : never
              : never
            : never
          : never
        : never
      : never
    : never;

/**
 * Layout for copy operations (not a view)
 */
interface CopyLayout extends LayoutFlags {
  readonly c_contiguous: true; // New tensors are C-contiguous
  readonly f_contiguous: false;
  readonly is_view: false; // Not a view
  readonly writeable: true;
  readonly aligned: true;
}

/**
 * Tile operation storage transformation
 *
 * Repeats tensor along specified dimensions.
 * This creates a new tensor - data is copied.
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3]>;
 * type B = TileOp<A, [2, 3]>; // Shape: [4, 9]
 * type C = TileOp<A, [2, 1, 3]>; // Shape: [2, 2, 9] (adds dim)
 */
export type TileOp<
  Input extends AnyTensorStorage,
  Reps extends readonly number[],
> = StorageTransformation<
  'tile',
  TensorStorage<
    Input['__dtype'],
    TileShape<Input['__shape'], Reps>,
    ComputeStrides<TileShape<Input['__shape'], Reps>>,
    CopyLayout
  > & {
    readonly __tileReps: Reps;
  },
  readonly [Input]
>;
