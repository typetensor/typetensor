/**
 * Einops operations for tensor storage with type-safe transformations
 *
 * This module provides einops-style operations that create different views of tensor data
 * with compile-time pattern validation and shape computation.
 */

import type { Shape, ShapeToString } from '../shape/types';
import type { AnyDType } from '../dtype/types';
import type {
  TensorStorage,
  StorageTransformation,
  LayoutFlags,
  AnyTensorStorage,
  ComputeStrides,
} from './layout';
import type { ResolveEinopsShape } from '../einops/type-shape-resolver-rearrange';
import type { ResolveReduceShape } from '../einops/type-shape-resolver-reduce';
import type { ReductionOp } from '../einops/reduce';

// =============================================================================
// Layout Types
// =============================================================================

/**
 * Compute output layout for rearrange operations
 * Rearrange operations create views with modified layout properties
 */
interface RearrangeLayout<InputLayout extends LayoutFlags> extends LayoutFlags {
  // Rearrange operations typically break contiguity
  readonly c_contiguous: false;
  readonly f_contiguous: false;
  // Rearrange operations create views, not copies
  readonly is_view: true;
  // Inherit writeability from input
  readonly writeable: InputLayout['writeable'];
  // Views of aligned data remain aligned
  readonly aligned: InputLayout['aligned'];
}

// =============================================================================
// Rearrange Operation
// =============================================================================

/**
 * Rearrange operation with einops pattern-based transformation
 *
 * Creates a new view of the tensor with dimensions rearranged according to
 * an einops pattern string.
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3]>;
 * type B = RearrangeOp<A, 'h w -> w h'>; // Shape: [3, 2]
 * type C = RearrangeOp<A, 'h w -> (h w)'>; // Shape: [6]
 */
export type RearrangeOp<
  Input extends AnyTensorStorage,
  Pattern extends string,
  Axes extends Record<string, number> | undefined = undefined,
> =
  ResolveEinopsShape<Pattern, Input['__shape'], Axes> extends infer OutputShape
    ? OutputShape extends Shape
      ? StorageTransformation<
          'rearrange',
          TensorStorage<
            Input['__dtype'],
            OutputShape,
            ComputeStrides<OutputShape>, // TODO: This may need special handling for non-contiguous results
            RearrangeLayout<Input['__layout']>
          >,
          readonly [Input]
        >
      : never & {
          __error: 'Failed to resolve einops pattern';
          __pattern: Pattern;
          __inputShape: ShapeToString<Input['__shape']>;
        }
    : never;

// =============================================================================
// Reduce Operation
// =============================================================================

/**
 * Compute output layout for reduce operations
 * Reduce operations always produce new data (not views)
 */
interface ReduceLayout extends LayoutFlags {
  // Reduce operations typically produce contiguous output
  readonly c_contiguous: true;
  readonly f_contiguous: false;
  // Reduce operations create new data, not views
  readonly is_view: false;
  // New data is always writeable
  readonly writeable: true;
  // New data is aligned
  readonly aligned: true;
}

/**
 * Reduce operation with einops pattern-based aggregation
 *
 * Reduces tensor dimensions according to an einops pattern and aggregation operation.
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3, 4]>;
 * type B = ReduceEinopsOp<A, 'h w c -> c', 'mean'>; // Shape: [4]
 * type C = ReduceEinopsOp<A, 'h w c -> h', 'sum'>; // Shape: [2]
 * type D = ReduceEinopsOp<A, '(h 2) w c -> h w c', 'max', false, {h: 1}>; // Shape: [1, 3, 4]
 */
/**
 * Helper type to extract input shape and dtype from tensor storage
 */
export type ExtractInputInfo<Input extends AnyTensorStorage> = Input extends {
  __shape: infer InputShape;
  __dtype: infer InputDtype;
}
  ? InputShape extends Shape
    ? InputDtype extends AnyDType
      ? { shape: InputShape; dtype: InputDtype }
      : never
    : never
  : never;

/**
 * Helper type to resolve reduce operation
 */
export type ResolveReduceOp<
  InputInfo extends { shape: Shape; dtype: AnyDType },
  Pattern extends string,
  Operation extends ReductionOp,
  KeepDims extends boolean,
  Axes extends Record<string, number> | undefined,
  Input extends AnyTensorStorage,
> =
  ResolveReduceShape<Pattern, InputInfo['shape'], KeepDims, Axes> extends infer OutputShape
    ? OutputShape extends Shape
      ? StorageTransformation<
          'reduce',
          TensorStorage<InputInfo['dtype'], OutputShape, ComputeStrides<OutputShape>, ReduceLayout>,
          readonly [Input]
        >
      : never & {
          __error: 'Failed to resolve reduce pattern';
          __pattern: Pattern;
          __operation: Operation;
          __inputShape: ShapeToString<InputInfo['shape']>;
        }
    : never;

export type ReduceEinopsOp<
  Input extends AnyTensorStorage,
  Pattern extends string,
  Operation extends ReductionOp,
  KeepDims extends boolean = false,
  Axes extends Record<string, number> | undefined = undefined,
> =
  ExtractInputInfo<Input> extends infer InputInfo
    ? InputInfo extends { shape: Shape; dtype: AnyDType }
      ? ResolveReduceOp<InputInfo, Pattern, Operation, KeepDims, Axes, Input>
      : never
    : never;
