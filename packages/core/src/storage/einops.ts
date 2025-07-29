/**
 * Einops operations for tensor storage with type-safe transformations
 *
 * This module provides einops-style operations that create different views of tensor data
 * with compile-time pattern validation and shape computation.
 */

import type { Shape, ShapeToString } from '../shape/types';
import type {
  TensorStorage,
  StorageTransformation,
  LayoutFlags,
  AnyTensorStorage,
  ComputeStrides,
} from './layout';
import type { ResolveEinopsShape } from '../einops/type-shape-resolver';

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
