/**
 * Reduction operations for tensor storage with type-safe axis validation
 *
 * This module provides reduction operations (sum, mean) that reduce tensor
 * dimensions along specified axes. The operations use compile-time validation
 * to ensure axis parameters are valid and compute output shapes correctly.
 */

import type { AnyDType, ToFloat } from '../dtype/types';
import type { ValidateReduction, ReduceShape, ValidateAxes } from '../shape/types';
import type {
  TensorStorage,
  StorageTransformation,
  LayoutFlags,
  AnyTensorStorage,
  ComputeStrides,
} from './layout';

// =============================================================================
// Reduction Operations
// =============================================================================

/**
 * Compute output layout for reduction operations
 * Reduction operations may preserve layout or produce C-contiguous output
 */
interface ReductionOpLayout<InputLayout extends LayoutFlags> extends LayoutFlags {
  // Backend can choose to preserve layout or make contiguous
  readonly c_contiguous: true | InputLayout['c_contiguous'];
  readonly f_contiguous: InputLayout['f_contiguous'] extends true ? true : false;
  readonly is_view: false; // Reduction ops always copy
  readonly writeable: true;
  readonly aligned: true;
}

/**
 * Helper to compute output shape for reduction operations
 * Handles the case where axes is undefined (reduce all dimensions)
 */
type ComputeReductionShape<
  InputShape extends readonly number[],
  Axes extends readonly number[] | undefined,
  KeepDims extends boolean,
> = Axes extends undefined
  ? KeepDims extends true
    ? ReduceShape<InputShape, readonly [], true> // Keep all dims as 1
    : readonly [] // Scalar result
  : Axes extends readonly number[]
    ? ValidateAxes<Axes, InputShape> extends readonly number[]
      ? ReduceShape<InputShape, ValidateAxes<Axes, InputShape>, KeepDims>
      : never // Invalid axes
    : never; // Invalid axes type

/**
 * Sum reduction operation
 * Computes the sum of tensor elements along specified axes
 *
 * @template Input - Input tensor storage type
 * @template Axes - Axes to reduce along (undefined means reduce all)
 * @template KeepDims - Whether to keep reduced dimensions as size 1
 *
 * @example
 * // Sum along axis 1, remove dimension
 * type Input = TensorStorage<Float32, [2, 3, 4], [12, 4, 1], DefaultLayoutFlags>;
 * type Result = SumOp<Input, [1], false>; // Shape: [2, 4]
 *
 * @example
 * // Sum all elements to scalar
 * type Scalar = SumOp<Input, undefined, false>; // Shape: []
 */
export type SumOp<
  Input extends AnyTensorStorage,
  Axes extends readonly number[] | undefined,
  KeepDims extends boolean = false,
  OutputDType extends AnyDType = Input['__dtype'],
> =
  ValidateReduction<Input['__shape'], Axes> extends true
    ? StorageTransformation<
        'sum',
        TensorStorage<
          OutputDType,
          ComputeReductionShape<Input['__shape'], Axes, KeepDims>,
          ComputeStrides<ComputeReductionShape<Input['__shape'], Axes, KeepDims>>,
          ReductionOpLayout<Input['__layout']>
        >,
        readonly [Input]
      > & {
        // Store reduction metadata for device implementation
        readonly __sumAxes: Axes;
        readonly __keepDims: KeepDims;
      }
    : never; // Invalid reduction parameters result in never type

/**
 * Mean reduction operation
 * Computes the mean of tensor elements along specified axes
 *
 * @template Input - Input tensor storage type
 * @template Axes - Axes to reduce along (undefined means reduce all)
 * @template KeepDims - Whether to keep reduced dimensions as size 1
 * @template OutputDType - Output data type (defaults to float version of input)
 *
 * @example
 * // Mean along last axis, keep dimension
 * type Input = TensorStorage<Int32, [2, 3, 4], [12, 4, 1], DefaultLayoutFlags>;
 * type Result = MeanOp<Input, [-1], true>; // Shape: [2, 3, 1], DType: Float32
 *
 * @example
 * // Global mean (all elements)
 * type GlobalMean = MeanOp<Input, undefined, false>; // Shape: [], DType: Float32
 */
export type MeanOp<
  Input extends AnyTensorStorage,
  Axes extends readonly number[] | undefined,
  KeepDims extends boolean = false,
  OutputDType extends AnyDType = ToFloat<Input['__dtype']>,
> =
  ValidateReduction<Input['__shape'], Axes> extends true
    ? StorageTransformation<
        'mean',
        TensorStorage<
          OutputDType,
          ComputeReductionShape<Input['__shape'], Axes, KeepDims>,
          ComputeStrides<ComputeReductionShape<Input['__shape'], Axes, KeepDims>>,
          ReductionOpLayout<Input['__layout']>
        >,
        readonly [Input]
      > & {
        // Store reduction metadata for device implementation
        readonly __meanAxes: Axes;
        readonly __keepDims: KeepDims;
      }
    : never; // Invalid reduction parameters result in never type

/**
 * Max reduction operation
 * Computes the maximum of tensor elements along specified axes
 *
 * @template Input - Input tensor storage type
 * @template Axes - Axes to reduce along (undefined means reduce all)
 * @template KeepDims - Whether to keep reduced dimensions as size 1
 * @template OutputDType - Output data type (defaults to same as input)
 *
 * @example
 * // Max along axis 1, remove dimension
 * type Input = TensorStorage<Float32, [2, 3, 4], [12, 4, 1], DefaultLayoutFlags>;
 * type Result = MaxOp<Input, [1], false>; // Shape: [2, 4]
 *
 * @example
 * // Global max (all elements)
 * type GlobalMax = MaxOp<Input, undefined, false>; // Shape: []
 */
export type MaxOp<
  Input extends AnyTensorStorage,
  Axes extends readonly number[] | undefined,
  KeepDims extends boolean = false,
  OutputDType extends AnyDType = Input['__dtype'],
> =
  ValidateReduction<Input['__shape'], Axes> extends true
    ? StorageTransformation<
        'max',
        TensorStorage<
          OutputDType,
          ComputeReductionShape<Input['__shape'], Axes, KeepDims>,
          ComputeStrides<ComputeReductionShape<Input['__shape'], Axes, KeepDims>>,
          ReductionOpLayout<Input['__layout']>
        >,
        readonly [Input]
      > & {
        // Store reduction metadata for device implementation
        readonly __maxAxes: Axes;
        readonly __keepDims: KeepDims;
      }
    : never; // Invalid reduction parameters result in never type

/**
 * Min reduction operation
 * Computes the minimum of tensor elements along specified axes
 *
 * @template Input - Input tensor storage type
 * @template Axes - Axes to reduce along (undefined means reduce all)
 * @template KeepDims - Whether to keep reduced dimensions as size 1
 * @template OutputDType - Output data type (defaults to same as input)
 *
 * @example
 * // Min along last axis, keep dimension
 * type Input = TensorStorage<Int32, [2, 3, 4], [12, 4, 1], DefaultLayoutFlags>;
 * type Result = MinOp<Input, [-1], true>; // Shape: [2, 3, 1]
 *
 * @example
 * // Global min (all elements)
 * type GlobalMin = MinOp<Input, undefined, false>; // Shape: []
 */
export type MinOp<
  Input extends AnyTensorStorage,
  Axes extends readonly number[] | undefined,
  KeepDims extends boolean = false,
  OutputDType extends AnyDType = Input['__dtype'],
> =
  ValidateReduction<Input['__shape'], Axes> extends true
    ? StorageTransformation<
        'min',
        TensorStorage<
          OutputDType,
          ComputeReductionShape<Input['__shape'], Axes, KeepDims>,
          ComputeStrides<ComputeReductionShape<Input['__shape'], Axes, KeepDims>>,
          ReductionOpLayout<Input['__layout']>
        >,
        readonly [Input]
      > & {
        // Store reduction metadata for device implementation
        readonly __minAxes: Axes;
        readonly __keepDims: KeepDims;
      }
    : never; // Invalid reduction parameters result in never type

// =============================================================================
// Type-Level Reduction Functions
// =============================================================================

/**
 * Type-level sum function for use in other type computations
 *
 * @template T - Input tensor storage type
 * @template Axes - Axes to reduce along
 * @template KeepDims - Whether to keep reduced dimensions
 *
 * @example
 * type Input = TensorStorage<Float32, [2, 3, 4], [12, 4, 1], DefaultLayoutFlags>;
 * type BatchSum = Sum<Input, [0]>; // Sum over batch dimension
 */
export type Sum<
  T extends AnyTensorStorage,
  Axes extends readonly number[] | undefined = undefined,
  KeepDims extends boolean = false,
> = SumOp<T, Axes, KeepDims>;

/**
 * Type-level mean function for use in other type computations
 *
 * @template T - Input tensor storage type
 * @template Axes - Axes to reduce along
 * @template KeepDims - Whether to keep reduced dimensions
 *
 * @example
 * type Input = TensorStorage<Int32, [32, 128], [128, 1], DefaultLayoutFlags>;
 * type FeatureMean = Mean<Input, [-1], true>; // Mean over features, keep dim
 */
export type Mean<
  T extends AnyTensorStorage,
  Axes extends readonly number[] | undefined = undefined,
  KeepDims extends boolean = false,
> = MeanOp<T, Axes, KeepDims>;

/**
 * Type-level max function for use in other type computations
 *
 * @template T - Input tensor storage type
 * @template Axes - Axes to reduce along
 * @template KeepDims - Whether to keep reduced dimensions
 *
 * @example
 * type Input = TensorStorage<Float32, [2, 3, 4], [12, 4, 1], DefaultLayoutFlags>;
 * type BatchMax = Max<Input, [0]>; // Max over batch dimension
 */
export type Max<
  T extends AnyTensorStorage,
  Axes extends readonly number[] | undefined = undefined,
  KeepDims extends boolean = false,
> = MaxOp<T, Axes, KeepDims>;

/**
 * Type-level min function for use in other type computations
 *
 * @template T - Input tensor storage type
 * @template Axes - Axes to reduce along
 * @template KeepDims - Whether to keep reduced dimensions
 *
 * @example
 * type Input = TensorStorage<Int32, [32, 128], [128, 1], DefaultLayoutFlags>;
 * type FeatureMin = Min<Input, [-1], true>; // Min over features, keep dim
 */
export type Min<
  T extends AnyTensorStorage,
  Axes extends readonly number[] | undefined = undefined,
  KeepDims extends boolean = false,
> = MinOp<T, Axes, KeepDims>;

// =============================================================================
// Convenience Types for Common Use Cases
// =============================================================================

/**
 * Sum over all dimensions (global sum)
 * Returns scalar result
 */
export type GlobalSum<T extends AnyTensorStorage> = Sum<T, undefined, false>;

/**
 * Mean over all dimensions (global mean)
 * Returns scalar result
 */
export type GlobalMean<T extends AnyTensorStorage> = Mean<T, undefined, false>;

/**
 * Sum over the last dimension (most common for features)
 * Removes the last dimension
 */
export type SumLastDim<T extends AnyTensorStorage> = Sum<T, readonly [-1], false>;

/**
 * Mean over the last dimension (most common for features)
 * Removes the last dimension
 */
export type MeanLastDim<T extends AnyTensorStorage> = Mean<T, readonly [-1], false>;

/**
 * Sum over the first dimension (batch reduction)
 * Removes the first dimension
 */
export type SumBatch<T extends AnyTensorStorage> = Sum<T, readonly [0], false>;

/**
 * Mean over the first dimension (batch reduction)
 * Removes the first dimension
 */
export type MeanBatch<T extends AnyTensorStorage> = Mean<T, readonly [0], false>;

/**
 * Max over all dimensions (global max)
 * Returns scalar result
 */
export type GlobalMax<T extends AnyTensorStorage> = Max<T, undefined, false>;

/**
 * Min over all dimensions (global min)
 * Returns scalar result
 */
export type GlobalMin<T extends AnyTensorStorage> = Min<T, undefined, false>;

/**
 * Max over the last dimension (most common for features)
 * Removes the last dimension
 */
export type MaxLastDim<T extends AnyTensorStorage> = Max<T, readonly [-1], false>;

/**
 * Min over the last dimension (most common for features)
 * Removes the last dimension
 */
export type MinLastDim<T extends AnyTensorStorage> = Min<T, readonly [-1], false>;

/**
 * Max over the first dimension (batch reduction)
 * Removes the first dimension
 */
export type MaxBatch<T extends AnyTensorStorage> = Max<T, readonly [0], false>;

/**
 * Min over the first dimension (batch reduction)
 * Removes the first dimension
 */
export type MinBatch<T extends AnyTensorStorage> = Min<T, readonly [0], false>;
