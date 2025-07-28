/**
 * Softmax operations for tensor storage with type-safe axis validation
 *
 * This module provides softmax operations that preserve tensor shape while
 * applying the softmax function along a specified axis. The axis parameter
 * is validated at compile-time using the dimension validation system.
 */

import type { AnyDType, ToFloat } from '../dtype/types';
import type { ValidateDim, DimensionError } from '../shape/types';
import type {
  TensorStorage,
  StorageTransformation,
  LayoutFlags,
  AnyTensorStorage,
  ComputeStrides,
} from './layout';

// =============================================================================
// Softmax Operations
// =============================================================================

/**
 * Compute output layout for softmax operations
 * Softmax operations preserve the input shape but may produce C-contiguous output
 */
interface SoftmaxOpLayout<InputLayout extends LayoutFlags> extends LayoutFlags {
  // Backend can choose to preserve layout or make contiguous
  readonly c_contiguous: true | InputLayout['c_contiguous'];
  readonly f_contiguous: InputLayout['f_contiguous'] extends true ? true : false;
  readonly is_view: false; // Softmax ops always copy
  readonly writeable: true;
  readonly aligned: true;
}

/**
 * Base softmax operation interface
 * Softmax preserves the input shape and applies normalization along a specified axis
 */
export type SoftmaxOp<
  Input extends AnyTensorStorage,
  Axis extends number,
  OutputDType extends AnyDType = ToFloat<Input['__dtype']>,
> =
  ValidateDim<Axis, Input['__shape']> extends DimensionError<string>
    ? never // Invalid axis results in never type
    : StorageTransformation<
        'softmax',
        TensorStorage<
          OutputDType,
          Input['__shape'], // Shape is preserved
          ComputeStrides<Input['__shape']> | Input['__strides'], // Union type for stride flexibility
          SoftmaxOpLayout<Input['__layout']>
        >,
        readonly [Input]
      > & {
        // Store the normalized axis as metadata for device implementation
        readonly __softmaxAxis: ValidateDim<Axis, Input['__shape']>;
      };

/**
 * Log-softmax operation interface
 * Computes log(softmax(x)) for numerical stability
 */
export type LogSoftmaxOp<
  Input extends AnyTensorStorage,
  Axis extends number,
  OutputDType extends AnyDType = ToFloat<Input['__dtype']>,
> =
  ValidateDim<Axis, Input['__shape']> extends DimensionError<string>
    ? never // Invalid axis results in never type
    : StorageTransformation<
        'log_softmax',
        TensorStorage<
          OutputDType,
          Input['__shape'], // Shape is preserved
          ComputeStrides<Input['__shape']> | Input['__strides'], // Union type for stride flexibility
          SoftmaxOpLayout<Input['__layout']>
        >,
        readonly [Input]
      > & {
        // Store the normalized axis as metadata for device implementation
        readonly __logSoftmaxAxis: ValidateDim<Axis, Input['__shape']>;
      };

// =============================================================================
// Type-Level Softmax Functions
// =============================================================================

/**
 * Standard softmax function: softmax(x) = exp(x) / sum(exp(x))
 *
 * @template T - Input tensor storage type
 * @template Axis - The axis along which to apply softmax (supports negative indexing)
 *
 * @example
 * type Input = TensorStorage<Float32, [32, 10], [10, 1], DefaultLayoutFlags>;
 * type Result = Softmax<Input, -1>; // Softmax over classes (last dimension)
 *
 * @example
 * type AttentionScores = TensorStorage<Float32, [32, 8, 128, 128], [131072, 16384, 128, 1], DefaultLayoutFlags>;
 * type AttentionWeights = Softmax<AttentionScores, -1>; // Softmax over key sequence
 */
export type Softmax<T extends AnyTensorStorage, Axis extends number> = SoftmaxOp<T, Axis>;

/**
 * Log-softmax function: log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
 *
 * Numerically more stable than computing log(softmax(x)) separately.
 * Commonly used in cross-entropy loss computation.
 *
 * @template T - Input tensor storage type
 * @template Axis - The axis along which to apply log-softmax (supports negative indexing)
 *
 * @example
 * type Logits = TensorStorage<Float32, [32, 1000], [1000, 1], DefaultLayoutFlags>;
 * type LogProbs = LogSoftmax<Logits, -1>; // Log probabilities over classes
 */
export type LogSoftmax<T extends AnyTensorStorage, Axis extends number> = LogSoftmaxOp<T, Axis>;

// =============================================================================
// Convenience Types for Common Use Cases
// =============================================================================

/**
 * Softmax over the last dimension (most common case)
 * Equivalent to Softmax<T, -1>
 */
export type SoftmaxLastDim<T extends AnyTensorStorage> = Softmax<T, -1>;

/**
 * Log-softmax over the last dimension (most common case)
 * Equivalent to LogSoftmax<T, -1>
 */
export type LogSoftmaxLastDim<T extends AnyTensorStorage> = LogSoftmax<T, -1>;

/**
 * Attention softmax - typically over the last dimension (key sequence)
 * for attention weight computation
 */
export type AttentionSoftmax<T extends AnyTensorStorage> = SoftmaxLastDim<T>;
