/**
 * Unary operations for tensor storage with type-safe propagation
 *
 * This module provides unary operations (neg, abs, sin, etc.) that
 * preserve tensor metadata through compile-time type checking.
 */

import type { AnyDType, ToFloat } from '../dtype/types';
import type {
  TensorStorage,
  StorageTransformation,
  LayoutFlags,
  AnyTensorStorage,
  ComputeStrides,
} from './layout';

// =============================================================================
// Unary Operations
// =============================================================================

/**
 * Compute output layout for unary operations
 * Unary ops can either preserve input layout OR produce C-contiguous output
 */
interface UnaryOpLayout<InputLayout extends LayoutFlags> extends LayoutFlags {
  // Backend can choose to preserve layout or make contiguous
  readonly c_contiguous: true | InputLayout['c_contiguous'];
  readonly f_contiguous: InputLayout['f_contiguous'] extends true ? true : false;
  readonly is_view: false; // Unary ops always copy
  readonly writeable: true;
  readonly aligned: true;
}

/**
 * Base unary operation interface
 * The output can have EITHER the input strides (preserved) OR C-contiguous strides
 */
export type UnaryOp<
  Input extends AnyTensorStorage,
  OutputDType extends AnyDType = Input['__dtype'],
  Op extends string = string,
> = StorageTransformation<
  Op,
  TensorStorage<
    OutputDType,
    Input['__shape'],
    ComputeStrides<Input['__shape']> | Input['__strides'], // Union type!
    UnaryOpLayout<Input['__layout']>
  >,
  readonly [Input]
>;

/**
 * Type-level unary operations
 * These preserve the flexibility of stride selection
 */
export type Neg<T extends AnyTensorStorage> = UnaryOp<T, T['__dtype'], 'neg'>;
export type Abs<T extends AnyTensorStorage> = UnaryOp<T, T['__dtype'], 'abs'>;
export type Sin<T extends AnyTensorStorage> = UnaryOp<T, ToFloat<T['__dtype']>, 'sin'>;
export type Cos<T extends AnyTensorStorage> = UnaryOp<T, ToFloat<T['__dtype']>, 'cos'>;
export type Exp<T extends AnyTensorStorage> = UnaryOp<T, ToFloat<T['__dtype']>, 'exp'>;
export type Log<T extends AnyTensorStorage> = UnaryOp<T, ToFloat<T['__dtype']>, 'log'>;
export type Sqrt<T extends AnyTensorStorage> = UnaryOp<T, ToFloat<T['__dtype']>, 'sqrt'>;
export type Square<T extends AnyTensorStorage> = UnaryOp<T, T['__dtype'], 'square'>;
