/**
 * Binary operations for tensor storage with type-safe broadcasting
 *
 * This module provides binary operations (add, sub, mul, etc.) that
 * handle shape broadcasting and dtype promotion at compile time.
 */

import type {
  BroadcastShapes,
  CanBroadcast,
  IncompatibleShapes,
  ShapeToString,
} from '../shape/types';
import type { Promote } from '../dtype/types';
import type {
  TensorStorage,
  StorageTransformation,
  LayoutFlags,
  AnyTensorStorage,
  ComputeStrides,
} from './layout';

// =============================================================================
// Binary Operations
// =============================================================================

/**
 * Compute output layout for binary operations
 * Binary ops typically produce new tensors with conservative layout guarantees
 */
interface BinaryOpLayout extends LayoutFlags {
  // Backend has freedom to choose the output layout for optimal performance
  readonly c_contiguous: boolean; // true | false
  // Binary ops rarely preserve f-contiguous due to broadcasting complexity
  readonly f_contiguous: false;
  // Binary ops always copy (create new tensor)
  readonly is_view: false;
  readonly writeable: true;
  readonly aligned: true;
}

/**
 * Base binary operation interface with broadcasting and dtype promotion
 *
 * This type validates that shapes can broadcast and computes the output type.
 * If shapes cannot broadcast, it returns never which will trigger error messages.
 */
export type BinaryOp<
  A extends AnyTensorStorage,
  B extends AnyTensorStorage,
  Op extends string = string,
> =
  CanBroadcast<A['__shape'], B['__shape']> extends true
    ? StorageTransformation<
        Op,
        TensorStorage<
          Promote<A['__dtype'], B['__dtype']>,
          BroadcastShapes<A['__shape'], B['__shape']>,
          ComputeStrides<BroadcastShapes<A['__shape'], B['__shape']>>, // Always C-contiguous for now
          BinaryOpLayout
        >,
        readonly [A, B]
      >
    : never & {
        __error: `Cannot broadcast shapes [${ShapeToString<A['__shape']>}] and [${ShapeToString<B['__shape']>}]`;
      };

/**
 * Add operation with broadcasting support
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3]>;
 * type B = TensorStorage<Float32, [1, 3]>;
 * type Result = Add<A, B>; // Output shape: [2, 3]
 */
export type Add<A extends AnyTensorStorage, B extends AnyTensorStorage> =
  CanBroadcast<A['__shape'], B['__shape']> extends true
    ? BinaryOp<A, B, 'add'>
    : IncompatibleShapes<A['__shape'], B['__shape']>;

/**
 * Subtract operation with broadcasting support
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3]>;
 * type B = TensorStorage<Float32, [1, 3]>;
 * type Result = Sub<A, B>; // Output shape: [2, 3]
 */
export type Sub<A extends AnyTensorStorage, B extends AnyTensorStorage> =
  CanBroadcast<A['__shape'], B['__shape']> extends true
    ? BinaryOp<A, B, 'sub'>
    : IncompatibleShapes<A['__shape'], B['__shape']>;

/**
 * Multiply operation with broadcasting support
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3]>;
 * type B = TensorStorage<Float32, [1, 3]>;
 * type Result = Mul<A, B>; // Output shape: [2, 3]
 */
export type Mul<A extends AnyTensorStorage, B extends AnyTensorStorage> =
  CanBroadcast<A['__shape'], B['__shape']> extends true
    ? BinaryOp<A, B, 'mul'>
    : IncompatibleShapes<A['__shape'], B['__shape']>;

/**
 * Divide operation with broadcasting support
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3]>;
 * type B = TensorStorage<Float32, [1, 3]>;
 * type Result = Div<A, B>; // Output shape: [2, 3]
 */
export type Div<A extends AnyTensorStorage, B extends AnyTensorStorage> =
  CanBroadcast<A['__shape'], B['__shape']> extends true
    ? BinaryOp<A, B, 'div'>
    : IncompatibleShapes<A['__shape'], B['__shape']>;
