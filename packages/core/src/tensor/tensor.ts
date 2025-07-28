/**
 * Core Tensor class implementation
 *
 * This module provides the main Tensor class that combines storage metadata
 * with backend execution to provide a high-level tensor API.
 */

import type { AnyStorageTransformation, ComputeStrides } from '../storage/layout';
import type { DeviceData, Device } from '../device/types';
import type {
  Shape,
  CanReshape,
  Product,
  CanBroadcast,
  ShapeToString,
  SliceIndex,
  Permute,
  CanMatmul,
  ValidateDim,
  DimensionError,
  ValidateReduction,
} from '../shape/types';
import type { Neg, Abs, Sin, Cos, Exp, Log, Sqrt, Square } from '../storage/unary';
import type { Add, Sub, Mul, Div } from '../storage/binary';
import type { ReshapeOp, Flatten, View, SliceOp, TransposeOp, PermuteOp } from '../storage/view';
import type { MatmulOp } from '../storage/matmul';
import type { SoftmaxOp, LogSoftmaxOp } from '../storage/softmax';
import type { SumOp, MeanOp, MaxOp, MinOp } from '../storage/reduction';
import type { DTypeValue } from '../dtype/types';
import type { NestedArray } from './types';
import { bufferToNestedArray } from './types';
import {
  formatShape,
  broadcastShapes,
  assertShapesCompatible,
  assertMatmulCompatible,
  matmulShape,
} from '../shape';
import {
  computeStrides,
  computeSize,
  computeSlicedShape,
  validateSliceIndices,
  computeTransposedShape,
  computeTransposedStrides,
  computePermutedShape,
  computePermutedStrides,
  validatePermutationAxes,
  normalizePermutationAxes,
} from './utils';
import { toFloatDType, toPromotedDType } from '../dtype';
import type { Mod } from 'ts-arithmetic';

/**
 * A Promise that supports method chaining for tensor operations
 *
 * Extends Promise<Tensor<S>> so it's completely backward compatible,
 * but adds chainable methods that allow fluid operation composition.
 *
 * @example
 * // Both syntaxes work:
 * const result1 = await (await a.add(b)).mul(c);  // Traditional
 * const result2 = await a.add(b).mul(c);          // Chainable
 */
export class ChainablePromise<S extends AnyStorageTransformation> extends Promise<Tensor<S>> {
  // eslint-disable-next-line @typescript-eslint/no-useless-constructor
  constructor(
    executor: (
      resolve: (value: Tensor<S> | PromiseLike<Tensor<S>>) => void,
      reject: (reason?: unknown) => void,
    ) => void,
  ) {
    super(executor);
  }

  // =============================================================================
  // Unary Operations (Chainable)
  // =============================================================================

  neg(): ChainablePromise<Neg<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.neg())
        .then(resolve)
        .catch(reject);
    });
  }

  abs(): ChainablePromise<Abs<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.abs())
        .then(resolve)
        .catch(reject);
    });
  }

  sin(): ChainablePromise<Sin<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.sin())
        .then(resolve)
        .catch(reject);
    });
  }

  cos(): ChainablePromise<Cos<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.cos())
        .then(resolve)
        .catch(reject);
    });
  }

  exp(): ChainablePromise<Exp<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.exp())
        .then(resolve)
        .catch(reject);
    });
  }

  log(): ChainablePromise<Log<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.log())
        .then(resolve)
        .catch(reject);
    });
  }

  sqrt(): ChainablePromise<Sqrt<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.sqrt())
        .then(resolve)
        .catch(reject);
    });
  }

  square(): ChainablePromise<Square<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.square())
        .then(resolve)
        .catch(reject);
    });
  }

  // =============================================================================
  // Binary Operations (Chainable)
  // =============================================================================

  add<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T> | ChainablePromise<T>
      : `[TypeTensor ❌] Cannot add tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): ChainablePromise<Add<S['__output'], T['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      Promise.all([
        this,
        other instanceof ChainablePromise ? other : Promise.resolve(other as Tensor<T>),
      ])
        // Unsafe cast here out of ... necessity? runtime tests validate this properly.
        // If you are reading this and you have an idea please fix :)
        .then(([tensor, otherTensor]) => tensor.add(otherTensor as never))
        .then(resolve)
        .catch(reject);
    });
  }

  sub<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T> | ChainablePromise<T>
      : `[TypeTensor ❌] Cannot subtract tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): ChainablePromise<Sub<S['__output'], T['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      Promise.all([
        this,
        other instanceof ChainablePromise ? other : Promise.resolve(other as Tensor<T>),
      ])
        .then(([tensor, otherTensor]) => tensor.sub(otherTensor as any))
        .then(resolve)
        .catch(reject);
    });
  }

  mul<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T> | ChainablePromise<T>
      : `[TypeTensor ❌] Cannot multiply tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): ChainablePromise<Mul<S['__output'], T['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      Promise.all([
        this,
        other instanceof ChainablePromise ? other : Promise.resolve(other as Tensor<T>),
      ])
        .then(([tensor, otherTensor]) => tensor.mul(otherTensor as any))
        .then(resolve)
        .catch(reject);
    });
  }

  div<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T> | ChainablePromise<T>
      : `[TypeTensor ❌] Cannot divide tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): ChainablePromise<Div<S['__output'], T['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      Promise.all([
        this,
        other instanceof ChainablePromise ? other : Promise.resolve(other as Tensor<T>),
      ])
        .then(([tensor, otherTensor]) => tensor.div(otherTensor as any))
        .then(resolve)
        .catch(reject);
    });
  }

  // =============================================================================
  // View Operations (Chainable)
  // =============================================================================

  reshape<NewShape extends readonly number[]>(
    shape: ValidReshapeShape<S['__output']['__shape'], NewShape>,
  ): ChainablePromise<ReshapeOp<S['__output'], NewShape>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.reshape(shape))
        .then(resolve)
        .catch(reject);
    });
  }

  flatten(): ChainablePromise<Flatten<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.flatten())
        .then(resolve)
        .catch(reject);
    });
  }

  contiguous(): ChainablePromise<S> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.contiguous())
        .then(resolve)
        .catch(reject);
    });
  }

  view<NewShape extends readonly (number | -1)[]>(
    shape: IsValidViewShape<S['__output']['__shape'], NewShape> extends true ? NewShape : never,
  ): ChainablePromise<View<S['__output'], NewShape>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => {
        resolve(tensor.view(shape));
      }).catch(reject);
    });
  }

  slice<const Indices extends readonly SliceIndex[]>(
    indices: Indices,
  ): ChainablePromise<SliceOp<S['__output'], Indices>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.slice(indices))
        .then(resolve)
        .catch(reject);
    });
  }

  transpose(): ChainablePromise<TransposeOp<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => {
        resolve(tensor.transpose());
      }).catch(reject);
    });
  }

  get T(): ChainablePromise<TransposeOp<S['__output']>> {
    return this.transpose();
  }

  permute<Axes extends readonly number[]>(
    axes: Axes & (Axes['length'] extends S['__output']['__shape']['length'] ? Axes : never),
  ): ChainablePromise<PermuteOp<S['__output'], Axes>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => {
        tensor
          .permute(axes as any)
          .then(resolve)
          .catch(reject);
      }).catch(reject);
    });
  }

  // =============================================================================
  // Matrix Operations (Chainable)
  // =============================================================================

  matmul<T extends AnyStorageTransformation>(
    other: CanMatmul<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T> | ChainablePromise<T>
      : `[TypeTensor ❌] Cannot multiply tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}] for matrix multiplication`,
  ): ChainablePromise<MatmulOp<S['__output'], T['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      Promise.all([
        this,
        other instanceof ChainablePromise ? other : Promise.resolve(other as Tensor<T>),
      ])
        .then(([tensor, otherTensor]) => tensor.matmul(otherTensor as any))
        .then(resolve)
        .catch(reject);
    });
  }

  // =============================================================================
  // Softmax Operations (Chainable)
  // =============================================================================

  softmax<Axis extends number>(
    axis: ValidateDim<Axis, S['__output']['__shape']> extends DimensionError<string>
      ? `[TypeTensor ❌] Invalid axis ${Axis} for tensor with shape [${ShapeToString<S['__output']['__shape']>}]. Use axis in range [-${S['__output']['__shape']['length']}, ${S['__output']['__shape']['length']})`
      : Axis,
  ): ChainablePromise<SoftmaxOp<S['__output'], Axis>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.softmax(axis))
        .then(resolve)
        .catch(reject);
    });
  }

  logSoftmax<Axis extends number>(
    axis: ValidateDim<Axis, S['__output']['__shape']> extends DimensionError<string>
      ? `[TypeTensor ❌] Invalid axis ${Axis} for tensor with shape [${ShapeToString<S['__output']['__shape']>}]. Use axis in range [-${S['__output']['__shape']['length']}, ${S['__output']['__shape']['length']})`
      : Axis,
  ): ChainablePromise<LogSoftmaxOp<S['__output'], Axis>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.logSoftmax(axis))
        .then(resolve)
        .catch(reject);
    });
  }

  // =============================================================================
  // Reduction Operations (Chainable)
  // =============================================================================

  sum<Axes extends readonly number[] | undefined = undefined, KeepDims extends boolean = false>(
    axes?: ValidateReduction<S['__output']['__shape'], Axes> extends true
      ? Axes
      : `[TypeTensor ❌] Invalid axes for sum reduction on tensor with shape [${ShapeToString<S['__output']['__shape']>}]`,
    keepdims?: KeepDims,
  ): ChainablePromise<SumOp<S['__output'], Axes, KeepDims>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.sum(axes, keepdims))
        .then(resolve)
        .catch(reject);
    });
  }

  mean<Axes extends readonly number[] | undefined = undefined, KeepDims extends boolean = false>(
    axes?: ValidateReduction<S['__output']['__shape'], Axes> extends true
      ? Axes
      : `[TypeTensor ❌] Invalid axes for mean reduction on tensor with shape [${ShapeToString<S['__output']['__shape']>}]`,
    keepdims?: KeepDims,
  ): ChainablePromise<MeanOp<S['__output'], Axes, KeepDims>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.mean(axes, keepdims))
        .then(resolve)
        .catch(reject);
    });
  }

  max<Axes extends readonly number[] | undefined = undefined, KeepDims extends boolean = false>(
    axes?: ValidateReduction<S['__output']['__shape'], Axes> extends true
      ? Axes
      : `[TypeTensor ❌] Invalid axes for max reduction on tensor with shape [${ShapeToString<S['__output']['__shape']>}]`,
    keepdims?: KeepDims,
  ): ChainablePromise<MaxOp<S['__output'], Axes, KeepDims>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.max(axes, keepdims))
        .then(resolve)
        .catch(reject);
    });
  }

  min<Axes extends readonly number[] | undefined = undefined, KeepDims extends boolean = false>(
    axes?: ValidateReduction<S['__output']['__shape'], Axes> extends true
      ? Axes
      : `[TypeTensor ❌] Invalid axes for min reduction on tensor with shape [${ShapeToString<S['__output']['__shape']>}]`,
    keepdims?: KeepDims,
  ): ChainablePromise<MinOp<S['__output'], Axes, KeepDims>> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.min(axes, keepdims))
        .then(resolve)
        .catch(reject);
    });
  }

  // =============================================================================
  // Utility Operations (Chainable)
  // =============================================================================

  to(device: Device): ChainablePromise<S> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.to(device))
        .then(resolve)
        .catch(reject);
    });
  }

  clone(): ChainablePromise<S> {
    return new ChainablePromise((resolve, reject) => {
      this.then((tensor) => tensor.clone())
        .then(resolve)
        .catch(reject);
    });
  }

  // =============================================================================
  // Data Access Operations (Chainable)
  // =============================================================================

  toArray(): Promise<NestedArray<DTypeValue<S['__output']['__dtype']>, S['__output']['__shape']>> {
    return this.then((tensor) => tensor.toArray());
  }

  item(): Promise<DTypeValue<S['__output']['__dtype']>> {
    return this.then((tensor) => tensor.item());
  }

  format(): Promise<string> {
    return this.then((tensor) => tensor.format());
  }

  // =============================================================================
  // Property Accessors (Need to await the tensor first)
  // =============================================================================

  get shape(): Promise<S['__output']['__shape']> {
    return this.then((tensor) => tensor.shape);
  }

  get dtype(): Promise<S['__output']['__dtype']> {
    return this.then((tensor) => tensor.dtype);
  }

  get device(): Promise<Device> {
    return this.then((tensor) => tensor.device);
  }

  get size(): Promise<S['__output']['__size']> {
    return this.then((tensor) => tensor.size);
  }

  get ndim(): Promise<number> {
    return this.then((tensor) => tensor.ndim);
  }

  get strides(): Promise<S['__output']['__strides']> {
    return this.then((tensor) => tensor.strides);
  }

  get layout(): Promise<S['__output']['__layout']> {
    return this.then((tensor) => tensor.layout);
  }
}

/**
 * Validation type for reshape operations
 * Provides clear error messages for invalid reshape attempts
 */
type ValidReshapeShape<
  Current extends Shape,
  New extends readonly number[],
> = number extends New['length']
  ? `[TypeTensor ❌] Shape must use 'as const' → reshape([2, 3] as const)`
  : Product<Current> extends Product<New>
    ? New
    : `[TypeTensor ❌] Cannot reshape: ${Product<Current> & number} ≠ ${Product<New> & number} elements`;

/**
 * Check if view shape is valid
 */
type IsValidViewShape<
  Current extends Shape,
  New extends readonly (number | -1)[],
> = number extends New['length']
  ? false // Not const
  : CountMinusOnes<New> extends 0
    ? Product<Current> extends Product<New>
      ? Product<New> extends Product<Current>
        ? true
        : false
      : false
    : CountMinusOnes<New> extends 1
      ? CanInferDimension<Current, New>
      : false; // More than one -1

/**
 * Count number of -1s in a shape
 */
type CountMinusOnes<S extends readonly (number | -1)[]> = CountMinusOnesHelper<S, 0>;

type CountMinusOnesHelper<
  S extends readonly (number | -1)[],
  Count extends number,
> = S extends readonly []
  ? Count
  : S extends readonly [infer Head, ...infer Tail]
    ? Head extends -1
      ? Tail extends readonly (number | -1)[]
        ? CountMinusOnesHelper<Tail, Inc<Count>>
        : never
      : Tail extends readonly (number | -1)[]
        ? CountMinusOnesHelper<Tail, Count>
        : never
    : never;

/**
 * Increment helper
 */
type Inc<N extends number> = N extends 0 ? 1 : N extends 1 ? 2 : N extends 2 ? 3 : number;

/**
 * Product of known dimensions (excluding -1)
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

/**
 * Check if dimension can be inferred
 */
type CanInferDimension<Current extends Shape, New extends readonly (number | -1)[]> =
  ProductOfKnown<New> extends 0
    ? false // Can't divide by zero
    : Product<Current> extends number
      ? ProductOfKnown<New> extends number
        ? ModuloCheck<Product<Current>, ProductOfKnown<New>> extends 0
          ? true
          : false
        : false
      : false;

/**
 * Check if A is divisible by B (A % B === 0)
 */
type ModuloCheck<A extends number, B extends number> = Mod<A, B>;

/**
 * Main Tensor class
 *
 * Combines compile-time type safety from the storage layer with
 * runtime execution via backends. All operations maintain type-level
 * guarantees about shapes, dtypes, and broadcasting.
 *
 * @template S - Storage transformation type containing tensor metadata
 *
 * @example
 * const a = await tensor([[1, 2], [3, 4]]);
 * const b = await a.neg();                    // Element-wise negation
 * const c = await a.add(b);                   // Element-wise addition
 * const d = a.reshape([4]);                   // Reshape to 1D
 */
export class Tensor<S extends AnyStorageTransformation = AnyStorageTransformation> {
  constructor(
    private readonly transform: S,
    private readonly data: DeviceData,
  ) {}

  // =============================================================================
  // Property Accessors
  // =============================================================================

  /** Extract storage metadata from transformation */
  get storage(): S['__output'] {
    return this.transform.__output;
  }

  /** Tensor shape */
  get shape(): S['__output']['__shape'] {
    return this.transform.__output.__shape;
  }

  /** Data type */
  get dtype(): S['__output']['__dtype'] {
    return this.transform.__output.__dtype;
  }

  /** Memory strides */
  get strides(): S['__output']['__strides'] {
    return this.transform.__output.__strides;
  }

  /** Device where tensor data resides */
  get device(): Device {
    // NOTE: typescript resolves this as device but eslint does not...
    return this.data.device satisfies Device as Device;
  }

  /** Total number of elements */
  get size(): S['__output']['__size'] {
    return this.transform.__output.__size;
  }

  /** Number of dimensions */
  get ndim(): number {
    return this.shape.length;
  }

  /** Layout flags */
  get layout(): S['__output']['__layout'] {
    return this.transform.__output.__layout;
  }

  // =============================================================================
  // Operation Execution Helper
  // =============================================================================

  /**
   * Helper to ensure tensor is contiguous if device doesn't support non-contiguous operations
   *
   * @param opType - The operation type to check
   * @returns The tensor itself if contiguous or device supports non-contiguous, otherwise a contiguous copy
   */
  private async _ensureContiguousIfNeeded(
    opType: AnyStorageTransformation['__op'],
  ): Promise<Tensor<S>> {
    // If already contiguous, no need to check device support
    if (this.layout.c_contiguous === true) {
      return this;
    }

    // Check if device supports non-contiguous for this operation
    if (this.device.supportsNonContiguous(opType)) {
      return this;
    }

    // Device doesn't support non-contiguous, make a contiguous copy
    return this._createContiguousCopy();
  }

  // =============================================================================
  // Unary Operations
  // =============================================================================

  /**
   * Element-wise negation
   *
   * @returns New tensor with negated values
   */
  neg(): ChainablePromise<Neg<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Ensure tensor is contiguous if device requires it
          const tensor = await this._ensureContiguousIfNeeded('neg');

          // Build the transformation with proper output metadata
          const negOp: Neg<S['__output']> = {
            __op: 'neg',
            __output: {
              __dtype: tensor.storage.__dtype,
              __shape: tensor.storage.__shape,
              __strides: tensor.storage.__strides,
              __size: tensor.storage.__size,
              __layout: {
                c_contiguous: true,
                f_contiguous: false,
                is_view: false,
                writeable: true,
                aligned: true,
              },
              __offset: 0,
            } as Neg<S['__output']>['__output'],
            __inputs: [tensor.storage] as const,
          };

          const resultData = await tensor.data.device.execute(negOp, [tensor.data]);
          resolve(new Tensor(negOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Element-wise absolute value
   *
   * @returns New tensor with absolute values
   */
  abs(): ChainablePromise<Abs<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Ensure tensor is contiguous if device requires it
          const tensor = await this._ensureContiguousIfNeeded('abs');

          // Build the transformation with proper output metadata
          const absOp: Abs<S['__output']> = {
            __op: 'abs',
            __output: {
              __dtype: tensor.storage.__dtype,
              __shape: tensor.storage.__shape,
              __strides: tensor.storage.__strides,
              __size: tensor.storage.__size,
              __layout: {
                c_contiguous: true,
                f_contiguous: false,
                is_view: false,
                writeable: true,
                aligned: true,
              },
              __offset: 0,
            } as Abs<S['__output']>['__output'],
            __inputs: [tensor.storage] as const,
          };

          const resultData = await tensor.data.device.execute(absOp, [tensor.data]);
          resolve(new Tensor(absOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Element-wise sine
   *
   * @returns New tensor with sine values
   */
  sin(): ChainablePromise<Sin<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Ensure tensor is contiguous if device requires it
          const tensor = await this._ensureContiguousIfNeeded('sin');

          const sinOp = {
            __op: 'sin' as const,
            __output: {
              ...tensor.storage,
              __dtype: toFloatDType(tensor.dtype),
            } as Sin<S['__output']>['__output'],
            __inputs: [tensor.storage] as const,
          } as Sin<S['__output']>;

          const resultData = await tensor.data.device.execute(sinOp, [tensor.data]);
          resolve(new Tensor(sinOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Element-wise cosine
   *
   * @returns New tensor with cosine values
   */
  cos(): ChainablePromise<Cos<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Ensure tensor is contiguous if device requires it
          const tensor = await this._ensureContiguousIfNeeded('cos');

          const cosOp = {
            __op: 'cos' as const,
            __output: {
              ...tensor.storage,
              __dtype: toFloatDType(tensor.dtype),
            } as Cos<S['__output']>['__output'],
            __inputs: [tensor.storage] as const,
          } as Cos<S['__output']>;

          const resultData = await tensor.data.device.execute(cosOp, [tensor.data]);
          resolve(new Tensor(cosOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Element-wise exponential
   *
   * @returns New tensor with exponential values
   */
  exp(): ChainablePromise<Exp<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Ensure tensor is contiguous if device requires it
          const tensor = await this._ensureContiguousIfNeeded('exp');

          const expOp = {
            __op: 'exp' as const,
            __output: {
              ...tensor.storage,
              __dtype: toFloatDType(tensor.dtype),
            } as Exp<S['__output']>['__output'],
            __inputs: [tensor.storage] as const,
          } as Exp<S['__output']>;

          const resultData = await tensor.data.device.execute(expOp, [tensor.data]);
          resolve(new Tensor(expOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Element-wise natural logarithm
   *
   * @returns New tensor with logarithm values
   */
  log(): ChainablePromise<Log<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Ensure tensor is contiguous if device requires it
          const tensor = await this._ensureContiguousIfNeeded('log');

          const logOp = {
            __op: 'log' as const,
            __output: {
              ...tensor.storage,
              __dtype: toFloatDType(tensor.dtype),
            } as Log<S['__output']>['__output'],
            __inputs: [tensor.storage] as const,
          } as Log<S['__output']>;

          const resultData = await this.data.device.execute(logOp, [this.data]);
          resolve(new Tensor(logOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Element-wise square root
   *
   * @returns New tensor with square root values
   */
  sqrt(): ChainablePromise<Sqrt<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Ensure tensor is contiguous if device requires it
          const tensor = await this._ensureContiguousIfNeeded('sqrt');

          const sqrtOp = {
            __op: 'sqrt' as const,
            __output: {
              ...tensor.storage,
              __dtype: toFloatDType(tensor.dtype),
            } as Sqrt<S['__output']>['__output'],
            __inputs: [tensor.storage] as const,
          } as Sqrt<S['__output']>;

          const resultData = await tensor.data.device.execute(sqrtOp, [tensor.data]);
          resolve(new Tensor(sqrtOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Element-wise square
   *
   * @returns New tensor with squared values
   */
  square(): ChainablePromise<Square<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Ensure tensor is contiguous if device requires it
          const tensor = await this._ensureContiguousIfNeeded('square');

          const squareOp = {
            __op: 'square' as const,
            __output: {
              ...tensor.storage,
            } as Square<S['__output']>['__output'],
            __inputs: [tensor.storage] as const,
          } as Square<S['__output']>;

          const resultData = await tensor.data.device.execute(squareOp, [tensor.data]);
          resolve(new Tensor(squareOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  // =============================================================================
  // Binary Operations
  // =============================================================================

  /**
   * Element-wise addition with broadcasting
   *
   * @param other - Tensor to add
   * @returns New tensor with sum
   * @throws {Error} If tensors are on different devices
   * @throws {Error} If shapes cannot broadcast
   */
  add<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T> | ChainablePromise<T>
      : `[TypeTensor ❌] Cannot add tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): ChainablePromise<Add<S['__output'], T['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Resolve the other tensor if it's a ChainablePromise
          const otherTensor = other instanceof ChainablePromise ? await other : other;

          if (!(otherTensor instanceof Tensor)) {
            throw new Error('Expected a Tensor instance');
          }

          if (otherTensor.device.id !== this.device.id) {
            // NOTE: the as string is a workaround, typescript properly infers the type as string... but eslint does not
            throw new Error(
              `Tensors must be on same device: ${this.device.id as string} vs ${otherTensor.device.id as string}`,
            );
          }

          // Validate shapes can broadcast with helpful error messages
          assertShapesCompatible(this.shape, otherTensor.shape, 'element-wise addition');

          // Ensure both tensors are contiguous if device requires it
          const tensor1 = await this._ensureContiguousIfNeeded('add');
          const tensor2 = await otherTensor._ensureContiguousIfNeeded('add');

          // Compute broadcast shape and promoted dtype
          const outputShape = broadcastShapes(tensor1.shape, tensor2.shape);
          const outputStrides = computeStrides(outputShape);
          const outputSize = computeSize(outputShape);
          const promotedDtype = toPromotedDType(tensor1.dtype, tensor2.dtype);

          // Build the add operation with proper output metadata
          const addOp = {
            __op: 'add' as const,
            __output: {
              __dtype: promotedDtype,
              __shape: outputShape,
              __strides: outputStrides,
              __size: outputSize,
              __layout: {
                c_contiguous: true,
                f_contiguous: false,
                is_view: false,
                writeable: true,
                aligned: true,
              },
              __offset: 0,
            } as Add<S['__output'], T['__output']>['__output'],
            __inputs: [tensor1.storage, tensor2.storage] as const,
          } as Add<S['__output'], T['__output']>;

          const resultData = await tensor1.data.device.execute(addOp, [tensor1.data, tensor2.data]);
          resolve(new Tensor(addOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Element-wise subtraction with broadcasting
   *
   * @param other - Tensor to subtract
   * @returns New tensor with difference
   * @throws {Error} If tensors are on different devices
   * @throws {Error} If shapes cannot broadcast
   */
  sub<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T> | ChainablePromise<T>
      : `[TypeTensor ❌] Cannot subtract tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): ChainablePromise<Sub<S['__output'], T['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Resolve the other tensor if it's a ChainablePromise
          const otherTensor = other instanceof ChainablePromise ? await other : other;

          if (!(otherTensor instanceof Tensor)) {
            throw new Error('Expected a Tensor instance');
          }

          if (otherTensor.device.id !== this.device.id) {
            throw new Error(
              `Tensors must be on same device: ${this.device.id as string} vs ${otherTensor.device.id as string}`,
            );
          }

          // Ensure both tensors are contiguous if device requires it
          const tensor1 = await this._ensureContiguousIfNeeded('sub');
          const tensor2 = await otherTensor._ensureContiguousIfNeeded('sub');

          // Compute broadcast shape and promoted dtype
          const outputShape = broadcastShapes(tensor1.shape, tensor2.shape);
          const outputStrides = computeStrides(outputShape);
          const outputSize = computeSize(outputShape);
          const promotedDtype = toPromotedDType(tensor1.dtype, tensor2.dtype);

          // Build the sub operation with proper output metadata
          const subOp = {
            __op: 'sub' as const,
            __output: {
              __dtype: promotedDtype,
              __shape: outputShape,
              __strides: outputStrides,
              __size: outputSize,
              __layout: {
                c_contiguous: true,
                f_contiguous: false,
                is_view: false,
                writeable: true,
                aligned: true,
              },
            } as Sub<S['__output'], T['__output']>['__output'],
            __inputs: [tensor1.storage, tensor2.storage] as const,
          } as Sub<S['__output'], T['__output']>;

          const resultData = await tensor1.data.device.execute(subOp, [tensor1.data, tensor2.data]);
          resolve(new Tensor(subOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Element-wise multiplication with broadcasting
   *
   * @param other - Tensor to multiply
   * @returns New tensor with product
   * @throws {Error} If tensors are on different devices
   * @throws {Error} If shapes cannot broadcast
   */
  mul<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T> | ChainablePromise<T>
      : `[TypeTensor ❌] Cannot multiply tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): ChainablePromise<Mul<S['__output'], T['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Resolve the other tensor if it's a ChainablePromise
          const otherTensor = other instanceof ChainablePromise ? await other : other;

          if (!(otherTensor instanceof Tensor)) {
            throw new Error('Expected a Tensor instance');
          }

          if (otherTensor.device.id !== this.device.id) {
            throw new Error(
              `Tensors must be on same device: ${this.device.id as string} vs ${otherTensor.device.id as string}`,
            );
          }

          // Ensure both tensors are contiguous if device requires it
          const tensor1 = await this._ensureContiguousIfNeeded('mul');
          const tensor2 = await otherTensor._ensureContiguousIfNeeded('mul');

          // Compute broadcast shape and promoted dtype
          const outputShape = broadcastShapes(tensor1.shape, tensor2.shape);
          const outputStrides = computeStrides(outputShape);
          const outputSize = computeSize(outputShape);
          const promotedDtype = toPromotedDType(tensor1.dtype, tensor2.dtype);

          // Build the mul operation with proper output metadata
          const mulOp = {
            __op: 'mul' as const,
            __output: {
              __dtype: promotedDtype,
              __shape: outputShape,
              __strides: outputStrides,
              __size: outputSize,
              __layout: {
                c_contiguous: true,
                f_contiguous: false,
                is_view: false,
                writeable: true,
                aligned: true,
              },
            } as Mul<S['__output'], T['__output']>['__output'],
            __inputs: [tensor1.storage, tensor2.storage] as const,
          } as Mul<S['__output'], T['__output']>;

          const resultData = await tensor1.data.device.execute(mulOp, [tensor1.data, tensor2.data]);
          resolve(new Tensor(mulOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Element-wise division with broadcasting
   *
   * @param other - Tensor to divide by
   * @returns New tensor with quotient
   * @throws {Error} If tensors are on different devices
   * @throws {Error} If shapes cannot broadcast
   */
  div<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T> | ChainablePromise<T>
      : `[TypeTensor ❌] Cannot divide tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): ChainablePromise<Div<S['__output'], T['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Resolve the other tensor if it's a ChainablePromise
          const otherTensor = other instanceof ChainablePromise ? await other : other;

          if (!(otherTensor instanceof Tensor)) {
            throw new Error('Expected a Tensor instance');
          }

          if (otherTensor.device.id !== this.device.id) {
            throw new Error(
              `Tensors must be on same device: ${this.device.id as string} vs ${otherTensor.device.id as string}`,
            );
          }

          // Ensure both tensors are contiguous if device requires it
          const tensor1 = await this._ensureContiguousIfNeeded('div');
          const tensor2 = await otherTensor._ensureContiguousIfNeeded('div');

          // Compute broadcast shape and promoted dtype
          const outputShape = broadcastShapes(tensor1.shape, tensor2.shape);
          const outputStrides = computeStrides(outputShape);
          const outputSize = computeSize(outputShape);
          const promotedDtype = toPromotedDType(tensor1.dtype, tensor2.dtype);

          // Build the div operation with proper output metadata
          const divOp = {
            __op: 'div' as const,
            __output: {
              __dtype: promotedDtype,
              __shape: outputShape,
              __strides: outputStrides,
              __size: outputSize,
              __layout: {
                c_contiguous: true,
                f_contiguous: false,
                is_view: false,
                writeable: true,
                aligned: true,
              },
            } as Div<S['__output'], T['__output']>['__output'],
            __inputs: [tensor1.storage, tensor2.storage] as const,
          } as Div<S['__output'], T['__output']>;

          const resultData = await tensor1.data.device.execute(divOp, [tensor1.data, tensor2.data]);
          resolve(new Tensor(divOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  // =============================================================================
  // View Operations
  // =============================================================================

  /**
   * Reshape tensor to new shape
   *
   * Creates a view of the tensor with a different shape. The total
   * number of elements must remain the same.
   *
   * @param shape - New shape (must use 'as const')
   * @returns View with new shape
   *
   * @example
   * const a = await tensor([1, 2, 3, 4, 5, 6]);
   * const b = a.reshape([2, 3] as const); // [[1, 2, 3], [4, 5, 6]]
   */
  reshape<NewShape extends readonly number[]>(
    shape: ValidReshapeShape<S['__output']['__shape'], NewShape>,
  ): ChainablePromise<ReshapeOp<S['__output'], NewShape>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Validate that reshaping is allowed
          // At runtime, shape will always be a valid array (TypeScript ensures this)
          // Cast to NewShape after validation
          const validShape = shape as NewShape;
          const totalElements = this.shape.reduce((a, b) => a * b, 1);
          const newTotalElements = validShape.reduce((a, b) => a * b, 1);
          if (totalElements !== newTotalElements) {
            throw new Error(
              `Cannot reshape from ${formatShape(this.shape as Shape)} to ${formatShape(validShape as Shape)}: different number of elements`,
            );
          }

          // Ensure tensor is contiguous if device requires it
          const tensor = await this._ensureContiguousIfNeeded('reshape');

          const reshapeOp = {
            __op: 'reshape' as const,
            __output: {
              __dtype: tensor.storage.__dtype,
              __shape: validShape,
              __strides: computeStrides(validShape) as ComputeStrides<NewShape>,
              __size: computeSize(validShape) as Product<NewShape>,
              __layout: {
                ...tensor.storage.__layout,
                is_view: true,
              },
              __offset: tensor.storage.__offset,
            } as ReshapeOp<S['__output'], NewShape>['__output'],
            __inputs: [tensor.storage] as const,
          } as ReshapeOp<S['__output'], NewShape>;

          // Reshape typically returns a view (same data, new metadata)
          resolve(
            new Tensor(reshapeOp, tensor.data) as CanReshape<
              S['__output']['__shape'],
              NewShape
            > extends true
              ? Tensor<ReshapeOp<S['__output'], NewShape>>
              : never,
          );
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Internal reshape without validation for runtime-generated shapes
   * Used by flatten() and other internal methods
   */
  private _reshapeUnsafe<NewShape extends readonly number[]>(
    shape: NewShape,
  ): Tensor<ReshapeOp<S['__output'], NewShape>> {
    const totalElements = this.shape.reduce((a, b) => a * b, 1);
    const newTotalElements = shape.reduce((a, b) => a * b, 1);
    if (totalElements !== newTotalElements) {
      throw new Error(
        `Cannot reshape from ${formatShape(this.shape as Shape)} to ${formatShape(shape as Shape)}: different number of elements`,
      );
    }

    const reshapeOp = {
      __op: 'reshape' as const,
      __output: {
        __dtype: this.storage.__dtype,
        __shape: shape,
        __strides: computeStrides(shape) as ComputeStrides<NewShape>,
        __size: computeSize(shape) as Product<NewShape>,
        __layout: {
          ...this.storage.__layout,
          is_view: true,
        },
        __offset: this.storage.__offset,
      } as ReshapeOp<S['__output'], NewShape>['__output'],
      __inputs: [this.storage] as const,
    } as ReshapeOp<S['__output'], NewShape>;

    return new Tensor(reshapeOp, this.data);
  }

  /**
   * Flatten tensor to 1D
   *
   * Creates a 1D view if the tensor is contiguous, otherwise creates a copy
   * with the correct data layout. This matches PyTorch's flatten() behavior.
   *
   * @returns 1D tensor with flattened data
   */
  flatten(): ChainablePromise<Flatten<S['__output']>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          const totalSize = this.size;

          // If tensor is contiguous, we can safely create a view (same as PyTorch)
          if (this.layout.c_contiguous === true) {
            resolve(this._reshapeUnsafe([totalSize] as const) as Tensor<Flatten<S['__output']>>);
            return;
          }

          // If tensor is not contiguous (e.g., after transpose), we need to create a copy
          // with data in the logical view order (same as PyTorch)
          const contiguousCopy = await this._createContiguousCopy();
          resolve(
            contiguousCopy._reshapeUnsafe([totalSize] as const) as Tensor<Flatten<S['__output']>>,
          );
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Returns a contiguous tensor with the same data
   *
   * If the tensor is already contiguous, returns itself (no-op).
   * Otherwise, creates a new tensor with C-contiguous memory layout.
   *
   * @returns Contiguous tensor
   */
  contiguous(): ChainablePromise<S> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // If already contiguous, return self
          if (this.layout.c_contiguous === true) {
            resolve(this);
            return;
          }
          // Otherwise create a contiguous copy
          const contiguousCopy = await this._createContiguousCopy();
          resolve(contiguousCopy);
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Create a contiguous copy of the tensor with data in logical order
   *
   * This reads the tensor data in logical order (respecting strides/views)
   * and creates a new C-contiguous tensor with that data.
   */
  private async _createContiguousCopy(): Promise<Tensor<S>> {
    // Get the logical view data (this respects strides correctly)
    const logicalData = await this.toArray();

    // Flatten the nested array to get data in logical order
    const flatData = this._flattenNestedArray(logicalData);

    // Create a new C-contiguous tensor from the logical data
    // Use the same approach that worked in the manual test
    const { tensor: tensorFn } = await import('./creation');
    return (await tensorFn(flatData as any, {
      device: this.device,
      dtype: this.dtype,
      shape: this.shape,
    })) as unknown as Tensor<S>;
  }

  /**
   * Recursively flatten a nested array to 1D
   */
  private _flattenNestedArray(arr: any): any[] {
    const result: any[] = [];

    function flatten(item: any): void {
      if (Array.isArray(item)) {
        for (const subItem of item) {
          flatten(subItem);
        }
      } else {
        result.push(item);
      }
    }

    flatten(arr);
    return result;
  }

  /**
   * Create view with dimension inference
   *
   * Allows using -1 for one dimension to be inferred from total size.
   *
   * @param shape - Shape with optional -1 for inference (must use 'as const')
   * @returns View with inferred shape
   *
   * @example
   * const a = await tensor([[1, 2, 3], [4, 5, 6]] as const);
   * const b = a.view([6] as const);     // Flatten to 1D
   * const c = a.view([-1, 2] as const); // Infer first dimension: [3, 2]
   */
  view<NewShape extends readonly (number | -1)[]>(
    shape: IsValidViewShape<S['__output']['__shape'], NewShape> extends true ? NewShape : never,
  ): ChainablePromise<View<S['__output'], NewShape>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // First ensure the tensor is contiguous if needed
          const contiguousTensor = await this._ensureContiguousIfNeeded('view');

          // Infer shape if -1 is present (using contiguous tensor's size)
          const inferredShape = this.inferShapeForTensor(shape, contiguousTensor.size);

          const viewOp = {
            __op: 'view' as const,
            __output: {
              __dtype: contiguousTensor.storage.__dtype,
              __shape: inferredShape,
              __strides: computeStrides(inferredShape),
              __size: contiguousTensor.storage.__size, // Size remains the same
              __layout: {
                ...contiguousTensor.storage.__layout,
                is_view: true,
              },
              __offset: contiguousTensor.storage.__offset,
            } as View<S['__output'], NewShape>['__output'],
            __inputs: [contiguousTensor.storage] as const,
          } as View<S['__output'], NewShape>;

          resolve(new Tensor(viewOp, contiguousTensor.data));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Infer shape dimension when -1 is present for a given tensor size
   */
  private inferShapeForTensor(
    shape: readonly (number | -1)[],
    totalSize: number,
  ): readonly number[] {
    let inferIndex = -1;
    let knownSize = 1;
    let minusOneCount = 0;

    for (let i = 0; i < shape.length; i++) {
      const element = shape[i];
      if (element === undefined) {
        throw new Error(
          `Invalid shape: shape[${i.toString()}] is undefined. Did you forget 'as const'?`,
        );
      } else if (element === -1) {
        minusOneCount++;
        if (inferIndex !== -1) {
          throw new Error(
            `Can only infer one dimension, but found ${minusOneCount} -1s in shape. ` +
              `Use -1 for at most one dimension.`,
          );
        }
        inferIndex = i;
      } else if (!Number.isInteger(element) || element < 0) {
        throw new Error(
          `Invalid dimension ${element} at index ${i}. Dimensions must be positive integers or -1.`,
        );
      } else {
        knownSize *= element;
      }
    }

    if (inferIndex === -1) {
      return shape as readonly number[];
    }

    if (totalSize % knownSize !== 0) {
      throw new Error(
        `Cannot infer dimension: total size ${totalSize} is not divisible by known dimensions product ${knownSize}. ` +
          `The inferred dimension would be ${totalSize / knownSize}, which is not an integer.`,
      );
    }

    const result = [...shape];
    result[inferIndex] = totalSize / knownSize;
    return result as readonly number[];
  }

  /**
   * Slice tensor along dimensions
   *
   * Creates a view of the tensor with a subset of elements along each dimension.
   * Supports integer indexing (removes dimension), slice notation with start/stop/step,
   * and null for keeping entire dimension.
   *
   * @param indices - Array of slice indices for each dimension
   * @returns New tensor with sliced data
   *
   * @example
   * const a = await tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
   * const b = await a.slice([0]);           // [[1, 2], [3, 4]] - select first element
   * const c = await a.slice([null, 1]);     // [[3, 4], [7, 8]] - select column
   * const d = await a.slice([{ start: 0, stop: 2, step: 1 }, null]); // full tensor
   */
  slice<const Indices extends readonly SliceIndex[]>(
    indices: Indices,
  ): ChainablePromise<SliceOp<S['__output'], Indices>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Validate indices
          validateSliceIndices(this.shape, indices);

          // Compute runtime shape and strides
          const slicedShape = computeSlicedShape(this.shape, indices);
          // Note: The CPU backend creates a contiguous copy for slices, not a view
          // So we use contiguous strides rather than view-based strides
          const slicedStrides = computeStrides(slicedShape);
          const slicedSize = computeSize(slicedShape);

          // Build the slice operation
          const sliceOp = {
            __op: 'slice' as const,
            __output: {
              __dtype: this.storage.__dtype,
              __shape: slicedShape,
              __strides: slicedStrides,
              __size: slicedSize,
              __layout: {
                c_contiguous: true,
                f_contiguous: false,
                is_view: false,
                writeable: true,
                aligned: true,
              },
              __offset: this.storage.__offset, // Backend will compute actual offset
              __sliceIndices: indices,
            } as unknown as SliceOp<S['__output'], Indices>['__output'] & {
              __sliceIndices: Indices;
            },
            __inputs: [this.storage] as const,
          } as SliceOp<S['__output'], Indices>;

          // Execute on device - slicing needs device support for proper memory handling
          const resultData = await this.data.device.execute(sliceOp, [this.data]);
          resolve(new Tensor(sliceOp, resultData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Transpose tensor by swapping the last two dimensions
   *
   * For tensors with rank < 2, returns a view of the original tensor unchanged.
   * For higher rank tensors, swaps the last two dimensions.
   *
   * @returns View of transposed tensor
   *
   * @example
   * const a = await tensor([[1, 2, 3], [4, 5, 6]]); // Shape: [2, 3]
   * const b = a.transpose(); // Shape: [3, 2]
   * // [[1, 4],
   * //  [2, 5],
   * //  [3, 6]]
   *
   * const c = await tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]); // Shape: [2, 2, 2]
   * const d = c.transpose(); // Shape: [2, 2, 2] (swaps last two dims)
   */
  transpose(): ChainablePromise<TransposeOp<S['__output']>> {
    if (this.ndim < 2) {
      // For scalars and 1D tensors, return unchanged
      const transposeOp = {
        __op: 'transpose' as const,
        __output: {
          ...this.storage,
          __layout: {
            ...this.storage.__layout,
            is_view: true,
          },
        } as TransposeOp<S['__output']>['__output'],
        __inputs: [this.storage] as const,
      } as TransposeOp<S['__output']>;

      return new ChainablePromise((resolve, reject) => {
        void (async () => {
          try {
            const tensor = await this._ensureContiguousIfNeeded('transpose');
            resolve(new Tensor(transposeOp, tensor.data));
          } catch (error) {
            reject(error);
          }
        })();
      });
    }

    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // First ensure the tensor is contiguous if needed
          const contiguousTensor = await this._ensureContiguousIfNeeded('transpose');

          // Compute transposed shape and strides based on the contiguous tensor
          const transposedShape = computeTransposedShape(contiguousTensor.shape);
          const transposedStrides = computeTransposedStrides(
            contiguousTensor.shape,
            contiguousTensor.strides,
          );

          const transposeOp = {
            __op: 'transpose' as const,
            __output: {
              __dtype: contiguousTensor.storage.__dtype,
              __shape: transposedShape,
              __strides: transposedStrides,
              __size: contiguousTensor.storage.__size,
              __layout: {
                c_contiguous: false,
                f_contiguous: false,
                is_view: true,
                writeable: contiguousTensor.storage.__layout.writeable,
                aligned: contiguousTensor.storage.__layout.aligned,
              },
              __offset: contiguousTensor.storage.__offset,
            } as TransposeOp<S['__output']>['__output'],
            __inputs: [contiguousTensor.storage] as const,
          } as TransposeOp<S['__output']>;

          resolve(new Tensor(transposeOp, contiguousTensor.data));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  /**
   * Shorthand for transpose()
   *
   * @returns View of transposed tensor
   *
   * @example
   * const a = await tensor([[1, 2, 3], [4, 5, 6]]);
   * const b = a.T; // Same as a.transpose()
   */
  get T(): ChainablePromise<TransposeOp<S['__output']>> {
    return this.transpose();
  }

  /**
   * Permute tensor dimensions according to specified axes
   *
   * Returns a view of the tensor with dimensions rearranged according to
   * the axes array. Each dimension index must appear exactly once.
   *
   * @param axes - New order of dimensions (must use 'as const')
   * @returns View with permuted dimensions
   *
   * @example
   * const a = await tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]); // Shape: [2, 2, 2]
   * const b = a.permute([2, 0, 1] as const); // Shape: [2, 2, 2]
   * const c = a.permute([1, 2, 0] as const); // Shape: [2, 2, 2]
   *
   * // For a tensor with shape [batch, height, width, channels]:
   * const img = await tensor(imageData); // Shape: [32, 224, 224, 3]
   * const channelsFirst = img.permute([0, 3, 1, 2] as const); // Shape: [32, 3, 224, 224]
   */
  permute<Axes extends readonly number[]>(
    axes: Axes & (Axes['length'] extends S['__output']['__shape']['length'] ? Axes : never),
  ): ChainablePromise<PermuteOp<S['__output'], Axes>> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          // Validate axes at runtime
          validatePermutationAxes(this.ndim, axes);

          // Normalize negative indices
          const normalizedAxes = normalizePermutationAxes(axes, this.ndim);

          // First ensure the tensor is contiguous if needed
          const contiguousTensor = await this._ensureContiguousIfNeeded('permute');

          // Compute permuted shape and strides based on the contiguous tensor
          const permutedShape = computePermutedShape(contiguousTensor.shape, normalizedAxes);
          const permutedStrides = computePermutedStrides(contiguousTensor.strides, normalizedAxes);

          const permuteOp = {
            __op: 'permute' as const,
            __output: {
              __dtype: contiguousTensor.storage.__dtype,
              __shape: permutedShape as unknown as Permute<S['__output']['__shape'], Axes>,
              __strides: permutedStrides as unknown as Permute<S['__output']['__strides'], Axes>,
              __size: contiguousTensor.storage.__size,
              __layout: {
                c_contiguous: false,
                f_contiguous: false,
                is_view: true,
                writeable: contiguousTensor.storage.__layout.writeable,
                aligned: contiguousTensor.storage.__layout.aligned,
              },
              __offset: contiguousTensor.storage.__offset,
              // Include the permutation axes as metadata
              __permuteAxes: normalizedAxes as unknown as Axes,
            } as PermuteOp<S['__output'], Axes>['__output'],
            __inputs: [contiguousTensor.storage] as const,
          } as PermuteOp<S['__output'], Axes>;

          resolve(
            new Tensor(permuteOp, contiguousTensor.data) as Tensor<PermuteOp<S['__output'], Axes>>,
          );
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  // =============================================================================
  // Device Operations
  // =============================================================================

  /**
   * Move tensor to different device
   *
   * @param device - Target device
   * @returns New tensor on target device
   */
  async to(device: Device): Promise<Tensor<S>> {
    // Already on target device
    if (device.id === this.device.id) {
      return this;
    }

    // Transfer via host memory (devices may optimize this)
    const buffer = await this.data.device.readData(this.data);
    const newData = device.createData(buffer.byteLength);
    await device.writeData(newData, buffer);

    return new Tensor(this.transform, newData);
  }

  // =============================================================================
  // Data Access
  // =============================================================================

  /**
   * Convert tensor to nested JavaScript array
   *
   * Reads tensor data from device and converts to nested array
   * matching the tensor shape.
   *
   * @returns Promise resolving to nested array
   */
  async toArray(): Promise<
    NestedArray<DTypeValue<S['__output']['__dtype']>, S['__output']['__shape']>
  > {
    const buffer = await this.data.device.readData(this.data);
    return bufferToNestedArray(buffer, this.shape, this.dtype, this.strides, this.storage.__offset);
  }

  /**
   * Get a scalar value from a tensor with exactly one element
   *
   * Works on scalars (0-dimensional tensors) and single-element tensors
   * of any shape (e.g., [1], [1,1], [1,1,1]). Matches PyTorch behavior.
   *
   * @returns Scalar value
   * @throws {Error} If tensor does not have exactly one element
   *
   * @example
   * const scalar = await tensor(42);
   * console.log(await scalar.item()); // 42
   *
   * @example
   * const singleElement = await tensor([42]);
   * console.log(await singleElement.item()); // 42
   *
   * @example
   * const matrix1x1 = await tensor([[42]]);
   * console.log(await matrix1x1.item()); // 42
   */
  async item(): Promise<DTypeValue<S['__output']['__dtype']>> {
    if (this.size !== 1) {
      throw new Error(
        `item() only works on tensors with exactly one element, got ${this.size} elements (shape ${formatShape(this.shape)})`,
      );
    }

    const array = await this.toArray();

    // For scalars, array is the value directly
    if (this.ndim === 0) {
      return array as DTypeValue<S['__output']['__dtype']>;
    }

    // For single-element tensors of any shape, extract the nested value
    let result: any = array;
    while (Array.isArray(result) && result.length === 1) {
      result = result[0];
    }

    return result as DTypeValue<S['__output']['__dtype']>;
  }

  // =============================================================================
  // Lifecycle
  // =============================================================================

  /**
   * Dispose tensor and free device memory
   *
   * After calling dispose(), the tensor should not be used.
   */
  dispose(): void {
    this.data.device.disposeData(this.data);
  }

  // =============================================================================
  // Utilities
  // =============================================================================

  /**
   * String representation of tensor
   */
  toString(): string {
    // For async toString, we just show metadata
    // Use toStringAsync() for full data representation
    const shapeStr = formatShape(this.shape);
    const dtypeStr = this.dtype.__dtype;
    const deviceStr = this.device.id;
    return `Tensor(shape=${shapeStr}, dtype=${dtypeStr}, device=${deviceStr})`;
  }

  /**
   * Format tensor for display with actual data values
   * Similar to PyTorch's tensor printing
   *
   * @example
   * const t = await tensor([[1, 2, 3], [4, 5, 6]]);
   * console.log(await t.format());
   * // tensor([[1, 2, 3],
   * //         [4, 5, 6]])
   */
  async format(): Promise<string> {
    const maxElements = 1000; // Threshold for truncation
    const edgeItems = 3; // Number of items to show at edges when truncated

    // Get the actual data
    const data = await this.toArray();

    // Format based on dimensions
    if (this.ndim === 0) {
      // Scalar
      return `tensor(${data})`;
    }

    if (this.size === 0) {
      // Empty tensor
      return `tensor([])`;
    }

    // Check if we need to truncate
    const shouldTruncate = this.size > maxElements;

    // Format the data
    const formatted = this.formatArray(data, this.shape, 0, shouldTruncate, edgeItems);

    // Build the final string
    let result = `tensor(${formatted}`;

    // Add dtype/device info if non-default
    const annotations: string[] = [];
    if (this.dtype.__dtype !== 'float32') {
      annotations.push(`dtype=${this.dtype.__dtype}`);
    }
    if (this.device.id !== 'cpu') {
      annotations.push(`device='${this.device.id}'`);
    }

    if (annotations.length > 0) {
      result += `, ${annotations.join(', ')}`;
    }

    result += ')';
    return result;
  }

  /**
   * Format nested array for printing
   */
  private formatArray(
    data: unknown,
    shape: readonly number[],
    depth: number,
    shouldTruncate: boolean,
    edgeItems: number,
  ): string {
    if (shape.length === 0) {
      // Scalar case
      return this.formatValue(data);
    }

    if (shape.length === 1) {
      // 1D array
      const items = data as unknown[];
      if (shouldTruncate && items.length > 2 * edgeItems) {
        const start = items.slice(0, edgeItems).map((v) => this.formatValue(v));
        const end = items.slice(-edgeItems).map((v) => this.formatValue(v));
        return `[${start.join(', ')}, ..., ${end.join(', ')}]`;
      }
      return `[${items.map((v) => this.formatValue(v)).join(', ')}]`;
    }

    // Multi-dimensional array
    const currentDim = shape[0];
    if (currentDim === undefined) {
      throw new Error('currentDim is undefined');
    }

    const remainingShape = shape.slice(1);
    const items = data as unknown[];

    const baseIndent = ' '.repeat(depth);
    const itemIndent = ' '.repeat(depth + 1);

    // Determine if we need spacing between elements
    const useBlockSeparator = remainingShape.length > 1;
    const separator = useBlockSeparator ? ',\n\n' : ',\n';

    let result = '[';

    if (shouldTruncate && currentDim > 2 * edgeItems) {
      // Show first few
      for (let i = 0; i < edgeItems; i++) {
        if (i > 0) {
          result += separator;
          if (useBlockSeparator) {
            result += ' ' + itemIndent;
          } else {
            result += itemIndent;
          }
        } else {
          result += '\n' + itemIndent;
        }
        result += this.formatArray(items[i], remainingShape, depth + 1, shouldTruncate, edgeItems);
      }

      result += separator + itemIndent + '...';

      // Show last few
      for (let i = currentDim - edgeItems; i < currentDim; i++) {
        result += separator;
        if (useBlockSeparator) {
          result += ' ' + itemIndent;
        } else {
          result += itemIndent;
        }
        result += this.formatArray(items[i], remainingShape, depth + 1, shouldTruncate, edgeItems);
      }
    } else {
      // Show all
      for (let i = 0; i < currentDim; i++) {
        if (i > 0) {
          result += separator;
          if (useBlockSeparator) {
            result += ' ' + itemIndent;
          } else {
            result += itemIndent;
          }
        } else {
          result += '\n' + itemIndent;
        }
        result += this.formatArray(items[i], remainingShape, depth + 1, shouldTruncate, edgeItems);
      }
    }

    result += '\n' + baseIndent + ']';
    return result;
  }

  /**
   * Format a single value for display
   */
  private formatValue(value: unknown): string {
    if (typeof value === 'number') {
      if (Number.isInteger(value)) {
        return value.toString();
      } else {
        // Format floats to 4 decimal places
        return value.toFixed(4).replace(/\.?0+$/, '');
      }
    }
    return String(value);
  }

  /**
   * Create a copy of the tensor
   *
   * @returns New tensor with copied data
   */
  clone(): ChainablePromise<S> {
    return new ChainablePromise((resolve, reject) => {
      void (async () => {
        try {
          const buffer = await this.data.device.readData(this.data);
          const newData = this.data.device.createData(buffer.byteLength);
          await this.data.device.writeData(newData, buffer);
          resolve(new Tensor(this.transform, newData));
        } catch (error) {
          reject(error);
        }
      })();
    });
  }

  // =============================================================================
  // Matrix Multiplication
  // =============================================================================

  /**
   * Matrix multiplication
   *
   * Performs matrix multiplication following NumPy/PyTorch conventions:
   * - 1D × 1D → scalar (dot product)
   * - 1D × 2D → 1D (vector-matrix multiply)
   * - 2D × 1D → 1D (matrix-vector multiply)
   * - 2D × 2D → 2D (matrix-matrix multiply)
   * - ND × ND → ND (batched matrix multiply)
   *
   * @param other - Tensor to multiply with
   * @returns New tensor with matrix product
   * @throws {Error} If tensors are on different devices
   * @throws {Error} If shapes are not compatible for matrix multiplication
   *
   * @example
   * // Matrix multiplication
   * const a = await tensor([[1, 2], [3, 4]]);     // shape: [2, 2]
   * const b = await tensor([[5, 6], [7, 8]]);     // shape: [2, 2]
   * const c = await a.matmul(b);                  // shape: [2, 2]
   *
   * @example
   * // Vector dot product
   * const x = await tensor([1, 2, 3]);            // shape: [3]
   * const y = await tensor([4, 5, 6]);            // shape: [3]
   * const dot = await x.matmul(y);               // shape: [] (scalar)
   *
   * @example
   * // Batch matrix multiplication
   * const batch1 = await tensor([
   *   [[1, 2], [3, 4]],
   *   [[5, 6], [7, 8]]
   * ]);                                           // shape: [2, 2, 2]
   * const batch2 = await tensor([
   *   [[1, 0], [0, 1]],
   *   [[2, 0], [0, 2]]
   * ]);                                           // shape: [2, 2, 2]
   * const result = await batch1.matmul(batch2);   // shape: [2, 2, 2]
   */
  async matmul<T extends AnyStorageTransformation>(
    other: CanMatmul<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T>
      : `[TypeTensor ❌] Cannot multiply tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}] for matrix multiplication`,
  ): Promise<Tensor<MatmulOp<S['__output'], T['__output']>>> {
    if (!(other instanceof Tensor)) {
      throw new Error('Expected a Tensor instance');
    }

    if (other.device.id !== this.device.id) {
      throw new Error(
        `Tensors must be on same device: ${this.device.id as string} vs ${other.device.id as string}`,
      );
    }

    // Validate shapes can be matrix multiplied with helpful error messages
    assertMatmulCompatible(this.shape, other.shape);

    // Ensure both tensors are contiguous if device requires it
    const tensor1 = await this._ensureContiguousIfNeeded('matmul');
    const tensor2 = await other._ensureContiguousIfNeeded('matmul');

    // Compute output shape
    const outputShape = matmulShape(tensor1.shape, tensor2.shape);
    if (!outputShape) {
      throw new Error(
        `Cannot perform matrix multiplication on shapes ${formatShape(tensor1.shape)} and ${formatShape(tensor2.shape)}`,
      );
    }

    const outputStrides = computeStrides(outputShape);
    const outputSize = computeSize(outputShape);
    const promotedDtype = toPromotedDType(tensor1.dtype, tensor2.dtype);

    // Build the matmul operation with proper output metadata
    const matmulOp = {
      __op: 'matmul' as const,
      __output: {
        __dtype: promotedDtype,
        __shape: outputShape,
        __strides: outputStrides,
        __size: outputSize,
        __layout: {
          c_contiguous: true,
          f_contiguous: false,
          is_view: false,
          writeable: true,
          aligned: true,
        },
        __offset: 0,
      } as MatmulOp<S['__output'], T['__output']>['__output'],
      __inputs: [tensor1.storage, tensor2.storage] as const,
    } as MatmulOp<S['__output'], T['__output']>;

    const resultData = await tensor1.data.device.execute(matmulOp, [tensor1.data, tensor2.data]);
    return new Tensor(matmulOp, resultData);
  }

  // =============================================================================
  // Softmax Operations
  // =============================================================================

  /**
   * Apply softmax function along the specified axis
   *
   * Computes softmax(x) = exp(x) / sum(exp(x)) along the given axis.
   * This is commonly used for converting logits to probabilities.
   *
   * @param axis - The axis along which to apply softmax (supports negative indexing)
   * @returns New tensor with softmax applied
   *
   * @example
   * // Classification logits -> probabilities
   * const logits = await tensor([[1, 2, 3], [4, 5, 6]]);  // shape: [2, 3]
   * const probs = await logits.softmax(-1);                // shape: [2, 3], softmax over classes
   *
   * @example
   * // Attention weights
   * const scores = await tensor([[[1, 2], [3, 4]]]);      // shape: [1, 2, 2]
   * const weights = await scores.softmax(-1);             // shape: [1, 2, 2], softmax over key sequence
   */
  async softmax<Axis extends number>(
    axis: ValidateDim<Axis, S['__output']['__shape']> extends DimensionError<string>
      ? `[TypeTensor ❌] Invalid axis ${Axis} for tensor with shape [${ShapeToString<S['__output']['__shape']>}]. Use axis in range [-${S['__output']['__shape']['length']}, ${S['__output']['__shape']['length']})`
      : Axis,
  ): Promise<Tensor<SoftmaxOp<S['__output'], Axis>>> {
    // Validate axis parameter
    if (typeof axis !== 'number' || !Number.isInteger(axis)) {
      throw new Error('Axis must be an integer');
    }

    const rank = this.shape.length;
    const normalizedAxis = axis < 0 ? rank + axis : axis;

    if (normalizedAxis < 0 || normalizedAxis >= rank) {
      throw new Error(
        `Axis ${axis} out of bounds for tensor with ${rank} dimensions. Valid range: [-${rank}, ${rank})`,
      );
    }

    // Ensure tensor is contiguous if device requires it
    const tensor = await this._ensureContiguousIfNeeded('softmax');

    // Convert to float dtype for softmax computation
    const outputDtype = toFloatDType(tensor.dtype);
    const outputStrides = computeStrides(tensor.shape);
    const outputSize = computeSize(tensor.shape);

    // Build the softmax operation with proper output metadata
    const softmaxOp = {
      __op: 'softmax' as const,
      __output: {
        __dtype: outputDtype,
        __shape: tensor.shape,
        __strides: outputStrides,
        __size: outputSize,
        __layout: {
          c_contiguous: true,
          f_contiguous: false,
          is_view: false,
          writeable: true,
          aligned: true,
        },
        __offset: 0,
      } as SoftmaxOp<S['__output'], Axis>['__output'],
      __inputs: [tensor.storage] as const,
      __softmaxAxis: normalizedAxis,
    } as SoftmaxOp<S['__output'], Axis>;

    const resultData = await tensor.data.device.execute(softmaxOp, [tensor.data]);
    return new Tensor(softmaxOp, resultData);
  }

  /**
   * Apply log-softmax function along the specified axis
   *
   * Computes log(softmax(x)) = log(exp(x) / sum(exp(x))) = x - log(sum(exp(x)))
   * This is numerically more stable than computing log(softmax(x)) separately
   * and is commonly used in cross-entropy loss computation.
   *
   * @param axis - The axis along which to apply log-softmax (supports negative indexing)
   * @returns New tensor with log-softmax applied
   *
   * @example
   * // Classification with log probabilities for numerical stability
   * const logits = await tensor([[1, 2, 3], [4, 5, 6]]);     // shape: [2, 3]
   * const logProbs = await logits.logSoftmax(-1);            // shape: [2, 3]
   *
   * @example
   * // Suitable for cross-entropy loss computation
   * const predictions = await model.forward(input);
   * const logProbs = await predictions.logSoftmax(-1);
   * const loss = await crossEntropyLoss(logProbs, targets);
   */
  async logSoftmax<Axis extends number>(
    axis: ValidateDim<Axis, S['__output']['__shape']> extends DimensionError<string>
      ? `[TypeTensor ❌] Invalid axis ${Axis} for tensor with shape [${ShapeToString<S['__output']['__shape']>}]. Use axis in range [-${S['__output']['__shape']['length']}, ${S['__output']['__shape']['length']})`
      : Axis,
  ): Promise<Tensor<LogSoftmaxOp<S['__output'], Axis>>> {
    // Validate axis parameter
    if (typeof axis !== 'number' || !Number.isInteger(axis)) {
      throw new Error('Axis must be an integer');
    }

    const rank = this.shape.length;
    const normalizedAxis = axis < 0 ? rank + axis : axis;

    if (normalizedAxis < 0 || normalizedAxis >= rank) {
      throw new Error(
        `Axis ${axis} out of bounds for tensor with ${rank} dimensions. Valid range: [-${rank}, ${rank})`,
      );
    }

    // Ensure tensor is contiguous if device requires it
    const tensor = await this._ensureContiguousIfNeeded('log_softmax');

    // Convert to float dtype for log-softmax computation
    const outputDtype = toFloatDType(tensor.dtype);
    const outputStrides = computeStrides(tensor.shape);
    const outputSize = computeSize(tensor.shape);

    // Build the log-softmax operation with proper output metadata
    const logSoftmaxOp = {
      __op: 'log_softmax' as const,
      __output: {
        __dtype: outputDtype,
        __shape: tensor.shape,
        __strides: outputStrides,
        __size: outputSize,
        __layout: {
          c_contiguous: true,
          f_contiguous: false,
          is_view: false,
          writeable: true,
          aligned: true,
        },
        __offset: 0,
      } as LogSoftmaxOp<S['__output'], Axis>['__output'],
      __inputs: [tensor.storage] as const,
      __logSoftmaxAxis: normalizedAxis,
    } as LogSoftmaxOp<S['__output'], Axis>;

    const resultData = await tensor.data.device.execute(logSoftmaxOp, [tensor.data]);
    return new Tensor(logSoftmaxOp, resultData);
  }

  // =============================================================================
  // Reduction Operations
  // =============================================================================

  /**
   * Sum of tensor elements along specified axes
   *
   * Computes the sum of tensor elements along the given axes. If no axes are
   * specified, sums all elements to produce a scalar result.
   *
   * @param axes - Axes along which to sum (supports negative indexing)
   * @param keepdims - Whether to keep reduced dimensions as size 1
   * @returns New tensor with summed values
   *
   * @example
   * // Sum along specific axis
   * const x = await tensor([[1, 2, 3], [4, 5, 6]]);  // shape: [2, 3]
   * const rowSums = await x.sum([1]);                 // shape: [2] - sum each row
   * const colSums = await x.sum([0]);                 // shape: [3] - sum each column
   *
   * @example
   * // Global sum (all elements)
   * const total = await x.sum();                      // shape: [] - scalar result
   *
   * @example
   * // Keep dimensions
   * const keepDims = await x.sum([0], true);          // shape: [1, 3] - keep batch dim
   */
  async sum<
    Axes extends readonly number[] | undefined = undefined,
    KeepDims extends boolean = false,
  >(
    axes?: ValidateReduction<S['__output']['__shape'], Axes> extends true
      ? Axes
      : `[TypeTensor ❌] Invalid axes for sum reduction on tensor with shape [${ShapeToString<S['__output']['__shape']>}]`,
    keepdims?: KeepDims,
  ): Promise<Tensor<SumOp<S['__output'], Axes, KeepDims>>> {
    // Normalize and validate axes at runtime
    const normalizedAxes = this.normalizeReductionAxes(axes as readonly number[] | undefined);
    const keepDimsFlag = keepdims ?? false;

    // Ensure tensor is contiguous if device requires it
    const tensor = await this._ensureContiguousIfNeeded('sum');

    // Compute output shape based on reduction
    const outputShape = this.computeReductionShape(normalizedAxes, keepDimsFlag);
    const outputStrides = computeStrides(outputShape);
    const outputSize = computeSize(outputShape);

    // Build the sum operation with proper output metadata
    const sumOp = {
      __op: 'sum' as const,
      __output: {
        __dtype: tensor.storage.__dtype,
        __shape: outputShape as any, // Runtime shape
        __strides: outputStrides as any, // Runtime strides
        __size: outputSize,
        __layout: {
          c_contiguous: true,
          f_contiguous: false,
          is_view: false,
          writeable: true,
          aligned: true,
        },
        __offset: 0,
      },
      __inputs: [tensor.storage] as const,
      __sumAxes: axes,
      __keepDims: keepDimsFlag,
    } as unknown as SumOp<S['__output'], Axes, KeepDims>;

    const resultData = await tensor.data.device.execute(sumOp as any, [tensor.data]);
    return new Tensor(sumOp, resultData) as Tensor<SumOp<S['__output'], Axes, KeepDims>>;
  }

  /**
   * Mean of tensor elements along specified axes
   *
   * Computes the arithmetic mean of tensor elements along the given axes.
   * If no axes are specified, computes the mean of all elements to produce
   * a scalar result. The output is always a floating-point type.
   *
   * @param axes - Axes along which to compute mean (supports negative indexing)
   * @param keepdims - Whether to keep reduced dimensions as size 1
   * @returns New tensor with mean values
   *
   * @example
   * // Mean along specific axis
   * const x = await tensor([[1, 2, 3], [4, 5, 6]]);  // shape: [2, 3]
   * const rowMeans = await x.mean([1]);               // shape: [2] - mean of each row
   * const colMeans = await x.mean([0]);               // shape: [3] - mean of each column
   *
   * @example
   * // Global mean (all elements)
   * const avgValue = await x.mean();                  // shape: [] - scalar result
   *
   * @example
   * // Layer normalization pattern
   * const features = await tensor([[[1, 2], [3, 4]]]); // shape: [1, 2, 2]
   * const layerMean = await features.mean([-1], true);  // shape: [1, 2, 1]
   */
  async mean<
    Axes extends readonly number[] | undefined = undefined,
    KeepDims extends boolean = false,
  >(
    axes?: ValidateReduction<S['__output']['__shape'], Axes> extends true
      ? Axes
      : `[TypeTensor ❌] Invalid axes for mean reduction on tensor with shape [${ShapeToString<S['__output']['__shape']>}]`,
    keepdims?: KeepDims,
  ): Promise<Tensor<MeanOp<S['__output'], Axes, KeepDims>>> {
    // Normalize and validate axes at runtime
    const normalizedAxes = this.normalizeReductionAxes(axes as readonly number[] | undefined);
    const keepDimsFlag = keepdims ?? false;

    // Ensure tensor is contiguous if device requires it
    const tensor = await this._ensureContiguousIfNeeded('mean');

    // Compute output shape based on reduction
    const outputShape = this.computeReductionShape(normalizedAxes, keepDimsFlag);
    const outputStrides = computeStrides(outputShape);
    const outputSize = computeSize(outputShape);
    const outputDtype = toFloatDType(tensor.dtype);

    // Build the mean operation with proper output metadata
    const meanOp = {
      __op: 'mean' as const,
      __output: {
        __dtype: outputDtype,
        __shape: outputShape as any, // Runtime shape
        __strides: outputStrides as any, // Runtime strides
        __size: outputSize,
        __layout: {
          c_contiguous: true,
          f_contiguous: false,
          is_view: false,
          writeable: true,
          aligned: true,
        },
        __offset: 0,
      },
      __inputs: [tensor.storage] as const,
      __meanAxes: axes,
      __keepDims: keepDimsFlag,
    } as unknown as MeanOp<S['__output'], Axes, KeepDims>;

    const resultData = await tensor.data.device.execute(meanOp as any, [tensor.data]);
    return new Tensor(meanOp, resultData) as Tensor<MeanOp<S['__output'], Axes, KeepDims>>;
  }

  /**
   * Maximum of tensor elements along specified axes
   *
   * Computes the maximum of tensor elements along the given axes.
   * If no axes are specified, computes the maximum of all elements to produce
   * a scalar result. The output preserves the input data type.
   *
   * @param axes - Axes along which to compute maximum (supports negative indexing)
   * @param keepdims - Whether to keep reduced dimensions as size 1
   * @returns New tensor with maximum values
   *
   * @example
   * // Max along specific axis
   * const x = await tensor([[1, 5, 3], [4, 2, 6]]);  // shape: [2, 3]
   * const rowMaxes = await x.max([1]);               // shape: [2] - max of each row
   * const colMaxes = await x.max([0]);               // shape: [3] - max of each column
   *
   * @example
   * // Global max (all elements)
   * const maxValue = await x.max();                  // shape: [] - scalar result
   *
   * @example
   * // Attention mechanism pattern
   * const scores = await tensor([[[0.1, 0.9], [0.8, 0.2]]]); // shape: [1, 2, 2]
   * const maxScores = await scores.max([-1], true);           // shape: [1, 2, 1]
   */
  async max<
    Axes extends readonly number[] | undefined = undefined,
    KeepDims extends boolean = false,
  >(
    axes?: ValidateReduction<S['__output']['__shape'], Axes> extends true
      ? Axes
      : `[TypeTensor ❌] Invalid axes for max reduction on tensor with shape [${ShapeToString<S['__output']['__shape']>}]`,
    keepdims?: KeepDims,
  ): Promise<Tensor<MaxOp<S['__output'], Axes, KeepDims>>> {
    // Normalize and validate axes at runtime
    const normalizedAxes = this.normalizeReductionAxes(axes as readonly number[] | undefined);
    const keepDimsFlag = keepdims ?? false;

    // Ensure tensor is contiguous if device requires it
    const tensor = await this._ensureContiguousIfNeeded('max');

    // Compute output shape based on reduction
    const outputShape = this.computeReductionShape(normalizedAxes, keepDimsFlag);
    const outputStrides = computeStrides(outputShape);
    const outputSize = computeSize(outputShape);

    // Build the max operation with proper output metadata
    const maxOp = {
      __op: 'max' as const,
      __output: {
        __dtype: tensor.storage.__dtype,
        __shape: outputShape as any, // Runtime shape
        __strides: outputStrides as any, // Runtime strides
        __size: outputSize,
        __layout: {
          c_contiguous: true,
          f_contiguous: false,
          is_view: false,
          writeable: true,
          aligned: true,
        },
        __offset: 0,
      },
      __inputs: [tensor.storage] as const,
      __maxAxes: axes,
      __keepDims: keepDimsFlag,
    } as unknown as MaxOp<S['__output'], Axes, KeepDims>;

    const resultData = await tensor.data.device.execute(maxOp as any, [tensor.data]);
    return new Tensor(maxOp, resultData) as Tensor<MaxOp<S['__output'], Axes, KeepDims>>;
  }

  /**
   * Minimum of tensor elements along specified axes
   *
   * Computes the minimum of tensor elements along the given axes.
   * If no axes are specified, computes the minimum of all elements to produce
   * a scalar result. The output preserves the input data type.
   *
   * @param axes - Axes along which to compute minimum (supports negative indexing)
   * @param keepdims - Whether to keep reduced dimensions as size 1
   * @returns New tensor with minimum values
   *
   * @example
   * // Min along specific axis
   * const x = await tensor([[1, 5, 3], [4, 2, 6]]);  // shape: [2, 3]
   * const rowMins = await x.min([1]);                // shape: [2] - min of each row
   * const colMins = await x.min([0]);                // shape: [3] - min of each column
   *
   * @example
   * // Global min (all elements)
   * const minValue = await x.min();                  // shape: [] - scalar result
   *
   * @example
   * // Attention masking pattern
   * const logits = await tensor([[[0.1, -1e9], [0.8, 0.2]]]); // shape: [1, 2, 2]
   * const minLogits = await logits.min([-1], true);            // shape: [1, 2, 1]
   */
  async min<
    Axes extends readonly number[] | undefined = undefined,
    KeepDims extends boolean = false,
  >(
    axes?: ValidateReduction<S['__output']['__shape'], Axes> extends true
      ? Axes
      : `[TypeTensor ❌] Invalid axes for min reduction on tensor with shape [${ShapeToString<S['__output']['__shape']>}]`,
    keepdims?: KeepDims,
  ): Promise<Tensor<MinOp<S['__output'], Axes, KeepDims>>> {
    // Normalize and validate axes at runtime
    const normalizedAxes = this.normalizeReductionAxes(axes as readonly number[] | undefined);
    const keepDimsFlag = keepdims ?? false;

    // Ensure tensor is contiguous if device requires it
    const tensor = await this._ensureContiguousIfNeeded('min');

    // Compute output shape based on reduction
    const outputShape = this.computeReductionShape(normalizedAxes, keepDimsFlag);
    const outputStrides = computeStrides(outputShape);
    const outputSize = computeSize(outputShape);

    // Build the min operation with proper output metadata
    const minOp = {
      __op: 'min' as const,
      __output: {
        __dtype: tensor.storage.__dtype,
        __shape: outputShape as any, // Runtime shape
        __strides: outputStrides as any, // Runtime strides
        __size: outputSize,
        __layout: {
          c_contiguous: true,
          f_contiguous: false,
          is_view: false,
          writeable: true,
          aligned: true,
        },
        __offset: 0,
      },
      __inputs: [tensor.storage] as const,
      __minAxes: axes,
      __keepDims: keepDimsFlag,
    } as unknown as MinOp<S['__output'], Axes, KeepDims>;

    const resultData = await tensor.data.device.execute(minOp as any, [tensor.data]);
    return new Tensor(minOp, resultData) as Tensor<MinOp<S['__output'], Axes, KeepDims>>;
  }

  // =============================================================================
  // Reduction Utilities
  // =============================================================================

  /**
   * Normalize and validate reduction axes
   * Handles undefined (all axes), negative indexing, and validation
   */
  private normalizeReductionAxes(axes?: readonly number[]): number[] | undefined {
    if (axes === undefined) {
      return undefined; // Reduce all axes
    }

    const rank = this.shape.length;
    const normalizedAxes: number[] = [];

    for (const axis of axes) {
      if (!Number.isInteger(axis)) {
        throw new Error(`Axis must be an integer, got ${axis}`);
      }

      const normalizedAxis = axis < 0 ? rank + axis : axis;

      if (normalizedAxis < 0 || normalizedAxis >= rank) {
        throw new Error(
          `Axis ${axis} out of bounds for tensor with ${rank} dimensions. Valid range: [-${rank}, ${rank})`,
        );
      }

      if (normalizedAxes.includes(normalizedAxis)) {
        throw new Error(`Duplicate axis ${axis} in reduction axes`);
      }

      normalizedAxes.push(normalizedAxis);
    }

    return normalizedAxes;
  }

  /**
   * Compute the output shape after reduction
   */
  private computeReductionShape(axes: number[] | undefined, keepdims: boolean): number[] {
    if (axes === undefined) {
      // Reduce all axes
      return keepdims ? this.shape.map(() => 1) : [];
    }

    const outputShape: number[] = [];
    for (let i = 0; i < this.shape.length; i++) {
      if (axes.includes(i)) {
        // This axis is being reduced
        if (keepdims) {
          outputShape.push(1);
        }
        // else: skip this dimension (remove it)
      } else {
        // This axis is preserved
        outputShape.push(this.shape[i]!);
      }
    }

    return outputShape;
  }
}
