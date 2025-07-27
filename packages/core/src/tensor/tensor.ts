/**
 * Core Tensor class implementation
 *
 * This module provides the main Tensor class that combines storage metadata
 * with backend execution to provide a high-level tensor API.
 */

import type { AnyStorageTransformation, ComputeStrides } from '../storage/layout';
import type { DeviceData, Device } from '../device/types';
import type { Shape, CanReshape, Product, CanBroadcast, ShapeToString, SliceIndex } from '../shape/types';
import type { Neg, Abs, Sin, Cos, Exp, Log, Sqrt, Square } from '../storage/unary';
import type { Add, Sub, Mul, Div } from '../storage/binary';
import type { ReshapeOp, Flatten, View, SliceOp } from '../storage/view';
import type { DTypeValue } from '../dtype/types';
import type { NestedArray } from './types';
import { bufferToNestedArray } from './types';
import { formatShape, broadcastShapes, assertShapesCompatible } from '../shape';
import { computeStrides, computeSize, computeSlicedShape, computeSlicedStrides, validateSliceIndices } from './utils';
import { toFloatDType, toPromotedDType } from '../dtype';
import type { Mod } from 'ts-arithmetic';

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
  // Unary Operations
  // =============================================================================

  /**
   * Element-wise negation
   *
   * @returns New tensor with negated values
   */
  async neg(): Promise<Tensor<Neg<S['__output']>>> {
    // Build the transformation with proper output metadata
    const negOp: Neg<S['__output']> = {
      __op: 'neg',
      __output: {
        __dtype: this.storage.__dtype,
        __shape: this.storage.__shape,
        __strides: this.storage.__strides,
        __size: this.storage.__size,
        __layout: {
          c_contiguous: true,
          f_contiguous: false,
          is_view: false,
          writeable: true,
          aligned: true,
        },
        __offset: 0,
      } as Neg<S['__output']>['__output'],
      __inputs: [this.storage] as const,
    };

    const resultData = await this.data.device.execute(negOp, [this.data]);
    return new Tensor(negOp, resultData);
  }

  /**
   * Element-wise absolute value
   *
   * @returns New tensor with absolute values
   */
  async abs(): Promise<Tensor<Abs<S['__output']>>> {
    // Build the transformation with proper output metadata
    const absOp: Abs<S['__output']> = {
      __op: 'abs',
      __output: {
        __dtype: this.storage.__dtype,
        __shape: this.storage.__shape,
        __strides: this.storage.__strides,
        __size: this.storage.__size,
        __layout: {
          c_contiguous: true,
          f_contiguous: false,
          is_view: false,
          writeable: true,
          aligned: true,
        },
        __offset: 0,
      } as Abs<S['__output']>['__output'],
      __inputs: [this.storage] as const,
    };

    const resultData = await this.data.device.execute(absOp, [this.data]);
    return new Tensor(absOp, resultData);
  }

  /**
   * Element-wise sine
   *
   * @returns New tensor with sine values
   */
  async sin(): Promise<Tensor<Sin<S['__output']>>> {
    const sinOp = {
      __op: 'sin' as const,
      __output: {
        ...this.storage,
        __dtype: toFloatDType(this.dtype),
      } as Sin<S['__output']>['__output'],
      __inputs: [this.storage] as const,
    } as Sin<S['__output']>;

    const resultData = await this.data.device.execute(sinOp, [this.data]);
    return new Tensor(sinOp, resultData);
  }

  /**
   * Element-wise cosine
   *
   * @returns New tensor with cosine values
   */
  async cos(): Promise<Tensor<Cos<S['__output']>>> {
    const cosOp = {
      __op: 'cos' as const,
      __output: {
        ...this.storage,
        __dtype: toFloatDType(this.dtype),
      } as Cos<S['__output']>['__output'],
      __inputs: [this.storage] as const,
    } as Cos<S['__output']>;

    const resultData = await this.data.device.execute(cosOp, [this.data]);
    return new Tensor(cosOp, resultData);
  }

  /**
   * Element-wise exponential
   *
   * @returns New tensor with exponential values
   */
  async exp(): Promise<Tensor<Exp<S['__output']>>> {
    const expOp = {
      __op: 'exp' as const,
      __output: {
        ...this.storage,
        __dtype: toFloatDType(this.dtype),
      } as Exp<S['__output']>['__output'],
      __inputs: [this.storage] as const,
    } as Exp<S['__output']>;

    const resultData = await this.data.device.execute(expOp, [this.data]);
    return new Tensor(expOp, resultData);
  }

  /**
   * Element-wise natural logarithm
   *
   * @returns New tensor with logarithm values
   */
  async log(): Promise<Tensor<Log<S['__output']>>> {
    const logOp = {
      __op: 'log' as const,
      __output: {
        ...this.storage,
        __dtype: toFloatDType(this.dtype),
      } as Log<S['__output']>['__output'],
      __inputs: [this.storage] as const,
    } as Log<S['__output']>;

    const resultData = await this.data.device.execute(logOp, [this.data]);
    return new Tensor(logOp, resultData);
  }

  /**
   * Element-wise square root
   *
   * @returns New tensor with square root values
   */
  async sqrt(): Promise<Tensor<Sqrt<S['__output']>>> {
    const sqrtOp = {
      __op: 'sqrt' as const,
      __output: {
        ...this.storage,
        __dtype: toFloatDType(this.dtype),
      } as Sqrt<S['__output']>['__output'],
      __inputs: [this.storage] as const,
    } as Sqrt<S['__output']>;

    const resultData = await this.data.device.execute(sqrtOp, [this.data]);
    return new Tensor(sqrtOp, resultData);
  }

  /**
   * Element-wise square
   *
   * @returns New tensor with squared values
   */
  async square(): Promise<Tensor<Square<S['__output']>>> {
    const squareOp = {
      __op: 'square' as const,
      __output: {
        ...this.storage,
      } as Square<S['__output']>['__output'],
      __inputs: [this.storage] as const,
    } as Square<S['__output']>;

    const resultData = await this.data.device.execute(squareOp, [this.data]);
    return new Tensor(squareOp, resultData);
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
  async add<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T>
      : `[TypeTensor ❌] Cannot add tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): Promise<Tensor<Add<S['__output'], T['__output']>>> {
    if (!(other instanceof Tensor)) {
      throw new Error('Expected a Tensor instance');
    }

    if (other.device.id !== this.device.id) {
      // NOTE: the as string is a workaround, typescript properly infers the type as string... but eslint does not
      throw new Error(
        `Tensors must be on same device: ${this.device.id as string} vs ${other.device.id as string}`,
      );
    }

    // Validate shapes can broadcast with helpful error messages
    assertShapesCompatible(this.shape, other.shape, 'element-wise addition');

    // Compute broadcast shape and promoted dtype
    const outputShape = broadcastShapes(this.shape, other.shape);
    const outputStrides = computeStrides(outputShape);
    const outputSize = computeSize(outputShape);
    const promotedDtype = toPromotedDType(this.dtype, other.dtype);

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
      __inputs: [this.storage, other.storage] as const,
    } as Add<S['__output'], T['__output']>;

    const resultData = await this.data.device.execute(addOp, [this.data, other.data]);
    return new Tensor(addOp, resultData);
  }

  /**
   * Element-wise subtraction with broadcasting
   *
   * @param other - Tensor to subtract
   * @returns New tensor with difference
   * @throws {Error} If tensors are on different devices
   * @throws {Error} If shapes cannot broadcast
   */
  async sub<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T>
      : `[TypeTensor ❌] Cannot subtract tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): Promise<Tensor<Sub<S['__output'], T['__output']>>> {
    if (!(other instanceof Tensor)) {
      throw new Error('Expected a Tensor instance');
    }

    if (other.device.id !== this.device.id) {
      throw new Error(
        `Tensors must be on same device: ${this.device.id as string} vs ${other.device.id as string}`,
      );
    }

    // Compute broadcast shape and promoted dtype
    const outputShape = broadcastShapes(this.shape, other.shape);
    const outputStrides = computeStrides(outputShape);
    const outputSize = computeSize(outputShape);
    const promotedDtype = toPromotedDType(this.dtype, other.dtype);

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
      __inputs: [this.storage, other.storage] as const,
    } as Sub<S['__output'], T['__output']>;

    const resultData = await this.data.device.execute(subOp, [this.data, other.data]);
    return new Tensor(subOp, resultData);
  }

  /**
   * Element-wise multiplication with broadcasting
   *
   * @param other - Tensor to multiply
   * @returns New tensor with product
   * @throws {Error} If tensors are on different devices
   * @throws {Error} If shapes cannot broadcast
   */
  async mul<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T>
      : `[TypeTensor ❌] Cannot multiply tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): Promise<Tensor<Mul<S['__output'], T['__output']>>> {
    if (!(other instanceof Tensor)) {
      throw new Error('Expected a Tensor instance');
    }

    if (other.device.id !== this.device.id) {
      throw new Error(
        `Tensors must be on same device: ${this.device.id as string} vs ${other.device.id as string}`,
      );
    }

    // Compute broadcast shape and promoted dtype
    const outputShape = broadcastShapes(this.shape, other.shape);
    const outputStrides = computeStrides(outputShape);
    const outputSize = computeSize(outputShape);
    const promotedDtype = toPromotedDType(this.dtype, other.dtype);

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
      __inputs: [this.storage, other.storage] as const,
    } as Mul<S['__output'], T['__output']>;

    const resultData = await this.data.device.execute(mulOp, [this.data, other.data]);
    return new Tensor(mulOp, resultData);
  }

  /**
   * Element-wise division with broadcasting
   *
   * @param other - Tensor to divide by
   * @returns New tensor with quotient
   * @throws {Error} If tensors are on different devices
   * @throws {Error} If shapes cannot broadcast
   */
  async div<T extends AnyStorageTransformation>(
    other: CanBroadcast<S['__output']['__shape'], T['__output']['__shape']> extends true
      ? Tensor<T>
      : `[TypeTensor ❌] Cannot divide tensors with shapes [${ShapeToString<S['__output']['__shape']>}] and [${ShapeToString<T['__output']['__shape']>}]. Shapes must be compatible for broadcasting.`,
  ): Promise<Tensor<Div<S['__output'], T['__output']>>> {
    if (!(other instanceof Tensor)) {
      throw new Error('Expected a Tensor instance');
    }

    if (other.device.id !== this.device.id) {
      throw new Error(
        `Tensors must be on same device: ${this.device.id as string} vs ${other.device.id as string}`,
      );
    }

    // Compute broadcast shape and promoted dtype
    const outputShape = broadcastShapes(this.shape, other.shape);
    const outputStrides = computeStrides(outputShape);
    const outputSize = computeSize(outputShape);
    const promotedDtype = toPromotedDType(this.dtype, other.dtype);

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
      __inputs: [this.storage, other.storage] as const,
    } as Div<S['__output'], T['__output']>;

    const resultData = await this.data.device.execute(divOp, [this.data, other.data]);
    return new Tensor(divOp, resultData);
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
  ): CanReshape<S['__output']['__shape'], NewShape> extends true
    ? Tensor<ReshapeOp<S['__output'], NewShape>>
    : never {
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

    const reshapeOp = {
      __op: 'reshape' as const,
      __output: {
        __dtype: this.storage.__dtype,
        __shape: validShape,
        __strides: computeStrides(validShape) as ComputeStrides<NewShape>,
        __size: computeSize(validShape) as Product<NewShape>,
        __layout: {
          ...this.storage.__layout,
          is_view: true,
        },
        __offset: this.storage.__offset,
      } as ReshapeOp<S['__output'], NewShape>['__output'],
      __inputs: [this.storage] as const,
    } as ReshapeOp<S['__output'], NewShape>;

    // Reshape typically returns a view (same data, new metadata)
    return new Tensor(reshapeOp, this.data) as CanReshape<
      S['__output']['__shape'],
      NewShape
    > extends true
      ? Tensor<ReshapeOp<S['__output'], NewShape>>
      : never;
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
   * @returns 1D view of the tensor
   */
  flatten(): Tensor<Flatten<S['__output']>> {
    const totalSize = this.size;
    return this._reshapeUnsafe([totalSize] as const) as Tensor<Flatten<S['__output']>>;
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
  ): Tensor<View<S['__output'], NewShape>> {
    // Infer shape if -1 is present
    const inferredShape = this.inferShape(shape);

    const viewOp = {
      __op: 'view' as const,
      __output: {
        __dtype: this.storage.__dtype,
        __shape: inferredShape,
        __strides: computeStrides(inferredShape),
        __size: this.storage.__size, // Size remains the same
        __layout: {
          ...this.storage.__layout,
          is_view: true,
        },
        __offset: this.storage.__offset,
      } as View<S['__output'], NewShape>['__output'],
      __inputs: [this.storage] as const,
    } as View<S['__output'], NewShape>;

    return new Tensor(viewOp, this.data);
  }

  /**
   * Infer shape dimension when -1 is present
   */
  private inferShape(shape: readonly (number | -1)[]): readonly number[] {
    const totalSize = this.size;
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
  async slice<Indices extends readonly SliceIndex[]>(
    indices: Indices,
  ): Promise<Tensor<SliceOp<S['__output'], Indices>>> {
    // Validate indices
    validateSliceIndices(this.shape, indices);

    // Compute runtime shape and strides
    const slicedShape = computeSlicedShape(this.shape, indices);
    const slicedStrides = computeSlicedStrides(this.strides, indices);
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
          ...this.storage.__layout,
          is_view: true,
          // Slicing may break contiguity
          c_contiguous: 'unknown' as const,
          f_contiguous: 'unknown' as const,
        },
        __offset: this.storage.__offset, // Backend will compute actual offset
        __sliceIndices: indices,
      } as SliceOp<S['__output'], Indices>['__output'] & { __sliceIndices: Indices },
      __inputs: [this.storage] as const,
    } as SliceOp<S['__output'], Indices>;

    // Execute on device - slicing needs device support for proper memory handling
    const resultData = await this.data.device.execute(sliceOp, [this.data]);
    return new Tensor(sliceOp, resultData);
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
    return bufferToNestedArray(buffer, this.shape, this.dtype);
  }

  /**
   * Get a scalar value (for 0-dimensional tensors)
   *
   * @returns Scalar value
   * @throws {Error} If tensor is not 0-dimensional
   */
  async item(): Promise<DTypeValue<S['__output']['__dtype']>> {
    if (this.ndim !== 0) {
      throw new Error(`item() only works on scalars, got shape ${formatShape(this.shape)}`);
    }

    const array = await this.toArray();
    return array as DTypeValue<S['__output']['__dtype']>;
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
  async clone(): Promise<Tensor<S>> {
    const buffer = await this.data.device.readData(this.data);
    const newData = this.data.device.createData(buffer.byteLength);
    await this.data.device.writeData(newData, buffer);
    return new Tensor(this.transform, newData);
  }
}
