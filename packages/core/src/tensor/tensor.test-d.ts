/**
 * Type tests for tensor module
 *
 * These tests validate compile-time type safety for tensor creation,
 * operations, and transformations.
 */

import { expectTypeOf } from 'expect-type';
import type { Tensor } from './tensor';
import type { TensorStorage, CreateOp } from '../storage/layout';
import type { Neg, Abs } from '../storage/unary';
import type { Add } from '../storage/binary';
import type { ReshapeOp, Flatten, View } from '../storage/view';
import type { NestedArray, InferShape, TensorOptions, DTypeValue } from './types';
import type { Float32, Int32, Bool, Int64, AnyDType } from '../dtype/types';
import type { Shape, CanReshape } from '../shape/types';

// Mock creation functions for type testing
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare function tensor<D extends AnyDType, S extends Shape>(
  data: NestedArray<number, S>,
  options?: TensorOptions<D>,
): Promise<Tensor<CreateOp<TensorStorage<D, S>>>>;

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare function zeros<D extends AnyDType, S extends Shape>(
  shape: S,
  options?: TensorOptions<D>,
): Promise<Tensor<CreateOp<TensorStorage<D, S>>>>;

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare function ones<D extends AnyDType, S extends Shape>(
  shape: S,
  options?: TensorOptions<D>,
): Promise<Tensor<CreateOp<TensorStorage<D, S>>>>;

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare function eye<D extends AnyDType>(
  n: number,
  options?: TensorOptions<D>,
): Promise<Tensor<CreateOp<TensorStorage<D, readonly [number, number]>>>>;

// =============================================================================
// Shape Inference Tests
// =============================================================================

// Test InferShape utility type
expectTypeOf<InferShape<number>>().toEqualTypeOf<readonly []>();
expectTypeOf<InferShape<[1, 2, 3]>>().toEqualTypeOf<readonly [3]>();
expectTypeOf<InferShape<[[1, 2], [3, 4]]>>().toEqualTypeOf<readonly [2, 2]>();
expectTypeOf<InferShape<[[[1, 2]], [[3, 4]]]>>().toEqualTypeOf<readonly [2, 1, 2]>();
expectTypeOf<InferShape<[[[[5]]]]>>().toEqualTypeOf<readonly [1, 1, 1, 1]>();

// =============================================================================
// NestedArray Type Tests
// =============================================================================

// Test NestedArray construction for various shapes
expectTypeOf<NestedArray<number, readonly []>>().toEqualTypeOf<number>();
expectTypeOf<NestedArray<number, readonly [3]>>().toEqualTypeOf<[number, number, number]>();
expectTypeOf<NestedArray<number, readonly [2, 2]>>().toEqualTypeOf<
  [[number, number], [number, number]]
>();
expectTypeOf<NestedArray<number, readonly [2, 1, 2]>>().toEqualTypeOf<
  [[[number, number]], [[number, number]]]
>();

// Test with different element types
expectTypeOf<NestedArray<boolean, readonly [2]>>().toEqualTypeOf<[boolean, boolean]>();
expectTypeOf<NestedArray<bigint, readonly [2, 1]>>().toEqualTypeOf<[[bigint], [bigint]]>();
expectTypeOf<NestedArray<string, readonly [1, 2]>>().toEqualTypeOf<[[string, string]]>();

// =============================================================================
// Tensor Creation Type Tests
// =============================================================================

// Test tensor creation with default dtype (Float32)
declare const t1: Awaited<ReturnType<typeof tensor<Float32, readonly [2, 2]>>>;
expectTypeOf(t1).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Float32, readonly [2, 2]>>>>();
expectTypeOf(t1.shape).toEqualTypeOf<readonly [2, 2]>();
expectTypeOf(t1.dtype).toEqualTypeOf<Float32>();

// Test tensor creation with explicit dtype
declare const t2: Awaited<ReturnType<typeof tensor<Int32, readonly [3]>>>;
expectTypeOf(t2).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Int32, readonly [3]>>>>();
expectTypeOf(t2.dtype).toEqualTypeOf<Int32>();

// Test zeros creation
declare const z1: Awaited<ReturnType<typeof zeros<Float32, readonly [2, 3]>>>;
expectTypeOf(z1).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Float32, readonly [2, 3]>>>>();
expectTypeOf(z1.shape).toEqualTypeOf<readonly [2, 3]>();

// Test ones creation with custom dtype
declare const o1: Awaited<ReturnType<typeof ones<Int32, readonly [4, 5, 6]>>>;
expectTypeOf(o1).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Int32, readonly [4, 5, 6]>>>>();
expectTypeOf(o1.shape).toEqualTypeOf<readonly [4, 5, 6]>();

// Test eye (identity matrix) creation
declare const I: Awaited<ReturnType<typeof eye<Float32>>>;
expectTypeOf(I).toEqualTypeOf<
  Tensor<CreateOp<TensorStorage<Float32, readonly [number, number]>>>
>();

// =============================================================================
// Tensor Property Type Tests
// =============================================================================

declare const tensorProps: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 3, 4]>>>;

// Test shape property
expectTypeOf(tensorProps.shape).toEqualTypeOf<readonly [2, 3, 4]>();

// Test dtype property
expectTypeOf(tensorProps.dtype).toEqualTypeOf<Float32>();

// Test size property (should be Product<Shape>)
expectTypeOf(tensorProps.size).toEqualTypeOf<24>();

// Test ndim property (always number)
expectTypeOf(tensorProps.ndim).toEqualTypeOf<number>();

// Test strides property
expectTypeOf(tensorProps.strides).toMatchTypeOf<readonly number[]>();

// Test layout property
expectTypeOf(tensorProps.layout).toMatchTypeOf<{
  c_contiguous: boolean;
  f_contiguous: boolean;
  is_view: boolean;
  writeable: boolean;
  aligned: boolean;
}>();

// =============================================================================
// Unary Operation Type Tests
// =============================================================================

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const unaryTensor: Tensor<CreateOp<TensorStorage<Int32, readonly [2, 2]>>>;

// Test negation preserves shape and dtype
declare const negResult: Awaited<ReturnType<typeof unaryTensor.neg>>;
expectTypeOf(negResult).toEqualTypeOf<Tensor<Neg<TensorStorage<Int32, readonly [2, 2]>>>>();
expectTypeOf(negResult.shape).toEqualTypeOf<readonly [2, 2]>();
expectTypeOf(negResult.dtype).toEqualTypeOf<Int32>();

// Test absolute value preserves shape and dtype
declare const absResult: Awaited<ReturnType<typeof unaryTensor.abs>>;
expectTypeOf(absResult).toEqualTypeOf<Tensor<Abs<TensorStorage<Int32, readonly [2, 2]>>>>();
expectTypeOf(absResult.shape).toEqualTypeOf<readonly [2, 2]>();
expectTypeOf(absResult.dtype).toEqualTypeOf<Int32>();

// =============================================================================
// Binary Operation Type Tests
// =============================================================================

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const addTensor1: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 2]>>>;
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const addTensor2: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 2]>>>;
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const addScalar: Tensor<CreateOp<TensorStorage<Float32, readonly []>>>;
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const addBroadcast1: Tensor<CreateOp<TensorStorage<Float32, readonly [1, 3]>>>;
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const addBroadcast2: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 1]>>>;

// Test same-shape addition
declare const addResult1: Awaited<
  ReturnType<typeof addTensor1.add<(typeof addTensor2)['transform']>>
>;
expectTypeOf(addResult1).toMatchTypeOf<
  Tensor<Add<TensorStorage<Float32, readonly [2, 2]>, TensorStorage<Float32, readonly [2, 2]>>>
>();

// Test scalar broadcasting
declare const addResult2: Awaited<
  ReturnType<typeof addTensor1.add<(typeof addScalar)['transform']>>
>;
expectTypeOf(addResult2).toMatchTypeOf<
  Tensor<Add<TensorStorage<Float32, readonly [2, 2]>, TensorStorage<Float32, readonly []>>>
>();

// Test broadcasting with different shapes
declare const addResult3: Awaited<
  ReturnType<typeof addBroadcast1.add<(typeof addBroadcast2)['transform']>>
>;
expectTypeOf(addResult3).toMatchTypeOf<
  Tensor<Add<TensorStorage<Float32, readonly [1, 3]>, TensorStorage<Float32, readonly [2, 1]>>>
>();

// =============================================================================
// View Operation Type Tests
// =============================================================================

declare const viewTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [6]>>>;

// Test valid reshape
const reshapeResult1 = viewTensor.reshape([2, 3] as const);
expectTypeOf(reshapeResult1).toMatchTypeOf<
  Tensor<ReshapeOp<TensorStorage<Float32, readonly [6]>, readonly [2, 3]>>
>();
expectTypeOf(reshapeResult1.shape).toEqualTypeOf<readonly [2, 3]>();

const reshapeResult2 = viewTensor.reshape([3, 2] as const);
expectTypeOf(reshapeResult2).toMatchTypeOf<
  Tensor<ReshapeOp<TensorStorage<Float32, readonly [6]>, readonly [3, 2]>>
>();

// Test reshape validation at type level
expectTypeOf<CanReshape<readonly [6], readonly [2, 3]>>().toEqualTypeOf<true>();
expectTypeOf<CanReshape<readonly [6], readonly [2, 2]>>().toEqualTypeOf<false>();
expectTypeOf<CanReshape<readonly [2, 3, 4], readonly [6, 4]>>().toEqualTypeOf<true>();
expectTypeOf<CanReshape<readonly [2, 3, 4], readonly [5, 5]>>().toEqualTypeOf<false>();

// Test flatten
declare const flattenTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [3, 2]>>>;
const flatResult = flattenTensor.flatten();
expectTypeOf(flatResult).toMatchTypeOf<Tensor<Flatten<TensorStorage<Float32, readonly [3, 2]>>>>();

// Test view with dimension inference
declare const viewInferenceTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 3]>>>;
const viewResult1 = viewInferenceTensor.view([-1, 2] as const);
expectTypeOf(viewResult1).toMatchTypeOf<
  Tensor<View<TensorStorage<Float32, readonly [2, 3]>, readonly [-1, 2]>>
>();

const viewResult2 = viewInferenceTensor.view([3, -1] as const);
expectTypeOf(viewResult2).toMatchTypeOf<
  Tensor<View<TensorStorage<Float32, readonly [2, 3]>, readonly [3, -1]>>
>();

// =============================================================================
// Data Access Type Tests
// =============================================================================

// Test toArray return types for different dtypes
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const floatTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 2]>>>;
declare const floatArray: Awaited<ReturnType<typeof floatTensor.toArray>>;
expectTypeOf(floatArray).toEqualTypeOf<NestedArray<number, readonly [2, 2]>>();
expectTypeOf(floatArray).toEqualTypeOf<[[number, number], [number, number]]>();

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const intTensor: Tensor<CreateOp<TensorStorage<Int32, readonly [3]>>>;
declare const intArray: Awaited<ReturnType<typeof intTensor.toArray>>;
expectTypeOf(intArray).toEqualTypeOf<NestedArray<number, readonly [3]>>();
expectTypeOf(intArray).toEqualTypeOf<[number, number, number]>();

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const bigintTensor: Tensor<CreateOp<TensorStorage<Int64, readonly [3]>>>;
declare const bigintArray: Awaited<ReturnType<typeof bigintTensor.toArray>>;
expectTypeOf(bigintArray).toEqualTypeOf<NestedArray<bigint, readonly [3]>>();
expectTypeOf(bigintArray).toEqualTypeOf<[bigint, bigint, bigint]>();

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const boolTensor: Tensor<CreateOp<TensorStorage<Bool, readonly [3]>>>;
declare const boolArray: Awaited<ReturnType<typeof boolTensor.toArray>>;
expectTypeOf(boolArray).toEqualTypeOf<NestedArray<boolean, readonly [3]>>();
expectTypeOf(boolArray).toEqualTypeOf<[boolean, boolean, boolean]>();

// Test item() return types for scalars
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const scalarFloat: Tensor<CreateOp<TensorStorage<Float32, readonly []>>>;
declare const scalarFloatValue: Awaited<ReturnType<typeof scalarFloat.item>>;
expectTypeOf(scalarFloatValue).toEqualTypeOf<number>();

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const scalarInt: Tensor<CreateOp<TensorStorage<Int32, readonly []>>>;
declare const scalarIntValue: Awaited<ReturnType<typeof scalarInt.item>>;
expectTypeOf(scalarIntValue).toEqualTypeOf<number>();

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const scalarBigint: Tensor<CreateOp<TensorStorage<Int64, readonly []>>>;
declare const scalarBigintValue: Awaited<ReturnType<typeof scalarBigint.item>>;
expectTypeOf(scalarBigintValue).toEqualTypeOf<bigint>();

// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const scalarBool: Tensor<CreateOp<TensorStorage<Bool, readonly []>>>;
declare const scalarBoolValue: Awaited<ReturnType<typeof scalarBool.item>>;
expectTypeOf(scalarBoolValue).toEqualTypeOf<boolean>();

// =============================================================================
// DTypeValue Extraction Tests
// =============================================================================

expectTypeOf<DTypeValue<Float32>>().toEqualTypeOf<number>();
expectTypeOf<DTypeValue<Int32>>().toEqualTypeOf<number>();
expectTypeOf<DTypeValue<Int64>>().toEqualTypeOf<bigint>();
expectTypeOf<DTypeValue<Bool>>().toEqualTypeOf<boolean>();

// =============================================================================
// Device and Lifecycle Operation Type Tests
// =============================================================================

declare const deviceTensor: Tensor<CreateOp<TensorStorage<Int32, readonly [2, 2]>>>;

// Test to() preserves tensor type
declare const gpuTensor: Awaited<ReturnType<typeof deviceTensor.to>>;
expectTypeOf(gpuTensor).toEqualTypeOf<typeof deviceTensor>();
expectTypeOf(gpuTensor.shape).toEqualTypeOf<readonly [2, 2]>();
expectTypeOf(gpuTensor.dtype).toEqualTypeOf<Int32>();

// Test clone() preserves tensor type
declare const clonedTensor: Awaited<ReturnType<typeof deviceTensor.clone>>;
expectTypeOf(clonedTensor).toEqualTypeOf<typeof deviceTensor>();
expectTypeOf(clonedTensor.shape).toEqualTypeOf<readonly [2, 2]>();
expectTypeOf(clonedTensor.dtype).toEqualTypeOf<Int32>();

// Test toString() returns string
expectTypeOf(deviceTensor.toString()).toEqualTypeOf<string>();

// Test dispose() returns void
// eslint-disable-next-line @typescript-eslint/no-confusing-void-expression
expectTypeOf(deviceTensor.dispose()).toEqualTypeOf<void>();
