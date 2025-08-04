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
import type { ReshapeOp, Flatten, View, SliceOp, TransposeOp, PermuteOp } from '../storage/view';
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
const reshapeResult1 = await viewTensor.reshape([2, 3] as const);
expectTypeOf(reshapeResult1).toMatchTypeOf<
  Tensor<ReshapeOp<TensorStorage<Float32, readonly [6]>, readonly [2, 3]>>
>();
expectTypeOf(reshapeResult1.shape).toEqualTypeOf<readonly [2, 3]>();

const reshapeResult2 = await viewTensor.reshape([3, 2] as const);
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
const flatResult = await flattenTensor.flatten();
expectTypeOf(flatResult).toMatchTypeOf<Tensor<Flatten<TensorStorage<Float32, readonly [3, 2]>>>>();

// Test view with dimension inference
declare const viewInferenceTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 3]>>>;
const viewResult1 = await viewInferenceTensor.view([-1, 2] as const);
expectTypeOf(viewResult1).toMatchTypeOf<
  Tensor<View<TensorStorage<Float32, readonly [2, 3]>, readonly [-1, 2]>>
>();

const viewResult2 = await viewInferenceTensor.view([3, -1] as const);
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

// =============================================================================
// Slice Operation Type Tests
// =============================================================================

// Test 1: Basic 1D slicing
declare const slice1DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [10]>>>;

// Integer indexing (removes dimension)
declare const slice1DResult1: Awaited<ReturnType<typeof slice1DTensor.slice<readonly [5]>>>;
expectTypeOf(slice1DResult1).toMatchTypeOf<
  Tensor<SliceOp<TensorStorage<Float32, readonly [10]>, readonly [5]>>
>();
expectTypeOf(slice1DResult1.shape).toEqualTypeOf<readonly []>(); // Scalar result

// Slice with start and stop
declare const slice1DResult2: Awaited<
  ReturnType<typeof slice1DTensor.slice<readonly [{ start: 2; stop: 8 }]>>
>;
expectTypeOf(slice1DResult2).toMatchTypeOf<
  Tensor<SliceOp<TensorStorage<Float32, readonly [10]>, readonly [{ start: 2; stop: 8 }]>>
>();
expectTypeOf(slice1DResult2.shape).toEqualTypeOf<readonly [6]>();

// Test 2: Multi-dimensional slicing
declare const slice2DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [10, 20]>>>;

// Integer index on first dimension
declare const slice2DResult1: Awaited<ReturnType<typeof slice2DTensor.slice<readonly [5]>>>;
expectTypeOf(slice2DResult1).toMatchTypeOf<
  Tensor<SliceOp<TensorStorage<Float32, readonly [10, 20]>, readonly [5]>>
>();
expectTypeOf(slice2DResult1.shape).toEqualTypeOf<readonly [20]>();

// Mixed indexing: integer and slice
declare const slice2DResult2: Awaited<
  ReturnType<typeof slice2DTensor.slice<readonly [5, { start: 5; stop: 15 }]>>
>;
expectTypeOf(slice2DResult2).toMatchTypeOf<
  Tensor<SliceOp<TensorStorage<Float32, readonly [10, 20]>, readonly [5, { start: 5; stop: 15 }]>>
>();
expectTypeOf(slice2DResult2.shape).toEqualTypeOf<readonly [10]>(); // First dim removed, second sliced

// Both dimensions sliced
declare const slice2DResult3: Awaited<
  ReturnType<typeof slice2DTensor.slice<readonly [{ start: 0; stop: 5 }, { start: 5; stop: 15 }]>>
>;
expectTypeOf(slice2DResult3).toMatchTypeOf<
  Tensor<
    SliceOp<
      TensorStorage<Float32, readonly [10, 20]>,
      readonly [{ start: 0; stop: 5 }, { start: 5; stop: 15 }]
    >
  >
>();
expectTypeOf(slice2DResult3.shape).toEqualTypeOf<readonly [5, 10]>();

// Test 3: 3D tensor slicing
declare const slice3DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [8, 12, 16]>>>;

// Complex slice from view.test-d.ts test 15
declare const slice3DResult1: Awaited<
  ReturnType<
    typeof slice3DTensor.slice<readonly [{ step: 2 }, { start: 2; stop: 10 }, { step: 3 }]>
  >
>;
expectTypeOf(slice3DResult1).toMatchTypeOf<
  Tensor<
    SliceOp<
      TensorStorage<Float32, readonly [8, 12, 16]>,
      readonly [{ step: 2 }, { start: 2; stop: 10 }, { step: 3 }]
    >
  >
>();
expectTypeOf(slice3DResult1.shape).toEqualTypeOf<readonly [4, 8, 6]>();

// Mixed null, integer, and slice
declare const slice3DResult2: Awaited<
  ReturnType<typeof slice3DTensor.slice<readonly [null, 5, { start: 0; stop: 10; step: 2 }]>>
>;
expectTypeOf(slice3DResult2).toMatchTypeOf<
  Tensor<
    SliceOp<
      TensorStorage<Float32, readonly [8, 12, 16]>,
      readonly [null, 5, { start: 0; stop: 10; step: 2 }]
    >
  >
>();
expectTypeOf(slice3DResult2.shape).toEqualTypeOf<readonly [8, 5]>(); // Middle dim removed

// Test 4: Null slicing (keeps dimension)
declare const sliceNullResult: Awaited<
  ReturnType<typeof slice2DTensor.slice<readonly [null, null]>>
>;
expectTypeOf(sliceNullResult).toMatchTypeOf<
  Tensor<SliceOp<TensorStorage<Float32, readonly [10, 20]>, readonly [null, null]>>
>();
expectTypeOf(sliceNullResult.shape).toEqualTypeOf<readonly [10, 20]>(); // Shape unchanged

// Test 5: Partial indexing (fewer indices than dimensions)
declare const slicePartialResult: Awaited<ReturnType<typeof slice3DTensor.slice<readonly [5]>>>;
expectTypeOf(slicePartialResult).toMatchTypeOf<
  Tensor<SliceOp<TensorStorage<Float32, readonly [8, 12, 16]>, readonly [5]>>
>();
expectTypeOf(slicePartialResult.shape).toEqualTypeOf<readonly [12, 16]>(); // First dim removed

// Test 6: Negative step slicing
declare const sliceNegStepResult: Awaited<
  ReturnType<typeof slice1DTensor.slice<readonly [{ step: -1 }]>>
>;
expectTypeOf(sliceNegStepResult).toMatchTypeOf<
  Tensor<SliceOp<TensorStorage<Float32, readonly [10]>, readonly [{ step: -1 }]>>
>();
expectTypeOf(sliceNegStepResult.shape).toEqualTypeOf<readonly [10]>(); // Full reversal

// Test 7: Empty slice (start >= stop with positive step)
declare const sliceEmptyResult: Awaited<
  ReturnType<typeof slice1DTensor.slice<readonly [{ start: 5; stop: 5 }]>>
>;
expectTypeOf(sliceEmptyResult).toMatchTypeOf<
  Tensor<SliceOp<TensorStorage<Float32, readonly [10]>, readonly [{ start: 5; stop: 5 }]>>
>();
expectTypeOf(sliceEmptyResult.shape).toEqualTypeOf<readonly [0]>(); // Empty

// Test 8: Dtype preservation
declare const sliceInt32Tensor: Tensor<CreateOp<TensorStorage<Int32, readonly [10, 20]>>>;
declare const sliceInt32Result: Awaited<
  ReturnType<typeof sliceInt32Tensor.slice<readonly [{ start: 2; stop: 8 }, null]>>
>;
expectTypeOf(sliceInt32Result).toMatchTypeOf<
  Tensor<SliceOp<TensorStorage<Int32, readonly [10, 20]>, readonly [{ start: 2; stop: 8 }, null]>>
>();
expectTypeOf(sliceInt32Result.dtype).toEqualTypeOf<Int32>();
expectTypeOf(sliceInt32Result.shape).toEqualTypeOf<readonly [6, 20]>();

// Test 9: Layout properties after slicing
declare const sliceLayoutTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [10, 20]>>>;
declare const sliceLayoutResult: Awaited<
  ReturnType<typeof sliceLayoutTensor.slice<readonly [{ start: 0; stop: 5 }, null]>>
>;
// Check that layout is preserved but contiguity becomes unknown
expectTypeOf(sliceLayoutResult.layout.is_view).toEqualTypeOf<true>();
expectTypeOf(sliceLayoutResult.layout.c_contiguous).toEqualTypeOf<true>();
expectTypeOf(sliceLayoutResult.layout.f_contiguous).toEqualTypeOf<false>();

// Test 10: Zero step should fail at runtime (type system allows it)
declare const sliceZeroStepTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [10]>>>;
declare const sliceZeroStepResult: Awaited<
  ReturnType<typeof sliceZeroStepTensor.slice<readonly [{ step: 0 }]>>
>;
// Type system allows this, but runtime will throw
expectTypeOf(sliceZeroStepResult).toMatchTypeOf<
  Tensor<SliceOp<TensorStorage<Float32, readonly [10]>, readonly [{ step: 0 }]>>
>();

// Test 11: Verify return type structure
declare const sliceReturnType: Awaited<
  ReturnType<typeof slice3DTensor.slice<readonly [5, { start: 2; stop: 8 }, null]>>
>;
// Verify the full type structure
type ExpectedSliceType = Tensor<
  SliceOp<TensorStorage<Float32, readonly [8, 12, 16]>, readonly [5, { start: 2; stop: 8 }, null]>
>;
expectTypeOf(sliceReturnType).toMatchTypeOf<ExpectedSliceType>();

// =============================================================================
// Transpose Operation Type Tests
// =============================================================================

// Test 1: 2D transpose
declare const transpose2DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [3, 4]>>>;
const transposeResult2D = await transpose2DTensor.transpose();
expectTypeOf(transposeResult2D).toMatchTypeOf<
  Tensor<TransposeOp<TensorStorage<Float32, readonly [3, 4]>>>
>();
expectTypeOf(transposeResult2D.shape).toEqualTypeOf<readonly [4, 3]>();
expectTypeOf(transposeResult2D.dtype).toEqualTypeOf<Float32>();

// Test 2: 3D transpose (swaps last two dimensions)
declare const transpose3DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 3, 4]>>>;
const transposeResult3D = await transpose3DTensor.transpose();
expectTypeOf(transposeResult3D).toMatchTypeOf<
  Tensor<TransposeOp<TensorStorage<Float32, readonly [2, 3, 4]>>>
>();
expectTypeOf(transposeResult3D.shape).toEqualTypeOf<readonly [2, 4, 3]>();

// Test 3: 4D transpose
declare const transpose4DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [5, 6, 7, 8]>>>;
const transposeResult4D = await transpose4DTensor.transpose();
expectTypeOf(transposeResult4D.shape).toEqualTypeOf<readonly [5, 6, 8, 7]>();

// Test 4: 1D tensor (no change)
declare const transpose1DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [10]>>>;
const transposeResult1D = await transpose1DTensor.transpose();
expectTypeOf(transposeResult1D).toMatchTypeOf<
  Tensor<TransposeOp<TensorStorage<Float32, readonly [10]>>>
>();
expectTypeOf(transposeResult1D.shape).toEqualTypeOf<readonly [10]>();

// Test 5: Scalar (no change)
declare const transposeScalarTensor: Tensor<CreateOp<TensorStorage<Float32, readonly []>>>;
const transposeResultScalar = await transposeScalarTensor.transpose();
expectTypeOf(transposeResultScalar.shape).toEqualTypeOf<readonly []>();

// Test 6: T property
const transposeResultT = await transpose2DTensor.T;
expectTypeOf(transposeResultT).toMatchTypeOf<
  Tensor<TransposeOp<TensorStorage<Float32, readonly [3, 4]>>>
>();
expectTypeOf(transposeResultT.shape).toEqualTypeOf<readonly [4, 3]>();

// Test 7: Preserves dtype
declare const transposeIntTensor: Tensor<CreateOp<TensorStorage<Int32, readonly [5, 6]>>>;
const transposeIntResult = await transposeIntTensor.transpose();
expectTypeOf(transposeIntResult.dtype).toEqualTypeOf<Int32>();
expectTypeOf(transposeIntResult.shape).toEqualTypeOf<readonly [6, 5]>();

// Test 8: Different dtypes
declare const transposeBigintTensor: Tensor<CreateOp<TensorStorage<Int64, readonly [3, 4]>>>;
const transposeBigintResult = await transposeBigintTensor.transpose();
expectTypeOf(transposeBigintResult.dtype).toEqualTypeOf<Int64>();

declare const transposeBoolTensor: Tensor<CreateOp<TensorStorage<Bool, readonly [2, 3]>>>;
const transposeBoolResult = await transposeBoolTensor.transpose();
expectTypeOf(transposeBoolResult.dtype).toEqualTypeOf<Bool>();

// Test 9: Layout properties
expectTypeOf(transposeResult2D.layout.is_view).toEqualTypeOf<true>();
expectTypeOf(transposeResult2D.layout.c_contiguous).toEqualTypeOf<false>();
expectTypeOf(transposeResult2D.layout.f_contiguous).toEqualTypeOf<false>();

// =============================================================================
// Permute Operation Type Tests
// =============================================================================

// Test 1: 3D permutation
declare const permute3DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 3, 4]>>>;
const permuteResult1 = await permute3DTensor.permute([2, 0, 1] as const);
expectTypeOf(permuteResult1).toMatchTypeOf<
  Tensor<PermuteOp<TensorStorage<Float32, readonly [2, 3, 4]>, readonly [2, 0, 1]>>
>();
expectTypeOf(permuteResult1.shape).toEqualTypeOf<readonly [4, 2, 3]>();
expectTypeOf(permuteResult1.dtype).toEqualTypeOf<Float32>();

// Test 2: Identity permutation
const permuteIdentity = await permute3DTensor.permute([0, 1, 2] as const);
expectTypeOf(permuteIdentity).toMatchTypeOf<
  Tensor<PermuteOp<TensorStorage<Float32, readonly [2, 3, 4]>, readonly [0, 1, 2]>>
>();
expectTypeOf(permuteIdentity.shape).toEqualTypeOf<readonly [2, 3, 4]>();

// Test 3: 2D transpose via permute
declare const permute2DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [10, 20]>>>;
const permute2DResult = await permute2DTensor.permute([1, 0] as const);
expectTypeOf(permute2DResult).toMatchTypeOf<
  Tensor<PermuteOp<TensorStorage<Float32, readonly [10, 20]>, readonly [1, 0]>>
>();
expectTypeOf(permute2DResult.shape).toEqualTypeOf<readonly [20, 10]>();

// Test 4: NHWC to NCHW conversion
declare const nhwcTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [32, 224, 224, 3]>>>;
const nchwTensor = await nhwcTensor.permute([0, 3, 1, 2] as const);
expectTypeOf(nchwTensor).toMatchTypeOf<
  Tensor<PermuteOp<TensorStorage<Float32, readonly [32, 224, 224, 3]>, readonly [0, 3, 1, 2]>>
>();
expectTypeOf(nchwTensor.shape).toEqualTypeOf<readonly [32, 3, 224, 224]>();

// Test 5: 4D permutation variations
declare const permute4DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 3, 4, 5]>>>;

// Reverse all dimensions
const permuteReverse = await permute4DTensor.permute([3, 2, 1, 0] as const);
expectTypeOf(permuteReverse.shape).toEqualTypeOf<readonly [5, 4, 3, 2]>();

// Rotate dimensions
const permuteRotate = await permute4DTensor.permute([1, 2, 3, 0] as const);
expectTypeOf(permuteRotate.shape).toEqualTypeOf<readonly [3, 4, 5, 2]>();

// Test 6: Preserves dtype
declare const permuteInt64Tensor: Tensor<CreateOp<TensorStorage<Int64, readonly [5, 6, 7]>>>;
const permuteInt64Result = await permuteInt64Tensor.permute([2, 1, 0] as const);
expectTypeOf(permuteInt64Result.dtype).toEqualTypeOf<Int64>();
expectTypeOf(permuteInt64Result.shape).toEqualTypeOf<readonly [7, 6, 5]>();

// Test 7: Layout properties
expectTypeOf(permuteResult1.layout.is_view).toEqualTypeOf<true>();
expectTypeOf(permuteResult1.layout.c_contiguous).toEqualTypeOf<false>();
expectTypeOf(permuteResult1.layout.f_contiguous).toEqualTypeOf<false>();

// Test 8: 1D permutation
declare const permute1DTensor: Tensor<CreateOp<TensorStorage<Float32, readonly [10]>>>;
const permute1DResult = await permute1DTensor.permute([0] as const);
expectTypeOf(permute1DResult.shape).toEqualTypeOf<readonly [10]>();

// Test 9: Different permutation patterns
const permutePattern1 = await permute3DTensor.permute([1, 0, 2] as const);
expectTypeOf(permutePattern1.shape).toEqualTypeOf<readonly [3, 2, 4]>();

const permutePattern2 = await permute3DTensor.permute([1, 2, 0] as const);
expectTypeOf(permutePattern2.shape).toEqualTypeOf<readonly [3, 4, 2]>();

const permutePattern3 = await permute3DTensor.permute([2, 1, 0] as const);
expectTypeOf(permutePattern3.shape).toEqualTypeOf<readonly [4, 3, 2]>();

// Test 10: Type-level validation
// The following should fail at compile time if uncommented:
// @ts-expect-error - axes length must match rank
const invalidPerm1 = permute3DTensor.permute([0, 1] as const);

// @ts-expect-error - duplicate axes
const invalidPerm2 = permute3DTensor.permute([0, 0, 1] as const);

// @ts-expect-error - out of bounds axis
const invalidPerm3 = permute3DTensor.permute([0, 1, 3] as const);

// =============================================================================
// Product Reduction Operation Tests
// =============================================================================

import type { ProdOp } from '../storage/reduction';

// Test tensors for prod operations
declare const prodFloat32Tensor: Tensor<CreateOp<TensorStorage<Float32, readonly [2, 3, 4]>>>;
declare const prodInt32Tensor: Tensor<CreateOp<TensorStorage<Int32, readonly [5, 6]>>>;

// Test 1: Product along specific axis (axis=1, remove dimension)
declare const prodAxis1Result: Awaited<
  ReturnType<typeof prodFloat32Tensor.prod<readonly [1], false>>
>;
expectTypeOf(prodAxis1Result).toMatchTypeOf<
  Tensor<ProdOp<TensorStorage<Float32, readonly [2, 3, 4]>, readonly [1], false>>
>();
expectTypeOf(prodAxis1Result.shape).toEqualTypeOf<readonly [2, 4]>();
expectTypeOf(prodAxis1Result.dtype).toEqualTypeOf<Float32>(); // Dtype preserved

// Test 2: Global product (all elements to scalar)
declare const prodGlobalResult: Awaited<
  ReturnType<typeof prodFloat32Tensor.prod<undefined, false>>
>;
expectTypeOf(prodGlobalResult).toMatchTypeOf<
  Tensor<ProdOp<TensorStorage<Float32, readonly [2, 3, 4]>, undefined, false>>
>();
expectTypeOf(prodGlobalResult.shape).toEqualTypeOf<readonly []>(); // Scalar result
expectTypeOf(prodGlobalResult.dtype).toEqualTypeOf<Float32>();

// Test 3: Product with keepdims=true
declare const prodKeepDimsResult: Awaited<
  ReturnType<typeof prodFloat32Tensor.prod<readonly [2], true>>
>;
expectTypeOf(prodKeepDimsResult).toMatchTypeOf<
  Tensor<ProdOp<TensorStorage<Float32, readonly [2, 3, 4]>, readonly [2], true>>
>();
expectTypeOf(prodKeepDimsResult.shape).toEqualTypeOf<readonly [2, 3, 1]>(); // Keep as size 1
expectTypeOf(prodKeepDimsResult.dtype).toEqualTypeOf<Float32>();

// Test 4: Product along multiple axes
declare const prodMultiAxesResult: Awaited<
  ReturnType<typeof prodFloat32Tensor.prod<readonly [0, 2], false>>
>;
expectTypeOf(prodMultiAxesResult).toMatchTypeOf<
  Tensor<ProdOp<TensorStorage<Float32, readonly [2, 3, 4]>, readonly [0, 2], false>>
>();
expectTypeOf(prodMultiAxesResult.shape).toEqualTypeOf<readonly [3]>(); // Only middle dimension remains

// Test 5: Integer dtype preservation (unlike mean which converts to float)
declare const prodIntResult: Awaited<ReturnType<typeof prodInt32Tensor.prod<readonly [1], false>>>;
expectTypeOf(prodIntResult).toMatchTypeOf<
  Tensor<ProdOp<TensorStorage<Int32, readonly [5, 6]>, readonly [1], false>>
>();
expectTypeOf(prodIntResult.shape).toEqualTypeOf<readonly [5]>();
expectTypeOf(prodIntResult.dtype).toEqualTypeOf<Int32>(); // Integer dtype preserved

// Test 6: Product with negative axis indexing
const prodNegativeAxis = await prodFloat32Tensor.prod<readonly [-1], false>();
expectTypeOf(prodNegativeAxis).toMatchTypeOf<
  Tensor<ProdOp<TensorStorage<Float32, readonly [2, 3, 4]>, readonly [-1], false>>
>();
expectTypeOf(prodNegativeAxis.shape).toEqualTypeOf<readonly [2, 3]>(); // Last dim removed

// Test 7: Chainable promise prod operation
const chainableProdResult = await prodFloat32Tensor.prod(undefined, false);
expectTypeOf(chainableProdResult.shape).toEqualTypeOf<readonly []>();
expectTypeOf(chainableProdResult.dtype).toEqualTypeOf<Float32>();

// Test 8: All axes reduction with keepdims
declare const prodAllKeepResult: Awaited<
  ReturnType<typeof prodFloat32Tensor.prod<readonly [0, 1, 2], true>>
>;
expectTypeOf(prodAllKeepResult.shape).toEqualTypeOf<readonly [1, 1, 1]>(); // All dims kept as 1

// Test 9: Type validation should prevent invalid axes
// The following should fail at compile time if uncommented:
// @ts-expect-error - axis out of bounds
const invalidProdAxis = prodFloat32Tensor.prod([5] as const);

// @ts-expect-error - invalid axis type
const invalidProdType = prodFloat32Tensor.prod(['invalid'] as any);

// Test 10: Comparison with other reduction operations
declare const sumResult: Awaited<ReturnType<typeof prodFloat32Tensor.sum<readonly [1], false>>>;
declare const meanResult: Awaited<ReturnType<typeof prodFloat32Tensor.mean<readonly [1], false>>>;
declare const maxResult: Awaited<ReturnType<typeof prodFloat32Tensor.max<readonly [1], false>>>;
declare const minResult: Awaited<ReturnType<typeof prodFloat32Tensor.min<readonly [1], false>>>;

// All should have same shape after reduction
expectTypeOf(prodAxis1Result.shape).toEqualTypeOf<typeof sumResult.shape>();
expectTypeOf(prodAxis1Result.shape).toEqualTypeOf<typeof meanResult.shape>();
expectTypeOf(prodAxis1Result.shape).toEqualTypeOf<typeof maxResult.shape>();
expectTypeOf(prodAxis1Result.shape).toEqualTypeOf<typeof minResult.shape>();

// Prod should preserve dtype like sum/max/min, unlike mean which promotes to float
expectTypeOf(prodAxis1Result.dtype).toEqualTypeOf<typeof sumResult.dtype>();
expectTypeOf(prodAxis1Result.dtype).toEqualTypeOf<typeof maxResult.dtype>();
expectTypeOf(prodAxis1Result.dtype).toEqualTypeOf<typeof minResult.dtype>();
// Note: mean converts to float, so it would be different
