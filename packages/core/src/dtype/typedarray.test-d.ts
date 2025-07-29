/**
 * Type tests for dtype/typedarray.ts
 *
 * Tests for TypedArray integration types and type safety
 */

import { expectTypeOf } from 'expect-type';
import type {
  AnyDType,
  Int32,
  Float32,
  Float64,
  Int64,
  Uint32,
  Bool,
  JSTypeOf,
  ArrayConstructorOf,
} from './types';
import type { RuntimeDType } from './runtime';
import {
  type DTypedArray,
  createTypedArray,
  createTypedArrayFromBuffer,
  createTypedArrayFromData,
  type createReadonlyTypedArray,
  convertTypedArray,
  createTypedArrayView,
  type wrapTypedArray,
  sharesSameBuffer,
  hasMemoryOverlap,
  type calculateMemoryUsage,
  validateTypedArray,
  createAlignedBuffer,
  type TypedArrayError,
  type AlignmentError,
  type BoundsError,
} from './typedarray';

// =============================================================================
// DTypedArray Interface Tests
// =============================================================================

// DTypedArray property types
declare const float32Array: DTypedArray<Float32>;
expectTypeOf(float32Array).toHaveProperty('dtype').toEqualTypeOf<RuntimeDType<Float32>>();
expectTypeOf(float32Array)
  .toHaveProperty('array')
  .toEqualTypeOf<InstanceType<ArrayConstructorOf<Float32>>>();
expectTypeOf(float32Array).toHaveProperty('buffer').toEqualTypeOf<ArrayBuffer>();
expectTypeOf(float32Array).toHaveProperty('byteOffset').toEqualTypeOf<number>();
expectTypeOf(float32Array).toHaveProperty('byteLength').toEqualTypeOf<number>();
expectTypeOf(float32Array).toHaveProperty('length').toEqualTypeOf<number>();
expectTypeOf(float32Array).toHaveProperty('readonly').toEqualTypeOf<boolean>();

// DTypedArray method signatures
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare const int32Array: DTypedArray<Int32>;
expectTypeOf<typeof int32Array.get>().toEqualTypeOf<(index: number) => JSTypeOf<Int32>>();
expectTypeOf<typeof int32Array.set>().toEqualTypeOf<
  (index: number, value: JSTypeOf<Int32>) => void
>();

// Subarray operations
expectTypeOf<typeof int32Array.subarray>().toEqualTypeOf<
  (begin?: number, end?: number) => DTypedArray<Int32>
>();
expectTypeOf<typeof int32Array.slice>().toEqualTypeOf<
  (begin?: number, end?: number) => DTypedArray<Int32>
>();

// Copy operations
expectTypeOf<typeof int32Array.copyWithin>().toEqualTypeOf<
  (target: number, start: number, end?: number) => DTypedArray<Int32>
>();
expectTypeOf<typeof int32Array.fill>().toEqualTypeOf<
  (value: JSTypeOf<Int32>, start?: number, end?: number) => DTypedArray<Int32>
>();

// Conversion
expectTypeOf<typeof int32Array.toArray>().toEqualTypeOf<() => readonly JSTypeOf<Int32>[]>();

// Map method typing
type MapMethod = DTypedArray<Float32>['map'];
expectTypeOf<MapMethod>().toMatchTypeOf<
  <U extends AnyDType>(
    callback: (value: number, index: number, array: DTypedArray<Float32>) => JSTypeOf<U>,
    targetDType: RuntimeDType<U>,
  ) => DTypedArray<U>
>();

// CreateView method typing
type CreateViewMethod = DTypedArray<Int32>['createView'];
expectTypeOf<CreateViewMethod>().toMatchTypeOf<
  <U extends AnyDType>(
    targetDType: RuntimeDType<U>,
    byteOffset?: number,
    length?: number,
  ) => DTypedArray<U>
>();

// Different JS types
declare const boolArray: DTypedArray<Bool>;
declare const int64Array: DTypedArray<Int64>;
declare const float64Array: DTypedArray<Float64>;
expectTypeOf(boolArray.get(0)).toEqualTypeOf<boolean>();
expectTypeOf(int64Array.get(0)).toEqualTypeOf<bigint>();
expectTypeOf(float64Array.get(0)).toEqualTypeOf<number>();

// =============================================================================
// Factory Function Type Tests
// =============================================================================

// Factory function return types
declare const float32Dtype: RuntimeDType<Float32>;
declare const int64Dtype: RuntimeDType<Int64>;
expectTypeOf(createTypedArray(float32Dtype, 10)).toEqualTypeOf<DTypedArray<Float32>>();
expectTypeOf(createTypedArray(int64Dtype, 10)).toEqualTypeOf<DTypedArray<Int64>>();

// Generic type inference
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare function createArray<T extends AnyDType>(
  dtype: RuntimeDType<T>,
  length: number,
): DTypedArray<T>;
declare const genericFloat32Array: ReturnType<typeof createArray<Float32>>;
expectTypeOf(genericFloat32Array).toEqualTypeOf<DTypedArray<Float32>>();

// createTypedArrayFromBuffer
declare const buffer: ArrayBuffer;
declare const int32Dtype: RuntimeDType<Int32>;
expectTypeOf(createTypedArrayFromBuffer(int32Dtype, buffer)).toEqualTypeOf<DTypedArray<Int32>>();
expectTypeOf(createTypedArrayFromBuffer(int32Dtype, buffer, 0)).toEqualTypeOf<
  DTypedArray<Int32>
>();
expectTypeOf(createTypedArrayFromBuffer(int32Dtype, buffer, 0, 10)).toEqualTypeOf<
  DTypedArray<Int32>
>();

// createTypedArrayFromData
declare const boolDtype: RuntimeDType<Bool>;
expectTypeOf(createTypedArrayFromData(float32Dtype, [1, 2, 3])).toEqualTypeOf<
  DTypedArray<Float32>
>();
expectTypeOf(createTypedArrayFromData(int64Dtype, [1n, 2n, 3n])).toEqualTypeOf<
  DTypedArray<Int64>
>();
expectTypeOf(createTypedArrayFromData(boolDtype, [true, false])).toEqualTypeOf<
  DTypedArray<Bool>
>();

// createReadonlyTypedArray
declare const readonlyArray: ReturnType<typeof createReadonlyTypedArray<Float32>>;
expectTypeOf(readonlyArray).toEqualTypeOf<DTypedArray<Float32>>();

// =============================================================================
// Conversion Function Type Tests
// =============================================================================

// convertTypedArray
declare const sourceInt32Array: DTypedArray<Int32>;
expectTypeOf(convertTypedArray(sourceInt32Array, float32Dtype)).toEqualTypeOf<
  DTypedArray<Float32>
>();
expectTypeOf(convertTypedArray(sourceInt32Array, float32Dtype, {})).toEqualTypeOf<
  DTypedArray<Float32>
>();

// Generic conversion function
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare function convertArray<From extends AnyDType, To extends AnyDType>(
  source: DTypedArray<From>,
  targetDType: RuntimeDType<To>,
): DTypedArray<To>;
declare const conversionResult: ReturnType<typeof convertArray<Int32, Float64>>;
expectTypeOf(conversionResult).toEqualTypeOf<DTypedArray<Float64>>();

// createTypedArrayView
declare const uint32Dtype: RuntimeDType<Uint32>;
expectTypeOf(createTypedArrayView(float32Array, uint32Dtype)).toEqualTypeOf<
  DTypedArray<Uint32>
>();
expectTypeOf(createTypedArrayView(float32Array, uint32Dtype, 0)).toEqualTypeOf<
  DTypedArray<Uint32>
>();
expectTypeOf(createTypedArrayView(float32Array, uint32Dtype, 0, 5)).toEqualTypeOf<
  DTypedArray<Uint32>
>();

// =============================================================================
// Utility Function Type Tests
// =============================================================================

// wrapTypedArray
declare const wrappedArray: ReturnType<typeof wrapTypedArray>;
expectTypeOf(wrappedArray).toEqualTypeOf<DTypedArray<AnyDType> | null>();

// Buffer comparison functions
declare const array1: DTypedArray<Float32>;
declare const array2: DTypedArray<Int32>;
expectTypeOf(sharesSameBuffer(array1, array2)).toEqualTypeOf<boolean>();
expectTypeOf(hasMemoryOverlap(array1, array2)).toEqualTypeOf<boolean>();

// calculateMemoryUsage
declare const memoryUsage: ReturnType<typeof calculateMemoryUsage>;
expectTypeOf(memoryUsage).toEqualTypeOf<{
  dataBytes: number;
  metadataBytes: number;
  totalBytes: number;
  elementsPerMB: number;
}>();

// validateTypedArray
declare const arrayToValidate: DTypedArray<Int32>;
expectTypeOf(validateTypedArray(arrayToValidate)).toEqualTypeOf<boolean>();
expectTypeOf(validateTypedArray(arrayToValidate, {})).toEqualTypeOf<boolean>();
expectTypeOf(
  validateTypedArray(arrayToValidate, {
    minLength: 50,
    maxLength: 200,
    alignment: 4,
    dtype: int32Dtype,
  }),
).toEqualTypeOf<boolean>();

// =============================================================================
// Performance Utility Type Tests
// =============================================================================

// createAlignedBuffer
expectTypeOf(createAlignedBuffer(1024)).toEqualTypeOf<ArrayBuffer>();
expectTypeOf(createAlignedBuffer(1024, 64)).toEqualTypeOf<ArrayBuffer>();

// =============================================================================
// Error Class Type Tests
// =============================================================================

// TypedArrayError
declare const error1: TypedArrayError;
expectTypeOf(error1).toMatchTypeOf<Error>();
expectTypeOf(error1.message).toEqualTypeOf<string>();
expectTypeOf(error1.name).toEqualTypeOf<string>();
expectTypeOf(error1.dtype).toEqualTypeOf<RuntimeDType | undefined>();
expectTypeOf(error1.arrayLength).toEqualTypeOf<number | undefined>();

// AlignmentError
declare const alignmentError: AlignmentError;
expectTypeOf(alignmentError).toMatchTypeOf<TypedArrayError>();
expectTypeOf(alignmentError.requiredAlignment).toEqualTypeOf<number>();
expectTypeOf(alignmentError.actualOffset).toEqualTypeOf<number>();

// BoundsError
declare const boundsError: BoundsError;
expectTypeOf(boundsError).toMatchTypeOf<TypedArrayError>();
expectTypeOf(boundsError.index).toEqualTypeOf<number>();
expectTypeOf(boundsError.arrayLength).toEqualTypeOf<number>();

// =============================================================================
// Generic Type Constraint Tests
// =============================================================================

// Generic function maintaining type relationships
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare function processArray<T extends AnyDType>(
  array: DTypedArray<T>,
  dtype: RuntimeDType<T>,
): DTypedArray<T>;
declare const processResult: ReturnType<typeof processArray<Float32>>;
expectTypeOf(processResult).toEqualTypeOf<DTypedArray<Float32>>();

// Cross-dtype operations
// eslint-disable-next-line @typescript-eslint/no-unused-vars
declare function convertAndProcess<From extends AnyDType, To extends AnyDType>(
  source: DTypedArray<From>,
  fromDType: RuntimeDType<From>,
  toDType: RuntimeDType<To>,
): DTypedArray<To>;
declare const crossDtypeResult: ReturnType<typeof convertAndProcess<Int32, Float64>>;
expectTypeOf(crossDtypeResult).toEqualTypeOf<DTypedArray<Float64>>();

// forEach callback parameters
float64Array.forEach((value, index, arr) => {
  expectTypeOf(value).toEqualTypeOf<number>();
  expectTypeOf(index).toEqualTypeOf<number>();
  expectTypeOf(arr).toEqualTypeOf<DTypedArray<Float64>>();
});

// map callback and return type
declare const mapResult: ReturnType<typeof int32Array.map<Float32>>;
expectTypeOf(mapResult).toEqualTypeOf<DTypedArray<Float32>>();
