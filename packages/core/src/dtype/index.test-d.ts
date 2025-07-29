/**
 * Type tests for dtype/index.ts
 *
 * Tests for module exports and type definitions
 */

import { expectTypeOf } from 'expect-type';
import {
  // Type exports
  type DType,
  type AnyDType,
  type DTypeName,
  type Bool,
  type Int8,
  type Uint8,
  type Int16,
  type Uint16,
  type Int32,
  type Uint32,
  type Int64,
  type Uint64,
  type Float32,
  type Float64,
  type JSTypeOf,
  type Promote,
  type RuntimeDType,
  type ConversionResult,
  type ArrayConstructorOf,
  type DTypedArray,

  // Runtime exports
  CommonDTypes,
  DTypeConstants,
  createInferredArray,
  getDTypeSystemInfo,
  validateDTypeSystem,
  getDType,
  promoteTypes,
  convertValue,
  createTypedArray,
} from './index';

// =============================================================================
// Core Type Export Tests
// =============================================================================

// Test that core DType types are properly exported
expectTypeOf<Bool>().toMatchTypeOf<DType<'bool', boolean, Uint8ArrayConstructor, 1, false, true>>();
expectTypeOf<Int32>().toMatchTypeOf<DType<'int32', number, Int32ArrayConstructor, 4, true, true>>();
expectTypeOf<Float64>().toMatchTypeOf<
  DType<'float64', number, Float64ArrayConstructor, 8, true, false>
>();
expectTypeOf<Int64>().toMatchTypeOf<
  DType<'int64', bigint, BigInt64ArrayConstructor, 8, true, true>
>();

// Test utility type exports
expectTypeOf<JSTypeOf<Bool>>().toEqualTypeOf<boolean>();
expectTypeOf<JSTypeOf<Int32>>().toEqualTypeOf<number>();
expectTypeOf<JSTypeOf<Int64>>().toEqualTypeOf<bigint>();

expectTypeOf<Promote<Int8, Int16>>().toEqualTypeOf<Int16>();
expectTypeOf<Promote<Int32, Float32>>().toEqualTypeOf<Float64>();

// Test DTypeName union export
type AllDTypeNames = DTypeName;
expectTypeOf<AllDTypeNames>().toEqualTypeOf<
  | 'bool'
  | 'int8'
  | 'uint8'
  | 'int16'
  | 'uint16'
  | 'int32'
  | 'uint32'
  | 'int64'
  | 'uint64'
  | 'float32'
  | 'float64'
>();

// Test runtime type exports
expectTypeOf<RuntimeDType<Float32>>().toHaveProperty('name');
expectTypeOf<RuntimeDType<Float32>>().toHaveProperty('jsType');
expectTypeOf<RuntimeDType<Float32>>().toHaveProperty('byteSize');
expectTypeOf<RuntimeDType<Float32>>().toHaveProperty('signed');
expectTypeOf<RuntimeDType<Float32>>().toHaveProperty('isInteger');

// =============================================================================
// CommonDTypes Type Tests
// =============================================================================

// Test CommonDTypes property types
expectTypeOf(CommonDTypes.bool).toEqualTypeOf<RuntimeDType<Bool>>();
expectTypeOf(CommonDTypes.int8).toEqualTypeOf<RuntimeDType<Int8>>();
expectTypeOf(CommonDTypes.uint8).toEqualTypeOf<RuntimeDType<Uint8>>();
expectTypeOf(CommonDTypes.int16).toEqualTypeOf<RuntimeDType<Int16>>();
expectTypeOf(CommonDTypes.uint16).toEqualTypeOf<RuntimeDType<Uint16>>();
expectTypeOf(CommonDTypes.int32).toEqualTypeOf<RuntimeDType<Int32>>();
expectTypeOf(CommonDTypes.uint32).toEqualTypeOf<RuntimeDType<Uint32>>();
expectTypeOf(CommonDTypes.int64).toEqualTypeOf<RuntimeDType<Int64>>();
expectTypeOf(CommonDTypes.uint64).toEqualTypeOf<RuntimeDType<Uint64>>();
expectTypeOf(CommonDTypes.float32).toEqualTypeOf<RuntimeDType<Float32>>();
expectTypeOf(CommonDTypes.float64).toEqualTypeOf<RuntimeDType<Float64>>();

// Test CommonDTypes is readonly
expectTypeOf(CommonDTypes).toMatchTypeOf<
  Readonly<{
    bool: RuntimeDType<Bool>;
    int8: RuntimeDType<Int8>;
    uint8: RuntimeDType<Uint8>;
    int16: RuntimeDType<Int16>;
    uint16: RuntimeDType<Uint16>;
    int32: RuntimeDType<Int32>;
    uint32: RuntimeDType<Uint32>;
    int64: RuntimeDType<Int64>;
    uint64: RuntimeDType<Uint64>;
    float32: RuntimeDType<Float32>;
    float64: RuntimeDType<Float64>;
  }>
>();

// =============================================================================
// DTypeConstants Type Tests
// =============================================================================

// Test DTypeConstants phantom type properties
expectTypeOf(DTypeConstants.bool).toEqualTypeOf<Bool>();
expectTypeOf(DTypeConstants.int8).toEqualTypeOf<Int8>();
expectTypeOf(DTypeConstants.uint8).toEqualTypeOf<Uint8>();
expectTypeOf(DTypeConstants.int16).toEqualTypeOf<Int16>();
expectTypeOf(DTypeConstants.uint16).toEqualTypeOf<Uint16>();
expectTypeOf(DTypeConstants.int32).toEqualTypeOf<Int32>();
expectTypeOf(DTypeConstants.uint32).toEqualTypeOf<Uint32>();
expectTypeOf(DTypeConstants.int64).toEqualTypeOf<Int64>();
expectTypeOf(DTypeConstants.uint64).toEqualTypeOf<Uint64>();
expectTypeOf(DTypeConstants.float32).toEqualTypeOf<Float32>();
expectTypeOf(DTypeConstants.float64).toEqualTypeOf<Float64>();

// Test DTypeConstants is readonly
expectTypeOf(DTypeConstants).toMatchTypeOf<
  Readonly<{
    bool: Bool;
    int8: Int8;
    uint8: Uint8;
    int16: Int16;
    uint16: Uint16;
    int32: Int32;
    uint32: Uint32;
    int64: Int64;
    uint64: Uint64;
    float32: Float32;
    float64: Float64;
  }>
>();

// =============================================================================
// Function Return Type Tests
// =============================================================================

// Test createInferredArray type
expectTypeOf(createInferredArray).toEqualTypeOf<
  (data: readonly unknown[]) => DTypedArray<AnyDType>
>();

// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
const inferredResult = createInferredArray([1, 2, 3]);
expectTypeOf(inferredResult).toEqualTypeOf<DTypedArray<AnyDType>>();

// Test getDTypeSystemInfo return type
const systemInfo = getDTypeSystemInfo();
expectTypeOf(systemInfo).toEqualTypeOf<{
  availableDTypes: readonly string[];
  defaultDTypes: Record<string, string>;
  promotionRules: string;
  memoryLayout: Record<string, { byteSize: number; signed: boolean; isInteger: boolean }>;
}>();

expectTypeOf(systemInfo.availableDTypes).toEqualTypeOf<readonly string[]>();
expectTypeOf(systemInfo.defaultDTypes).toEqualTypeOf<Record<string, string>>();
expectTypeOf(systemInfo.promotionRules).toEqualTypeOf<string>();
expectTypeOf(systemInfo.memoryLayout).toEqualTypeOf<
  Record<string, { byteSize: number; signed: boolean; isInteger: boolean }>
>();

// Test validateDTypeSystem return type
const validationResult = validateDTypeSystem();
expectTypeOf(validationResult).toEqualTypeOf<{
  success: boolean;
  errors: string[];
}>();

expectTypeOf(validationResult.success).toEqualTypeOf<boolean>();
expectTypeOf(validationResult.errors).toEqualTypeOf<string[]>();

// =============================================================================
// Re-exported Function Type Tests
// =============================================================================

// Test getDType maintains correct types
expectTypeOf(getDType('float32')).toEqualTypeOf<RuntimeDType<Float32>>();
expectTypeOf(getDType('int64')).toEqualTypeOf<RuntimeDType<Int64>>();
expectTypeOf(getDType('bool')).toEqualTypeOf<RuntimeDType<Bool>>();

// Test promoteTypes maintains correct types
const int32Dtype = getDType('int32');
const float32Dtype = getDType('float32');
expectTypeOf(promoteTypes(int32Dtype, float32Dtype)).toEqualTypeOf<RuntimeDType<Float64>>();

// Test convertValue maintains correct types
expectTypeOf(convertValue(42, int32Dtype, float32Dtype)).toMatchTypeOf<ConversionResult<Float32>>();

// Test createTypedArray maintains correct types
const float64Dtype = getDType('float64');
expectTypeOf(createTypedArray(float64Dtype, 100)).toEqualTypeOf<DTypedArray<Float64>>();

// =============================================================================
// Generic Usage Type Tests
// =============================================================================

// Test generic function usage
function processArray<T extends AnyDType>(
  dtype: RuntimeDType<T>,
  data: JSTypeOf<T>[],
): DTypedArray<T> {
  return createTypedArray(dtype, data.length);
}

// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
const float32GenericResult = processArray(getDType('float32'), [1, 2, 3]);
expectTypeOf(float32GenericResult).toEqualTypeOf<DTypedArray<Float32>>();

// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
const int64GenericResult = processArray(getDType('int64'), [1n, 2n, 3n]);
expectTypeOf(int64GenericResult).toEqualTypeOf<DTypedArray<Int64>>();

// Test CommonDTypes in generic contexts
const commonFloat32Array = createTypedArray(CommonDTypes.float32, 10);
expectTypeOf(commonFloat32Array).toEqualTypeOf<DTypedArray<Float32>>();

// Test DTypeConstants for type-level operations
type PromotedType = Promote<typeof DTypeConstants.int32, typeof DTypeConstants.float32>;
expectTypeOf<PromotedType>().toEqualTypeOf<Float64>();

type JSType = JSTypeOf<typeof DTypeConstants.bool>;
expectTypeOf<JSType>().toEqualTypeOf<boolean>();

// =============================================================================
// Module Completeness Type Tests
// =============================================================================

// Test all necessary types are exported
// eslint-disable-next-line @typescript-eslint/no-explicit-any
expectTypeOf<DType<any, any, any, any, any, any>>().not.toBeNever();
expectTypeOf<AnyDType>().not.toBeNever();
expectTypeOf<DTypeName>().not.toBeNever();

// eslint-disable-next-line @typescript-eslint/no-unnecessary-type-arguments
expectTypeOf<RuntimeDType<AnyDType>>().toBeObject();
expectTypeOf<ConversionResult<AnyDType>>().toBeObject();
expectTypeOf<ArrayConstructorOf<AnyDType>>().toBeObject();

expectTypeOf<JSTypeOf<AnyDType>>().not.toBeNever();
expectTypeOf<Promote<AnyDType, AnyDType>>().not.toBeNever();

// Test complete type coverage for operations
const completeDtype = getDType('float32');
expectTypeOf(completeDtype).toEqualTypeOf<RuntimeDType<Float32>>();

const completeArray = createTypedArray(completeDtype, 10);
expectTypeOf(completeArray).toEqualTypeOf<DTypedArray<Float32>>();

const completePromoted = promoteTypes(completeDtype, getDType('int32'));
expectTypeOf(completePromoted).toEqualTypeOf<RuntimeDType<Float64>>();

const completeConversion = convertValue(42, getDType('int32'), completeDtype);
expectTypeOf(completeConversion).toMatchTypeOf<ConversionResult<Float32>>();
