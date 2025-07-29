/**
 * Type-level tests for tensor storage layout and base types
 */

import type {
  TensorStorage,
  LayoutFlags,
  DTypeOf,
  ShapeOf,
  StridesOf,
  LayoutOf,
} from './layout';

import type { Float32, Int32, Float64 } from '../dtype/types';
import { expectTypeOf } from 'expect-type';

// =============================================================================
// Test Helpers
// =============================================================================

// Create test tensors with different shapes and dtypes
type Float32Scalar = TensorStorage<Float32, readonly []>;
type Float32Vector1D = TensorStorage<Float32, readonly [5]>;
type Float32Matrix2D = TensorStorage<Float32, readonly [3, 4]>;
type Float32Tensor3D = TensorStorage<Float32, readonly [2, 3, 4]>;
type Float32Tensor4D = TensorStorage<Float32, readonly [2, 3, 4, 5]>;

// Test tensors with different dtypes
type Int32Vector = TensorStorage<Int32, readonly [5]>;
type Float64Matrix = TensorStorage<Float64, readonly [3, 4]>;

// Test tensors with custom strides (non-contiguous)
interface NonContiguousLayout extends LayoutFlags {
  readonly c_contiguous: false;
  readonly f_contiguous: false;
  readonly is_view: false;
  readonly writeable: true;
  readonly aligned: true;
}

interface FortranLayout extends LayoutFlags {
  readonly c_contiguous: false;
  readonly f_contiguous: true;
  readonly is_view: false;
  readonly writeable: true;
  readonly aligned: true;
}

// Non-contiguous tensor (e.g., from a slice)
type NonContiguousTensor = TensorStorage<
  Float32,
  readonly [3, 4],
  readonly [8, 2], // Non C-contiguous strides
  NonContiguousLayout
>;

// Fortran-order tensor
type FortranTensor = TensorStorage<
  Float32,
  readonly [3, 4],
  readonly [1, 3], // Fortran strides
  FortranLayout
>;

// =============================================================================
// Base Storage Type Tests
// =============================================================================

// Test 1: DTypeOf extraction
{
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Float32Scalar>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Float32Vector1D>>();
  expectTypeOf<Int32>().toEqualTypeOf<DTypeOf<Int32Vector>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<Float64Matrix>>();
}

// Test 2: ShapeOf extraction
{
  expectTypeOf<readonly []>().toEqualTypeOf<ShapeOf<Float32Scalar>>();
  expectTypeOf<readonly [5]>().toEqualTypeOf<ShapeOf<Float32Vector1D>>();
  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<Float32Matrix2D>>();
  expectTypeOf<readonly [2, 3, 4]>().toEqualTypeOf<ShapeOf<Float32Tensor3D>>();
  expectTypeOf<readonly [2, 3, 4, 5]>().toEqualTypeOf<ShapeOf<Float32Tensor4D>>();
}

// Test 3: StridesOf extraction and computation
{
  // C-order stride computation
  expectTypeOf<readonly []>().toEqualTypeOf<StridesOf<Float32Scalar>>();
  expectTypeOf<readonly [1]>().toEqualTypeOf<StridesOf<Float32Vector1D>>();
  expectTypeOf<readonly [4, 1]>().toEqualTypeOf<StridesOf<Float32Matrix2D>>();
  expectTypeOf<readonly [12, 4, 1]>().toEqualTypeOf<StridesOf<Float32Tensor3D>>();
  expectTypeOf<readonly [60, 20, 5, 1]>().toEqualTypeOf<StridesOf<Float32Tensor4D>>();

  // Custom strides
  expectTypeOf<readonly [8, 2]>().toEqualTypeOf<StridesOf<NonContiguousTensor>>();
  expectTypeOf<readonly [1, 3]>().toEqualTypeOf<StridesOf<FortranTensor>>();
}

// Test 4: LayoutOf extraction
{
  type DefaultLayout = LayoutOf<Float32Matrix2D>;
  expectTypeOf<true>().toEqualTypeOf<DefaultLayout['c_contiguous']>();
  expectTypeOf<false>().toEqualTypeOf<DefaultLayout['f_contiguous']>();
  expectTypeOf<false>().toEqualTypeOf<DefaultLayout['is_view']>();
  expectTypeOf<true>().toEqualTypeOf<DefaultLayout['writeable']>();
  expectTypeOf<true>().toEqualTypeOf<DefaultLayout['aligned']>();

  type NonContigLayout = LayoutOf<NonContiguousTensor>;
  expectTypeOf<false>().toEqualTypeOf<NonContigLayout['c_contiguous']>();
  expectTypeOf<false>().toEqualTypeOf<NonContigLayout['f_contiguous']>();

  type FortranLayoutType = LayoutOf<FortranTensor>;
  expectTypeOf<false>().toEqualTypeOf<FortranLayoutType['c_contiguous']>();
  expectTypeOf<true>().toEqualTypeOf<FortranLayoutType['f_contiguous']>();
}

// Test 5: Storage size computation
{
  type ScalarSize = Float32Scalar['__size'];
  type VectorSize = Float32Vector1D['__size'];
  type MatrixSize = Float32Matrix2D['__size'];
  type Tensor3DSize = Float32Tensor3D['__size'];

  expectTypeOf<1>().toEqualTypeOf<ScalarSize>();
  expectTypeOf<5>().toEqualTypeOf<VectorSize>();
  expectTypeOf<12>().toEqualTypeOf<MatrixSize>(); // 3 * 4
  expectTypeOf<24>().toEqualTypeOf<Tensor3DSize>(); // 2 * 3 * 4
}