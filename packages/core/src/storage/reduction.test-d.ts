/**
 * Type-level tests for reduction operations
 */

import { expectTypeOf } from 'expect-type';
import type { SumOp, MeanOp, MaxOp, MinOp, ProdOp } from './reduction';
import type { TensorStorage, DefaultLayoutFlags, ComputeStrides } from './layout';
import type { Float32, Int32, ToFloat } from '../dtype/types';
import type { Product } from '../shape/types';

// Test tensor storage types
type TestTensorStorage2D = TensorStorage<Float32, readonly [2, 3], ComputeStrides<readonly [2, 3]>, DefaultLayoutFlags>;
type TestTensorStorage3D = TensorStorage<Int32, readonly [2, 3, 4], ComputeStrides<readonly [2, 3, 4]>, DefaultLayoutFlags>;

// =============================================================================
// SumOp Tests
// =============================================================================

// Sum along specific axis
{
  type SumAxis1 = SumOp<TestTensorStorage2D, readonly [1], false>;
  expectTypeOf<SumAxis1['__output']['__shape']>().toEqualTypeOf<readonly [2]>();
  expectTypeOf<SumAxis1['__output']['__dtype']>().toEqualTypeOf<Float32>();
  expectTypeOf<SumAxis1['__op']>().toEqualTypeOf<'sum'>();
}

// Global sum (all axes)
{
  type GlobalSum = SumOp<TestTensorStorage2D, undefined, false>;
  expectTypeOf<GlobalSum['__output']['__shape']>().toEqualTypeOf<readonly []>();
  expectTypeOf<GlobalSum['__output']['__dtype']>().toEqualTypeOf<Float32>();
}

// Sum with keepdims
{
  type SumKeepDims = SumOp<TestTensorStorage2D, readonly [1], true>;
  expectTypeOf<SumKeepDims['__output']['__shape']>().toEqualTypeOf<readonly [2, 1]>();
}

// =============================================================================
// MeanOp Tests
// =============================================================================

// Mean preserves floating point type
{
  type MeanAxis1 = MeanOp<TestTensorStorage2D, readonly [1], false>;
  expectTypeOf<MeanAxis1['__output']['__shape']>().toEqualTypeOf<readonly [2]>();
  expectTypeOf<MeanAxis1['__output']['__dtype']>().toEqualTypeOf<Float32>();
  expectTypeOf<MeanAxis1['__op']>().toEqualTypeOf<'mean'>();
}

// Mean converts integer to float
{
  type MeanIntToFloat = MeanOp<TestTensorStorage3D, readonly [2], false>;
  expectTypeOf<MeanIntToFloat['__output']['__shape']>().toEqualTypeOf<readonly [2, 3]>();
  expectTypeOf<MeanIntToFloat['__output']['__dtype']>().toEqualTypeOf<ToFloat<Int32>>();
}

// =============================================================================
// MaxOp Tests
// =============================================================================

// Max preserves dtype
{
  type MaxAxis0 = MaxOp<TestTensorStorage3D, readonly [0], false>;
  expectTypeOf<MaxAxis0['__output']['__shape']>().toEqualTypeOf<readonly [3, 4]>();
  expectTypeOf<MaxAxis0['__output']['__dtype']>().toEqualTypeOf<Int32>();
  expectTypeOf<MaxAxis0['__op']>().toEqualTypeOf<'max'>();
}

// Max with keepdims
{
  type MaxKeepDims = MaxOp<TestTensorStorage3D, readonly [1, 2], true>;
  expectTypeOf<MaxKeepDims['__output']['__shape']>().toEqualTypeOf<readonly [2, 1, 1]>();
}

// =============================================================================
// MinOp Tests
// =============================================================================

// Min preserves dtype
{
  type MinAxis2 = MinOp<TestTensorStorage3D, readonly [2], false>;
  expectTypeOf<MinAxis2['__output']['__shape']>().toEqualTypeOf<readonly [2, 3]>();
  expectTypeOf<MinAxis2['__output']['__dtype']>().toEqualTypeOf<Int32>();
  expectTypeOf<MinAxis2['__op']>().toEqualTypeOf<'min'>();
}

// Global min
{
  type GlobalMin = MinOp<TestTensorStorage2D, undefined, false>;
  expectTypeOf<GlobalMin['__output']['__shape']>().toEqualTypeOf<readonly []>();
  expectTypeOf<GlobalMin['__output']['__dtype']>().toEqualTypeOf<Float32>();
}

// =============================================================================
// ProdOp Tests (New Implementation)
// =============================================================================

// Product along specific axis
{
  type ProdAxis1 = ProdOp<TestTensorStorage2D, readonly [1], false>;
  expectTypeOf<ProdAxis1['__output']['__shape']>().toEqualTypeOf<readonly [2]>();
  expectTypeOf<ProdAxis1['__output']['__dtype']>().toEqualTypeOf<Float32>();
  expectTypeOf<ProdAxis1['__op']>().toEqualTypeOf<'prod'>();
  expectTypeOf<ProdAxis1['__prodAxes']>().toEqualTypeOf<readonly [1]>();
  expectTypeOf<ProdAxis1['__keepDims']>().toEqualTypeOf<false>();
}

// Global product (all axes)
{
  type GlobalProd = ProdOp<TestTensorStorage2D, undefined, false>;
  expectTypeOf<GlobalProd['__output']['__shape']>().toEqualTypeOf<readonly []>();
  expectTypeOf<GlobalProd['__output']['__dtype']>().toEqualTypeOf<Float32>();
  expectTypeOf<GlobalProd['__prodAxes']>().toEqualTypeOf<undefined>();
  expectTypeOf<GlobalProd['__keepDims']>().toEqualTypeOf<false>();
}

// Product with keepdims
{
  type ProdKeepDims = ProdOp<TestTensorStorage2D, readonly [1], true>;
  expectTypeOf<ProdKeepDims['__output']['__shape']>().toEqualTypeOf<readonly [2, 1]>();
  expectTypeOf<ProdKeepDims['__keepDims']>().toEqualTypeOf<true>();
}

// Product preserves integer dtype (unlike mean)
{
  type ProdIntPreserved = ProdOp<TestTensorStorage3D, readonly [2], false>;
  expectTypeOf<ProdIntPreserved['__output']['__shape']>().toEqualTypeOf<readonly [2, 3]>();
  expectTypeOf<ProdIntPreserved['__output']['__dtype']>().toEqualTypeOf<Int32>();
}

// Product along multiple axes
{
  type ProdMultiAxis = ProdOp<TestTensorStorage3D, readonly [0, 2], false>;
  expectTypeOf<ProdMultiAxis['__output']['__shape']>().toEqualTypeOf<readonly [3]>();
  expectTypeOf<ProdMultiAxis['__output']['__dtype']>().toEqualTypeOf<Int32>();
  expectTypeOf<ProdMultiAxis['__prodAxes']>().toEqualTypeOf<readonly [0, 2]>();
}

// Product with all axes removed (scalar result)
{
  type ProdAllAxes = ProdOp<TestTensorStorage3D, readonly [0, 1, 2], false>;
  expectTypeOf<ProdAllAxes['__output']['__shape']>().toEqualTypeOf<readonly []>();
  expectTypeOf<ProdAllAxes['__output']['__size']>().toEqualTypeOf<Product<readonly []>>();
}

// Product with all axes kept as 1s
{
  type ProdAllKeep = ProdOp<TestTensorStorage3D, readonly [0, 1, 2], true>;
  expectTypeOf<ProdAllKeep['__output']['__shape']>().toEqualTypeOf<readonly [1, 1, 1]>();
  expectTypeOf<ProdAllKeep['__output']['__size']>().toEqualTypeOf<Product<readonly [1, 1, 1]>>();
  expectTypeOf<ProdAllKeep['__keepDims']>().toEqualTypeOf<true>();
}

// =============================================================================
// Reduction Operation Layout Tests
// =============================================================================

// All reduction operations should produce contiguous output
{
  type SumLayout = SumOp<TestTensorStorage2D, readonly [1], false>;
  expectTypeOf<SumLayout['__output']['__layout']['c_contiguous']>().toEqualTypeOf<true | TestTensorStorage2D['__layout']['c_contiguous']>();
  expectTypeOf<SumLayout['__output']['__layout']['is_view']>().toEqualTypeOf<false>();
  expectTypeOf<SumLayout['__output']['__layout']['writeable']>().toEqualTypeOf<true>();
  
  type ProdLayout = ProdOp<TestTensorStorage2D, readonly [1], false>;
  expectTypeOf<ProdLayout['__output']['__layout']['c_contiguous']>().toEqualTypeOf<true | TestTensorStorage2D['__layout']['c_contiguous']>();
  expectTypeOf<ProdLayout['__output']['__layout']['is_view']>().toEqualTypeOf<false>();
  expectTypeOf<ProdLayout['__output']['__layout']['writeable']>().toEqualTypeOf<true>();
}

// =============================================================================
// Error Cases (should resolve to never)
// =============================================================================

// Invalid axes should result in never type
{
  type InvalidProd = ProdOp<TestTensorStorage2D, readonly [5], false>;
  expectTypeOf<InvalidProd>().toEqualTypeOf<never>();
  
  type InvalidSum = SumOp<TestTensorStorage2D, readonly [-5], false>;
  expectTypeOf<InvalidSum>().toEqualTypeOf<never>();
}

// =============================================================================
// Metadata Preservation Tests
// =============================================================================

// All reduction ops should preserve input metadata correctly
{
  type ProdMeta = ProdOp<TestTensorStorage3D, readonly [1], false>;
  expectTypeOf<ProdMeta['__inputs']>().toEqualTypeOf<readonly [TestTensorStorage3D]>();
  expectTypeOf<ProdMeta['__inputs']['length']>().toEqualTypeOf<1>();
  
  type SumMeta = SumOp<TestTensorStorage3D, readonly [0], true>;  
  expectTypeOf<SumMeta['__inputs']>().toEqualTypeOf<readonly [TestTensorStorage3D]>();
  expectTypeOf<SumMeta['__sumAxes']>().toEqualTypeOf<readonly [0]>();
  expectTypeOf<SumMeta['__keepDims']>().toEqualTypeOf<true>();
}