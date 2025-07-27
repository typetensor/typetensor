/**
 * Type-level tests for matrix multiplication operations
 */

import type {
  TensorStorage,
  DTypeOf,
  ShapeOf,
  StridesOf,
  OutputOf,
  LayoutOf,
} from './layout';
import type { Matmul, MatmulOp } from './matmul';
import type { Float32, Int32, Float64 } from '../dtype/types';
import { expectTypeOf } from 'expect-type';

// =============================================================================
// Test Helpers
// =============================================================================

// Standard test tensors
type Float32Matrix23 = TensorStorage<Float32, readonly [2, 3]>;
type Float32Matrix34 = TensorStorage<Float32, readonly [3, 4]>;
type Float32Vector3 = TensorStorage<Float32, readonly [3]>;
type Float32Scalar = TensorStorage<Float32, readonly []>;

type Int32Matrix23 = TensorStorage<Int32, readonly [2, 3]>;
type Float64Matrix34 = TensorStorage<Float64, readonly [3, 4]>;

// Batch tensors
type BatchMatrix523 = TensorStorage<Float32, readonly [5, 2, 3]>;
type BatchMatrix534 = TensorStorage<Float32, readonly [5, 3, 4]>;

// =============================================================================
// Basic Matrix Multiplication Tests
// =============================================================================

// 2D × 2D matrix multiplication
{
  type Result = Matmul<Float32Matrix23, Float32Matrix34>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<ShapeOf<Output>>().toEqualTypeOf<readonly [2, 4]>();
  expectTypeOf<DTypeOf<Output>>().toEqualTypeOf<Float32>();
  expectTypeOf<LayoutOf<Output>['c_contiguous']>().toEqualTypeOf<true>();
  expectTypeOf<LayoutOf<Output>['is_view']>().toEqualTypeOf<false>();
}

// 1D × 1D (dot product) -> scalar
{
  type Result = Matmul<Float32Vector3, Float32Vector3>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<ShapeOf<Output>>().toEqualTypeOf<readonly []>();
  expectTypeOf<DTypeOf<Output>>().toEqualTypeOf<Float32>();
}

// 1D × 2D
{
  type Result = Matmul<Float32Vector3, Float32Matrix34>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<ShapeOf<Output>>().toEqualTypeOf<readonly [4]>();
  expectTypeOf<DTypeOf<Output>>().toEqualTypeOf<Float32>();
}

// 2D × 1D
{
  type Result = Matmul<Float32Matrix23, Float32Vector3>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<ShapeOf<Output>>().toEqualTypeOf<readonly [2]>();
  expectTypeOf<DTypeOf<Output>>().toEqualTypeOf<Float32>();
}

// =============================================================================
// Batch Matrix Multiplication Tests
// =============================================================================

// Batch matmul with same batch dimensions
{
  type Result = Matmul<BatchMatrix523, BatchMatrix534>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<ShapeOf<Output>>().toEqualTypeOf<readonly [5, 2, 4]>();
  expectTypeOf<DTypeOf<Output>>().toEqualTypeOf<Float32>();
}

// Higher dimensional batch
{
  type Batch4D_A = TensorStorage<Float32, readonly [10, 8, 2, 3]>;
  type Batch4D_B = TensorStorage<Float32, readonly [10, 8, 3, 4]>;
  type Result = Matmul<Batch4D_A, Batch4D_B>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<ShapeOf<Output>>().toEqualTypeOf<readonly [10, 8, 2, 4]>();
}

// =============================================================================
// Mixed Precision Tests
// =============================================================================

// Int32 × Float32 -> Float64 (large integers need float64 for precision)
{
  type Result = Matmul<Int32Matrix23, Float32Matrix34>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<DTypeOf<Output>>().toEqualTypeOf<Float64>();
  expectTypeOf<ShapeOf<Output>>().toEqualTypeOf<readonly [2, 4]>();
}

// Float32 × Float64 -> Float64
{
  type Result = Matmul<Float32Matrix23, Float64Matrix34>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<DTypeOf<Output>>().toEqualTypeOf<Float64>();
  expectTypeOf<ShapeOf<Output>>().toEqualTypeOf<readonly [2, 4]>();
}

// =============================================================================
// Error Cases (Should Result in Never)
// =============================================================================

// Incompatible inner dimensions
{
  type Matrix24 = TensorStorage<Float32, readonly [2, 4]>;
  type Matrix35 = TensorStorage<Float32, readonly [3, 5]>;
  type Result = MatmulOp<Matrix24, Matrix35>;
  
  expectTypeOf<Result>().toEqualTypeOf<never>();
}

// Scalar inputs (not allowed)
{
  type Result = MatmulOp<Float32Scalar, Float32Scalar>;
  expectTypeOf<Result>().toEqualTypeOf<never>();
}

// Mismatched batch dimensions
{
  type Batch523 = TensorStorage<Float32, readonly [5, 2, 3]>;
  type Batch1034 = TensorStorage<Float32, readonly [10, 3, 4]>;
  type Result = MatmulOp<Batch523, Batch1034>;
  
  expectTypeOf<Result>().toEqualTypeOf<never>();
}

// =============================================================================
// Edge Cases
// =============================================================================

// 1D × 3D (special broadcasting case)
{
  type Vector4 = TensorStorage<Float32, readonly [4]>;
  type Tensor3D = TensorStorage<Float32, readonly [3, 4, 5]>;
  type Result = Matmul<Vector4, Tensor3D>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<ShapeOf<Output>>().toEqualTypeOf<readonly [3, 5]>();
}

// Large shapes (transformer-like)
{
  type QKV = TensorStorage<Float32, readonly [32, 8, 128, 64]>; // [batch, heads, seq, dim]
  type QKV_T = TensorStorage<Float32, readonly [32, 8, 64, 128]>; // transposed
  type Result = Matmul<QKV, QKV_T>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<ShapeOf<Output>>().toEqualTypeOf<readonly [32, 8, 128, 128]>();
}

// =============================================================================
// Strided Output Tests
// =============================================================================

// Verify output has correct strides
{
  type Result = Matmul<Float32Matrix23, Float32Matrix34>;
  type Output = OutputOf<Result>;
  
  // For shape [2, 4], C-contiguous strides should be [4, 1]
  expectTypeOf<StridesOf<Output>>().toEqualTypeOf<readonly [4, 1]>();
}

// Batch output strides
{
  type Result = Matmul<BatchMatrix523, BatchMatrix534>;
  type Output = OutputOf<Result>;
  
  // For shape [5, 2, 4], C-contiguous strides should be [8, 4, 1]
  expectTypeOf<StridesOf<Output>>().toEqualTypeOf<readonly [8, 4, 1]>();
}

// =============================================================================
// User-Facing Matmul Type with Error Messages
// =============================================================================

// The Matmul type should provide good error messages
{
  // Valid case
  type ValidResult = Matmul<Float32Matrix23, Float32Matrix34>;
  expectTypeOf<ValidResult>().toMatchTypeOf<MatmulOp<Float32Matrix23, Float32Matrix34>>();
  
  // Invalid case should give ShapeError (not never)
  type Matrix24 = TensorStorage<Float32, readonly [2, 4]>;
  type Matrix35 = TensorStorage<Float32, readonly [3, 5]>;
  type InvalidResult = Matmul<Matrix24, Matrix35>;
  // This will be a ShapeError type with a helpful message
  expectTypeOf<InvalidResult>().not.toEqualTypeOf<never>();
}