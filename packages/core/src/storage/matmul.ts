/**
 * Matrix multiplication operations for tensor storage
 *
 * This module provides type-safe matrix multiplication with compile-time
 * shape validation and output shape inference.
 */

import type { CanMatmul, MatMulShape, IsMatMulCompatible } from '../shape/types';
import type { Promote } from '../dtype/types';
import type {
  TensorStorage,
  StorageTransformation,
  LayoutFlags,
  AnyTensorStorage,
  ComputeStrides,
} from './layout';

// =============================================================================
// Matrix Multiplication Operation
// =============================================================================

/**
 * Compute output layout for matrix multiplication
 * Matrix multiplication always produces contiguous output
 */
interface MatmulOpLayout extends LayoutFlags {
  readonly c_contiguous: true;
  readonly f_contiguous: false;
  readonly is_view: false;
  readonly writeable: true;
  readonly aligned: true;
}

/**
 * Matrix multiplication operation with shape validation
 *
 * This type performs compile-time validation that:
 * 1. Both inputs are at least 1D
 * 2. Inner dimensions match (A's last dim == B's second-to-last dim)
 * 3. Batch dimensions are compatible
 *
 * The output shape follows NumPy/PyTorch conventions:
 * - 1D × 1D → scalar
 * - 1D × 2D → 1D
 * - 2D × 1D → 1D
 * - 2D × 2D → 2D
 * - ND × ND → ND with broadcasted batch dimensions
 *
 * @template A - First tensor storage
 * @template B - Second tensor storage
 */
export type MatmulOp<A extends AnyTensorStorage, B extends AnyTensorStorage> =
  IsMatMulCompatible<A['__shape'], B['__shape']> extends true
    ? MatMulShape<A['__shape'], B['__shape']> extends never
      ? never // This shouldn't happen if IsMatMulCompatible is true, but safeguard
      : StorageTransformation<
          'matmul',
          TensorStorage<
            Promote<A['__dtype'], B['__dtype']>,
            MatMulShape<A['__shape'], B['__shape']>,
            ComputeStrides<MatMulShape<A['__shape'], B['__shape']>>,
            MatmulOpLayout
          >,
          readonly [A, B]
        >
    : never;

/**
 * User-facing matrix multiplication type with enhanced error messages
 *
 * This provides better developer experience by using CanMatmul which
 * gives detailed error messages for incompatible shapes.
 *
 * @template A - First tensor storage
 * @template B - Second tensor storage
 *
 * @example
 * type A = TensorStorage<Float32, [2, 3]>;
 * type B = TensorStorage<Float32, [3, 4]>;
 * type Result = Matmul<A, B>; // Output shape: [2, 4]
 *
 * @example
 * type Vector = TensorStorage<Float32, [5]>;
 * type DotProduct = Matmul<Vector, Vector>; // Output shape: [] (scalar)
 */
export type Matmul<A extends AnyTensorStorage, B extends AnyTensorStorage> =
  CanMatmul<A['__shape'], B['__shape']> extends true
    ? MatmulOp<A, B>
    : CanMatmul<A['__shape'], B['__shape']>; // This will be a ShapeError with details
