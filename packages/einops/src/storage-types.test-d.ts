/**
 * Type-level tests for einops rearrange operations
 */

import type { TensorStorage, LayoutFlags, DTypeOf, ShapeOf, OutputOf, LayoutOf } from '@typetensor/core';
import type { RearrangeOp } from './storage-types';
import type { Float32, Int32, Float64 } from '@typetensor/core';
import { expectTypeOf } from 'expect-type';

// =============================================================================
// Test Helpers
// =============================================================================

// Create test tensors with different shapes and dtypes
type Float32Vector = TensorStorage<Float32, readonly [6]>;
type Float32Matrix2D = TensorStorage<Float32, readonly [2, 3]>;
type Float32Tensor3D = TensorStorage<Float32, readonly [2, 3, 4]>;
type Float32Tensor4D = TensorStorage<Float32, readonly [2, 3, 4, 5]>;

type Int32Matrix = TensorStorage<Int32, readonly [3, 4]>;
type Float64Tensor = TensorStorage<Float64, readonly [2, 3, 4]>;

// Custom layout types for testing
interface FortranLayout extends LayoutFlags {
  readonly c_contiguous: false;
  readonly f_contiguous: true;
  readonly is_view: false;
  readonly writeable: true;
  readonly aligned: true;
}

type FortranTensor = TensorStorage<Float32, readonly [3, 4], readonly [1, 3], FortranLayout>;

// =============================================================================
// Simple Transpose Tests
// =============================================================================

// Test 1: Basic 2D transpose
{
  type Transposed = RearrangeOp<Float32Matrix2D, 'h w -> w h'>;
  type Output = OutputOf<Transposed>;

  expectTypeOf<'rearrange'>().toEqualTypeOf<Transposed['__op']>();
  expectTypeOf<readonly [3, 2]>().toEqualTypeOf<ShapeOf<Output>>(); // [2,3] -> [3,2]
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Output>>();
}

// Test 2: 3D permutation patterns
{
  // Transpose last two dimensions
  type Transpose3D = RearrangeOp<Float32Tensor3D, 'batch height width -> batch width height'>;
  expectTypeOf<readonly [2, 4, 3]>().toEqualTypeOf<ShapeOf<OutputOf<Transpose3D>>>();

  // Move batch to end
  type MoveBatch = RearrangeOp<Float32Tensor3D, 'batch height width -> height width batch'>;
  expectTypeOf<readonly [3, 4, 2]>().toEqualTypeOf<ShapeOf<OutputOf<MoveBatch>>>();

  // Full permutation
  type FullPerm = RearrangeOp<Float32Tensor3D, 'a b c -> c a b'>;
  expectTypeOf<readonly [4, 2, 3]>().toEqualTypeOf<ShapeOf<OutputOf<FullPerm>>>();
}

// Test 3: Identity patterns
{
  type Identity2D = RearrangeOp<Float32Matrix2D, 'h w -> h w'>;
  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<OutputOf<Identity2D>>>();

  type Identity3D = RearrangeOp<Float32Tensor3D, 'a b c -> a b c'>;
  expectTypeOf<readonly [2, 3, 4]>().toEqualTypeOf<ShapeOf<OutputOf<Identity3D>>>();
}

// =============================================================================
// Composite Pattern Tests
// =============================================================================

// Test 4: Merge patterns (simple to composite)
{
  type Merge2D = RearrangeOp<Float32Matrix2D, 'h w -> (h w)'>;
  expectTypeOf<readonly [6]>().toEqualTypeOf<ShapeOf<OutputOf<Merge2D>>>(); // [2,3] -> [6]

  type MergeLast = RearrangeOp<Float32Tensor3D, 'batch h w -> batch (h w)'>;
  expectTypeOf<readonly [2, 12]>().toEqualTypeOf<ShapeOf<OutputOf<MergeLast>>>(); // [2,3,4] -> [2,12]

  type MergeFirst = RearrangeOp<Float32Tensor3D, 'a b c -> (a b) c'>;
  expectTypeOf<readonly [6, 4]>().toEqualTypeOf<ShapeOf<OutputOf<MergeFirst>>>(); // [2,3,4] -> [6,4]
}

// Test 5: Split patterns (composite to simple) with axes
{
  type Split1D = RearrangeOp<Float32Vector, '(h w) -> h w', { h: 2 }>;
  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<OutputOf<Split1D>>>(); // [6] -> [2,3]

  type Split2D = RearrangeOp<Float32Matrix2D, '(h w) c -> h w c', { h: 2 }>;
  expectTypeOf<readonly [2, 1, 3]>().toEqualTypeOf<ShapeOf<OutputOf<Split2D>>>(); // [2,3] -> [2,1,3]

  // Split composite dimension  
  type Tensor2D_12 = TensorStorage<Float32, readonly [2, 12]>;
  type FlattenLast = RearrangeOp<Tensor2D_12, 'batch (h w) -> batch h w', { h: 3 }>;
  expectTypeOf<readonly [2, 3, 4]>().toEqualTypeOf<ShapeOf<OutputOf<FlattenLast>>>(); // [2,12] -> [2,3,4]
}

// =============================================================================
// Ellipsis Pattern Tests
// =============================================================================

// Test 6: Basic ellipsis patterns
{
  // Move last dimension to front
  type MoveLast = RearrangeOp<Float32Tensor3D, '... c -> c ...'>;
  expectTypeOf<readonly [4, 2, 3]>().toEqualTypeOf<ShapeOf<OutputOf<MoveLast>>>(); // [2,3,4] -> [4,2,3]

  // Move first dimension to end
  type MoveFirst = RearrangeOp<Float32Tensor3D, 'a ... -> ... a'>;
  expectTypeOf<readonly [3, 4, 2]>().toEqualTypeOf<ShapeOf<OutputOf<MoveFirst>>>(); // [2,3,4] -> [3,4,2]
}

// Test 7: Ellipsis with named axes
{
  type Pattern1 = RearrangeOp<Float32Tensor4D, 'batch ... height width -> height width batch ...'>;
  expectTypeOf<readonly [4, 5, 2, 3]>().toEqualTypeOf<ShapeOf<OutputOf<Pattern1>>>(); // [2,3,4,5] -> [4,5,2,3]

  type Pattern2 = RearrangeOp<Float32Tensor3D, 'a ... z -> z a ...'>;
  expectTypeOf<readonly [4, 2, 3]>().toEqualTypeOf<ShapeOf<OutputOf<Pattern2>>>(); // [2,3,4] -> [4,2,3]
}

// =============================================================================
// Singleton Pattern Tests
// =============================================================================

// Test 8: Adding singleton dimensions
{
  type AddSingleton1 = RearrangeOp<Float32Matrix2D, 'h w -> h 1 w'>;
  expectTypeOf<readonly [2, 1, 3]>().toEqualTypeOf<ShapeOf<OutputOf<AddSingleton1>>>(); // [2,3] -> [2,1,3]

  type AddSingleton2 = RearrangeOp<Float32Matrix2D, 'h w -> 1 h w 1'>;
  expectTypeOf<readonly [1, 2, 3, 1]>().toEqualTypeOf<ShapeOf<OutputOf<AddSingleton2>>>(); // [2,3] -> [1,2,3,1]
}

// Test 9: Removing singleton dimensions
{
  type Squeezable = TensorStorage<Float32, readonly [2, 1, 3, 1]>;
  type RemoveSingleton = RearrangeOp<Squeezable, 'a 1 b 1 -> a b'>;
  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<OutputOf<RemoveSingleton>>>(); // [2,1,3,1] -> [2,3]
}

// =============================================================================
// Type Safety and Layout Tests
// =============================================================================

// Test 10: Preserve dtype through rearrange
{
  type RearrangeInt32 = RearrangeOp<Int32Matrix, 'h w -> w h'>;
  expectTypeOf<Int32>().toEqualTypeOf<DTypeOf<OutputOf<RearrangeInt32>>>();
  expectTypeOf<readonly [4, 3]>().toEqualTypeOf<ShapeOf<OutputOf<RearrangeInt32>>>(); // [3,4] -> [4,3]

  type RearrangeFloat64 = RearrangeOp<Float64Tensor, 'a b c -> c b a'>;
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<OutputOf<RearrangeFloat64>>>();
  expectTypeOf<readonly [4, 3, 2]>().toEqualTypeOf<ShapeOf<OutputOf<RearrangeFloat64>>>(); // [2,3,4] -> [4,3,2]
}

// Test 11: Layout properties
{
  type Rearranged = RearrangeOp<Float32Matrix2D, 'h w -> w h'>;
  type Output = OutputOf<Rearranged>;

  // Rearrange creates views
  expectTypeOf<true>().toEqualTypeOf<LayoutOf<Output>['is_view']>();

  // Simple transpose breaks contiguity
  expectTypeOf<false>().toEqualTypeOf<LayoutOf<Output>['c_contiguous']>();
  expectTypeOf<false>().toEqualTypeOf<LayoutOf<Output>['f_contiguous']>();
}

// Test 12: Writeable property is preserved
{
  interface ReadOnlyLayout extends LayoutFlags {
    readonly c_contiguous: true;
    readonly f_contiguous: false;
    readonly is_view: false;
    readonly writeable: false;
    readonly aligned: true;
  }

  type ReadOnlyTensor = TensorStorage<Float32, readonly [3, 4], readonly [4, 1], ReadOnlyLayout>;
  type RearrangedReadOnly = RearrangeOp<ReadOnlyTensor, 'h w -> w h'>;

  expectTypeOf<false>().toEqualTypeOf<LayoutOf<OutputOf<RearrangedReadOnly>>['writeable']>();
  expectTypeOf<readonly [4, 3]>().toEqualTypeOf<ShapeOf<OutputOf<RearrangedReadOnly>>>();
}

// Test 13: Fortran layout tensor
{
  type RearrangedF = RearrangeOp<FortranTensor, 'h w -> w h'>;
  type Output = OutputOf<RearrangedF>;

  expectTypeOf<readonly [4, 3]>().toEqualTypeOf<ShapeOf<Output>>(); // [3,4] -> [4,3]
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Output>>();
  expectTypeOf<true>().toEqualTypeOf<LayoutOf<Output>['is_view']>();
  expectTypeOf<false>().toEqualTypeOf<LayoutOf<Output>['c_contiguous']>();
  expectTypeOf<false>().toEqualTypeOf<LayoutOf<Output>['f_contiguous']>();
}

// =============================================================================
// Complex Real-World Patterns
// =============================================================================

// Test 14: Vision transformer patterns
{
  // Image to patches
  type Image = TensorStorage<Float32, readonly [3, 224, 224]>; // CHW
  type Patches = RearrangeOp<Image, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', { p1: 16; p2: 16 }>;
  expectTypeOf<readonly [196, 768]>().toEqualTypeOf<ShapeOf<OutputOf<Patches>>>();

  // Multi-head attention reshape
  type Attention = TensorStorage<Float32, readonly [8, 64, 512]>; // batch, seq, hidden
  type MultiHead = RearrangeOp<
    Attention,
    'batch seq (heads dim) -> batch heads seq dim',
    { heads: 8 }
  >;
  expectTypeOf<readonly [8, 8, 64, 64]>().toEqualTypeOf<ShapeOf<OutputOf<MultiHead>>>();
}

// Test 15: Channel first/last conversion
{
  // NCHW to NHWC
  type NCHW = TensorStorage<Float32, readonly [32, 3, 224, 224]>;
  type NHWC = RearrangeOp<NCHW, 'batch channel height width -> batch height width channel'>;
  expectTypeOf<readonly [32, 224, 224, 3]>().toEqualTypeOf<ShapeOf<OutputOf<NHWC>>>();

  // NHWC to NCHW
  type NHWC2 = TensorStorage<Float32, readonly [32, 224, 224, 3]>;
  type NCHW2 = RearrangeOp<NHWC2, 'batch height width channel -> batch channel height width'>;
  expectTypeOf<readonly [32, 3, 224, 224]>().toEqualTypeOf<ShapeOf<OutputOf<NCHW2>>>();
}

// =============================================================================
// Error Cases (should resolve to never)
// =============================================================================

// Test 16: Invalid patterns
{
  // Unknown axis in output
  type InvalidAxis = RearrangeOp<Float32Matrix2D, 'h w -> h w c'>;
  expectTypeOf<InvalidAxis>().toBeNever();

  // Duplicate axes in input
  type DuplicateInput = RearrangeOp<Float32Matrix2D, 'h h -> h'>;
  expectTypeOf<DuplicateInput>().toBeNever();

  // Multiple ellipsis
  type MultiEllipsis = RearrangeOp<Float32Tensor3D, '... a ... -> a'>;
  expectTypeOf<MultiEllipsis>().toBeNever();
}

// Test 17: Shape mismatch in composite patterns
{
  // Wrong decomposition - 6 != 2*4
  type InvalidDecomp = RearrangeOp<Float32Vector, '(h w) -> h w', { h: 2; w: 4 }>;
  expectTypeOf<InvalidDecomp>().toBeNever();

  // Missing required axis
  type MissingAxis = RearrangeOp<Float32Matrix2D, '(h w) c -> h w c'>; // Missing h dimension
  expectTypeOf<MissingAxis>().toBeNever();
}
