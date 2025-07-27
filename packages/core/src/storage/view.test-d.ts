/**
 * Type-level tests for view operations
 */

import type {
  TensorStorage,
  LayoutFlags,
  DTypeOf,
  ShapeOf,
  StridesOf,
  OutputOf,
  LayoutOf,
} from './layout';
import type { ReshapeOp, Flatten, View } from './view';
import type { Float32, Int32, Float64 } from '../dtype/types';
import { expectTypeOf } from 'expect-type';

// =============================================================================
// Test Helpers
// =============================================================================

// Create test tensors with different shapes and dtypes
type Float32Scalar = TensorStorage<Float32, readonly []>;
type Float32Matrix2D = TensorStorage<Float32, readonly [3, 4]>;
type Float32Tensor3D = TensorStorage<Float32, readonly [2, 3, 4]>;
type Float32Tensor4D = TensorStorage<Float32, readonly [2, 3, 4, 5]>;

type Int32Matrix = TensorStorage<Int32, readonly [3, 4]>;
type Float64Tensor = TensorStorage<Float64, readonly [2, 3, 4]>;

// Custom layout types
interface NonContiguousLayout extends LayoutFlags {
  readonly c_contiguous: false;
  readonly f_contiguous: false;
  readonly is_view: true;
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

interface ReadOnlyLayout extends LayoutFlags {
  readonly c_contiguous: true;
  readonly f_contiguous: false;
  readonly is_view: false;
  readonly writeable: false;
  readonly aligned: true;
}

// Non-contiguous tensor (e.g., from a slice)
type NonContiguousTensor = TensorStorage<
  Float32,
  readonly [3, 4],
  readonly [8, 2],
  NonContiguousLayout
>;

// Fortran-order tensor
type FortranTensor = TensorStorage<Float32, readonly [3, 4], readonly [1, 3], FortranLayout>;

// Read-only tensor
type ReadOnlyTensor = TensorStorage<Float32, readonly [3, 4], readonly [4, 1], ReadOnlyLayout>;

// =============================================================================
// Reshape Operation Tests
// =============================================================================

// Test 1: Basic reshape preserves total elements
{
  type Reshaped1 = ReshapeOp<Float32Matrix2D, readonly [6, 2]>;
  type Reshaped2 = ReshapeOp<Float32Matrix2D, readonly [4, 3]>;
  type Reshaped3 = ReshapeOp<Float32Matrix2D, readonly [12]>;
  type Reshaped4 = ReshapeOp<Float32Matrix2D, readonly [2, 2, 3]>;

  expectTypeOf<readonly [6, 2]>().toEqualTypeOf<ShapeOf<OutputOf<Reshaped1>>>();
  expectTypeOf<readonly [4, 3]>().toEqualTypeOf<ShapeOf<OutputOf<Reshaped2>>>();
  expectTypeOf<readonly [12]>().toEqualTypeOf<ShapeOf<OutputOf<Reshaped3>>>();
  expectTypeOf<readonly [2, 2, 3]>().toEqualTypeOf<ShapeOf<OutputOf<Reshaped4>>>();
}

// Test 2: Reshape preserves dtype
{
  type ReshapedFloat32 = ReshapeOp<Float32Matrix2D, readonly [6, 2]>;
  type ReshapedInt32 = ReshapeOp<Int32Matrix, readonly [6, 2]>;
  type ReshapedFloat64 = ReshapeOp<Float64Tensor, readonly [6, 4]>;

  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<ReshapedFloat32>>>();
  expectTypeOf<Int32>().toEqualTypeOf<DTypeOf<OutputOf<ReshapedInt32>>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<OutputOf<ReshapedFloat64>>>();
}

// Test 3: Reshape with C-contiguous input produces C-contiguous strides
{
  type Reshaped = ReshapeOp<Float32Matrix2D, readonly [6, 2]>;
  type Output = OutputOf<Reshaped>;

  // C-contiguous strides for [6, 2] shape
  expectTypeOf<readonly [2, 1]>().toEqualTypeOf<StridesOf<Output>>();

  // Layout should preserve C-contiguous
  type OutputLayout = LayoutOf<Output>;
  expectTypeOf<true>().toEqualTypeOf<OutputLayout['c_contiguous']>();
  expectTypeOf<false>().toEqualTypeOf<OutputLayout['f_contiguous']>();
  expectTypeOf<true>().toEqualTypeOf<OutputLayout['is_view']>(); // Reshape creates a view
}

// Test 4: Reshape with F-contiguous input produces F-contiguous strides
{
  type Reshaped = ReshapeOp<FortranTensor, readonly [6, 2]>;
  type Output = OutputOf<Reshaped>;

  // F-contiguous strides for [6, 2] shape
  expectTypeOf<readonly [1, 6]>().toEqualTypeOf<StridesOf<Output>>();

  // Layout should preserve F-contiguous
  type OutputLayout = LayoutOf<Output>;
  expectTypeOf<false>().toEqualTypeOf<OutputLayout['c_contiguous']>();
  expectTypeOf<true>().toEqualTypeOf<OutputLayout['f_contiguous']>();
  expectTypeOf<true>().toEqualTypeOf<OutputLayout['is_view']>();
}

// Test 5: Non-contiguous tensor cannot be reshaped
{
  // This resolves to never (with error metadata)
  type InvalidReshape = ReshapeOp<NonContiguousTensor, readonly [6, 2]>;
  expectTypeOf<InvalidReshape>().toBeNever();

  // The error type should contain helpful message
  type ErrorType = ReshapeOp<NonContiguousTensor, readonly [6, 2]>;
  expectTypeOf<ErrorType>().toBeNever();
}

// Test 6: Invalid reshape (wrong total elements)
{
  // These resolve to never - Total elements don't match: 12 != 10
  type InvalidReshape1 = ReshapeOp<Float32Matrix2D, readonly [5, 2]>;
  expectTypeOf<InvalidReshape1>().toBeNever();

  // Total elements don't match: 12 != 15
  type InvalidReshape2 = ReshapeOp<Float32Matrix2D, readonly [3, 5]>;
  expectTypeOf<InvalidReshape2>().toBeNever();
}

// Test 7: Reshape preserves writeable property
{
  type ReshapedWriteable = ReshapeOp<Float32Matrix2D, readonly [6, 2]>;
  type ReshapedReadOnly = ReshapeOp<ReadOnlyTensor, readonly [6, 2]>;

  type WriteableLayout = LayoutOf<OutputOf<ReshapedWriteable>>;
  type ReadOnlyLayout = LayoutOf<OutputOf<ReshapedReadOnly>>;

  expectTypeOf<true>().toEqualTypeOf<WriteableLayout['writeable']>();
  expectTypeOf<false>().toEqualTypeOf<ReadOnlyLayout['writeable']>();
}

// Test 8: Reshape operation metadata
{
  type ReshapeOpResult = ReshapeOp<Float32Matrix2D, readonly [6, 2]>;
  expectTypeOf<'reshape'>().toEqualTypeOf<ReshapeOpResult['__op']>();
  expectTypeOf<readonly [Float32Matrix2D]>().toEqualTypeOf<ReshapeOpResult['__inputs']>();
}

// =============================================================================
// Flatten Operation Tests
// =============================================================================

// Test 1: Flatten creates 1D tensor
{
  type Flattened1 = Flatten<Float32Matrix2D>;
  type Flattened2 = Flatten<Float32Tensor3D>;
  type Flattened3 = Flatten<Float32Tensor4D>;
  type FlattenedScalar = Flatten<Float32Scalar>;

  expectTypeOf<readonly [12]>().toEqualTypeOf<ShapeOf<OutputOf<Flattened1>>>();
  expectTypeOf<readonly [24]>().toEqualTypeOf<ShapeOf<OutputOf<Flattened2>>>();
  expectTypeOf<readonly [120]>().toEqualTypeOf<ShapeOf<OutputOf<Flattened3>>>();
  expectTypeOf<readonly [1]>().toEqualTypeOf<ShapeOf<OutputOf<FlattenedScalar>>>();
}

// Test 2: Flatten preserves dtype and other properties
{
  type FlattenedInt = Flatten<Int32Matrix>;
  type Output = OutputOf<FlattenedInt>;

  expectTypeOf<Int32>().toEqualTypeOf<DTypeOf<Output>>();
  expectTypeOf<readonly [1]>().toEqualTypeOf<StridesOf<Output>>();
  expectTypeOf<true>().toEqualTypeOf<LayoutOf<Output>['is_view']>();
}

// Test 3: Flatten on non-contiguous fails
{
  // This resolves to never - Cannot flatten non-contiguous tensor
  type InvalidFlatten = Flatten<NonContiguousTensor>;
  expectTypeOf<InvalidFlatten>().toBeNever();
}

// =============================================================================
// View Operation Tests (with -1 inference)
// =============================================================================

// Test 1: View with no -1 (same as Reshape)
{
  type ViewResult = View<Float32Matrix2D, readonly [6, 2]>;
  type ReshapeResult = ReshapeOp<Float32Matrix2D, readonly [6, 2]>;

  // Should be equivalent
  expectTypeOf<ShapeOf<OutputOf<ViewResult>>>().toEqualTypeOf<ShapeOf<OutputOf<ReshapeResult>>>();
}

// Test 2: View with -1 in last dimension
{
  type ViewResult1 = View<Float32Matrix2D, readonly [3, -1]>; // Should infer [3, 4]
  type ViewResult2 = View<Float32Matrix2D, readonly [6, -1]>; // Should infer [6, 2]
  type ViewResult3 = View<Float32Matrix2D, readonly [2, -1]>; // Should infer [2, 6]

  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<OutputOf<ViewResult1>>>();
  expectTypeOf<readonly [6, 2]>().toEqualTypeOf<ShapeOf<OutputOf<ViewResult2>>>();
  expectTypeOf<readonly [2, 6]>().toEqualTypeOf<ShapeOf<OutputOf<ViewResult3>>>();
}

// Test 3: View with -1 in first dimension
{
  type ViewResult1 = View<Float32Matrix2D, readonly [-1, 4]>; // Should infer [3, 4]
  type ViewResult2 = View<Float32Matrix2D, readonly [-1, 2]>; // Should infer [6, 2]
  type ViewResult3 = View<Float32Matrix2D, readonly [-1, 6]>; // Should infer [2, 6]

  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<OutputOf<ViewResult1>>>();
  expectTypeOf<readonly [6, 2]>().toEqualTypeOf<ShapeOf<OutputOf<ViewResult2>>>();
  expectTypeOf<readonly [2, 6]>().toEqualTypeOf<ShapeOf<OutputOf<ViewResult3>>>();
}

// Test 4: View with -1 in middle dimension (3D tensor)
{
  type Tensor3D = TensorStorage<Float32, readonly [2, 3, 4]>; // 24 elements
  type ViewResult1 = View<Tensor3D, readonly [2, -1, 2]>; // Should infer [2, 6, 2]
  type ViewResult2 = View<Tensor3D, readonly [4, -1, 2]>; // Should infer [4, 3, 2]

  expectTypeOf<readonly [2, 6, 2]>().toEqualTypeOf<ShapeOf<OutputOf<ViewResult1>>>();
  expectTypeOf<readonly [4, 3, 2]>().toEqualTypeOf<ShapeOf<OutputOf<ViewResult2>>>();

  // Test invalid middle dimension inference
  type InvalidMid = View<Tensor3D, readonly [5, -1, 1]>; // 24 / (5*1) = 4.8 (invalid)
  expectTypeOf<InvalidMid>().toExtend<never>();
}

// Test 5: Invalid view with multiple -1s
{
  // These resolve to never - Only one dimension can be -1
  type InvalidView = View<Float32Matrix2D, readonly [-1, -1]>;
  expectTypeOf<InvalidView>().toBeNever();

  // Multiple -1s not allowed
  type InvalidView2 = View<Float32Tensor3D, readonly [-1, 2, -1]>;
  expectTypeOf<InvalidView2>().toBeNever();
}

// Test 6: View with incompatible dimensions
{
  // Actually, these should work! 12/3 = 4 and 12/5 = 2.4 (invalid)
  type ValidView1 = View<Float32Matrix2D, readonly [3, -1]>; // Should be [3, 4]
  type ValidView2 = View<Float32Matrix2D, readonly [-1, 4]>; // Should be [3, 4]

  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<OutputOf<ValidView1>>>();
  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<OutputOf<ValidView2>>>();

  // This one should actually fail - 5 doesn't divide 12 evenly
  // This resolves to never - 12 is not divisible by 5
  type InvalidView = View<Float32Matrix2D, readonly [-1, 5]>;
  expectTypeOf<InvalidView>().toExtend<never>();
}

// Test 7: More divisibility edge cases
{
  // Test various invalid divisions
  type Tensor20 = TensorStorage<Float32, readonly [20]>;

  // 20 / 3 = 6.666... (invalid)
  type InvalidDiv1 = View<Tensor20, readonly [-1, 3]>;
  expectTypeOf<InvalidDiv1>().toExtend<never>();

  // 20 / 7 = 2.857... (invalid)
  type InvalidDiv2 = View<Tensor20, readonly [7, -1]>;
  expectTypeOf<InvalidDiv2>().toExtend<never>();

  // Valid divisions
  type ValidDiv1 = View<Tensor20, readonly [-1, 4]>; // 20/4 = 5
  type ValidDiv2 = View<Tensor20, readonly [5, -1]>; // 20/5 = 4
  type ValidDiv3 = View<Tensor20, readonly [-1, 10]>; // 20/10 = 2

  expectTypeOf<readonly [5, 4]>().toEqualTypeOf<ShapeOf<OutputOf<ValidDiv1>>>();
  expectTypeOf<readonly [5, 4]>().toEqualTypeOf<ShapeOf<OutputOf<ValidDiv2>>>();
  expectTypeOf<readonly [2, 10]>().toEqualTypeOf<ShapeOf<OutputOf<ValidDiv3>>>();
}

// =============================================================================
// Chained Operation Tests
// =============================================================================

// Test 1: Reshape -> Flatten
{
  type Step1 = ReshapeOp<Float32Tensor3D, readonly [6, 4]>;
  type Step2 = Flatten<OutputOf<Step1>>;
  type FinalOutput = OutputOf<Step2>;

  expectTypeOf<readonly [24]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<FinalOutput>>();
}

// Test 2: View -> Reshape -> View
{
  type Step1 = View<Float32Tensor3D, readonly [4, -1]>; // [4, 6]
  type Step2 = ReshapeOp<OutputOf<Step1>, readonly [2, 12]>;
  type Step3 = View<OutputOf<Step2>, readonly [3, -1]>; // [3, 8]
  type FinalOutput = OutputOf<Step3>;

  expectTypeOf<readonly [3, 8]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
}

// Test 3: Flatten -> View (reshape 1D tensor)
{
  type Step1 = Flatten<Float32Matrix2D>; // [12]
  type Step2 = View<OutputOf<Step1>, readonly [3, -1]>; // [3, 4]
  type FinalOutput = OutputOf<Step2>;

  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
}

// =============================================================================
// Edge Case Tests
// =============================================================================

// Test 1: Scalar reshape
{
  type ReshapedScalar1 = ReshapeOp<Float32Scalar, readonly []>; // Scalar to scalar
  type ReshapedScalar2 = ReshapeOp<Float32Scalar, readonly [1]>; // Scalar to [1]

  expectTypeOf<readonly []>().toEqualTypeOf<ShapeOf<OutputOf<ReshapedScalar1>>>();
  expectTypeOf<readonly [1]>().toEqualTypeOf<ShapeOf<OutputOf<ReshapedScalar2>>>();

  // This resolves to never - Cannot reshape scalar to shape with >1 elements
  type InvalidScalarReshape = ReshapeOp<Float32Scalar, readonly [2]>;
  expectTypeOf<InvalidScalarReshape>().toBeNever();
}

// Test 2: 1D to various shapes
{
  type Vec12 = TensorStorage<Float32, readonly [12]>;
  type To2D = ReshapeOp<Vec12, readonly [3, 4]>;
  type To3D = ReshapeOp<Vec12, readonly [2, 2, 3]>;
  type To4D = ReshapeOp<Vec12, readonly [1, 2, 2, 3]>;

  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<OutputOf<To2D>>>();
  expectTypeOf<readonly [2, 2, 3]>().toEqualTypeOf<ShapeOf<OutputOf<To3D>>>();
  expectTypeOf<readonly [1, 2, 2, 3]>().toEqualTypeOf<ShapeOf<OutputOf<To4D>>>();
}

// Test 3: Large tensor reshape
{
  type LargeTensor = TensorStorage<Float32, readonly [10, 20, 30]>; // 6000 elements
  type Reshaped1 = ReshapeOp<LargeTensor, readonly [100, 60]>;
  type Reshaped2 = ReshapeOp<LargeTensor, readonly [6000]>;
  type Reshaped3 = View<LargeTensor, readonly [10, -1]>; // [10, 600]

  expectTypeOf<readonly [100, 60]>().toEqualTypeOf<ShapeOf<OutputOf<Reshaped1>>>();
  expectTypeOf<readonly [6000]>().toEqualTypeOf<ShapeOf<OutputOf<Reshaped2>>>();
  expectTypeOf<readonly [10, 600]>().toEqualTypeOf<ShapeOf<OutputOf<Reshaped3>>>();
}
