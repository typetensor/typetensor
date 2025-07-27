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
import type { ReshapeOp, Flatten, View, SliceOp } from './view';
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

// =============================================================================
// Slice Operation Tests
// =============================================================================

// Test 1: Basic slicing with SliceSpec
{
  type Tensor3D = TensorStorage<Float32, readonly [10, 20, 30]>;

  // Slice with SliceSpec on first dimension
  type Sliced1 = SliceOp<Tensor3D, readonly [{ start: 0; stop: 5 }, null, null]>;
  type Sliced1Output = OutputOf<Sliced1>;
  type Sliced1Shape = ShapeOf<Sliced1Output>;
  expectTypeOf<Sliced1Shape>().toEqualTypeOf<readonly [5, 20, 30]>();

  // Slice with SliceSpec on multiple dimensions
  type Sliced2 = SliceOp<Tensor3D, readonly [{ start: 0; stop: 5 }, { start: 5; stop: 15 }, null]>;
  type Sliced2Output = OutputOf<Sliced2>;
  type Sliced2Shape = ShapeOf<Sliced2Output>;
  expectTypeOf<Sliced2Shape>().toEqualTypeOf<readonly [5, 10, 30]>();
}

// Test 2: Integer indexing (removes dimensions)
{
  type Tensor3D = TensorStorage<Float32, readonly [10, 20, 30]>;

  // Index first dimension
  type Indexed1 = SliceOp<Tensor3D, readonly [5, null, null]>;
  type Indexed1Output = OutputOf<Indexed1>;
  type Indexed1Shape = ShapeOf<Indexed1Output>;
  expectTypeOf<Indexed1Shape>().toEqualTypeOf<readonly [20, 30]>();

  // Index multiple dimensions
  type Indexed2 = SliceOp<Tensor3D, readonly [5, 10, null]>;
  type Indexed2Output = OutputOf<Indexed2>;
  type Indexed2Shape = ShapeOf<Indexed2Output>;
  expectTypeOf<Indexed2Shape>().toEqualTypeOf<readonly [30]>();

  // Index all dimensions (scalar)
  type Indexed3 = SliceOp<Tensor3D, readonly [5, 10, 15]>;
  type Indexed3Output = OutputOf<Indexed3>;
  type Indexed3Shape = ShapeOf<Indexed3Output>;
  expectTypeOf<Indexed3Shape>().toEqualTypeOf<readonly []>();
}

// Test 3: Mixed slicing and indexing
{
  type Tensor4D = TensorStorage<Float32, readonly [5, 10, 15, 20]>;

  // Mix of slice, index, and null
  type Mixed1 = SliceOp<
    Tensor4D,
    readonly [{ start: 1; stop: 4 }, 5, null, { start: 0; stop: 10; step: 2 }]
  >;
  type Mixed1Output = OutputOf<Mixed1>;
  type Mixed1Shape = ShapeOf<Mixed1Output>;
  expectTypeOf<Mixed1Shape>().toEqualTypeOf<readonly [3, 15, 5]>();

  // Different combination
  type Mixed2 = SliceOp<Tensor4D, readonly [null, 0, { start: 5 }, 10]>;
  type Mixed2Output = OutputOf<Mixed2>;
  type Mixed2Shape = ShapeOf<Mixed2Output>;
  expectTypeOf<Mixed2Shape>().toEqualTypeOf<readonly [5, 10]>();
}

// Test 4: Partial indexing (fewer indices than dimensions)
{
  type Tensor3D = TensorStorage<Float32, readonly [10, 20, 30]>;

  // Only index first dimension
  type Partial1 = SliceOp<Tensor3D, readonly [5]>;
  type Partial1Output = OutputOf<Partial1>;
  type Partial1Shape = ShapeOf<Partial1Output>;
  expectTypeOf<Partial1Shape>().toEqualTypeOf<readonly [20, 30]>();

  // Slice first dimension, leave others unchanged
  type Partial2 = SliceOp<Tensor3D, readonly [{ start: 2; stop: 8 }, null]>;
  type Partial2Output = OutputOf<Partial2>;
  type Partial2Shape = ShapeOf<Partial2Output>;
  expectTypeOf<Partial2Shape>().toEqualTypeOf<readonly [6, 20, 30]>();
}

// Test 5: Stride computation
{
  type Tensor2D = TensorStorage<Float32, readonly [10, 20], readonly [20, 1]>;

  // Slicing preserves strides (but as number type)
  type Sliced1 = SliceOp<Tensor2D, readonly [{ start: 0; stop: 5 }, null]>;
  type Sliced1Output = OutputOf<Sliced1>;
  type Sliced1Strides = StridesOf<Sliced1Output>;
  expectTypeOf<Sliced1Strides>().toEqualTypeOf<readonly [number, number]>();

  // Integer indexing removes strides
  type Indexed1 = SliceOp<Tensor2D, readonly [5, null]>;
  type Indexed1Output = OutputOf<Indexed1>;
  type Indexed1Strides = StridesOf<Indexed1Output>;
  expectTypeOf<Indexed1Strides>().toEqualTypeOf<readonly [number]>();
}

// Test 6: Layout preservation
{
  type Tensor2D = TensorStorage<Float32, readonly [10, 20]>;
  type Sliced = SliceOp<Tensor2D, readonly [{ start: 0; stop: 5 }, null]>;
  type SlicedLayout = LayoutOf<OutputOf<Sliced>>;

  // Slicing creates a view
  expectTypeOf<true>().toEqualTypeOf<SlicedLayout['is_view']>();
  expectTypeOf<true>().toEqualTypeOf<SlicedLayout['c_contiguous']>();
}

// Test 7: Dtype preservation
{
  type Int32Tensor = TensorStorage<Int32, readonly [10, 20]>;
  type Float64Tensor = TensorStorage<Float64, readonly [10, 20]>;

  type SlicedInt32 = SliceOp<Int32Tensor, readonly [5, null]>;
  type SlicedFloat64 = SliceOp<Float64Tensor, readonly [null, 10]>;

  expectTypeOf<Int32>().toEqualTypeOf<DTypeOf<OutputOf<SlicedInt32>>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<OutputOf<SlicedFloat64>>>();
}

// Test 8: Edge cases
{
  // Scalar tensor
  type Scalar = TensorStorage<Float32, readonly []>;
  type SlicedScalar = SliceOp<Scalar, readonly []>; // No indices
  expectTypeOf<readonly []>().toEqualTypeOf<ShapeOf<OutputOf<SlicedScalar>>>();

  // 1D tensor
  type Vec = TensorStorage<Float32, readonly [10]>;
  type SlicedVec = SliceOp<Vec, readonly [{ start: 2; stop: 8 }]>;
  expectTypeOf<readonly [6]>().toEqualTypeOf<ShapeOf<OutputOf<SlicedVec>>>();
}

// =============================================================================
// Edge Case Tests for Slicing
// =============================================================================

// Test 9: Negative indices
{
  type Tensor2D = TensorStorage<Float32, readonly [10, 20]>;

  // Negative stop index: -2 means up to (but not including) the 2nd last element
  // For dim size 10: stop=-2 → stop=8, so [0:8] has size 8
  type NegStop = SliceOp<Tensor2D, readonly [{ stop: -2 }, null]>;
  expectTypeOf<ShapeOf<OutputOf<NegStop>>>().toEqualTypeOf<readonly [8, 20]>();

  // Negative start index: -5 means start from 5th element from end
  // For dim size 10: start=-5 → start=5, so [5:10] has size 5
  type NegStart = SliceOp<Tensor2D, readonly [{ start: -5 }, null]>;
  expectTypeOf<ShapeOf<OutputOf<NegStart>>>().toEqualTypeOf<readonly [5, 20]>();

  // Both negative: start=-8 → 2, stop=-2 → 8, so [2:8] has size 6
  type BothNeg = SliceOp<Tensor2D, readonly [{ start: -8; stop: -2 }, null]>;
  expectTypeOf<ShapeOf<OutputOf<BothNeg>>>().toEqualTypeOf<readonly [6, 20]>();

  // Extreme negative indices
  type ExtremeNeg = SliceOp<Tensor2D, readonly [{ start: -10 }, null]>; // -10 = 0 (at boundary)
  expectTypeOf<ShapeOf<OutputOf<ExtremeNeg>>>().toEqualTypeOf<readonly [10, 20]>();

  type BeyondNeg = SliceOp<Tensor2D, readonly [{ start: -15 }, null]>; // -15 clamps to 0
  expectTypeOf<ShapeOf<OutputOf<BeyondNeg>>>().toEqualTypeOf<readonly [10, 20]>();

  // Mixed positive and negative
  type MixedPosNeg = SliceOp<Tensor2D, readonly [{ start: -8; stop: 9 }, null]>; // [2:9] = size 7
  expectTypeOf<ShapeOf<OutputOf<MixedPosNeg>>>().toEqualTypeOf<readonly [7, 20]>();
}

// Test 10: Negative step
{
  type Tensor1D = TensorStorage<Float32, readonly [20]>;

  // Reverse entire dimension: step=-1 with no bounds reverses whole array
  // Default start=19 (last elem), stop=-1 (before first), step=-1
  type Reverse = SliceOp<Tensor1D, readonly [{ step: -1 }]>;
  expectTypeOf<ShapeOf<OutputOf<Reverse>>>().toEqualTypeOf<readonly [20]>();

  // Reverse with bounds: [15:5:-1] goes from index 15 down to (but not including) 5
  // Elements: 15,14,13,12,11,10,9,8,7,6 = 10 elements
  type ReversePartial = SliceOp<Tensor1D, readonly [{ start: 15; stop: 5; step: -1 }]>;
  expectTypeOf<ShapeOf<OutputOf<ReversePartial>>>().toEqualTypeOf<readonly [10]>();

  // Reverse with step > 1: [18:4:-2] takes every 2nd element going backwards
  // Elements: 18,16,14,12,10,8,6 = 7 elements
  type ReverseStep = SliceOp<Tensor1D, readonly [{ start: 18; stop: 4; step: -2 }]>;
  expectTypeOf<ShapeOf<OutputOf<ReverseStep>>>().toEqualTypeOf<readonly [7]>();
}

// Test 11: Out of bounds indices
{
  type Tensor2D = TensorStorage<Float32, readonly [10, 20]>;

  // Start beyond dimension size: clamps to dim size, results in empty slice
  // [15:10] with dim size 10 → [10:10] → size 0
  type StartOOB = SliceOp<Tensor2D, readonly [{ start: 15 }, null]>;
  expectTypeOf<ShapeOf<OutputOf<StartOOB>>>().toEqualTypeOf<readonly [0, 20]>();

  // Stop beyond dimension size: clamps stop to dim size
  // [5:25] with dim size 10 → [5:10] → size 5
  type StopOOB = SliceOp<Tensor2D, readonly [{ start: 5; stop: 25 }, null]>;
  expectTypeOf<ShapeOf<OutputOf<StopOOB>>>().toEqualTypeOf<readonly [5, 20]>();

  // Both beyond bounds: results in empty slice
  // [20:30] with dim size 10 → [10:10] → size 0
  type BothOOB = SliceOp<Tensor2D, readonly [{ start: 20; stop: 30 }, null]>;
  expectTypeOf<ShapeOf<OutputOf<BothOOB>>>().toEqualTypeOf<readonly [0, 20]>();

  // Additional test: Very large out-of-bounds values
  type VeryLargeOOB = SliceOp<Tensor2D, readonly [{ start: 1000; stop: 2000 }, null]>;
  expectTypeOf<ShapeOf<OutputOf<VeryLargeOOB>>>().toEqualTypeOf<readonly [0, 20]>();

  // Negative bounds that become valid after conversion
  // These tests will fail until we implement negative index support

  // Negative beyond bounds: -15 in dim 10 → clamps to 0
  type NegativeBeyondBounds = SliceOp<Tensor2D, readonly [{ start: -15 }, null]>;
  // Should clamp to [0:10] → size 10 (when implemented)
  expectTypeOf<ShapeOf<OutputOf<NegativeBeyondBounds>>>().toEqualTypeOf<readonly [10, 20]>();

  // Mixed negative and positive out of bounds
  type MixedNegPosBounds = SliceOp<Tensor2D, readonly [{ start: -20; stop: 50 }, null]>;
  // Should clamp to [0:10] → size 10 (when implemented)
  expectTypeOf<ShapeOf<OutputOf<MixedNegPosBounds>>>().toEqualTypeOf<readonly [10, 20]>();
}

// Test 12: Zero step (should be error)
{
  type Tensor1D = TensorStorage<Float32, readonly [10]>;

  // Zero step should be compile-time error
  type ZeroStep = SliceOp<Tensor1D, readonly [{ step: 0 }]>;
  // Should produce never type - zero step is invalid
  expectTypeOf<ZeroStep>().toEqualTypeOf<never>();

  // Zero step with start/stop
  type ZeroStepWithBounds = SliceOp<Tensor1D, readonly [{ start: 2; stop: 8; step: 0 }]>;
  expectTypeOf<ZeroStepWithBounds>().toEqualTypeOf<never>();

  // Zero step in multi-dimensional slice
  type ZeroStepMultiDim = SliceOp<
    TensorStorage<Float32, readonly [10, 20]>,
    readonly [{ step: 2 }, { step: 0 }]
  >;
  expectTypeOf<ZeroStepMultiDim>().toEqualTypeOf<never>();
}

// Test 13: Empty slices (start >= stop)
{
  type Tensor1D = TensorStorage<Float32, readonly [20]>;

  // Start equals stop: [5:5] = empty slice
  type EmptySlice1 = SliceOp<Tensor1D, readonly [{ start: 5; stop: 5 }]>;
  expectTypeOf<ShapeOf<OutputOf<EmptySlice1>>>().toEqualTypeOf<readonly [0]>();

  // Start greater than stop with positive step: [10:5] = empty slice
  type EmptySlice2 = SliceOp<Tensor1D, readonly [{ start: 10; stop: 5 }]>;
  expectTypeOf<ShapeOf<OutputOf<EmptySlice2>>>().toEqualTypeOf<readonly [0]>();

  // With negative step this would be valid: [10:5:-1] = 5 elements
  type ValidWithNegStep = SliceOp<Tensor1D, readonly [{ start: 10; stop: 5; step: -1 }]>;
  expectTypeOf<ShapeOf<OutputOf<ValidWithNegStep>>>().toEqualTypeOf<readonly [5]>();

  // Additional edge cases for empty slices
  type EmptyAtStart = SliceOp<Tensor1D, readonly [{ start: 0; stop: 0 }]>;
  expectTypeOf<ShapeOf<OutputOf<EmptyAtStart>>>().toEqualTypeOf<readonly [0]>();

  type EmptyAtEnd = SliceOp<Tensor1D, readonly [{ start: 20; stop: 20 }]>;
  expectTypeOf<ShapeOf<OutputOf<EmptyAtEnd>>>().toEqualTypeOf<readonly [0]>();
}

// Test 14: Large numbers and arithmetic limits
{
  // Very large dimensions
  type LargeTensor = TensorStorage<Float32, readonly [1000000]>;

  // Large slice: [100000:900000:1000] = 800000/1000 = 800 elements
  type LargeSlice = SliceOp<LargeTensor, readonly [{ start: 100000; stop: 900000; step: 1000 }]>;
  expectTypeOf<ShapeOf<OutputOf<LargeSlice>>>().toEqualTypeOf<readonly [800]>();

  // Multiple large dimensions with step
  type LargeTensor3D = TensorStorage<Float32, readonly [1000, 1000, 1000]>;
  type LargeSlice3D = SliceOp<LargeTensor3D, readonly [{ step: 10 }, { step: 10 }, { step: 10 }]>;
  expectTypeOf<ShapeOf<OutputOf<LargeSlice3D>>>().toEqualTypeOf<readonly [100, 100, 100]>();
}

// Test 15: Complex stride patterns
{
  type Tensor3D = TensorStorage<Float32, readonly [8, 12, 16], readonly [192, 16, 1]>;

  // Slice that affects strides
  type StridedSlice = SliceOp<
    Tensor3D,
    readonly [{ step: 2 }, { start: 2; stop: 10 }, { step: 3 }]
  >;
  // Shape: [8/2=4, 10-2=8, 16/3=6 (rounded up)]
  expectTypeOf<ShapeOf<OutputOf<StridedSlice>>>().toEqualTypeOf<readonly [4, 8, 6]>();

  // Strides should be computed exactly:
  // - First dim has step=2, so stride 192*2=384
  // - Second dim is contiguous slice, so stride stays 16
  // - Third dim has step=3, so stride 1*3=3
  type StridedStrides = StridesOf<OutputOf<StridedSlice>>;
  expectTypeOf<StridedStrides>().toEqualTypeOf<readonly [384, 16, 3]>();
}

// Test 16: Mixed edge cases
{
  type Tensor4D = TensorStorage<Float32, readonly [10, 20, 30, 40]>;

  // Mix of challenging cases
  type Complex1 = SliceOp<
    Tensor4D,
    readonly [
      { start: 8 }, // Size: 2 (10-8)
      5, // Removes dimension
      { stop: 5 }, // Size: 5
      { start: 35; stop: 45 }, // Out of bounds: clamps to [35:40] = 5
    ]
  >;
  // Expected shape after handling all cases
  expectTypeOf<ShapeOf<OutputOf<Complex1>>>().toEqualTypeOf<readonly [2, 5, 5]>();

  // Partial indexing with edge cases
  type Complex2 = SliceOp<Tensor4D, readonly [{ start: 15 }, 0]>;
  // start=15 > dim=10, so clamps to [10:10] = 0, then removes second dim
  expectTypeOf<ShapeOf<OutputOf<Complex2>>>().toEqualTypeOf<readonly [0, 30, 40]>();
}
