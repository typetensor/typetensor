/**
 * Type-level tests for binary tensor operations
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
import type { Add, Sub, Mul, Div } from './binary';
import type { Float32, Int32, Float64, Bool } from '../dtype/types';
import { expectTypeOf } from 'expect-type';
import type { Neg } from './unary';

// =============================================================================
// Test Helpers
// =============================================================================

// Standard tensors
type Float32Matrix = TensorStorage<Float32, readonly [2, 3]>;
type Float32Vector = TensorStorage<Float32, readonly [3]>;
type Float32Scalar = TensorStorage<Float32, readonly []>;
type Int32Matrix = TensorStorage<Int32, readonly [2, 3]>;
type Float64Matrix = TensorStorage<Float64, readonly [2, 3]>;

// Broadcasting test cases
type BroadcastRow = TensorStorage<Float32, readonly [1, 3]>;
type BroadcastCol = TensorStorage<Float32, readonly [2, 1]>;
type Broadcast3D = TensorStorage<Float32, readonly [4, 2, 3]>;

// Non-contiguous tensor
interface NonContiguousLayout extends LayoutFlags {
  readonly c_contiguous: false;
  readonly f_contiguous: false;
  readonly is_view: false;
  readonly writeable: true;
  readonly aligned: true;
}

type NonContiguousTensor = TensorStorage<
  Float32,
  readonly [2, 3],
  readonly [6, 2], // Non C-contiguous strides
  NonContiguousLayout
>;

// =============================================================================
// Add Operation Tests - Basic
// =============================================================================

// Test 1: Same shape addition
{
  type Result = Add<Float32Matrix, Float32Matrix>;
  type Output = OutputOf<Result>;

  // Shape preserved
  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();

  // DType preserved
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Output>>();

  // Output is C-contiguous
  expectTypeOf<readonly [3, 1]>().toEqualTypeOf<StridesOf<Output>>();

  // Operation metadata
  expectTypeOf<'add'>().toEqualTypeOf<Result['__op']>();
  expectTypeOf<readonly [Float32Matrix, Float32Matrix]>().toEqualTypeOf<Result['__inputs']>();
}

// Test 2: DType promotion - Int32 + Float32 -> Float64 (for precision)
{
  type Result = Add<Int32Matrix, Float32Matrix>;
  type Output = OutputOf<Result>;

  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<Output>>(); // Int32 requires Float64 for full precision
  expectTypeOf<Output>().not.toBeNever();
}

// Test 3: DType promotion - Float32 + Float64 -> Float64
{
  type Result = Add<Float32Matrix, Float64Matrix>;
  type Output = OutputOf<Result>;

  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<Output>>();
}

// =============================================================================
// Add Operation Tests - Broadcasting
// =============================================================================

// Test 4: Row broadcasting [1, 3] + [2, 3] -> [2, 3]
{
  type Result = Add<BroadcastRow, Float32Matrix>;
  type Output = OutputOf<Result>;

  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Output>>();
}

// Test 5: Column broadcasting [2, 1] + [2, 3] -> [2, 3]
{
  type Result = Add<BroadcastCol, Float32Matrix>;
  type Output = OutputOf<Result>;

  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
}

// Test 6: 1D broadcasting [3] + [2, 3] -> [2, 3]
{
  type Result = Add<Float32Vector, Float32Matrix>;
  type Output = OutputOf<Result>;

  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
}

// Test 7: Scalar broadcasting [] + [2, 3] -> [2, 3]
{
  type Result = Add<Float32Scalar, Float32Matrix>;
  type Output = OutputOf<Result>;

  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
}

// Test 8: Higher dimensional broadcasting [2, 3] + [4, 2, 3] -> [4, 2, 3]
{
  type Result = Add<Float32Matrix, Broadcast3D>;
  type Output = OutputOf<Result>;

  expectTypeOf<readonly [4, 2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
}

// =============================================================================
// Add Operation Tests - Error Cases (Negative Tests)
// =============================================================================

// Test 9: Incompatible shapes should produce error
{
  type Matrix23 = TensorStorage<Float32, readonly [2, 3]>;
  type Matrix45 = TensorStorage<Float32, readonly [4, 5]>;

  // This should resolve to IncompatibleShapes type
  type Result = Add<Matrix23, Matrix45>;

  // The result should be never (with error message attached)
  expectTypeOf<Result>().toBeNever();

  // Result is never, so these also resolve to never
  type ShouldError = Result['__output'];
  type AlsoShouldError = OutputOf<Result>;
  expectTypeOf<ShouldError>().toBeNever();
  expectTypeOf<AlsoShouldError>().toBeNever();
}

// Test 10: More incompatible shape examples
{
  type Vec3 = TensorStorage<Float32, readonly [3]>;
  type Vec4 = TensorStorage<Float32, readonly [4]>;

  type BadAdd1 = Add<Vec3, Vec4>;
  expectTypeOf<BadAdd1>().toBeNever();

  type Matrix34 = TensorStorage<Float32, readonly [3, 4]>;
  type Matrix43 = TensorStorage<Float32, readonly [4, 3]>;

  type BadAdd2 = Add<Matrix34, Matrix43>;
  expectTypeOf<BadAdd2>().toBeNever();

  type ErrorResult = Add<Vec3, Vec4>;
  expectTypeOf<ErrorResult>().toBeNever();
}

// Test 11: Broadcasting edge cases that should fail
{
  type Shape234 = TensorStorage<Float32, readonly [2, 3, 4]>;
  type Shape345 = TensorStorage<Float32, readonly [3, 4, 5]>;

  // These resolve to never - Cannot broadcast [2,3,4] with [3,4,5] - dimension 0: 2 vs 3
  type BadBroadcast = Add<Shape234, Shape345>;
  expectTypeOf<BadBroadcast>().toBeNever();

  type Shape23 = TensorStorage<Float32, readonly [2, 3]>;
  type Shape45 = TensorStorage<Float32, readonly [4, 5]>;

  // These resolve to never - No dimensions are compatible between [2,3] and [4,5]
  type BadBroadcast2 = Add<Shape23, Shape45>;
  expectTypeOf<BadBroadcast2>().toBeNever();
}

// =============================================================================
// Add Operation Tests - Layout Propagation
// =============================================================================

// Test 10: Layout flags with contiguous inputs
{
  type Result = Add<Float32Matrix, Float32Matrix>;
  type Output = OutputOf<Result>;
  type OutputLayout = LayoutOf<Output>;

  expectTypeOf<false>().toEqualTypeOf<OutputLayout['is_view']>();
  expectTypeOf<true>().toEqualTypeOf<OutputLayout['writeable']>();
  expectTypeOf<true>().toEqualTypeOf<OutputLayout['aligned']>();

  // c_contiguous is backend's choice, even when both inputs are c_contiguous
  expectTypeOf<boolean>().toEqualTypeOf<OutputLayout['c_contiguous']>();

  // f_contiguous is always false for binary ops
  expectTypeOf<false>().toEqualTypeOf<OutputLayout['f_contiguous']>();
}

// Test 11: Layout flags with non-contiguous input
{
  type Result = Add<NonContiguousTensor, Float32Matrix>;
  type Output = OutputOf<Result>;
  type OutputLayout = LayoutOf<Output>;

  // c_contiguous is backend's choice regardless of input layouts
  expectTypeOf<boolean>().toEqualTypeOf<OutputLayout['c_contiguous']>();
}

// =============================================================================
// Add Operation Tests - Complex Scenarios
// =============================================================================

// Test 12: Chained operations - Unary then Binary
{
  type A = TensorStorage<Int32, readonly [2, 3]>;
  type B = TensorStorage<Float32, readonly [1, 3]>;

  // -A + B
  type NegA = Neg<A>;
  type Result = Add<OutputOf<NegA>, B>;
  type FinalOutput = OutputOf<Result>;

  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<FinalOutput>>(); // Int32 + Float32 -> Float64 for precision
}

// Test 13: Multiple broadcasting dimensions
{
  type A = TensorStorage<Float32, readonly [1, 3, 1]>;
  type B = TensorStorage<Float32, readonly [2, 1, 4]>;

  type Result = Add<A, B>;
  type Output = OutputOf<Result>;

  // Should broadcast to [2, 3, 4]
  expectTypeOf<readonly [2, 3, 4]>().toEqualTypeOf<ShapeOf<Output>>();
}

// =============================================================================
// Sub Operation Tests
// =============================================================================

// Test 1: Same shape subtraction
{
  type Result = Sub<Float32Matrix, Float32Matrix>;
  type Output = OutputOf<Result>;

  // Shape preserved
  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
  
  // DType preserved
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Output>>();
  
  // Operation metadata
  expectTypeOf<'sub'>().toEqualTypeOf<Result['__op']>();
  expectTypeOf<readonly [Float32Matrix, Float32Matrix]>().toEqualTypeOf<Result['__inputs']>();
}

// Test 2: Broadcasting with subtraction [2, 3] - [1, 3] -> [2, 3]
{
  type Result = Sub<Float32Matrix, BroadcastRow>;
  type Output = OutputOf<Result>;

  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Output>>();
}

// Test 3: DType promotion in subtraction
{
  type Result = Sub<Int32Matrix, Float64Matrix>;
  type Output = OutputOf<Result>;

  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<Output>>();
}

// =============================================================================
// Mul Operation Tests
// =============================================================================

// Test 1: Same shape multiplication
{
  type Result = Mul<Float32Matrix, Float32Matrix>;
  type Output = OutputOf<Result>;

  // Shape preserved
  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
  
  // DType preserved
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Output>>();
  
  // Operation metadata
  expectTypeOf<'mul'>().toEqualTypeOf<Result['__op']>();
  expectTypeOf<readonly [Float32Matrix, Float32Matrix]>().toEqualTypeOf<Result['__inputs']>();
}

// Test 2: Scalar multiplication (broadcasting)
{
  type Result = Mul<Float32Matrix, Float32Scalar>;
  type Output = OutputOf<Result>;

  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Output>>();
}

// Test 3: Column broadcasting in multiplication [2, 3] * [2, 1] -> [2, 3]
{
  type Result = Mul<Float32Matrix, BroadcastCol>;
  type Output = OutputOf<Result>;

  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
}

// =============================================================================
// Div Operation Tests
// =============================================================================

// Test 1: Same shape division
{
  type Result = Div<Float32Matrix, Float32Matrix>;
  type Output = OutputOf<Result>;

  // Shape preserved
  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<Output>>();
  
  // DType preserved
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Output>>();
  
  // Operation metadata
  expectTypeOf<'div'>().toEqualTypeOf<Result['__op']>();
  expectTypeOf<readonly [Float32Matrix, Float32Matrix]>().toEqualTypeOf<Result['__inputs']>();
}

// Test 2: Integer division promotes to float
{
  type Result = Div<Int32Matrix, Int32Matrix>;
  type Output = OutputOf<Result>;

  // Integer division still uses same promotion rules
  expectTypeOf<Int32>().toEqualTypeOf<DTypeOf<Output>>();
}

// Test 3: Broadcasting in division
{
  type A = TensorStorage<Float64, readonly [3, 1, 5]>;
  type B = TensorStorage<Float64, readonly [1, 4, 1]>;
  
  type Result = Div<A, B>;
  type Output = OutputOf<Result>;

  expectTypeOf<readonly [3, 4, 5]>().toEqualTypeOf<ShapeOf<Output>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<Output>>();
}

// =============================================================================
// Mixed Binary Operations Tests
// =============================================================================

// Test 1: Complex expression (a + b) * c - d
{
  type A = TensorStorage<Float32, readonly [2, 3]>;
  type B = TensorStorage<Float32, readonly [1, 3]>;
  type C = TensorStorage<Float32, readonly [2, 1]>;
  type D = TensorStorage<Float32, readonly [1]>;

  type Sum = Add<A, B>;
  type Product = Mul<OutputOf<Sum>, C>;
  type Result = Sub<OutputOf<Product>, D>;
  type FinalOutput = OutputOf<Result>;

  expectTypeOf<readonly [2, 3]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<FinalOutput>>();
}

// Test 2: Division chain with different dtypes
{
  type A = TensorStorage<Int32, readonly [4, 5]>;
  type B = TensorStorage<Float32, readonly [5]>;
  type C = TensorStorage<Float64, readonly [4, 1]>;

  type Div1 = Div<A, B>;
  type Div2 = Div<OutputOf<Div1>, C>;
  type FinalOutput = OutputOf<Div2>;

  expectTypeOf<readonly [4, 5]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<FinalOutput>>(); // Promoted to highest precision
}

// Test 3: All operations combined with broadcasting
{
  type Scalar = TensorStorage<Float32, readonly []>;
  type Vector = TensorStorage<Float32, readonly [5]>;
  type Matrix = TensorStorage<Float32, readonly [3, 5]>;

  // scalar + vector -> vector
  type Step1 = Add<Scalar, Vector>;
  // vector * matrix -> matrix (via broadcasting)
  type Step2 = Mul<OutputOf<Step1>, Matrix>;
  // matrix - vector -> matrix
  type Step3 = Sub<OutputOf<Step2>, Vector>;
  // matrix / scalar -> matrix
  type Step4 = Div<OutputOf<Step3>, Scalar>;
  
  type FinalOutput = OutputOf<Step4>;

  expectTypeOf<readonly [3, 5]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<FinalOutput>>();
}

// Test 4: Bool type in binary operations
{
  type BoolMatrix = TensorStorage<Bool, readonly [2, 3]>;
  type IntMatrix = TensorStorage<Int32, readonly [2, 3]>;
  
  // Bool promotes to the other type
  type Result = Add<BoolMatrix, IntMatrix>;
  type Output = OutputOf<Result>;
  
  expectTypeOf<Int32>().toEqualTypeOf<DTypeOf<Output>>();
}
