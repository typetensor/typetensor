/**
 * Type-level tests for unary tensor operations
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
import type { Neg, Abs, Sin, Cos, Exp, Log, Sqrt, Square } from './unary';
import type { Float32, Int32, Float64, Bool } from '../dtype/types';
import { expectTypeOf } from 'expect-type';

// =============================================================================
// Test Helpers
// =============================================================================

// Create test tensors with different shapes and dtypes
type Float32Scalar = TensorStorage<Float32, readonly []>;
type Float32Vector1D = TensorStorage<Float32, readonly [5]>;
type Float32Matrix2D = TensorStorage<Float32, readonly [3, 4]>;

type Int32Vector = TensorStorage<Int32, readonly [5]>;
type Float64Matrix = TensorStorage<Float64, readonly [3, 4]>;
type BoolVector = TensorStorage<Bool, readonly [5]>;

// Custom layout types
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

// Non-contiguous tensor
type NonContiguousTensor = TensorStorage<
  Float32,
  readonly [3, 4],
  readonly [8, 2],
  NonContiguousLayout
>;

// Fortran-order tensor
type FortranTensor = TensorStorage<Float32, readonly [3, 4], readonly [1, 3], FortranLayout>;

// =============================================================================
// Neg Operation Tests
// =============================================================================

// Test 1: Neg preserves shape
{
  type NegScalar = Neg<Float32Scalar>;
  type NegVector = Neg<Float32Vector1D>;
  type NegMatrix = Neg<Float32Matrix2D>;

  expectTypeOf<readonly []>().toEqualTypeOf<ShapeOf<OutputOf<NegScalar>>>();
  expectTypeOf<readonly [5]>().toEqualTypeOf<ShapeOf<OutputOf<NegVector>>>();
  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<OutputOf<NegMatrix>>>();
}

// Test 2: Neg preserves dtype
{
  type NegFloat32 = Neg<Float32Vector1D>;
  type NegInt32 = Neg<Int32Vector>;
  type NegFloat64 = Neg<Float64Matrix>;

  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<NegFloat32>>>();
  expectTypeOf<Int32>().toEqualTypeOf<DTypeOf<OutputOf<NegInt32>>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<OutputOf<NegFloat64>>>();
}

// Test 3: Neg with non-contiguous input
{
  type NegNonContig = Neg<NonContiguousTensor>;
  type Output = OutputOf<NegNonContig>;
  type OutputStrides = StridesOf<Output>;

  // Output strides are a union type
  expectTypeOf<readonly [8, 2] | readonly [4, 1]>().toEqualTypeOf<OutputStrides>();
}

// Test 4: Neg operation metadata
{
  type NegOp = Neg<Float32Vector1D>;

  expectTypeOf<'neg'>().toEqualTypeOf<NegOp['__op']>();
  expectTypeOf<readonly [Float32Vector1D]>().toEqualTypeOf<NegOp['__inputs']>();
}

// =============================================================================
// Abs Operation Tests
// =============================================================================

// Test 1: Abs preserves shape and dtype
{
  type AbsFloat32 = Abs<Float32Vector1D>;
  type AbsInt32 = Abs<Int32Vector>;

  expectTypeOf<readonly [5]>().toEqualTypeOf<ShapeOf<OutputOf<AbsFloat32>>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<AbsFloat32>>>();
  expectTypeOf<Int32>().toEqualTypeOf<DTypeOf<OutputOf<AbsInt32>>>();
}

// Test 2: Abs operation metadata
{
  type AbsOp = Abs<Float32Vector1D>;

  expectTypeOf<'abs'>().toEqualTypeOf<AbsOp['__op']>();
  expectTypeOf<readonly [Float32Vector1D]>().toEqualTypeOf<AbsOp['__inputs']>();
}

// =============================================================================
// Sin Operation Tests (DType Conversion)
// =============================================================================

// Test 1: Sin converts integers to Float32, preserves float types
{
  type SinFloat32 = Sin<Float32Vector1D>;
  type SinInt32 = Sin<Int32Vector>;
  type SinFloat64 = Sin<Float64Matrix>;
  type SinBool = Sin<BoolVector>;

  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<SinFloat32>>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<SinInt32>>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<OutputOf<SinFloat64>>>(); // Preserves Float64!
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<SinBool>>>();
}

// Test 2: Sin with integer input (dtype conversion)
{
  type Int32NonContig = TensorStorage<Int32, readonly [3, 4], readonly [8, 2], NonContiguousLayout>;
  type SinInt = Sin<Int32NonContig>;
  type Output = OutputOf<SinInt>;

  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<Output>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<Output>>();
  expectTypeOf<StridesOf<Output>>().toEqualTypeOf<readonly [8, 2] | readonly [4, 1]>();
}

// =============================================================================
// Chained Operation Tests
// =============================================================================

// Test 1: Simple chain - Neg -> Abs
{
  type Step1 = Neg<Float32Vector1D>;
  type Step2 = Abs<OutputOf<Step1>>;
  type FinalOutput = OutputOf<Step2>;

  expectTypeOf<readonly [5]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<FinalOutput>>();
}

// Test 2: Chain with dtype conversion - Abs -> Sin
{
  type Step1 = Abs<Int32Vector>;
  type Step2 = Sin<OutputOf<Step1>>;
  type FinalOutput = OutputOf<Step2>;

  expectTypeOf<readonly [5]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<FinalOutput>>();
}

// Test 3: Longer chain - Neg -> Abs -> Sin -> Neg
{
  type Step1 = Neg<Float64Matrix>;
  type Step2 = Abs<OutputOf<Step1>>;
  type Step3 = Sin<OutputOf<Step2>>;
  type Step4 = Neg<OutputOf<Step3>>;
  type FinalOutput = OutputOf<Step4>;

  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<FinalOutput>>();
}

// =============================================================================
// Cos Operation Tests
// =============================================================================

// Test 1: Cos converts integers to Float32, preserves float types
{
  type CosFloat32 = Cos<Float32Vector1D>;
  type CosInt32 = Cos<Int32Vector>;
  type CosFloat64 = Cos<Float64Matrix>;
  type CosBool = Cos<BoolVector>;

  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<CosFloat32>>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<CosInt32>>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<OutputOf<CosFloat64>>>(); // Preserves Float64!
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<CosBool>>>();
}

// =============================================================================
// Exp Operation Tests
// =============================================================================

// Test 1: Exp converts integers to Float32, preserves float types
{
  type ExpFloat32 = Exp<Float32Vector1D>;
  type ExpInt32 = Exp<Int32Vector>;
  type ExpFloat64 = Exp<Float64Matrix>;
  type ExpBool = Exp<BoolVector>;

  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<ExpFloat32>>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<ExpInt32>>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<OutputOf<ExpFloat64>>>(); // Preserves Float64!
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<ExpBool>>>();
}

// =============================================================================
// Log Operation Tests
// =============================================================================

// Test 1: Log converts integers to Float32, preserves float types
{
  type LogFloat32 = Log<Float32Vector1D>;
  type LogInt32 = Log<Int32Vector>;
  type LogFloat64 = Log<Float64Matrix>;
  type LogBool = Log<BoolVector>;

  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<LogFloat32>>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<LogInt32>>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<OutputOf<LogFloat64>>>(); // Preserves Float64!
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<LogBool>>>();
}

// =============================================================================
// Sqrt Operation Tests
// =============================================================================

// Test 1: Sqrt converts integers to Float32, preserves float types
{
  type SqrtFloat32 = Sqrt<Float32Vector1D>;
  type SqrtInt32 = Sqrt<Int32Vector>;
  type SqrtFloat64 = Sqrt<Float64Matrix>;
  type SqrtBool = Sqrt<BoolVector>;

  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<SqrtFloat32>>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<SqrtInt32>>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<OutputOf<SqrtFloat64>>>(); // Preserves Float64!
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<SqrtBool>>>();
}

// =============================================================================
// Square Operation Tests
// =============================================================================

// Test 1: Square preserves all dtypes (no conversion)
{
  type SquareFloat32 = Square<Float32Vector1D>;
  type SquareInt32 = Square<Int32Vector>;
  type SquareFloat64 = Square<Float64Matrix>;
  type SquareBool = Square<BoolVector>;

  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<OutputOf<SquareFloat32>>>();
  expectTypeOf<Int32>().toEqualTypeOf<DTypeOf<OutputOf<SquareInt32>>>(); // Preserves Int32!
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<OutputOf<SquareFloat64>>>();
  expectTypeOf<Bool>().toEqualTypeOf<DTypeOf<OutputOf<SquareBool>>>(); // Bool squared is still Bool
}

// Test 2: Square operation metadata
{
  type SquareOp = Square<Float32Vector1D>;

  expectTypeOf<'square'>().toEqualTypeOf<SquareOp['__op']>();
  expectTypeOf<readonly [Float32Vector1D]>().toEqualTypeOf<SquareOp['__inputs']>();
}

// =============================================================================
// Complex Chain Tests with New Operations
// =============================================================================

// Test 1: Mathematical function chain - Square -> Sqrt -> Log -> Exp
{
  type Step1 = Square<Float32Vector1D>;
  type Step2 = Sqrt<OutputOf<Step1>>;
  type Step3 = Log<OutputOf<Step2>>;
  type Step4 = Exp<OutputOf<Step3>>;
  type FinalOutput = OutputOf<Step4>;

  expectTypeOf<readonly [5]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<FinalOutput>>();
}

// Test 2: Trigonometric chain with Float64 preservation
{
  type Step1 = Sin<Float64Matrix>;
  type Step2 = Cos<OutputOf<Step1>>;
  type Step3 = Square<OutputOf<Step2>>;
  type FinalOutput = OutputOf<Step3>;

  expectTypeOf<readonly [3, 4]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float64>().toEqualTypeOf<DTypeOf<FinalOutput>>(); // Float64 preserved throughout!
}

// Test 3: Integer to float conversion chain
{
  type Step1 = Square<Int32Vector>; // Still Int32
  type Step2 = Sqrt<OutputOf<Step1>>; // Converts to Float32
  type Step3 = Neg<OutputOf<Step2>>; // Stays Float32
  type FinalOutput = OutputOf<Step3>;

  expectTypeOf<readonly [5]>().toEqualTypeOf<ShapeOf<FinalOutput>>();
  expectTypeOf<Float32>().toEqualTypeOf<DTypeOf<FinalOutput>>();
}

// Test 4: Layout propagation through chain
{
  type Step1 = Neg<FortranTensor>;
  type Step2 = Abs<OutputOf<Step1>>;
  type Layout1 = LayoutOf<OutputOf<Step1>>;
  type Layout2 = LayoutOf<OutputOf<Step2>>;

  expectTypeOf<true>().toEqualTypeOf<Layout1['f_contiguous']>();
  expectTypeOf<true>().toEqualTypeOf<Layout2['f_contiguous']>();
  expectTypeOf<true | false>().toEqualTypeOf<Layout2['c_contiguous']>();
}

// =============================================================================
// Negative Tests - Type System Should Catch These Errors
// =============================================================================

// Test 5: Invalid input types should be rejected
{
  // @ts-expect-error - Cannot use non-tensor type as input to Neg
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  type InvalidNeg = Neg<number>;

  // @ts-expect-error - Cannot use plain object as tensor
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  type InvalidAbs = Abs<{ shape: readonly [2, 3] }>;

  // TypeScript's structural typing means we need proper TensorStorage
  // eslint-disable-next-line @typescript-eslint/consistent-type-definitions
  type NotATensor = {
    __dtype: Float32;
    __shape: readonly [2, 3];
    // Missing __strides, __size, __layout
  };

  // @ts-expect-error - Missing required tensor properties (__strides, __size, __layout)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  type IncompleteOp = Sin<NotATensor>;
}

// Test 6: Operation metadata validation
{
  type ValidOp = Neg<Float32Vector1D>;

  // These should work
  type OpType = ValidOp['__op'];
  type Inputs = ValidOp['__inputs'];
  type Output = ValidOp['__output'];

  // Verify correct metadata
  expectTypeOf<'neg'>().toEqualTypeOf<OpType>();
  expectTypeOf<readonly [Float32Vector1D]>().toEqualTypeOf<Inputs>();
  expectTypeOf<Float32Vector1D>().toEqualTypeOf<Output>();

  // @ts-expect-error - Cannot assign wrong operation type
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const wrongOp: ValidOp['__op'] = 'add';

  // @ts-expect-error - Inputs should be readonly tuple with one element
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const wrongInputs: ValidOp['__inputs'] = [];
}

// Test 7: Invalid operations on OutputOf
{
  // This should work
  type ValidChain = Neg<OutputOf<Abs<Float32Vector1D>>>;
  expectTypeOf<ValidChain>().not.toBeNever();

  // @ts-expect-error - OutputOf requires a StorageTransformation, not a TensorStorage
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  type InvalidOutputOf = OutputOf<Float32Vector1D>;

  // @ts-expect-error - Cannot use OutputOf on non-transformation types
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  type BadOutputOf = OutputOf<number>;
}
