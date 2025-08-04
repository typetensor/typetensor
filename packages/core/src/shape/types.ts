/**
 * Type-level shape operations for compile-time tensor shape validation
 *
 * This module provides TypeScript types for representing and manipulating
 * tensor shapes at the type level, enabling compile-time shape checking
 * and inference for tensor operations.
 */

import type { Multiply, Subtract, Compare, Add } from 'ts-arithmetic';
import type { Ceil, Max, Min, Abs, Decrement } from '../arithmetic';

// =============================================================================
// Basic Shape Types
// =============================================================================

/**
 * A tensor shape represented as a readonly tuple of positive integers
 * Each number represents the size of a dimension
 */
export type Shape = readonly number[];

/**
 * A specifically constrained shape that is readonly and of known length
 * i.e. [1, 2] as const, [] as const etc...
 */
export type ConstShape = Shape extends readonly [number, ...number[]] | readonly [] ? Shape : never;

/**
 * A shape that may contain dynamic dimensions represented by -1
 * Used when some dimensions are only known at runtime
 */
export type DynamicShape = readonly (number | -1)[];

/**
 * A fully resolved shape with all concrete dimensions
 * Branded type to prevent mixing with unresolved shapes
 */
export type ResolvedShape = Shape & { readonly __resolved: true };

/**
 * A partially resolved shape that may contain -1 for unknown dimensions
 * Branded type to indicate partial resolution state
 */
export type PartialShape = DynamicShape & { readonly __partial: true };

/**
 * A symbolic dimension with a string name for constraint solving
 */
export interface SymbolicDim<Name extends string = string> {
  readonly __symbolic: Name;
}

/**
 * A shape that may contain symbolic dimensions
 * Used for advanced shape inference and constraint solving
 */
export type SymbolicShape = readonly (number | SymbolicDim)[];

/**
 * Maximum supported tensor rank (number of dimensions)
 * Set to 8 to balance type performance with practical needs
 */
export type MaxRank = 8;

// =============================================================================
// Shape Arithmetic and Utilities
// =============================================================================

/**
 * Calculate the product of all dimensions in a shape (total number of elements)
 *
 * @example
 * type Size = Product<[2, 3, 4]> // 24
 */
export type Product<T extends Shape> = T extends readonly []
  ? 1
  : T extends readonly [infer Head, ...infer Tail]
    ? Head extends number
      ? Tail extends Shape
        ? Head extends 0
          ? 0 // Zero dimension makes entire product zero
          : Multiply<Head, Product<Tail>>
        : never
      : never
    : never;

/**
 * Get the length (rank) of a shape
 *
 * @example
 * type Rank = Length<[2, 3, 4]> // 3
 */
export type Length<T extends Shape> = T['length'];

/**
 * Check if a shape is empty (scalar)
 *
 * @example
 * type IsScalar = IsEmpty<[]> // true
 * type IsNotScalar = IsEmpty<[2, 3]> // false
 */
export type IsEmpty<T extends Shape> = T extends readonly [] ? true : false;

/**
 * Get the first dimension of a shape
 *
 * @example
 * type First = Head<[2, 3, 4]> // 2
 */
export type Head<T extends Shape> = T extends readonly [infer H, ...unknown[]] ? H : never;

/**
 * Get all dimensions except the first
 *
 * @example
 * type Rest = Tail<[2, 3, 4]> // [3, 4]
 */
export type Tail<T extends Shape> = T extends readonly [unknown, ...infer Rest]
  ? Rest extends readonly unknown[]
    ? Rest
    : never
  : never;

/**
 * Get the last dimension of a shape
 *
 * @example
 * type Last = Last<[2, 3, 4]> // 4
 */
export type Last<T extends Shape> = T extends readonly [...unknown[], infer L] ? L : never;

/**
 * Get the last dimension with proper number constraint
 *
 * @example
 * type LastDim = LastDim<[2, 3, 4]> // 4
 */
export type LastDim<S extends Shape> = S extends readonly [...unknown[], infer Last]
  ? Last extends number
    ? Last
    : never
  : never;

/**
 * Get all dimensions except the last one
 *
 * @example
 * type AllButLast = AllButLast<[2, 3, 4]> // readonly [2, 3]
 */
export type AllButLast<S extends Shape> = S extends readonly [...infer Init, unknown]
  ? Init extends readonly unknown[]
    ? Init
    : never
  : never;

/**
 * Get the second-to-last dimension
 *
 * @example
 * type SecondToLast = SecondToLast<[2, 3, 4]> // 3
 */
export type SecondToLast<S extends Shape> = S extends readonly [...unknown[], infer SL, unknown]
  ? SL extends number
    ? SL
    : never
  : never;

/**
 * Get all dimensions except the last two (batch dimensions for matrix ops)
 *
 * @example
 * type BatchDims = BatchDims<[5, 2, 3, 4]> // readonly [5, 2]
 */
export type BatchDims<S extends Shape> = S extends readonly [...infer Batch, unknown, unknown]
  ? Batch extends readonly unknown[]
    ? Batch
    : never
  : never;

/**
 * Get all dimensions except the last
 *
 * @example
 * type Init = Init<[2, 3, 4]> // [2, 3]
 */
export type Init<T extends Shape> = T extends readonly [...infer Init, unknown]
  ? Init extends readonly unknown[]
    ? Init
    : never
  : never;

// =============================================================================
// Shape Transformations
// =============================================================================

/**
 * Concatenate two shapes
 *
 * @example
 * type Combined = Concat<[2, 3], [4, 5]> // [2, 3, 4, 5]
 */
export type Concat<A extends Shape, B extends Shape> = readonly [...A, ...B];

/**
 * Reverse the order of dimensions in a shape
 *
 * @example
 * type Reversed = Reverse<[2, 3, 4]> // [4, 3, 2]
 */
export type Reverse<T extends Shape> = T extends readonly [...infer Rest, infer Last]
  ? Last extends number
    ? Rest extends Shape
      ? readonly [Last, ...Reverse<Rest>]
      : readonly [Last]
    : readonly []
  : readonly [];

/**
 * Take the first N dimensions from a shape
 *
 * @example
 * type FirstTwo = Take<[2, 3, 4, 5], 2> // [2, 3]
 */
export type Take<T extends Shape, N extends number> = TakeHelper<T, N, []>;

type TakeHelper<T extends Shape, N extends number, Acc extends Shape> = Acc['length'] extends N
  ? Acc
  : T extends readonly [infer Head, ...infer Rest]
    ? Head extends number
      ? Rest extends Shape
        ? TakeHelper<Rest, N, readonly [...Acc, Head]>
        : Acc
      : Acc
    : Acc;

/**
 * Drop the first N dimensions from a shape
 *
 * @example
 * type LastTwo = Drop<[2, 3, 4, 5], 2> // [4, 5]
 */
export type Drop<T extends Shape, N extends number> = DropHelper<T, N>;

type DropHelper<T extends Shape, N extends number> = N extends 0
  ? T
  : T extends readonly [unknown, ...infer Rest]
    ? Rest extends Shape
      ? DropHelper<Rest, Subtract<N, 1>>
      : readonly []
    : readonly [];

/**
 * Permute dimensions according to an axis order
 *
 * @example
 * type Transposed = Permute<[2, 3, 4], [2, 0, 1]> // [4, 2, 3]
 */
export type Permute<T extends Shape, Order extends readonly number[]> = Order extends readonly [
  infer First,
  ...infer Rest,
]
  ? First extends number
    ? Rest extends readonly number[]
      ? T extends Shape
        ? readonly [At<T, First>, ...Permute<T, Rest>]
        : never
      : never
    : never
  : readonly [];

/**
 * Get the dimension at a specific index
 *
 * @example
 * type Second = At<[2, 3, 4], 1> // 3
 * type OutOfBounds = At<[2, 3, 4], 3> // never
 * type Negative = At<[2, 3, 4], -1> // never
 */
export type At<T extends Shape, Index extends number> = Index extends number
  ? `${Index}` extends `-${string}`
    ? never // Negative indices not supported
    : T extends readonly unknown[]
      ? Index extends keyof T
        ? T[Index] extends number
          ? T[Index]
          : never
        : never // Out of bounds
      : never
  : never;

/**
 * Transpose dimensions of a shape
 *
 * If no axes are provided, swaps the last two dimensions (like matrix transpose)
 * If axes are provided, permutes dimensions according to the axes array
 *
 * @example
 * type Transposed = Transpose<[2, 3, 4]> // [2, 4, 3] - default behavior
 * type Custom = Transpose<[2, 3, 4], [2, 0, 1]> // [4, 2, 3] - custom permutation
 */
export type Transpose<
  T extends Shape,
  Axes extends readonly number[] | undefined = undefined,
> = Axes extends undefined
  ? TransposeLastTwo<T>
  : Axes extends readonly number[]
    ? Permute<T, Axes>
    : never;

/**
 * Helper: Transpose the last two dimensions of a shape
 */
type TransposeLastTwo<T extends Shape> = T extends readonly [infer A, infer B]
  ? readonly [B, A]
  : T extends readonly [...infer Init, infer A, infer B]
    ? readonly [...Init, B, A]
    : T;

/**
 * Remove dimensions of size 1 from a shape
 *
 * @example
 * type Squeezed = Squeeze<[2, 1, 3, 1]> // [2, 3]
 */
export type Squeeze<T extends Shape, Axis extends number = -1> = Axis extends -1
  ? SqueezeAll<T>
  : RemoveAt<T, Axis>;

/**
 * Remove all dimensions of size 1
 */
type SqueezeAll<T extends Shape> = T extends readonly [infer First, ...infer Rest]
  ? First extends 1
    ? Rest extends Shape
      ? SqueezeAll<Rest>
      : readonly []
    : First extends number
      ? Rest extends Shape
        ? readonly [First, ...SqueezeAll<Rest>]
        : readonly [First]
      : readonly []
  : readonly [];

/**
 * Remove dimension at specific index
 */
type RemoveAt<T extends Shape, Index extends number> = RemoveAtHelper<T, Index, 0, []>;

type RemoveAtHelper<
  T extends Shape,
  Index extends number,
  Current extends number,
  Acc extends Shape,
> = T extends readonly [infer First, ...infer Rest]
  ? First extends number
    ? Rest extends Shape
      ? Current extends Index
        ? readonly [...Acc, ...Rest] // Skip this element
        : RemoveAtHelper<Rest, Index, Increment<Current>, readonly [...Acc, First]>
      : Acc
    : Acc
  : Acc;

/**
 * Add a dimension of size 1 at a specific position
 *
 * @example
 * type Unsqueezed = Unsqueeze<[2, 3], 1> // [2, 1, 3]
 */
export type Unsqueeze<T extends Shape, Axis extends number> = InsertAt<T, Axis, 1>;

/**
 * Insert a value at a specific index in a shape
 */
type InsertAt<T extends Shape, Index extends number, Value extends number> = InsertAtHelper<
  T,
  Index,
  Value,
  0,
  []
>;

type InsertAtHelper<
  T extends Shape,
  Index extends number,
  Value extends number,
  Current extends number,
  Acc extends Shape,
> = Current extends Index
  ? readonly [...Acc, Value, ...T] // Insert here
  : T extends readonly [infer First, ...infer Rest]
    ? First extends number
      ? Rest extends Shape
        ? InsertAtHelper<Rest, Index, Value, Increment<Current>, readonly [...Acc, First]>
        : readonly [...Acc, First, Value] // Insert at end
      : readonly [...Acc, Value]
    : readonly [...Acc, Value];

/**
 * Type-level increment (limited to small numbers)
 */
type Increment<N extends number> = N extends 0
  ? 1
  : N extends 1
    ? 2
    : N extends 2
      ? 3
      : N extends 3
        ? 4
        : N extends 4
          ? 5
          : N extends 5
            ? 6
            : N extends 6
              ? 7
              : N extends 7
                ? 8
                : N extends 8
                  ? 9
                  : N extends 9
                    ? 10
                    : number;

// =============================================================================
// Broadcasting and Compatibility
// =============================================================================

/**
 * A special never type that includes our error message
 * TypeScript will show this in error messages
 */
// Ignored since this is used for bubbling up errors
export type IncompatibleShapes<A extends Shape, B extends Shape> = never & {
  __error: `Cannot broadcast shapes [${ShapeToString<A>}] and [${ShapeToString<B>}]`;
};

/**
 * Convert a shape to a readable string for error messages
 */
export type ShapeToString<S extends Shape> = S extends readonly []
  ? ''
  : S extends readonly [infer H]
    ? `${H & (string | number)}`
    : S extends readonly [infer H, ...infer T]
      ? T extends readonly []
        ? H & string
        : T extends Shape
          ? `${H & string}, ${ShapeToString<T>}`
          : never
      : never;

/**
 * Check if two shapes can be broadcast together (NumPy-style)
 *
 * @example
 * type CanAdd = CanBroadcast<[1, 3], [2, 1]> // true
 * type CannotAdd = CanBroadcast<[2, 3], [4, 5]> // false
 */
export type CanBroadcast<A extends Shape, B extends Shape> =
  BroadcastShapes<A, B> extends never ? false : true;

/**
 * Compute the resulting shape when broadcasting two shapes
 * Returns IncompatibleShapes (a special never type) if shapes are incompatible
 *
 * @example
 * type Result = BroadcastShapes<[1, 3], [2, 1]> // [2, 3]
 * type Error = BroadcastShapes<[2, 3], [3, 2]> // IncompatibleShapes<[2, 3], [3, 2]>
 */
export type BroadcastShapes<A extends Shape, B extends Shape> =
  BroadcastReversed<Reverse<A>, Reverse<B>, A, B> extends infer Result
    ? Result extends Shape
      ? Reverse<Result>
      : IncompatibleShapes<A, B>
    : IncompatibleShapes<A, B>;

/**
 * Broadcast shapes working from the rightmost dimension (helper)
 * Now takes original shapes to create proper error messages
 */
type BroadcastReversed<
  A extends Shape,
  B extends Shape,
  OriginalA extends Shape,
  OriginalB extends Shape,
> = A extends readonly [infer A1, ...infer ARest]
  ? B extends readonly [infer B1, ...infer BRest]
    ? A1 extends number
      ? B1 extends number
        ? ARest extends Shape
          ? BRest extends Shape
            ? BroadcastDim<A1, B1, OriginalA, OriginalB> extends infer Dim
              ? Dim extends number
                ? readonly [Dim, ...BroadcastReversed<ARest, BRest, OriginalA, OriginalB>]
                : IncompatibleShapes<OriginalA, OriginalB>
              : IncompatibleShapes<OriginalA, OriginalB>
            : readonly [A1]
          : readonly [A1]
        : IncompatibleShapes<OriginalA, OriginalB>
      : IncompatibleShapes<OriginalA, OriginalB>
    : A // B is empty, return A
  : B; // A is empty, return B

/**
 * Broadcast two individual dimensions
 * Returns IncompatibleShapes if incompatible
 */
type BroadcastDim<
  A extends number,
  B extends number,
  OriginalA extends Shape,
  OriginalB extends Shape,
> = A extends 1 ? B : B extends 1 ? A : A extends B ? A : IncompatibleShapes<OriginalA, OriginalB>;

/**
 * Check if shapes are exactly equal
 *
 * @example
 * type Same = Equals<[2, 3], [2, 3]> // true
 * type Different = Equals<[2, 3], [3, 2]> // false
 */
export type Equals<A extends Shape, B extends Shape> = A extends B
  ? B extends A
    ? true
    : false
  : false;

/**
 * Check if first shape is assignable to second (covariant)
 *
 * @example
 * type IsAssignable = IsAssignableTo<[2, 3], readonly number[]> // true
 */
export type IsAssignableTo<A extends Shape, B extends Shape> = A extends B ? true : false;

// =============================================================================
// Shape Validation and Constraints
// =============================================================================

/**
 * Reshape a tensor to a new shape
 * Returns the new shape if valid (same total elements), never if invalid
 *
 * @example
 * type Reshaped = Reshape<[2, 3, 4], [6, 4]> // [6, 4] - valid (24 = 24)
 * type Invalid = Reshape<[2, 3], [4, 5]> // never - invalid (6 ≠ 20)
 */
export type Reshape<From extends Shape, To extends Shape> =
  CanReshape<From, To> extends true ? To : never;

/**
 * Validate that a reshape is valid (same total elements)
 *
 * @example
 * type Valid = CanReshape<[2, 6], [3, 4]> // true (both have 12 elements)
 * type Invalid = CanReshape<[2, 3], [4, 5]> // false (6 vs 20 elements)
 */
export type CanReshape<From extends Shape, To extends Shape> =
  Product<From> extends Product<To> ? (Product<To> extends Product<From> ? true : false) : false;

/**
 * Validate that tensor shapes are compatible for matrix multiplication
 *
 * @example
 * type CanMultiply = IsMatMulCompatible<[2, 3], [3, 4]> // true
 * type CannotMultiply = IsMatMulCompatible<[2, 3], [4, 5]> // false
 */
export type IsMatMulCompatible<A extends Shape, B extends Shape> = HasMatchingInnerDims<A, B>;

/**
 * Validate if two shapes can be matrix multiplied with detailed error messages
 *
 * Returns true for compatible shapes, or a ShapeError for incompatible ones.
 * This provides better developer experience than IsMatMulCompatible which returns boolean.
 *
 * @example
 * type Valid = CanMatmul<[2, 3], [3, 4]> // true
 * type Invalid = CanMatmul<[2, 3], [4, 5]> // ShapeError with detailed message
 */
export type CanMatmul<A extends Shape, B extends Shape> =
  IsMatMulCompatible<A, B> extends true
    ? true
    : A extends readonly []
      ? ShapeError<
          `Cannot multiply scalar tensors. Matrix multiplication requires at least 1D tensors.`,
          { shapeA: A; shapeB: B }
        >
      : B extends readonly []
        ? ShapeError<
            `Cannot multiply with scalar tensor. Matrix multiplication requires at least 1D tensors.`,
            { shapeA: A; shapeB: B }
          >
        : ShapeError<
            `Cannot multiply tensors with shapes [${ShapeToString<A>}] and [${ShapeToString<B>}]. Matrix multiplication requires the last dimension of the first tensor (${LastDim<A> & number}) to match the ${B extends readonly [unknown] ? 'dimension' : 'second-to-last dimension'} of the second tensor (${B extends readonly [unknown] ? B[0] & number : SecondToLast<B> & number}).`,
            { shapeA: A; shapeB: B }
          >;

/**
 * Check if inner dimensions match for matrix multiplication
 * A's last dimension must equal B's second-to-last dimension
 *
 * @example
 * type Match = HasMatchingInnerDims<[2, 3], [3, 4]> // true
 * type NoMatch = HasMatchingInnerDims<[2, 3], [4, 5]> // false
 */
export type HasMatchingInnerDims<A extends Shape, B extends Shape> =
  // Handle edge cases first
  A extends readonly []
    ? false // Scalar can't multiply
    : B extends readonly []
      ? false // Scalar can't multiply
      : B extends readonly [unknown] // B is 1D
        ? LastDim<A> extends B[0]
          ? true
          : false
        : // B is 2D+, check A's last vs B's second-to-last
          LastDim<A> extends SecondToLast<B>
          ? true
          : false;

/**
 * Infer the result shape of matrix multiplication
 *
 * Handles all standard matrix multiplication cases:
 * - 1D × 1D: [K] × [K] → scalar (returns readonly [])
 * - 1D × 2D: [K] × [K, N] → [N]
 * - 2D × 1D: [M, K] × [K] → [M]
 * - 2D × 2D: [M, K] × [K, N] → [M, N]
 * - ND × ND: [...batch, M, K] × [...batch, K, N] → [...batch, M, N]
 *
 * @example
 * type Result1 = MatMulShape<[2, 3], [3, 4]> // readonly [2, 4]
 * type Result2 = MatMulShape<[10], [10, 5]> // readonly [5]
 * type Result3 = MatMulShape<[5, 2, 3], [5, 3, 4]> // readonly [5, 2, 4]
 */
export type MatMulShape<A extends Shape, B extends Shape> =
  // Validate compatibility first - fail fast if incompatible
  HasMatchingInnerDims<A, B> extends false
    ? never
    : // Dispatch to specific case handlers
      A extends readonly [infer K1]
      ? K1 extends number
        ? B extends readonly [infer K2]
          ? K2 extends number
            ? readonly [] // 1D × 1D → scalar
            : never
          : VectorMatrixMul<readonly [K1], B> // 1D × 2D+
        : never
      : B extends readonly [infer K2]
        ? K2 extends number
          ? MatrixVectorMul<A, readonly [K2]> // 2D+ × 1D
          : never
        : MatrixMatrixMul<A, B>; // 2D+ × 2D+

/**
 * Vector × Matrix multiplication: [K] × [K, N] or [K] × [..., K, N] → [N] or [..., N]
 */
type VectorMatrixMul<A extends readonly [number], B extends Shape> =
  // Simple 2D case: [K] × [K, N] → [N]
  B extends readonly [infer K, infer N]
    ? A[0] extends K
      ? N extends number
        ? readonly [N]
        : never
      : never
    : // ND case: [K] × [..., K, N] → [..., N]
      B extends readonly [...infer BPrefix, infer K, infer N]
      ? A[0] extends K
        ? N extends number
          ? BPrefix extends readonly []
            ? readonly [N] // Actually was 2D case
            : BPrefix extends Shape
              ? readonly [...BPrefix, N]
              : never
          : never
        : never
      : never;

/**
 * Matrix × Vector multiplication: [M, K] or [..., M, K] × [K] → [M] or [..., M]
 */
type MatrixVectorMul<A extends Shape, B extends readonly [number]> =
  // Simple 2D case: [M, K] × [K] → [M]
  A extends readonly [infer M, infer K]
    ? K extends B[0]
      ? M extends number
        ? readonly [M]
        : never
      : never
    : // ND case: [..., M, K] × [K] → [..., M]
      A extends readonly [...infer APrefix, infer M, infer K]
      ? K extends B[0]
        ? M extends number
          ? APrefix extends readonly []
            ? readonly [M] // Actually was 2D case
            : APrefix extends Shape
              ? readonly [...APrefix, M]
              : never
          : never
        : never
      : never;

/**
 * Matrix × Matrix multiplication: [..., M, K] × [..., K, N] → [..., M, N]
 */
type MatrixMatrixMul<A extends Shape, B extends Shape> = A extends readonly [
  ...infer APre,
  infer M,
  infer K1,
]
  ? B extends readonly [...infer BPre, infer K2, infer N]
    ? K1 extends K2
      ? M extends number
        ? N extends number
          ? // Broadcast batch dimensions
            BroadcastShapes<
              APre extends Shape ? APre : readonly [],
              BPre extends Shape ? BPre : readonly []
            > extends infer BatchShape
            ? BatchShape extends Shape
              ? readonly [...BatchShape, M, N]
              : never // BroadcastShapes returned IncompatibleShapes
            : never
          : never
        : never
      : never
    : never
  : never;

// =============================================================================
// Error Types and Messages
// =============================================================================

/**
 * Shape error with descriptive message
 */
export interface ShapeError<Message extends string, Context = unknown> {
  readonly __error: 'ShapeError';
  readonly message: Message;
  readonly context: Context;
}

/**
 * Create a descriptive error for shape mismatches
 */
export type ShapeMismatchError<Expected extends Shape, Actual extends Shape> = ShapeError<
  `Shape mismatch: expected shape with ${Expected['length']} dimensions but got ${Actual['length']} dimensions`,
  { expected: Expected; actual: Actual }
>;

// =============================================================================
// Utility Types for Common Patterns
// =============================================================================

/**
 * Create a tuple of N identical values
 *
 * @example
 * type Ones = TupleOf<1, 3> // [1, 1, 1]
 */
export type TupleOf<T, N extends number> = TupleOfHelper<T, N, []>;

type TupleOfHelper<T, N extends number, Acc extends readonly unknown[]> = Acc['length'] extends N
  ? Acc
  : TupleOfHelper<T, N, readonly [...Acc, T]>;

/**
 * Common shape patterns for neural networks
 */
export type Shape1D = readonly [number];
export type Shape2D = readonly [number, number];
export type Shape3D = readonly [number, number, number];
export type Shape4D = readonly [number, number, number, number];

/**
 * Batch shape (first dimension can vary)
 */
export type BatchShape<S extends Shape> = readonly [number, ...S];

/**
 * Image shape with channels-last format
 */
export type ImageShape = readonly [number, number, number, number]; // [batch, height, width, channels]

/**
 * Sequence shape for NLP
 */
export type SequenceShape = readonly [number, number]; // [batch, sequence_length]

/**
 * Attention shape for transformers
 */
export type AttentionShape = readonly [number, number, number]; // [batch, sequence_length, features]

// =============================================================================
// Slicing Operations
// =============================================================================

/**
 * Slice specification for a single dimension
 */
export interface SliceSpec {
  readonly start?: number;
  readonly stop?: number;
  readonly step?: number;
}

/**
 * Slice index can be:
 * - number: for indexing (removes dimension)
 * - SliceSpec: for slicing (computes new dimension size)
 * - null: for full slice (preserves dimension)
 */
export type SliceIndex = number | SliceSpec | null;

/**
 * Normalize a potentially negative index to a positive index
 * Negative indices count from the end: -1 = last element, -2 = second to last, etc.
 *
 * @example
 * type Idx1 = NormalizeIndex<-1, 10> // 9
 * type Idx2 = NormalizeIndex<-5, 10> // 5
 * type Idx3 = NormalizeIndex<5, 10> // 5
 * type Idx4 = NormalizeIndex<-15, 10> // 0 (clamped)
 */
type NormalizeIndex<
  Index extends number,
  Dim extends number,
> = `${Index}` extends `-${infer AbsValue extends number}`
  ? Max<0, Subtract<Dim, AbsValue>> // Negative: dim - abs(index), clamped to 0
  : Index; // Positive: use as-is

/**
 * Compute the size of a sliced dimension
 *
 * @example
 * type Size1 = ComputeSlicedDimSize<10, { start: 0, stop: 5 }> // 5
 * type Size2 = ComputeSlicedDimSize<10, { start: 2, stop: 8, step: 2 }> // 3
 * type Size3 = ComputeSlicedDimSize<10, null> // 10
 */
type ComputeSlicedDimSize<Dim extends number, Index extends SliceSpec | null> = Index extends null
  ? Dim // Full slice preserves dimension
  : Index extends SliceSpec
    ? Index['step'] extends number
      ? Index['step'] extends 0
        ? never // Zero step is invalid
        : Compare<Index['step'], 0> extends -1
          ? ComputeNegativeStepSize<Dim, Index> // Negative step
          : ComputePositiveStepSize<Dim, Index> // Positive step
      : ComputePositiveStepSize<Dim, Index> // Default step = 1
    : never;

/**
 * Compute size for positive step slices
 */
type ComputePositiveStepSize<
  Dim extends number,
  Index extends SliceSpec,
> = Index['stop'] extends number
  ? Index['start'] extends number
    ? Index['step'] extends number
      ? Max<
          0,
          Ceil<
            Subtract<
              Min<Max<0, NormalizeIndex<Index['stop'], Dim>>, Dim>,
              Max<0, NormalizeIndex<Index['start'], Dim>>
            >,
            Index['step']
          >
        >
      : Max<
          0,
          Subtract<
            Min<Max<0, NormalizeIndex<Index['stop'], Dim>>, Dim>,
            Max<0, NormalizeIndex<Index['start'], Dim>>
          >
        > // Default step = 1
    : Index['step'] extends number
      ? Ceil<Min<Max<0, NormalizeIndex<Index['stop'], Dim>>, Dim>, Index['step']> // Default start = 0
      : Min<Max<0, NormalizeIndex<Index['stop'], Dim>>, Dim> // Default start = 0, step = 1
  : // No stop provided
    Index['start'] extends number
    ? Index['step'] extends number
      ? Max<0, Ceil<Subtract<Dim, Max<0, NormalizeIndex<Index['start'], Dim>>>, Index['step']>> // stop defaults to Dim
      : Max<0, Subtract<Dim, Max<0, NormalizeIndex<Index['start'], Dim>>>> // stop defaults to Dim, step = 1
    : Index['step'] extends number
      ? Ceil<Dim, Index['step']> // start = 0, stop = Dim
      : Dim; // No start, stop, or step - full slice

/**
 * Compute size for negative step slices
 * For negative steps: size = ceil((start - stop) / abs(step))
 * Default start = dim - 1, default stop = -1 (before index 0)
 */
type ComputeNegativeStepSize<
  Dim extends number,
  Index extends SliceSpec,
> = Index['start'] extends number
  ? Index['stop'] extends number
    ? Max<
        0,
        Ceil<
          Subtract<
            Min<Max<0, NormalizeIndex<Index['start'], Dim>>, Subtract<Dim, 1>>,
            Max<0, NormalizeIndex<Index['stop'], Dim>>
          >,
          Abs<Index['step'] extends number ? Index['step'] : -1>
        >
      >
    : // No stop: defaults to -1 (before index 0), so we count to index 0
      Max<
        0,
        Ceil<
          Add<Min<Max<0, NormalizeIndex<Index['start'], Dim>>, Subtract<Dim, 1>>, 1>,
          Abs<Index['step'] extends number ? Index['step'] : -1>
        >
      >
  : Index['stop'] extends number
    ? // No start: defaults to dim - 1
      Max<
        0,
        Ceil<
          Subtract<Subtract<Dim, 1>, Max<0, NormalizeIndex<Index['stop'], Dim>>>,
          Abs<Index['step'] extends number ? Index['step'] : -1>
        >
      >
    : // Neither start nor stop: full reverse
      Ceil<Dim, Abs<Index['step'] extends number ? Index['step'] : -1>>;

/**
 * Compute the shape after slicing operations
 *
 * @example
 * type Original = readonly [10, 20, 30];
 * type Sliced1 = SlicedShape<Original, [{ start: 0, stop: 5 }, null, null]> // [5, 20, 30]
 * type Sliced2 = SlicedShape<Original, [5, null, { start: 0, stop: 10 }]> // [20, 10]
 * type Sliced3 = SlicedShape<Original, [null, { start: 5, stop: 15 }, null]> // [10, 10, 30]
 */
export type SlicedShape<
  InputShape extends Shape,
  Indices extends readonly SliceIndex[],
> = SlicedShapeHelper<InputShape, Indices, readonly []>;

type SlicedShapeHelper<
  RemainingShape extends Shape,
  RemainingIndices extends readonly SliceIndex[],
  Acc extends Shape,
> = RemainingIndices extends readonly []
  ? readonly [...Acc, ...RemainingShape] // No more indices, keep remaining dims
  : RemainingShape extends readonly []
    ? Acc // No more shape dims
    : RemainingIndices extends readonly [infer FirstIndex, ...infer RestIndices]
      ? RemainingShape extends readonly [infer FirstDim, ...infer RestDims]
        ? FirstDim extends number
          ? RestDims extends Shape
            ? RestIndices extends readonly SliceIndex[]
              ? FirstIndex extends number
                ? SlicedShapeHelper<RestDims, RestIndices, Acc> // Integer index: remove dimension
                : FirstIndex extends SliceSpec | null
                  ? ComputeSlicedDimSize<FirstDim, FirstIndex> extends never
                    ? never // If dimension computation fails, propagate never
                    : SlicedShapeHelper<
                        RestDims,
                        RestIndices,
                        readonly [...Acc, ComputeSlicedDimSize<FirstDim, FirstIndex>]
                      >
                  : never
              : never
            : never
          : never
        : never
      : never;

// =============================================================================
// Dimension Validation and Normalization
// =============================================================================

/**
 * Error type for dimension validation
 *
 * @example
 * type Error = DimensionError<"Invalid dimension -3 for axis 0 of shape [2, 3]">
 */
export interface DimensionError<Message extends string> {
  readonly __error: 'DimensionError';
  readonly message: Message;
}

/**
 * Normalize a dimension index to handle negative indexing
 * Negative indices count from the end: -1 = last dimension, -2 = second to last, etc.
 *
 * @example
 * type Normalized1 = NormalizeDim<-1, 3> // 2 (last dimension of rank-3 tensor)
 * type Normalized2 = NormalizeDim<-2, 3> // 1 (second to last dimension)
 * type Normalized3 = NormalizeDim<1, 3> // 1 (positive index unchanged)
 * type Error1 = NormalizeDim<-5, 3> // DimensionError (out of bounds)
 * type Error2 = NormalizeDim<4, 3> // DimensionError (out of bounds)
 */
export type NormalizeDim<
  Dim extends number,
  Rank extends number,
> = `${Dim}` extends `-${infer AbsStr}`
  ? AbsStr extends `${infer Abs extends number}`
    ? Abs extends number
      ? Compare<Abs, Rank> extends 1
        ? DimensionError<`Invalid dimension ${Dim} for tensor with ${Rank} dimensions (must be >= -${Rank})`>
        : Subtract<Rank, Abs>
      : DimensionError<`Invalid dimension ${Dim} for tensor with ${Rank} dimensions`>
    : DimensionError<`Invalid dimension ${Dim} for tensor with ${Rank} dimensions`>
  : Compare<Dim, Rank> extends -1
    ? Dim
    : DimensionError<`Invalid dimension ${Dim} for tensor with ${Rank} dimensions (must be < ${Rank})`>;

/**
 * Validate a dimension index for a given tensor rank
 * Returns the normalized dimension or a DimensionError
 *
 * @example
 * type Valid1 = ValidateDim<-1, [2, 3, 4]> // 2 (valid negative index)
 * type Valid2 = ValidateDim<1, [2, 3, 4]> // 1 (valid positive index)
 * type Invalid1 = ValidateDim<-5, [2, 3, 4]> // DimensionError (out of bounds)
 * type Invalid2 = ValidateDim<3, [2, 3, 4]> // DimensionError (out of bounds)
 */
export type ValidateDim<Dim extends number, S extends Shape> = S extends Shape
  ? NormalizeDim<Dim, S['length']>
  : DimensionError<`Cannot validate dimension for non-shape type`>;

// =============================================================================
// Reduction Operations
// =============================================================================

/**
 * Validate that all axes in an array are valid for the given shape
 * Returns the normalized axes or a DimensionError
 *
 * @example
 * type Valid = ValidateAxes<[0, -1], [2, 3, 4]> // [0, 2] (normalized)
 * type Invalid = ValidateAxes<[0, 5], [2, 3, 4]> // DimensionError (axis 5 out of bounds)
 */
export type ValidateAxes<Axes extends readonly number[], S extends Shape> = ValidateAxesHelper<
  Axes,
  S,
  readonly []
>;

type ValidateAxesHelper<
  Axes extends readonly number[],
  S extends Shape,
  Acc extends readonly number[],
> = Axes extends readonly [infer First, ...infer Rest]
  ? First extends number
    ? Rest extends readonly number[]
      ? ValidateDim<First, S> extends DimensionError<string>
        ? ValidateDim<First, S> // Propagate error
        : ValidateDim<First, S> extends number
          ? ValidateAxesHelper<Rest, S, readonly [...Acc, ValidateDim<First, S>]>
          : DimensionError<`Failed to validate axis ${First}`>
      : readonly [...Acc, ValidateDim<First, S>] // Last element
    : DimensionError<`Invalid axis type: expected number, got ${First & string}`>
  : Acc; // Empty array, return accumulated result

/**
 * Compute the output shape after reduction along specified axes
 *
 * @param InputShape - Original tensor shape
 * @param Axes - Axes to reduce (must be normalized/validated)
 * @param KeepDims - Whether to keep reduced dimensions as size 1
 *
 * @example
 * type Reduced1 = ReduceShape<[2, 3, 4], [1], false> // [2, 4] (remove axis 1)
 * type Reduced2 = ReduceShape<[2, 3, 4], [1], true>  // [2, 1, 4] (keep axis 1 as size 1)
 * type Reduced3 = ReduceShape<[2, 3, 4], [0, 2], false> // [3] (remove axes 0 and 2)
 */
export type ReduceShape<
  InputShape extends Shape,
  Axes extends readonly number[],
  KeepDims extends boolean = false,
> = KeepDims extends true
  ? ReduceShapeKeepDims<InputShape, Axes>
  : ReduceShapeRemoveDims<InputShape, Axes>;

/**
 * Helper: Compute reduction shape with keepdims=true (set reduced dims to 1)
 */
type ReduceShapeKeepDims<
  InputShape extends Shape,
  Axes extends readonly number[],
> = ReduceShapeKeepDimsHelper<InputShape, Axes, 0, readonly []>;

type ReduceShapeKeepDimsHelper<
  InputShape extends Shape,
  Axes extends readonly number[],
  CurrentIndex extends number,
  Acc extends Shape,
> = InputShape extends readonly [infer First, ...infer Rest]
  ? First extends number
    ? Rest extends Shape
      ? IsInArray<CurrentIndex, Axes> extends true
        ? ReduceShapeKeepDimsHelper<Rest, Axes, Increment<CurrentIndex>, readonly [...Acc, 1]>
        : ReduceShapeKeepDimsHelper<Rest, Axes, Increment<CurrentIndex>, readonly [...Acc, First]>
      : readonly [...Acc, First] // Last element
    : Acc
  : Acc;

/**
 * Helper: Compute reduction shape with keepdims=false (remove reduced dims)
 */
type ReduceShapeRemoveDims<
  InputShape extends Shape,
  Axes extends readonly number[],
> = ReduceShapeRemoveDimsHelper<InputShape, Axes, 0, readonly []>;

type ReduceShapeRemoveDimsHelper<
  InputShape extends Shape,
  Axes extends readonly number[],
  CurrentIndex extends number,
  Acc extends Shape,
> = InputShape extends readonly [infer First, ...infer Rest]
  ? First extends number
    ? Rest extends Shape
      ? IsInArray<CurrentIndex, Axes> extends true
        ? ReduceShapeRemoveDimsHelper<Rest, Axes, Increment<CurrentIndex>, Acc> // Skip this dimension
        : ReduceShapeRemoveDimsHelper<Rest, Axes, Increment<CurrentIndex>, readonly [...Acc, First]>
      : IsInArray<CurrentIndex, Axes> extends true
        ? Acc // Last element, skip if in axes
        : readonly [...Acc, First] // Last element, keep if not in axes
    : Acc
  : Acc;

/**
 * Check if a value is present in a readonly array
 */
type IsInArray<Value, A extends readonly unknown[]> = A extends readonly [
  infer First,
  ...infer Rest,
]
  ? Value extends First
    ? true
    : IsInArray<Value, Rest>
  : false;

/**
 * Validate reduction operation parameters
 * Returns true if valid, or a descriptive error message
 *
 * @example
 * type Valid = ValidateReduction<[2, 3, 4], [1], false> // true
 * type Invalid = ValidateReduction<[2, 3, 4], [5], false> // DimensionError
 */
export type ValidateReduction<
  InputShape extends Shape,
  Axes extends readonly number[] | undefined,
> = Axes extends undefined
  ? true // No axes means reduce all dimensions (return scalar or [1,1,1...])
  : Axes extends readonly number[]
    ? ValidateAxes<Axes, InputShape> extends DimensionError<string>
      ? ValidateAxes<Axes, InputShape> // Propagate validation error
      : ValidateAxes<Axes, InputShape> extends readonly number[]
        ? CheckDuplicateAxes<ValidateAxes<Axes, InputShape>> extends DimensionError<string>
          ? CheckDuplicateAxes<ValidateAxes<Axes, InputShape>>
          : true
        : DimensionError<`Failed to validate axes`>
    : DimensionError<`Invalid axes type: expected readonly number[] or undefined`>;

/**
 * Check for duplicate axes after normalization
 */
type CheckDuplicateAxes<Axes extends readonly number[]> =
  HasDuplicates<Axes> extends true
    ? DimensionError<`Duplicate axes found in reduction. Each axis can only appear once.`>
    : Axes;

/**
 * Check if an array has duplicate values
 */
type HasDuplicates<A extends readonly unknown[]> = HasDuplicatesHelper<A, readonly []>;

type HasDuplicatesHelper<
  A extends readonly unknown[],
  Seen extends readonly unknown[],
> = A extends readonly [infer First, ...infer Rest]
  ? IsInArray<First, Seen> extends true
    ? true // Found duplicate
    : HasDuplicatesHelper<Rest, readonly [...Seen, First]>
  : false; // No duplicates found

// =============================================================================
// Expand Operation Types
// =============================================================================

/**
 * Compute the resulting shape from an expand operation
 *
 * Rules:
 * - Can only expand dimensions of size 1
 * - Use -1 to keep existing dimension size
 * - Can add new dimensions on the left
 *
 * @example
 * type E1 = ExpandShape<[2, 1, 3], [2, 5, 3]> // [2, 5, 3]
 * type E2 = ExpandShape<[1, 3], [4, 3]> // [4, 3]
 * type E3 = ExpandShape<[3], [2, 3]> // [2, 3] - adds new dim
 */
export type ExpandShape<InputShape extends Shape, TargetShape extends readonly (number | -1)[]> =
  CanExpand<InputShape, TargetShape> extends true
    ? ResolveExpandShape<InputShape, TargetShape>
    : never;

/**
 * Resolve -1 values and compute final expanded shape
 */
type ResolveExpandShape<
  InputShape extends Shape,
  TargetShape extends readonly (number | -1)[],
> = TargetShape extends readonly []
  ? readonly []
  : Length<TargetShape> extends infer TargetLen
    ? TargetLen extends number
      ? Length<InputShape> extends infer InputLen
        ? InputLen extends number
          ? Compare<TargetLen, InputLen> extends 1 // TargetLen > InputLen
            ? ResolveWithNewDims<InputShape, TargetShape, Subtract<TargetLen, InputLen>>
            : ResolveWithoutNewDims<InputShape, TargetShape>
          : never
        : never
      : never
    : never;

/**
 * Resolve shape when target has more dimensions (adding new dims on left)
 */
type ResolveWithNewDims<
  InputShape extends Shape,
  TargetShape extends readonly (number | -1)[],
  NewDimsCount extends number,
> = NewDimsCount extends 0
  ? ResolveWithoutNewDims<InputShape, TargetShape>
  : TargetShape extends readonly [infer First, ...infer Rest]
    ? First extends number | -1
      ? Rest extends readonly (number | -1)[]
        ? First extends -1
          ? never // Can't use -1 for new dimensions
          : readonly [First, ...ResolveWithNewDims<InputShape, Rest, Decrement<NewDimsCount>>]
        : never
      : never
    : never;

/**
 * Resolve shape without adding new dimensions
 */
type ResolveWithoutNewDims<
  InputShape extends Shape,
  TargetShape extends readonly (number | -1)[],
> = InputShape extends readonly []
  ? TargetShape extends readonly []
    ? readonly []
    : never
  : InputShape extends readonly [infer InputFirst, ...infer InputRest]
    ? TargetShape extends readonly [infer TargetFirst, ...infer TargetRest]
      ? InputFirst extends number
        ? InputRest extends Shape
          ? TargetFirst extends number | -1
            ? TargetRest extends readonly (number | -1)[]
              ? TargetFirst extends -1
                ? readonly [InputFirst, ...ResolveWithoutNewDims<InputRest, TargetRest>]
                : readonly [TargetFirst, ...ResolveWithoutNewDims<InputRest, TargetRest>]
              : never
            : never
          : never
        : never
      : never
    : never;

/**
 * Validate if expansion is legal
 *
 * @example
 * type V1 = CanExpand<[2, 1, 3], [2, 5, 3]> // true
 * type V2 = CanExpand<[2, 3], [2, 5]> // false - can't expand 3 to 5
 */
export type CanExpand<InputShape extends Shape, TargetShape extends readonly (number | -1)[]> =
  Length<TargetShape> extends infer TargetLen
    ? TargetLen extends number
      ? Length<InputShape> extends infer InputLen
        ? InputLen extends number
          ? Compare<TargetLen, InputLen> extends 1 // TargetLen > InputLen
            ? CanExpandWithNewDims<InputShape, TargetShape, Subtract<TargetLen, InputLen>>
            : CanExpandSameDims<InputShape, TargetShape>
          : false
        : false
      : false
    : false;

/**
 * Check expansion when adding new dimensions
 */
type CanExpandWithNewDims<
  InputShape extends Shape,
  TargetShape extends readonly (number | -1)[],
  NewDimsCount extends number,
> = NewDimsCount extends 0
  ? CanExpandSameDims<InputShape, TargetShape>
  : TargetShape extends readonly [infer First, ...infer Rest]
    ? First extends -1
      ? false // Can't use -1 for new dimensions
      : Rest extends readonly (number | -1)[]
        ? CanExpandWithNewDims<InputShape, Rest, Decrement<NewDimsCount>>
        : false
    : false;

/**
 * Check expansion with same number of dimensions
 */
type CanExpandSameDims<
  InputShape extends Shape,
  TargetShape extends readonly (number | -1)[],
> = InputShape extends readonly []
  ? TargetShape extends readonly []
    ? true
    : false
  : InputShape extends readonly [infer InputFirst, ...infer InputRest]
    ? TargetShape extends readonly [infer TargetFirst, ...infer TargetRest]
      ? InputFirst extends number
        ? InputRest extends Shape
          ? TargetFirst extends number | -1
            ? TargetRest extends readonly (number | -1)[]
              ? TargetFirst extends -1
                ? CanExpandSameDims<InputRest, TargetRest> // -1 means keep dimension
                : InputFirst extends 1
                  ? CanExpandSameDims<InputRest, TargetRest> // Can expand from 1
                  : InputFirst extends TargetFirst
                    ? CanExpandSameDims<InputRest, TargetRest> // Same size is ok
                    : false // Can't expand non-singleton to different size
              : false
            : false
          : false
        : false
      : false
    : false;

// =============================================================================
// Tile Operation Types
// =============================================================================

/**
 * Compute the resulting shape from a tile operation
 *
 * Rules:
 * - If fewer reps than dims: tiles rightmost dimensions
 * - If more reps than dims: adds new dimensions on the left
 * - Each dimension is multiplied by its repetition count
 *
 * @example
 * type T1 = TileShape<[2, 3], [2, 3]> // [4, 9]
 * type T2 = TileShape<[3], [2, 3]> // [2, 9] - adds dim
 * type T3 = TileShape<[2, 3, 4], [2]> // [2, 3, 8] - tiles last dim
 */
export type TileShape<InputShape extends Shape, Reps extends readonly number[]> =
  Length<Reps> extends 0
    ? Readonly<InputShape> // Empty reps returns original shape as readonly
    : Length<InputShape> extends infer InputLen
      ? InputLen extends number
        ? Length<Reps> extends infer RepsLen
          ? RepsLen extends number
            ? Compare<RepsLen, InputLen> extends 1 // More reps than dims
              ? TileWithNewDims<InputShape, Reps, Subtract<RepsLen, InputLen>>
              : TileWithoutNewDims<InputShape, Reps, Subtract<InputLen, RepsLen>>
            : never
          : never
        : never
      : never;

/**
 * Tile with more reps than dimensions (adds new dims on left)
 */
type TileWithNewDims<
  InputShape extends Shape,
  Reps extends readonly number[],
  NewDimsCount extends number,
> = NewDimsCount extends 0
  ? TileWithoutNewDims<InputShape, Reps, 0>
  : Reps extends readonly [infer First, ...infer Rest]
    ? First extends number
      ? Rest extends readonly number[]
        ? readonly [First, ...TileWithNewDims<InputShape, Rest, Decrement<NewDimsCount>>]
        : never
      : never
    : never;

/**
 * Tile without adding new dimensions
 */
type TileWithoutNewDims<
  InputShape extends Shape,
  Reps extends readonly number[],
  SkipCount extends number,
> = SkipCount extends 0
  ? TileMatchingDims<InputShape, Reps>
  : InputShape extends readonly [infer First, ...infer Rest]
    ? First extends number
      ? Rest extends Shape
        ? readonly [First, ...TileWithoutNewDims<Rest, Reps, Decrement<SkipCount>>]
        : never
      : never
    : never;

/**
 * Tile with matching dimensions
 */
type TileMatchingDims<
  InputShape extends Shape,
  Reps extends readonly number[],
> = InputShape extends readonly []
  ? readonly []
  : InputShape extends readonly [infer InputFirst, ...infer InputRest]
    ? Reps extends readonly [infer RepFirst, ...infer RepRest]
      ? InputFirst extends number
        ? RepFirst extends number
          ? InputRest extends Shape
            ? RepRest extends readonly number[]
              ? readonly [Multiply<InputFirst, RepFirst>, ...TileMatchingDims<InputRest, RepRest>]
              : never
            : never
          : never
        : never
      : InputFirst extends number
        ? InputRest extends Shape
          ? readonly [InputFirst, ...TileMatchingDims<InputRest, Reps>] // No more reps, keep original
          : never
        : never
    : never;
