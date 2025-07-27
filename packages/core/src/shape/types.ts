/**
 * Type-level shape operations for compile-time tensor shape validation
 *
 * This module provides TypeScript types for representing and manipulating
 * tensor shapes at the type level, enabling compile-time shape checking
 * and inference for tensor operations.
 */

import type { Multiply as TSMultiply } from 'ts-arithmetic';

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
 * Type-level multiplication using ts-arithmetic for precise calculations
 * Supports arbitrary precision arithmetic up to 1e+21 without recursion limits
 */
type Multiply<A extends number, B extends number> = TSMultiply<A, B>;

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
 * Type-level subtraction (limited to small numbers)
 */
type Subtract<A extends number, B extends number> = [
  never,
  0,
  1,
  2,
  3,
  4,
  5,
  6,
  7,
  8,
  9,
  10,
][A extends 0 ? never : A] extends infer Result
  ? Result extends number
    ? B extends 0
      ? A
      : B extends 1
        ? A extends 1
          ? 0
          : A extends 2
            ? 1
            : A extends 3
              ? 2
              : A extends 4
                ? 3
                : number
        : number
    : never
  : never;

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
