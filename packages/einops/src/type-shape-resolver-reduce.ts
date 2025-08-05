/**
 * Type-level shape resolver for einops reduce patterns
 *
 * This module provides compile-time shape computation for reduce operations,
 * determining the output shape based on which axes are being reduced.
 */

import type { Shape } from '@typetensor/core';
import type {
  ParsePattern,
  TypeEinopsAST,
  TypeAxisPattern,
  TypeSimpleAxis,
  TypeCompositeAxis,
  TypeEllipsisAxis,
  TypeSingletonAxis,
  TypeParseError,
} from './type-parser';
import type {
  BuildAxisMap,
  ExtractEllipsisDims,
  CountEllipsis,
  HasDuplicateAxisNames,
  Drop,
  CountConsumingAxes,
  ValidateComposites,
  AxisMap,
} from './type-shape-resolver-utils';
import type { Multiply } from 'ts-arithmetic';
import type { CollectAxisNames } from './validation';
import type { Add, Subtract, IsInt } from 'ts-arithmetic';
import type {
  ReduceParseError,
  ReduceAxisError,
  ReduceNewAxisError,
  ReduceDuplicateAxisError,
  ReduceRankMismatchError,
  ReduceCompositeResolutionError,
  ReduceFractionalDimensionError,
} from './errors';
import type { CollectAxisNamesIntersectionSafe, FindUnknownAxes } from './type-validation';

// =============================================================================
// Helper Types for Reduce
// =============================================================================

/**
 * Check if an axis from input is present in output
 */
type IsAxisReduced<
  AxisName extends string,
  OutputPatterns extends readonly TypeAxisPattern[],
> = AxisName extends CollectAxisNames<OutputPatterns>[number] ? false : true;

/**
 * Validate that all output axes exist in input
 */
type ValidateOutputAxesExist<
  OutputAxes extends readonly string[],
  InputAxes extends readonly string[],
> = OutputAxes extends readonly [infer Head, ...infer Tail]
  ? Head extends string
    ? Head extends InputAxes[number]
      ? Tail extends readonly string[]
        ? ValidateOutputAxesExist<Tail, InputAxes>
        : true
      : false // Output axis not in input
    : false
  : true;

// =============================================================================
// Enhanced Validation Types for Reduce
// =============================================================================

/**
 * Utility to check if a type is exactly never
 * This works correctly unlike [T] extends [string] which fails for never
 */
type IsNever<T> = [T] extends [never] ? true : false;

/**
 * Safer integer validation that works with computed types
 * Uses multiple fallback strategies to handle complex number types
 */
type IsSafeInteger<N extends number> =
  // First try: Direct IsInt check for literal numbers
  IsInt<N> extends 1
    ? true
    : // Second try: Check for obvious fractions using template literals
      `${N}` extends `${string}.${string}`
      ? false
      : // Third try: Accept if it's a basic number type (not obviously fractional)
        N extends number
        ? true
        : false;

/**
 * Check if a shape contains safe integer dimensions
 * This is more lenient with computed types while still catching obvious fractions
 */
type IsValidIntegerShape<S extends Shape> = S extends readonly []
  ? true // Empty shape is valid
  : S extends readonly [infer Head, ...infer Tail]
    ? Head extends number
      ? IsSafeInteger<Head> extends true
        ? Tail extends Shape
          ? IsValidIntegerShape<Tail>
          : false
        : false // Non-integer dimension found
      : false // Non-number dimension
    : false;

/**
 * Extract the first duplicate axis name from a list of patterns
 */
type ExtractFirstDuplicate<
  Patterns extends readonly TypeAxisPattern[],
  Seen extends readonly string[] = readonly [],
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends Seen[number]
      ? Head['name'] // Found first duplicate
      : Tail extends readonly TypeAxisPattern[]
        ? ExtractFirstDuplicate<Tail, readonly [...Seen, Head['name']]>
        : never
    : Head extends TypeCompositeAxis
      ? ExtractFirstDuplicate<Head['axes']> extends infer Duplicate
        ? Duplicate extends string
          ? Duplicate
          : Tail extends readonly TypeAxisPattern[]
            ? ExtractFirstDuplicate<Tail, Seen>
            : never
        : never
      : Tail extends readonly TypeAxisPattern[]
        ? ExtractFirstDuplicate<Tail, Seen>
        : never
  : never;

// =============================================================================
// Shape Computation for Reduce
// =============================================================================

/**
 * Build output shape for reduce operation
 */
export type BuildReduceShape<
  InputPatterns extends readonly TypeAxisPattern[],
  OutputPatterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  AxisMap extends Record<string, number>,
  EllipsisDims extends Shape,
  KeepDims extends boolean,
> = KeepDims extends true
  ? BuildReduceShapeKeepDims<InputPatterns, OutputPatterns, InputShape, AxisMap, EllipsisDims>
  : ComputeOutputShapeFromPatterns<OutputPatterns, AxisMap, EllipsisDims>;

/**
 * Build shape when keepDims is true (preserve input structure with 1s for reduced dims)
 */
type BuildReduceShapeKeepDims<
  InputPatterns extends readonly TypeAxisPattern[],
  OutputPatterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  AxisMap extends Record<string, number>,
  EllipsisDims extends Shape,
  CurrentIndex extends number = 0,
  Result extends Shape = readonly [],
> = InputPatterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? IsAxisReduced<Head['name'], OutputPatterns> extends true
      ? Tail extends readonly TypeAxisPattern[]
        ? BuildReduceShapeKeepDims<
            Tail,
            OutputPatterns,
            InputShape,
            AxisMap,
            EllipsisDims,
            Add<CurrentIndex, 1>,
            readonly [...Result, 1]
          >
        : readonly [...Result, 1]
      : CurrentIndex extends keyof InputShape
        ? InputShape[CurrentIndex] extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? BuildReduceShapeKeepDims<
                Tail,
                OutputPatterns,
                InputShape,
                AxisMap,
                EllipsisDims,
                Add<CurrentIndex, 1>,
                readonly [...Result, InputShape[CurrentIndex]]
              >
            : readonly [...Result, InputShape[CurrentIndex]]
          : never
        : never
    : Head extends TypeCompositeAxis
      ? ProcessCompositeForReduceKeepDims<
          Head,
          OutputPatterns,
          InputShape,
          CurrentIndex,
          AxisMap
        > extends infer CompositeResult
        ? CompositeResult extends { dims: Shape; nextIndex: number }
          ? Tail extends readonly TypeAxisPattern[]
            ? BuildReduceShapeKeepDims<
                Tail,
                OutputPatterns,
                InputShape,
                AxisMap,
                EllipsisDims,
                CompositeResult['nextIndex'],
                readonly [...Result, ...CompositeResult['dims']]
              >
            : readonly [...Result, ...CompositeResult['dims']]
          : never
        : never
      : Head extends TypeEllipsisAxis
        ? ProcessEllipsisForReduceKeepDims<
            OutputPatterns,
            EllipsisDims
          > extends infer EllipsisResult
          ? EllipsisResult extends Shape
            ? Tail extends readonly TypeAxisPattern[]
              ? CountConsumingAxes<Tail> extends infer TailCount
                ? TailCount extends number
                  ? InputShape['length'] extends number
                    ? BuildReduceShapeKeepDims<
                        Tail,
                        OutputPatterns,
                        Drop<InputShape, Subtract<InputShape['length'], TailCount>>,
                        AxisMap,
                        EllipsisDims,
                        0,
                        readonly [...Result, ...EllipsisResult]
                      >
                    : never
                  : never
                : never
              : readonly [...Result, ...EllipsisResult]
            : never
          : never
        : Head extends TypeSingletonAxis
          ? Tail extends readonly TypeAxisPattern[]
            ? BuildReduceShapeKeepDims<
                Tail,
                OutputPatterns,
                InputShape,
                AxisMap,
                EllipsisDims,
                Add<CurrentIndex, 1>,
                readonly [...Result, 1]
              >
            : readonly [...Result, 1]
          : never
  : Result;

/**
 * Compute output shape directly from output patterns (for keepDims=false)
 */
type ComputeOutputShapeFromPatterns<
  OutputPatterns extends readonly TypeAxisPattern[],
  AxisMap extends Record<string, number>,
  EllipsisDims extends Shape,
  Result extends Shape = readonly [],
> = OutputPatterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof AxisMap
      ? AxisMap[Head['name']] extends number
        ? Tail extends readonly TypeAxisPattern[]
          ? ComputeOutputShapeFromPatterns<
              Tail,
              AxisMap,
              EllipsisDims,
              readonly [...Result, AxisMap[Head['name']]]
            >
          : readonly [...Result, AxisMap[Head['name']]]
        : never
      : never
    : Head extends TypeCompositeAxis
      ? ComputeCompositeOutputDim<Head['axes'], AxisMap> extends infer CompDim
        ? CompDim extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? ComputeOutputShapeFromPatterns<
                Tail,
                AxisMap,
                EllipsisDims,
                readonly [...Result, CompDim]
              >
            : readonly [...Result, CompDim]
          : never
        : never
      : Head extends TypeEllipsisAxis
        ? Tail extends readonly TypeAxisPattern[]
          ? ComputeOutputShapeFromPatterns<
              Tail,
              AxisMap,
              EllipsisDims,
              readonly [...Result, ...EllipsisDims]
            >
          : readonly [...Result, ...EllipsisDims]
        : Head extends TypeSingletonAxis
          ? Tail extends readonly TypeAxisPattern[]
            ? ComputeOutputShapeFromPatterns<Tail, AxisMap, EllipsisDims, readonly [...Result, 1]>
            : readonly [...Result, 1]
          : never
  : Result;

/**
 * Compute dimension for composite axis in output
 */
type ComputeCompositeOutputDim<
  Axes extends readonly TypeAxisPattern[],
  AxisMap extends Record<string, number>,
  Product extends number = 1,
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof AxisMap
      ? AxisMap[Head['name']] extends number
        ? Tail extends readonly TypeAxisPattern[]
          ? ComputeCompositeOutputDim<Tail, AxisMap, Multiply<Product, AxisMap[Head['name']]>>
          : Multiply<Product, AxisMap[Head['name']]>
        : never
      : never
    : Head extends TypeCompositeAxis
      ? ComputeCompositeOutputDim<Head['axes'], AxisMap> extends infer InnerProd
        ? InnerProd extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? ComputeCompositeOutputDim<Tail, AxisMap, Multiply<Product, InnerProd>>
            : Multiply<Product, InnerProd>
          : never
        : never
      : never
  : Product;

/**
 * Process composite axis for reduction with keepDims
 */
type ProcessCompositeForReduceKeepDims<
  Composite extends TypeCompositeAxis,
  OutputPatterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  CurrentIndex extends number,
  AxisMap extends Record<string, number>,
> = CurrentIndex extends keyof InputShape
  ? InputShape[CurrentIndex] extends number
    ? ComputeCompositeReduceShapeKeepDims<
        Composite['axes'],
        OutputPatterns,
        InputShape[CurrentIndex],
        AxisMap
      > extends infer CompositeDims
      ? CompositeDims extends Shape
        ? { dims: CompositeDims; nextIndex: Add<CurrentIndex, 1> }
        : never
      : never
    : never
  : never;

/**
 * Compute shape for composite pattern in reduction with keepDims
 */
type ComputeCompositeReduceShapeKeepDims<
  Axes extends readonly TypeAxisPattern[],
  OutputPatterns extends readonly TypeAxisPattern[],
  TotalDim extends number,
  AxisMap extends Record<string, number>,
  Result extends Shape = readonly [],
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? IsAxisReduced<Head['name'], OutputPatterns> extends true
      ? Tail extends readonly TypeAxisPattern[]
        ? ComputeCompositeReduceShapeKeepDims<
            Tail,
            OutputPatterns,
            TotalDim,
            AxisMap,
            readonly [...Result, 1]
          >
        : readonly [...Result, 1]
      : Head['name'] extends keyof AxisMap
        ? AxisMap[Head['name']] extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? ComputeCompositeReduceShapeKeepDims<
                Tail,
                OutputPatterns,
                TotalDim,
                AxisMap,
                readonly [...Result, AxisMap[Head['name']]]
              >
            : readonly [...Result, AxisMap[Head['name']]]
          : never
        : never
    : never
  : Result;

/**
 * Process ellipsis for reduction with keepDims
 */
type ProcessEllipsisForReduceKeepDims<
  OutputPatterns extends readonly TypeAxisPattern[],
  EllipsisDims extends Shape,
> =
  HasEllipsis<OutputPatterns> extends true
    ? EllipsisDims // Ellipsis preserved in output
    : CreateOnesShape<EllipsisDims['length']>; // All ellipsis dims become 1

/**
 * Check if pattern has ellipsis
 */
type HasEllipsis<Patterns extends readonly TypeAxisPattern[]> = Patterns extends readonly [
  infer Head,
  ...infer Tail,
]
  ? Head extends TypeEllipsisAxis
    ? true
    : Tail extends readonly TypeAxisPattern[]
      ? HasEllipsis<Tail>
      : false
  : false;

/**
 * Create a shape of all 1s with given length
 */
type CreateOnesShape<
  Length extends number,
  Result extends Shape = readonly [],
  Counter extends readonly unknown[] = readonly [],
> = Counter['length'] extends Length
  ? Result
  : CreateOnesShape<Length, readonly [...Result, 1], readonly [...Counter, unknown]>;

// =============================================================================
// Validation for Reduce Patterns
// =============================================================================

/**
 * Validate reduce pattern with inferred types
 */
export type ValidateReducePatternInferred<
  InputPatterns extends readonly TypeAxisPattern[],
  OutputPatterns extends readonly TypeAxisPattern[],
> =
  // Check no duplicate axes
  HasDuplicateAxisNames<InputPatterns> extends true
    ? { valid: false; error: 'Duplicate axes in input pattern' }
    : HasDuplicateAxisNames<OutputPatterns> extends true
      ? { valid: false; error: 'Duplicate axes in output pattern' }
      : // Check at most one ellipsis
        CountEllipsis<InputPatterns> extends 0 | 1
        ? CountEllipsis<OutputPatterns> extends 0 | 1
          ? // Check all output axes exist in input
            ValidateOutputAxesExist<
              CollectAxisNames<OutputPatterns>,
              CollectAxisNames<InputPatterns>
            > extends true
            ? { valid: true } // Allow patterns where nothing is reduced (like einops)
            : { valid: false; error: 'Output contains axes not present in input' }
          : { valid: false; error: 'Multiple ellipsis in output' }
        : { valid: false; error: 'Multiple ellipsis in input' };

/**
 * Validate reduce pattern (legacy interface version)
 */
export type ValidateReducePattern<AST extends TypeEinopsAST> =
  // Check no duplicate axes
  HasDuplicateAxisNames<AST['input']> extends true
    ? { valid: false; error: 'Duplicate axes in input pattern' }
    : HasDuplicateAxisNames<AST['output']> extends true
      ? { valid: false; error: 'Duplicate axes in output pattern' }
      : // Check at most one ellipsis
        CountEllipsis<AST['input']> extends 0 | 1
        ? CountEllipsis<AST['output']> extends 0 | 1
          ? // Check all output axes exist in input
            ValidateOutputAxesExist<
              CollectAxisNames<AST['output']>,
              CollectAxisNames<AST['input']>
            > extends true
            ? { valid: true } // Allow patterns where nothing is reduced (like einops)
            : { valid: false; error: 'Output contains axes not present in input' }
          : { valid: false; error: 'Multiple ellipsis in output' }
        : { valid: false; error: 'Multiple ellipsis in input' };

// =============================================================================
// Main Resolver for Reduce
// =============================================================================

export type ValidateAndComputeOutput<
  AST extends TypeEinopsAST,
  AxisMapping extends AxisMap,
  EllipsisDims extends Shape,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
  KeepDims extends boolean,
> =
  ValidateComposites<AST['input'], InputShape, Axes> extends true
    ? BuildReduceShape<AST['input'], AST['output'], InputShape, AxisMapping, EllipsisDims, KeepDims>
    : never;

/**
 * Resolve reduce pattern to output shape
 */
export type ResolveReduceShape<
  Pattern extends string,
  InputShape extends Shape,
  KeepDims extends boolean = false,
  Axes extends Record<string, number> | undefined = undefined,
> =
  ParsePattern<Pattern> extends infer ParsedAST
    ? ParsedAST extends { input: infer InputPatterns; output: infer OutputPatterns }
      ? InputPatterns extends readonly TypeAxisPattern[]
        ? OutputPatterns extends readonly TypeAxisPattern[]
          ? ValidateReducePatternInferred<InputPatterns, OutputPatterns> extends { valid: true }
            ? BuildAxisMap<InputPatterns, InputShape, Axes> extends infer AxisMapping
              ? AxisMapping extends Record<string, number>
                ? ExtractEllipsisDims<InputPatterns, InputShape> extends infer EllipsisDims
                  ? EllipsisDims extends Shape
                    ? // Validate composite patterns have correct dimensions
                      ValidateComposites<InputPatterns, InputShape, Axes> extends true
                      ? BuildReduceShape<
                          InputPatterns,
                          OutputPatterns,
                          InputShape,
                          AxisMapping,
                          EllipsisDims,
                          KeepDims
                        >
                      : never
                    : never
                  : never
                : never
              : never
            : ValidateReducePatternInferred<InputPatterns, OutputPatterns> extends {
                  valid: false;
                  error: infer E;
                }
              ? never & { __error: E }
              : never
          : never
        : never
      : ParsedAST extends TypeParseError<infer E>
        ? never & { __error: E }
        : never
    : never;

// =============================================================================
// Enhanced Reduce Pattern Validation - Main Entry Point
// =============================================================================

/**
 * Validate reduce pattern and return output shape or specific error
 *
 * Enhanced validation with progressive error detection and specific error messages
 * - Quick syntax validation (missing arrow, empty patterns)
 * - Progressive validation chain with targeted error messages
 * - Returns Shape on success, specific branded error string on failure
 *
 * @example
 * type Valid = ValidReducePattern<'h w c -> c', [2, 3, 4]>; // [4]
 * type NoArrow = ValidReducePattern<'h w c', [2, 3, 4]>; // Parse error
 * type NewAxis = ValidReducePattern<'h w -> h w c', [2, 3]>; // Axis error
 * type DuplicateAxis = ValidReducePattern<'h h -> h', [2, 3]>; // Axis error
 *
 * @param Pattern - The reduce pattern string
 * @param InputShape - The input tensor shape
 * @param KeepDims - Whether to keep reduced dimensions as size 1
 * @param Axes - Optional axis dimension specifications
 */
export type ValidReducePattern<
  Pattern extends string,
  InputShape extends Shape,
  KeepDims extends boolean = false,
  Axes extends Record<string, number> | undefined = undefined,
> =
  // Step 1: Quick syntax validation (like ValidEinopsPattern)
  Pattern extends `${infer Input}->${infer Output}`
    ? // Step 2: Check for empty input/output, but allow them for scalar operations
      Input extends '' | ' ' | `  ${string}` | `${string}  `
      ? // Empty input is only valid for scalar tensors (shape [])
        InputShape extends readonly []
        ? ValidateReducePatternStructure<Pattern, InputShape, KeepDims, Axes>
        : ReduceParseError<"Empty input pattern. Specify input axes before '->'">
      : Output extends '' | ' ' | `  ${string}` | `${string}  `
        ? // Empty output is valid for reduce operations (global reduction)
          ValidateReducePatternStructure<Pattern, InputShape, KeepDims, Axes>
        : // Step 3: Progressive validation chain
          ValidateReducePatternStructure<Pattern, InputShape, KeepDims, Axes>
    : // No arrow operator found
      ReduceParseError<"Missing arrow operator '->'. Pattern must be 'input -> output'">;

/**
 * Simplified validation chain for reduce operations
 * Uses lightweight validation before delegating to existing system
 */
type ValidateReducePatternStructure<
  Pattern extends string,
  InputShape extends Shape,
  KeepDims extends boolean,
  Axes extends Record<string, number> | undefined,
> =
  // Try the existing resolver first - it handles most cases correctly
  ResolveReduceShape<Pattern, InputShape, KeepDims, Axes> extends infer Result
    ? [Result] extends [never]
      ? // Only when existing system fails, provide specific error detection
        DetectReduceSpecificError<Pattern, InputShape, KeepDims, Axes>
      : Result extends Shape
        ? IsValidIntegerShape<Result> extends true
          ? Result // Valid integer shape
          : ReduceFractionalDimensionError<Pattern> // Invalid fractional dimensions
        : DetectReduceSpecificError<Pattern, InputShape, KeepDims, Axes>
    : DetectReduceSpecificError<Pattern, InputShape, KeepDims, Axes>;

/**
 * Detect specific errors when the existing system returns never
 * This provides targeted error messages for common failure patterns
 */
export type DetectReduceSpecificError<
  Pattern extends string,
  InputShape extends Shape,
  _KeepDims extends boolean,
  Axes extends Record<string, number> | undefined,
> =
  // Parse the pattern to analyze what went wrong
  ParsePattern<Pattern> extends infer ParsedAST
    ? ParsedAST extends TypeEinopsAST
      ? // Check for specific error patterns using IsNever to distinguish never from strings
        IsNever<CheckForReduceDuplicateAxes<ParsedAST>> extends true
        ? // No duplicate error, check for multiple ellipsis
          IsNever<CheckForReduceMultipleEllipsis<ParsedAST>> extends true
          ? // No ellipsis error, check for new axes in output
            IsNever<CheckForReduceNewAxes<ParsedAST>> extends true
            ? // No new axis error, check for rank mismatch
              IsNever<CheckForReduceRankMismatch<ParsedAST, InputShape>> extends true
              ? // No rank error, check for composite errors
                CheckForReduceCompositeErrors<
                  ParsedAST,
                  InputShape,
                  Axes
                > extends infer CompositeError
                ? IsNever<CompositeError> extends true
                  ? ReduceParseError<'Pattern validation failed'> // Generic fallback
                  : CompositeError // Return specific composite error
                : ReduceParseError<'Pattern validation failed'>
              : CheckForReduceRankMismatch<ParsedAST, InputShape> // Return specific rank mismatch error
            : CheckForReduceNewAxes<ParsedAST> // Return specific new axis error
          : CheckForReduceMultipleEllipsis<ParsedAST> // Return specific multiple ellipsis error
        : CheckForReduceDuplicateAxes<ParsedAST> // Return specific duplicate axis error
      : ParsedAST extends TypeParseError<infer ErrorMsg>
        ? ReduceParseError<ErrorMsg>
        : ReduceParseError<'Pattern parsing failed'>
    : ReduceParseError<'Pattern parsing failed'>;

// =============================================================================
// Specific Error Detection Functions for Reduce
// =============================================================================

/**
 * Check for duplicate axes in input or output patterns
 */
export type CheckForReduceDuplicateAxes<AST extends TypeEinopsAST> =
  HasDuplicateAxisNames<AST['input']> extends true
    ? ReduceDuplicateAxisError<ExtractFirstDuplicate<AST['input']>, 'input'>
    : HasDuplicateAxisNames<AST['output']> extends true
      ? ReduceDuplicateAxisError<ExtractFirstDuplicate<AST['output']>, 'output'>
      : never; // No duplicates found

/**
 * Check for new axes in output that don't exist in input (invalid for reduce)
 */
export type CheckForReduceNewAxes<AST extends TypeEinopsAST> =
  FindUnknownAxes<
    CollectAxisNamesIntersectionSafe<AST['input']>,
    CollectAxisNamesIntersectionSafe<AST['output']>
  > extends readonly [infer FirstNew, ...any[]]
    ? FirstNew extends string
      ? ReduceNewAxisError<FirstNew, CollectAxisNamesIntersectionSafe<AST['input']>>
      : never
    : never; // No new axes found

/**
 * Check for rank mismatch between pattern and input shape
 */
export type CheckForReduceRankMismatch<AST extends TypeEinopsAST, InputShape extends Shape> =
  CountConsumingAxes<AST['input']> extends infer ExpectedRank
    ? ExpectedRank extends number
      ? InputShape['length'] extends ExpectedRank
        ? never // Ranks match
        : ReduceRankMismatchError<ExpectedRank, InputShape['length']>
      : never
    : never;

/**
 * Check for multiple ellipsis patterns (only one ellipsis per side allowed)
 */
export type CheckForReduceMultipleEllipsis<AST extends TypeEinopsAST> =
  CountEllipsis<AST['input']> extends infer InputEllipsisCount
    ? InputEllipsisCount extends number
      ? InputEllipsisCount extends 0 | 1
        ? // Input ellipsis count is valid, check output
          CountEllipsis<AST['output']> extends infer OutputEllipsisCount
          ? OutputEllipsisCount extends number
            ? OutputEllipsisCount extends 0 | 1
              ? never // Both sides have valid ellipsis counts
              : ReduceAxisError<"Multiple ellipsis '...' in output. Only one ellipsis allowed per side">
            : never
          : never
        : ReduceAxisError<"Multiple ellipsis '...' in input. Only one ellipsis allowed per side">
      : never
    : never;

/**
 * Check for composite pattern resolution errors
 */
type CheckForReduceCompositeErrors<
  AST extends TypeEinopsAST,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
> = CheckReduceCompositePatterns<AST['input'], InputShape, Axes>;

/**
 * Check composite patterns in input for resolution errors
 */
type CheckReduceCompositePatterns<
  Patterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
  CurrentIndex extends number = 0,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeCompositeAxis
    ? // Found a composite pattern - check if it can be resolved
      CurrentIndex extends keyof InputShape
      ? InputShape[CurrentIndex] extends number
        ? CheckReduceCompositeResolution<
            Head,
            InputShape[CurrentIndex],
            Axes
          > extends infer CompositeError
          ? CompositeError extends string
            ? CompositeError // Return the composite error
            : Tail extends readonly TypeAxisPattern[]
              ? CheckReduceCompositePatterns<Tail, InputShape, Axes, Add<CurrentIndex, 1>>
              : never
          : never
        : never
      : never
    : // Not a composite, continue to next pattern
      Tail extends readonly TypeAxisPattern[]
      ? CheckReduceCompositePatterns<Tail, InputShape, Axes, Add<CurrentIndex, 1>>
      : never
  : never; // No composite patterns found

/**
 * Check if a specific composite pattern can be resolved for reduce operations
 */
type CheckReduceCompositeResolution<
  Composite extends TypeCompositeAxis,
  Dimension extends number,
  Axes extends Record<string, number> | undefined,
> =
  // Use the same logic as rearrange but with reduce-specific error messages
  Axes extends Record<string, number>
    ? // Has provided axes - validate they work
      ReduceCompositeResolutionError<FormatReduceCompositePattern<Composite>, Dimension>
    : // No provided axes - need at least some for complex composites
      ReduceCompositeResolutionError<FormatReduceCompositePattern<Composite>, Dimension>;

/**
 * Format composite pattern for reduce error messages
 */
type FormatReduceCompositePattern<Composite extends TypeCompositeAxis> =
  `(${FormatReduceAxesInComposite<Composite['axes']>})`;

/**
 * Format the axes inside a composite pattern for reduce operations
 */
type FormatReduceAxesInComposite<
  Axes extends readonly TypeAxisPattern[],
  Result extends string = '',
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? Tail extends readonly []
        ? `${Result}${Head['name']}` // Last axis, no space after
        : FormatReduceAxesInComposite<Tail, `${Result}${Head['name']} `> // Add space after
      : `${Result}${Head['name']}`
    : Result // Skip non-simple axes for now
  : Result;
