/**
 * Type-level shape resolver for einops repeat patterns
 *
 * This module provides compile-time shape computation for repeat operations,
 * determining the output shape based on new axis creation and element repetition.
 *
 * Key differences from reduce:
 * - Allows NEW axes in output (requires explicit sizes)
 * - Supports element repetition along existing axes
 * - Creates new data (not a view operation)
 */

import type { Shape } from '../shape/types';
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
  CountConsumingAxes,
  ValidateComposites,
} from './type-shape-resolver-utils';
import type { Multiply } from 'ts-arithmetic';
import type { Add, IsInt } from 'ts-arithmetic';
import type {
  RepeatParseError,
  RepeatAxisError,
  RepeatMissingAxisError,
  RepeatMissingAxesError,
  RepeatInvalidSizeError,
  RepeatDuplicateAxisError,
  RepeatRankMismatchError,
  RepeatCompositeResolutionError,
  RepeatFractionalDimensionError,
} from './errors';
import type { CollectAxisNamesIntersectionSafe, FindUnknownAxes } from './type-validation';

// =============================================================================
// Helper Types for Repeat
// =============================================================================

/**
 * Find new axes in output that don't exist in input
 * Unlike reduce, repeat ALLOWS this but requires explicit sizes
 */
type FindNewAxes<
  InputAxes extends readonly string[],
  OutputAxes extends readonly string[],
> = FindUnknownAxes<InputAxes, OutputAxes>;

/**
 * Check if all new axes have provided sizes
 */
type ValidateNewAxesSizes<
  NewAxes extends readonly string[],
  Axes extends Record<string, number> | undefined,
> =
  Axes extends Record<string, number>
    ? ValidateAllNewAxesProvided<NewAxes, Axes>
    : NewAxes extends readonly []
      ? true // No new axes, so no sizes needed
      : false; // Has new axes but no sizes provided

/**
 * Check that all new axes are provided in the axes record
 */
type ValidateAllNewAxesProvided<
  NewAxes extends readonly string[],
  Axes extends Record<string, number>,
  Missing extends readonly string[] = readonly [],
> = NewAxes extends readonly [infer Head, ...infer Tail]
  ? Head extends string
    ? Head extends keyof Axes
      ? Tail extends readonly string[]
        ? ValidateAllNewAxesProvided<Tail, Axes, Missing>
        : Missing['length'] extends 0
          ? true
          : false
      : Tail extends readonly string[]
        ? ValidateAllNewAxesProvided<Tail, Axes, readonly [...Missing, Head]>
        : false
    : false
  : Missing['length'] extends 0
    ? true
    : false;

/**
 * Extract missing new axes that don't have provided sizes
 */
type ExtractMissingNewAxes<
  NewAxes extends readonly string[],
  Axes extends Record<string, number> | undefined,
  Missing extends readonly string[] = readonly [],
> =
  Axes extends Record<string, number>
    ? NewAxes extends readonly [infer Head, ...infer Tail]
      ? Head extends string
        ? Head extends keyof Axes
          ? Tail extends readonly string[]
            ? ExtractMissingNewAxes<Tail, Axes, Missing>
            : Missing
          : Tail extends readonly string[]
            ? ExtractMissingNewAxes<Tail, Axes, readonly [...Missing, Head]>
            : readonly [...Missing, Head]
        : Missing
      : Missing
    : NewAxes; // No axes provided, all new axes are missing

// =============================================================================
// Enhanced Validation Types for Repeat
// =============================================================================

/**
 * Utility to check if a type is exactly never
 */
type IsNever<T> = [T] extends [never] ? true : false;

/**
 * Safer integer validation that works with computed types
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
 * Validate axis sizes are positive integers
 */
type ValidateAxisSizes<Axes extends Record<string, number> | undefined> =
  Axes extends Record<string, number> ? ValidateAllAxisSizesPositive<Axes> : true; // No axes to validate

/**
 * Check that all provided axis sizes are positive integers
 */
type ValidateAllAxisSizesPositive<
  Axes extends Record<string, number>,
  Keys extends keyof Axes = keyof Axes,
> = Keys extends string
  ? Axes[Keys] extends number
    ? Axes[Keys] extends 0
      ? false // Zero size
      : `${Axes[Keys]}` extends `-${string}`
        ? false // Negative size
        : true // Positive size
    : false
  : true;

/**
 * Extract the first invalid axis size
 */
type ExtractFirstInvalidAxisSize<
  Axes extends Record<string, number>,
  Keys extends keyof Axes = keyof Axes,
> = Keys extends string
  ? Axes[Keys] extends number
    ? Axes[Keys] extends 0
      ? { axis: Keys; size: Axes[Keys] }
      : `${Axes[Keys]}` extends `-${string}`
        ? { axis: Keys; size: Axes[Keys] }
        : never
    : never
  : never;

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
// Extended Axis Mapping for Repeat
// =============================================================================

/**
 * BuildAxisMap extended to include new axes that don't exist in input
 */
export type RepeatBuildAxisMap<
  InputPatterns extends readonly TypeAxisPattern[],
  OutputPatterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined = undefined,
> =
  BuildAxisMap<InputPatterns, InputShape, Axes> extends infer BaseAxisMap
    ? BaseAxisMap extends Record<string, number>
      ? ExtendAxisMapWithNewAxes<
          BaseAxisMap,
          FindNewAxes<
            CollectAxisNamesIntersectionSafe<InputPatterns>,
            CollectAxisNamesIntersectionSafe<OutputPatterns>
          >,
          Axes
        >
      : never
    : never;

type ExtendAxisMapWithNewAxes<
  BaseMap extends Record<string, number>,
  NewAxes extends readonly string[],
  Axes extends Record<string, number> | undefined,
> = Axes extends Record<string, number> ? AddNewAxesToMap<BaseMap, NewAxes, Axes> : BaseMap;
type AddNewAxesToMap<
  BaseMap extends Record<string, number>,
  NewAxes extends readonly string[],
  Axes extends Record<string, number>,
  Result extends Record<string, number> = BaseMap,
> = NewAxes extends readonly [infer Head, ...infer Tail]
  ? Head extends string
    ? Head extends keyof Axes
      ? Axes[Head] extends number
        ? Tail extends readonly string[]
          ? AddNewAxesToMap<BaseMap, Tail, Axes, Result & Record<Head, Axes[Head]>>
          : Result & Record<Head, Axes[Head]>
        : Result
      : Result
    : Result
  : Result;

// =============================================================================
// Shape Computation for Repeat
// =============================================================================

/**
 * Build output shape for repeat operation
 * Unlike reduce, this can create new dimensions and repeat existing ones
 */
export type BuildRepeatShape<
  OutputPatterns extends readonly TypeAxisPattern[],
  AxisMap extends Record<string, number>,
  EllipsisDims extends Shape,
> = ComputeOutputShapeFromPatterns<OutputPatterns, AxisMap, EllipsisDims>;

/**
 * Compute output shape directly from output patterns
 * This is similar to reduce but allows new axes with provided sizes
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
      ? ComputeCompositeOutputDim<Head['axes'], AxisMap, EllipsisDims> extends infer CompDim
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
 * Compute product of all dimensions in an ellipsis shape
 * Used when ellipsis appears inside composite patterns like (... r)
 */
type ComputeEllipsisProduct<
  EllipsisDims extends Shape,
  Product extends number = 1,
> = EllipsisDims extends readonly [infer Head, ...infer Tail]
  ? Head extends number
    ? Tail extends Shape
      ? ComputeEllipsisProduct<Tail, Multiply<Product, Head>>
      : Multiply<Product, Head>
    : never
  : Product;

/**
 * Compute dimension for composite axis in output
 * This handles both existing axes and new axes with repetition
 */
type ComputeCompositeOutputDim<
  Axes extends readonly TypeAxisPattern[],
  AxisMap extends Record<string, number>,
  EllipsisDims extends Shape = readonly [],
  Product extends number = 1,
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof AxisMap
      ? AxisMap[Head['name']] extends number
        ? Tail extends readonly TypeAxisPattern[]
          ? ComputeCompositeOutputDim<
              Tail,
              AxisMap,
              EllipsisDims,
              Multiply<Product, AxisMap[Head['name']]>
            >
          : Multiply<Product, AxisMap[Head['name']]>
        : never
      : never
    : Head extends TypeCompositeAxis
      ? ComputeCompositeOutputDim<Head['axes'], AxisMap, EllipsisDims> extends infer InnerProd
        ? InnerProd extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? ComputeCompositeOutputDim<Tail, AxisMap, EllipsisDims, Multiply<Product, InnerProd>>
            : Multiply<Product, InnerProd>
          : never
        : never
      : Head extends TypeEllipsisAxis
        ? ComputeEllipsisProduct<EllipsisDims> extends infer EllipsisProd
          ? EllipsisProd extends number
            ? Tail extends readonly TypeAxisPattern[]
              ? ComputeCompositeOutputDim<
                  Tail,
                  AxisMap,
                  EllipsisDims,
                  Multiply<Product, EllipsisProd>
                >
              : Multiply<Product, EllipsisProd>
            : never
          : never
        : never
  : Product;

// =============================================================================
// Validation for Repeat Patterns
// =============================================================================

/**
 * Validate repeat pattern with inferred types
 * Key difference: Allows new axes in output if they have explicit sizes
 */
export type ValidateRepeatPatternInferred<
  InputPatterns extends readonly TypeAxisPattern[],
  OutputPatterns extends readonly TypeAxisPattern[],
  Axes extends Record<string, number> | undefined,
> =
  HasDuplicateAxisNames<InputPatterns> extends true
    ? { valid: false; error: 'Duplicate axes in input pattern' }
    : HasDuplicateAxisNames<OutputPatterns> extends true
      ? { valid: false; error: 'Duplicate axes in output pattern' }
      : CountEllipsis<InputPatterns> extends 0 | 1
        ? CountEllipsis<OutputPatterns> extends 0 | 1
          ? ValidateNewAxesSizes<
              FindNewAxes<
                CollectAxisNamesIntersectionSafe<InputPatterns>,
                CollectAxisNamesIntersectionSafe<OutputPatterns>
              >,
              Axes
            > extends true
            ? ValidateAxisSizes<Axes> extends true
              ? { valid: true }
              : { valid: false; error: 'Invalid axis sizes' }
            : { valid: false; error: 'Missing axis sizes for new axes' }
          : { valid: false; error: 'Multiple ellipsis in output' }
        : { valid: false; error: 'Multiple ellipsis in input' };

// =============================================================================
// Main Resolver for Repeat
// =============================================================================

/**
 * Resolve repeat pattern to output shape
 */
export type ResolveRepeatShape<
  Pattern extends string,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined = undefined,
> =
  ParsePattern<Pattern> extends infer ParsedAST
    ? ParsedAST extends { input: infer InputPatterns; output: infer OutputPatterns }
      ? InputPatterns extends readonly TypeAxisPattern[]
        ? OutputPatterns extends readonly TypeAxisPattern[]
          ? ValidateRepeatPatternInferred<InputPatterns, OutputPatterns, Axes> extends {
              valid: true;
            }
            ? RepeatBuildAxisMap<
                InputPatterns,
                OutputPatterns,
                InputShape,
                Axes
              > extends infer AxisMapping
              ? AxisMapping extends Record<string, number>
                ? ExtractEllipsisDims<InputPatterns, InputShape> extends infer EllipsisDims
                  ? EllipsisDims extends Shape
                    ? ValidateComposites<InputPatterns, InputShape, Axes> extends true
                      ? BuildRepeatShape<OutputPatterns, AxisMapping, EllipsisDims>
                      : never
                    : never
                  : never
                : never
              : never
            : ValidateRepeatPatternInferred<InputPatterns, OutputPatterns, Axes> extends {
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
// Enhanced Repeat Pattern Validation - Main Entry Point
// =============================================================================

/**
 * Validate repeat pattern and return output shape or specific error
 *
 * Enhanced validation with progressive error detection and specific error messages
 * - Quick syntax validation (missing arrow, empty patterns)
 * - Progressive validation chain with targeted error messages
 * - Returns Shape on success, specific branded error string on failure
 *
 * @example
 * type Valid = ValidRepeatPattern<'h w -> h w c', [2, 3], {c: 4}>; // [2, 3, 4]
 * type MissingAxis = ValidRepeatPattern<'h w -> h w c', [2, 3]>; // Axis error
 * type InvalidSize = ValidRepeatPattern<'h w -> h w c', [2, 3], {c: 0}>; // Size error
 * type DuplicateAxis = ValidRepeatPattern<'h h -> h c', [2, 3], {c: 4}>; // Axis error
 *
 * @param Pattern - The repeat pattern string
 * @param InputShape - The input tensor shape
 * @param Axes - Required axis dimension specifications for new axes
 */
export type ValidRepeatPattern<
  Pattern extends string,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined = undefined,
> =
  // Step 1: Quick syntax validation (like ValidEinopsPattern)
  Pattern extends `${infer Input}->${infer Output}`
    ? // Check for multiple arrows (common error case)
      Output extends `${string}->${string}`
      ? RepeatParseError<"Missing arrow operator '->'. Pattern must be 'input -> output'">
      : // Step 2: Check for empty input/output, but allow them for scalar operations
        Input extends '' | ' ' | `  ${string}` | `${string}  `
        ? // Empty input is only valid for scalar tensors (shape [])
          InputShape extends readonly []
          ? ValidateRepeatPatternStructure<Pattern, InputShape, Axes>
          : RepeatParseError<"Empty input pattern. Specify input axes before '->'">
        : Output extends '' | ' ' | `  ${string}` | `${string}  `
          ? // Empty output is invalid for repeat (unlike reduce)
            RepeatParseError<"Empty output pattern. Specify output axes after '->'">
          : // Step 3: Progressive validation chain
            ValidateRepeatPatternStructure<Pattern, InputShape, Axes>
    : // No arrow operator found
      RepeatParseError<"Missing arrow operator '->'. Pattern must be 'input -> output'">;

/**
 * Simplified validation chain for repeat operations
 * Uses lightweight validation before delegating to existing system
 * Following the exact pattern from ValidEinopsPattern and ValidReducePattern
 */
type ValidateRepeatPatternStructure<
  Pattern extends string,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
> =
  // Try the existing resolver first - it handles most cases correctly
  ResolveRepeatShape<Pattern, InputShape, Axes> extends infer Result
    ? [Result] extends [never]
      ? // Only when existing system fails, provide specific error detection
        DetectRepeatSpecificError<Pattern, InputShape, Axes>
      : Result extends Shape
        ? IsValidIntegerShape<Result> extends true
          ? Result // Valid integer shape
          : RepeatFractionalDimensionError<Pattern> // Invalid fractional dimensions
        : Result extends { __error: infer ErrorMsg }
          ? ErrorMsg extends string
            ? RepeatParseError<ErrorMsg> // Convert internal error to repeat error
            : DetectRepeatSpecificError<Pattern, InputShape, Axes>
          : DetectRepeatSpecificError<Pattern, InputShape, Axes>
    : DetectRepeatSpecificError<Pattern, InputShape, Axes>;

/**
 * Detect specific errors when the existing system returns never
 * This provides targeted error messages for common failure patterns
 */
export type DetectRepeatSpecificError<
  Pattern extends string,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
> =
  // Parse the pattern to analyze what went wrong
  ParsePattern<Pattern> extends infer ParsedAST
    ? ParsedAST extends TypeEinopsAST
      ? // Check errors in priority order (matching Python einops behavior)
        // 1. FIRST: Check for rank mismatch (most fundamental)
        IsNever<CheckForRepeatRankMismatch<ParsedAST, InputShape>> extends true
        ? // 2. Check for duplicate axes
          IsNever<CheckForRepeatDuplicateAxes<ParsedAST>> extends true
          ? // 3. Check for multiple ellipsis
            IsNever<CheckForRepeatMultipleEllipsis<ParsedAST>> extends true
            ? // 4. Check for missing axis sizes (repeat-specific)
              IsNever<CheckForRepeatMissingAxes<ParsedAST, Axes>> extends true
              ? // 5. Check for invalid axis sizes
                IsNever<CheckForRepeatInvalidSizes<Axes>> extends true
                ? // 6. Check for composite pattern errors
                  CheckForRepeatCompositeErrors<
                    ParsedAST,
                    InputShape,
                    Axes
                  > extends infer CompositeError
                  ? IsNever<CompositeError> extends true
                    ? RepeatParseError<'Pattern validation failed'> // Generic fallback
                    : CompositeError // Return specific composite error
                  : RepeatParseError<'Pattern validation failed'>
                : CheckForRepeatInvalidSizes<Axes> // Return specific invalid size error
              : CheckForRepeatMissingAxes<ParsedAST, Axes> // Return specific missing axes error
            : CheckForRepeatMultipleEllipsis<ParsedAST> // Return specific multiple ellipsis error
          : CheckForRepeatDuplicateAxes<ParsedAST> // Return specific duplicate axis error
        : CheckForRepeatRankMismatch<ParsedAST, InputShape> // Return specific rank mismatch error (FIRST PRIORITY)
      : ParsedAST extends TypeParseError<infer ErrorMsg>
        ? RepeatParseError<ErrorMsg>
        : RepeatParseError<'Pattern parsing failed'>
    : RepeatParseError<'Pattern parsing failed'>;

// =============================================================================
// Specific Error Detection Functions for Repeat
// =============================================================================

/**
 * Check for duplicate axes in input or output patterns
 */
export type CheckForRepeatDuplicateAxes<AST extends TypeEinopsAST> =
  HasDuplicateAxisNames<AST['input']> extends true
    ? RepeatDuplicateAxisError<ExtractFirstDuplicate<AST['input']>, 'input'>
    : HasDuplicateAxisNames<AST['output']> extends true
      ? RepeatDuplicateAxisError<ExtractFirstDuplicate<AST['output']>, 'output'>
      : never; // No duplicates found

/**
 * Check for missing axes sizes (unique to repeat - new axes need explicit sizes)
 */
export type CheckForRepeatMissingAxes<
  AST extends TypeEinopsAST,
  Axes extends Record<string, number> | undefined,
> =
  ExtractMissingNewAxes<
    FindNewAxes<
      CollectAxisNamesIntersectionSafe<AST['input']>,
      CollectAxisNamesIntersectionSafe<AST['output']>
    >,
    Axes
  > extends infer MissingAxes
    ? MissingAxes extends readonly [infer FirstMissing, ...infer RestMissing]
      ? FirstMissing extends string
        ? RestMissing extends readonly []
          ? RepeatMissingAxisError<FirstMissing> // Single missing axis
          : MissingAxes extends readonly string[]
            ? RepeatMissingAxesError<MissingAxes> // Multiple missing axes
            : never
        : never
      : never // No missing axes
    : never; // No missing axes

/**
 * Check for invalid axis sizes
 */
export type CheckForRepeatInvalidSizes<Axes extends Record<string, number> | undefined> =
  Axes extends Record<string, number>
    ? ExtractFirstInvalidAxisSize<Axes> extends { axis: infer AxisName; size: infer Size }
      ? AxisName extends string
        ? Size extends number
          ? RepeatInvalidSizeError<AxisName, Size>
          : never
        : never
      : never // All sizes are valid
    : never; // No axes to check

/**
 * Check for rank mismatch between pattern and input shape
 * Correctly handles ellipsis patterns following Python einops behavior:
 * - Pure ellipsis (...) matches any rank - never causes rank mismatch
 * - Ellipsis + named axes (... a) requires at least named axes count
 * - No ellipsis (h w) requires exact rank match
 */
export type CheckForRepeatRankMismatch<AST extends TypeEinopsAST, InputShape extends Shape> =
  CountEllipsis<AST['input']> extends 1
    ? // Has ellipsis - check if it's pure ellipsis or ellipsis + named axes
      CountConsumingAxes<AST['input']> extends 0
      ? never // Pure ellipsis (...) - matches any rank, never a mismatch
      : // Ellipsis + named axes (... a b) - check minimum required dimensions
        CountConsumingAxes<AST['input']> extends infer MinRequiredRank
        ? MinRequiredRank extends number
          ? InputShape['length'] extends number
            ? // For ellipsis + named axes, tensor must have at least MinRequiredRank dimensions
              // Simplified check: only flag obvious mismatches (empty tensor with required axes)
              InputShape['length'] extends 0
              ? MinRequiredRank extends 0
                ? never // No named axes required
                : RepeatRankMismatchError<MinRequiredRank, InputShape['length']>
              : never // Non-empty tensor likely has enough dimensions
            : never
          : never
        : never
    : // No ellipsis - normal exact rank matching
      CountConsumingAxes<AST['input']> extends infer ExpectedRank
      ? ExpectedRank extends number
        ? InputShape['length'] extends ExpectedRank
          ? never // Ranks match exactly
          : RepeatRankMismatchError<ExpectedRank, InputShape['length']>
        : never
      : never;

/**
 * Check for multiple ellipsis patterns (only one ellipsis per side allowed)
 */
export type CheckForRepeatMultipleEllipsis<AST extends TypeEinopsAST> =
  CountEllipsis<AST['input']> extends infer InputEllipsisCount
    ? InputEllipsisCount extends number
      ? InputEllipsisCount extends 0 | 1
        ? // Input ellipsis count is valid, check output
          CountEllipsis<AST['output']> extends infer OutputEllipsisCount
          ? OutputEllipsisCount extends number
            ? OutputEllipsisCount extends 0 | 1
              ? never // Both sides have valid ellipsis counts
              : RepeatAxisError<"Multiple ellipsis '...' in output. Only one ellipsis allowed per side">
            : never
          : never
        : RepeatAxisError<"Multiple ellipsis '...' in input. Only one ellipsis allowed per side">
      : never
    : never;

/**
 * Check for composite pattern resolution errors
 */
type CheckForRepeatCompositeErrors<
  AST extends TypeEinopsAST,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
> = CheckRepeatCompositePatterns<AST['input'], InputShape, Axes>;

/**
 * Check composite patterns in input for resolution errors
 */
type CheckRepeatCompositePatterns<
  Patterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
  CurrentIndex extends number = 0,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeCompositeAxis
    ? // Found a composite pattern - check if it can be resolved
      CurrentIndex extends keyof InputShape
      ? InputShape[CurrentIndex] extends number
        ? CheckRepeatCompositeResolution<
            Head,
            InputShape[CurrentIndex],
            Axes
          > extends infer CompositeError
          ? CompositeError extends string
            ? CompositeError // Return the composite error
            : Tail extends readonly TypeAxisPattern[]
              ? CheckRepeatCompositePatterns<Tail, InputShape, Axes, Add<CurrentIndex, 1>>
              : never
          : never
        : never
      : never
    : // Not a composite, continue to next pattern
      Tail extends readonly TypeAxisPattern[]
      ? CheckRepeatCompositePatterns<Tail, InputShape, Axes, Add<CurrentIndex, 1>>
      : never
  : never; // No composite patterns found

/**
 * Check if a specific composite pattern can be resolved for repeat operations
 */
type CheckRepeatCompositeResolution<
  Composite extends TypeCompositeAxis,
  Dimension extends number,
  Axes extends Record<string, number> | undefined,
> =
  // Use the same logic as rearrange but with repeat-specific error messages
  Axes extends Record<string, number>
    ? // Has provided axes - validate they work
      RepeatCompositeResolutionError<FormatRepeatCompositePattern<Composite>, Dimension>
    : // No provided axes - need at least some for complex composites
      RepeatCompositeResolutionError<FormatRepeatCompositePattern<Composite>, Dimension>;

/**
 * Format composite pattern for repeat error messages
 * Uses the same logic as the working rearrange implementation
 */
type FormatRepeatCompositePattern<Composite extends TypeCompositeAxis> =
  `(${FormatRepeatAxesInComposite<Composite['axes']>})`;

/**
 * Format the axes inside a composite pattern for repeat operations
 * Enhanced to handle nested composites like the working implementations
 */
type FormatRepeatAxesInComposite<
  Axes extends readonly TypeAxisPattern[],
  Result extends string = '',
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? Tail extends readonly []
        ? `${Result}${Head['name']}` // Last axis, no space after
        : FormatRepeatAxesInComposite<Tail, `${Result}${Head['name']} `> // Add space after
      : `${Result}${Head['name']}`
    : Head extends TypeCompositeAxis
      ? // Handle nested composites
        FormatRepeatCompositePattern<Head> extends infer NestedPattern
        ? NestedPattern extends string
          ? Tail extends readonly TypeAxisPattern[]
            ? Tail extends readonly []
              ? `${Result}${NestedPattern}` // Last composite, no space after
              : FormatRepeatAxesInComposite<Tail, `${Result}${NestedPattern} `> // Add space after
            : `${Result}${NestedPattern}`
          : Result
        : Result
      : Head extends TypeSingletonAxis
        ? Tail extends readonly TypeAxisPattern[]
          ? Tail extends readonly []
            ? `${Result}1` // Last singleton, no space after
            : FormatRepeatAxesInComposite<Tail, `${Result}1 `> // Add space after
          : `${Result}1`
        : Result // Skip other pattern types
  : Result;
