/**
 * Top-level validation for einops patterns - Step 2: Basic Implementation
 *
 * This module provides the main validation entry point following the pattern
 * established by ValidReshapeShape. This is a minimal working version that
 * adds syntax validation while delegating to the existing system.
 */

import type { Shape } from '../shape/types';
import type {
  EinopsParseError,
  UnknownAxisError,
  DuplicateAxisError,
  RankMismatchError,
  CompositeResolutionError,
} from './errors';
import type { ResolveEinopsShape } from './type-shape-resolver-rearrange';
import type {
  ParsePattern,
  TypeEinopsAST,
  TypeParseError,
  TypeAxisPattern,
  TypeSimpleAxis,
  TypeCompositeAxis,
} from './type-parser';
import type {
  HasDuplicateAxisNames,
  CountConsumingAxes,
  FlattenAxes,
  CountUnknownAxes,
  AllAxesProvided,
} from './type-shape-resolver-utils';
import type { Add } from 'ts-arithmetic';

// =============================================================================
// Helper Types for Error Extraction
// =============================================================================

/**
 * Utility to check if a type is exactly never
 * This works correctly unlike [T] extends [string] which fails for never
 */
type IsNever<T> = [T] extends [never] ? true : false;

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

/**
 * Find unknown axes in output that don't exist in input
 */
export type FindUnknownAxes<
  InputAxes extends readonly string[],
  OutputAxes extends readonly string[],
> = OutputAxes extends readonly [infer Head, ...infer Tail]
  ? Head extends string
    ? Head extends InputAxes[number]
      ? Tail extends readonly string[]
        ? FindUnknownAxes<InputAxes, Tail> // Known axis, continue
        : readonly []
      : readonly [Head] // Unknown axis found
    : Tail extends readonly string[]
      ? FindUnknownAxes<InputAxes, Tail>
      : readonly []
  : readonly [];

// =============================================================================
// Main Validation Entry Point - Enhanced Version
// =============================================================================

/**
 * Validate einops pattern and return output shape or specific error
 *
 * Step 3 Implementation: Progressive validation with specific error messages
 * - Quick syntax validation (missing arrow, empty patterns)
 * - Progressive validation chain with targeted error messages
 * - Returns Shape on success, specific branded error string on failure
 *
 * @example
 * type Valid = ValidEinopsPattern<'h w -> w h', [2, 3]>; // [3, 2]
 * type NoArrow = ValidEinopsPattern<'h w', [2, 3]>; // Parse error
 * type UnknownAxis = ValidEinopsPattern<'h w -> h w c', [2, 3]>; // Axis error
 * type DuplicateAxis = ValidEinopsPattern<'h h -> h', [2, 3]>; // Axis error
 *
 * @param Pattern - The einops pattern string
 * @param InputShape - The input tensor shape
 * @param Axes - Optional axis dimension specifications
 */
export type ValidEinopsPattern<
  Pattern extends string,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined = undefined,
> =
  // Step 1: Quick syntax validation (like ValidReshapeShape const check)
  Pattern extends `${infer Input}->${infer Output}`
    ? // Step 2: Check for empty input/output, but allow them for scalar operations
      Input extends '' | ' ' | `  ${string}` | `${string}  `
      ? // Empty input is only valid for scalar tensors (shape [])
        InputShape extends readonly []
        ? ValidateEinopsPatternStructure<Pattern, InputShape, Axes>
        : EinopsParseError<"Empty input pattern. Specify input axes before '->'">
      : Output extends '' | ' ' | `  ${string}` | `${string}  `
        ? // Empty output is only valid when producing scalars
          ValidateEinopsPatternStructure<Pattern, InputShape, Axes>
        : // Step 3: Progressive validation chain
          ValidateEinopsPatternStructure<Pattern, InputShape, Axes>
    : // No arrow operator found
      EinopsParseError<"Missing arrow operator '->'. Pattern must be 'input -> output'">;

/**
 * Simplified validation chain - Step 3 implementation
 * Uses lightweight validation before delegating to existing system
 */
type ValidateEinopsPatternStructure<
  Pattern extends string,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
> =
  // Try the existing resolver first - it handles most cases correctly
  ResolveEinopsShape<Pattern, InputShape, Axes> extends infer Result
    ? [Result] extends [never]
      ? // Only when existing system fails, provide specific error detection
        DetectSpecificError<Pattern, InputShape, Axes>
      : Result extends Shape
        ? Result
        : DetectSpecificError<Pattern, InputShape, Axes>
    : DetectSpecificError<Pattern, InputShape, Axes>;

/**
 * Detect specific errors when the existing system returns never
 * This provides targeted error messages for common failure patterns
 */
export type DetectSpecificError<
  Pattern extends string,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
> =
  // Parse the pattern to analyze what went wrong
  ParsePattern<Pattern> extends infer ParsedAST
    ? ParsedAST extends TypeEinopsAST
      ? // Check for specific error patterns using IsNever to distinguish never from strings
        IsNever<CheckForDuplicateAxes<ParsedAST>> extends true
        ? // No duplicate error, check for unknown axes
          IsNever<CheckForUnknownAxes<ParsedAST>> extends true
          ? // No unknown error, check for rank mismatch
            IsNever<CheckForRankMismatch<ParsedAST, InputShape>> extends true
            ? // No rank error, check for composite errors
              CheckForCompositeErrors<ParsedAST, InputShape, Axes> extends infer CompositeError
              ? IsNever<CompositeError> extends true
                ? EinopsParseError<'Pattern validation failed'> // Generic fallback
                : CompositeError // Return specific composite error
              : EinopsParseError<'Pattern validation failed'>
            : CheckForRankMismatch<ParsedAST, InputShape> // Return specific rank mismatch error
          : CheckForUnknownAxes<ParsedAST> // Return specific unknown axis error
        : CheckForDuplicateAxes<ParsedAST> // Return specific duplicate axis error
      : ParsedAST extends TypeParseError<infer ErrorMsg>
        ? EinopsParseError<ErrorMsg>
        : EinopsParseError<'Pattern parsing failed'>
    : EinopsParseError<'Pattern parsing failed'>;

// =============================================================================
// Intersection-Safe Axis Collection (Fixes the Root Issue)
// =============================================================================

/**
 * Collect axis names from patterns that works with intersection types
 * This is the key fix - it doesn't rely on problematic tail inference
 */
export type CollectAxisNamesIntersectionSafe<
  Patterns, // Accept any type including intersections
  Result extends readonly string[] = readonly [],
> = Patterns extends readonly []
  ? Result
  : Patterns extends readonly [infer Head]
    ? Head extends TypeSimpleAxis
      ? readonly [...Result, Head['name']]
      : Head extends TypeCompositeAxis
        ? CollectAxisNamesIntersectionSafe<Head['axes']> extends infer CompositeNames
          ? CompositeNames extends readonly string[]
            ? readonly [...Result, ...CompositeNames]
            : Result
          : Result
        : Result
    : Patterns extends readonly [infer First, infer Second]
      ? First extends TypeSimpleAxis
        ? CollectAxisNamesIntersectionSafe<readonly [Second], readonly [...Result, First['name']]>
        : First extends TypeCompositeAxis
          ? CollectAxisNamesIntersectionSafe<First['axes']> extends infer CompositeNames
            ? CompositeNames extends readonly string[]
              ? CollectAxisNamesIntersectionSafe<
                  readonly [Second],
                  readonly [...Result, ...CompositeNames]
                >
              : CollectAxisNamesIntersectionSafe<readonly [Second], Result>
            : CollectAxisNamesIntersectionSafe<readonly [Second], Result>
          : CollectAxisNamesIntersectionSafe<readonly [Second], Result>
      : Patterns extends readonly [infer First, infer Second, infer Third]
        ? First extends TypeSimpleAxis
          ? CollectAxisNamesIntersectionSafe<
              readonly [Second, Third],
              readonly [...Result, First['name']]
            >
          : First extends TypeCompositeAxis
            ? CollectAxisNamesIntersectionSafe<First['axes']> extends infer CompositeNames
              ? CompositeNames extends readonly string[]
                ? CollectAxisNamesIntersectionSafe<
                    readonly [Second, Third],
                    readonly [...Result, ...CompositeNames]
                  >
                : CollectAxisNamesIntersectionSafe<readonly [Second, Third], Result>
              : CollectAxisNamesIntersectionSafe<readonly [Second, Third], Result>
            : CollectAxisNamesIntersectionSafe<readonly [Second, Third], Result>
        : Result;

// =============================================================================
// Specific Error Detection Functions
// =============================================================================

/**
 * Check for duplicate axes in input or output patterns
 */
export type CheckForDuplicateAxes<AST extends TypeEinopsAST> =
  HasDuplicateAxisNames<AST['input']> extends true
    ? DuplicateAxisError<ExtractFirstDuplicate<AST['input']>, 'input'>
    : HasDuplicateAxisNames<AST['output']> extends true
      ? DuplicateAxisError<ExtractFirstDuplicate<AST['output']>, 'output'>
      : never; // No duplicates found

/**
 * Check for unknown axes in output that don't exist in input
 */
export type CheckForUnknownAxes<AST extends TypeEinopsAST> =
  FindUnknownAxes<
    CollectAxisNamesIntersectionSafe<AST['input']>,
    CollectAxisNamesIntersectionSafe<AST['output']>
  > extends readonly [infer FirstUnknown, ...any[]]
    ? FirstUnknown extends string
      ? UnknownAxisError<FirstUnknown, CollectAxisNamesIntersectionSafe<AST['input']>>
      : never
    : never; // No unknown axes found

/**
 * Check for rank mismatch between pattern and input shape
 */
export type CheckForRankMismatch<AST extends TypeEinopsAST, InputShape extends Shape> =
  CountConsumingAxes<AST['input']> extends infer ExpectedRank
    ? ExpectedRank extends number
      ? InputShape['length'] extends ExpectedRank
        ? never // Ranks match
        : RankMismatchError<ExpectedRank, InputShape['length']>
      : never
    : never;

/**
 * Check for composite pattern resolution errors
 */
type CheckForCompositeErrors<
  AST extends TypeEinopsAST,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
> = CheckCompositePatterns<AST['input'], InputShape, Axes>;

/**
 * Check composite patterns in input for resolution errors
 */
type CheckCompositePatterns<
  Patterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
  CurrentIndex extends number = 0,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeCompositeAxis
    ? // Found a composite pattern - check if it can be resolved
      CurrentIndex extends keyof InputShape
      ? InputShape[CurrentIndex] extends number
        ? CheckCompositeResolution<
            Head,
            InputShape[CurrentIndex],
            Axes
          > extends infer CompositeError
          ? CompositeError extends string
            ? CompositeError // Return the composite error
            : Tail extends readonly TypeAxisPattern[]
              ? CheckCompositePatterns<Tail, InputShape, Axes, Add<CurrentIndex, 1>>
              : never
          : never
        : never
      : never
    : // Not a composite, continue to next pattern
      Tail extends readonly TypeAxisPattern[]
      ? CheckCompositePatterns<Tail, InputShape, Axes, Add<CurrentIndex, 1>>
      : never
  : never; // No composite patterns found

/**
 * Check if a specific composite pattern can be resolved
 */
type CheckCompositeResolution<
  Composite extends TypeCompositeAxis,
  Dimension extends number,
  Axes extends Record<string, number> | undefined,
> =
  FlattenAxes<Composite['axes']> extends infer FlatAxes
    ? FlatAxes extends readonly TypeSimpleAxis[]
      ? Axes extends Record<string, number>
        ? // Check if all axes are provided or if we can resolve with missing axes
          AllAxesProvided<FlatAxes, Axes> extends true
          ? never // All axes provided, should be resolvable
          : CountUnknownAxes<FlatAxes, Axes> extends infer UnknownCount
            ? UnknownCount extends number
              ? UnknownCount extends 0
                ? never // No unknown axes, should be resolvable
                : UnknownCount extends 1
                  ? never // One unknown axis, can be inferred
                  : // Multiple unknown axes, cannot resolve
                    CompositeResolutionError<FormatCompositePattern<Composite>, Dimension>
              : never
            : never
        : // No axes provided at all
          FlatAxes extends readonly [TypeSimpleAxis]
          ? never // Single axis composite can be inferred
          : // Multiple axes, need at least one specified
            CompositeResolutionError<FormatCompositePattern<Composite>, Dimension>
      : never
    : never;

/**
 * Format composite pattern for error message
 */
type FormatCompositePattern<Composite extends TypeCompositeAxis> =
  `(${FormatAxesInComposite<Composite['axes']>})`;

/**
 * Format the axes inside a composite pattern
 */
type FormatAxesInComposite<
  Axes extends readonly TypeAxisPattern[],
  Result extends string = '',
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? Tail extends readonly []
        ? `${Result}${Head['name']}` // Last axis, no space after
        : FormatAxesInComposite<Tail, `${Result}${Head['name']} `> // Add space after
      : `${Result}${Head['name']}`
    : Result // Skip non-simple axes for now
  : Result;

// =============================================================================
// Step 3 Notes
// =============================================================================

/*
 * Step 3 simplified approach provides specific error messages:
 *
 * ✅ Leverages existing ResolveEinopsShape for all successful cases
 * ✅ Detects specific errors only when existing system returns never
 * ✅ Avoids pattern reconstruction problem by keeping original pattern
 * ✅ Targeted error messages: duplicates, unknown axes, rank mismatches
 *
 * Approach:
 * 1. Try existing ResolveEinopsShape first (handles success cases perfectly)
 * 2. If it returns never, analyze the pattern to determine specific error
 * 3. Return branded error strings with helpful context
 *
 * This is simpler, more maintainable, and avoids architectural issues.
 */
