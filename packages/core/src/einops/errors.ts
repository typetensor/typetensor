/**
 * Error types for einops operations
 *
 * This module defines error interfaces and branded error types following
 * the established patterns in TypeTensor (ShapeError, DimensionError).
 * Provides consistent error messaging across all einops operations.
 */

// =============================================================================
// Core Error Interfaces
// =============================================================================

/**
 * Base interface for einops errors with structured context
 * Follows the pattern established by ShapeError and DimensionError
 */
export interface EinopsError<Message extends string, Context = unknown> {
  readonly __error: 'EinopsError';
  readonly message: Message;
  readonly context: Context;
}

// =============================================================================
// Branded Error String Types
// =============================================================================

/**
 * Parse error - issues with pattern syntax
 * Following the ValidReshapeShape pattern of branded string errors
 *
 * @example
 * type Error = EinopsParseError<"Missing arrow operator '->'">;
 * // Result: "[Einops ❌] Parse Error: Missing arrow operator '->'"
 */
export type EinopsParseError<Message extends string> = `[Einops ❌] Parse Error: ${Message}`;

/**
 * Axis error - issues with axis names and usage
 *
 * @example
 * type Error = EinopsAxisError<"Unknown axis 'z' in output">;
 * // Result: "[Einops ❌] Axis Error: Unknown axis 'z' in output"
 */
export type EinopsAxisError<Message extends string> = `[Einops ❌] Axis Error: ${Message}`;

/**
 * Shape error - issues with dimension compatibility
 *
 * @example
 * type Error = EinopsShapeError<"Cannot resolve '(h w)' from dimension 24">;
 * // Result: "[Einops ❌] Shape Error: Cannot resolve '(h w)' from dimension 24"
 */
export type EinopsShapeError<Message extends string> = `[Einops ❌] Shape Error: ${Message}`;

// =============================================================================
// Specific Error Messages
// =============================================================================

/**
 * Common parse error messages
 */
export interface ParseErrorMessages {
  MissingArrow: EinopsParseError<"Missing arrow operator '->'. Pattern must be 'input -> output'">;
  EmptyInput: EinopsParseError<"Empty input pattern. Specify input axes before '->'">;
  EmptyOutput: EinopsParseError<"Empty output pattern. Specify output axes after '->'">;
  InvalidSyntax: EinopsParseError<'Invalid pattern syntax. Check parentheses and axis names'>;
}

/**
 * Common axis error messages
 */
export interface AxisErrorMessages {
  DuplicateInput: EinopsAxisError<'Duplicate axes in input. Each axis can appear at most once per side'>;
  DuplicateOutput: EinopsAxisError<'Duplicate axes in output. Each axis can appear at most once per side'>;
  MultipleEllipsisInput: EinopsAxisError<"Multiple ellipsis '...' in input. Only one ellipsis allowed per side">;
  MultipleEllipsisOutput: EinopsAxisError<"Multiple ellipsis '...' in output. Only one ellipsis allowed per side">;
  UnknownAxis: EinopsAxisError<'Unknown axis in output. All output axes must exist in input'>;
}

/**
 * Common shape error messages
 */
export interface ShapeErrorMessages {
  RankMismatch: EinopsShapeError<'Pattern rank does not match tensor dimensions'>;
  CompositeResolution: EinopsShapeError<'Cannot resolve composite pattern. Specify axis dimensions'>;
  DimensionMismatch: EinopsShapeError<'Composite pattern dimension mismatch. Check axis values'>;
  InvalidDecomposition: EinopsShapeError<'Cannot decompose dimension with given axes'>;
}

// =============================================================================
// Helper Types for Error Formatting
// =============================================================================

/**
 * Format an array of axis names as a readable string list
 *
 * @example
 * FormatAxesList<['h', 'w', 'c']> → "'h', 'w', 'c'"
 * FormatAxesList<['batch']> → "'batch'"
 */
type FormatAxesList<T extends readonly string[]> = T extends readonly [
  infer First extends string,
  ...infer Rest extends readonly string[],
]
  ? Rest['length'] extends 0
    ? `'${First}'`
    : `'${First}', ${FormatAxesList<Rest>}`
  : never;

/**
 * Format an array of duplicate axis names for error messages
 *
 * @example
 * FormatDuplicates<['h', 'w']> → "['h', 'w']"
 */
export type FormatDuplicates<T extends readonly string[]> = T extends readonly string[]
  ? `[${FormatAxesList<T>}]`
  : never;

// =============================================================================
// Error Factory Functions (Type-Level)
// =============================================================================

/**
 * Create specific unknown axis error with context
 */
export type UnknownAxisError<
  AxisName extends string,
  AvailableAxes extends readonly string[],
> = EinopsAxisError<`Unknown axis '${AxisName}' in output. Available axes: [${FormatAxesList<AvailableAxes>}]`>;

/**
 * Create specific duplicate axis error
 */
export type DuplicateAxisError<
  AxisName extends string,
  Side extends 'input' | 'output',
> = EinopsAxisError<`Duplicate axis '${AxisName}' in ${Side}. Each axis can appear at most once per side`>;

/**
 * Create specific rank mismatch error
 */
export type RankMismatchError<
  Expected extends number,
  Actual extends number,
> = EinopsShapeError<`Pattern expects ${Expected} dimensions but tensor has ${Actual}`>;

/**
 * Create specific composite resolution error
 */
export type CompositeResolutionError<
  CompositePattern extends string,
  Dimension extends number,
> = EinopsShapeError<`Cannot resolve '${CompositePattern}' from dimension ${Dimension}. Specify axis values: rearrange(tensor, pattern, {axis: number})`>;

/**
 * Create specific dimension product mismatch error
 */
export type ProductMismatchError<
  CompositePattern extends string,
  Expected extends number,
  Actual extends number,
> = EinopsShapeError<`Composite '${CompositePattern}' expects product ${Expected} but axes give ${Actual}. Check axis values`>;

/**
 * Create specific fractional dimension error (simplified version)
 */
export type FractionalDimensionError<Pattern extends string> =
  EinopsShapeError<`Pattern '${Pattern}' produces fractional dimensions. Composite axes must divide evenly. Use integer axis values: rearrange(tensor, pattern, {axis: integer})`>;

// =============================================================================
// Reduce-Specific Error Types
// =============================================================================

/**
 * Reduce parse error - issues with reduce pattern syntax
 *
 * @example
 * type Error = ReduceParseError<"Missing arrow operator '->'">;
 * // Result: "[Reduce ❌] Parse Error: Missing arrow operator '->'"
 */
export type ReduceParseError<Message extends string> = `[Reduce ❌] Parse Error: ${Message}`;

/**
 * Reduce axis error - issues with reduce axis names and usage
 *
 * @example
 * type Error = ReduceAxisError<"Cannot create new axis 'z' in reduce output">;
 * // Result: "[Reduce ❌] Axis Error: Cannot create new axis 'z' in reduce output"
 */
export type ReduceAxisError<Message extends string> = `[Reduce ❌] Axis Error: ${Message}`;

/**
 * Reduce shape error - issues with reduce dimension compatibility
 *
 * @example
 * type Error = ReduceShapeError<"Cannot resolve '(h w)' in reduce pattern">;
 * // Result: "[Reduce ❌] Shape Error: Cannot resolve '(h w)' in reduce pattern"
 */
export type ReduceShapeError<Message extends string> = `[Reduce ❌] Shape Error: ${Message}`;

// =============================================================================
// Reduce-Specific Error Factory Functions
// =============================================================================

/**
 * Create specific error for attempting to create new axes in reduce output
 * Reduce operations can only preserve or remove axes, never create new ones
 */
export type ReduceNewAxisError<
  AxisName extends string,
  InputAxes extends readonly string[],
> = ReduceAxisError<`Cannot create new axis '${AxisName}' in reduce output. Available input axes: [${FormatAxesList<InputAxes>}]. Reduce can only preserve or remove axes`>;

/**
 * Create specific duplicate axis error for reduce operations
 */
export type ReduceDuplicateAxisError<
  AxisName extends string,
  Side extends 'input' | 'output',
> = ReduceAxisError<`Duplicate axis '${AxisName}' in ${Side}. Each axis can appear at most once per side`>;

/**
 * Create specific rank mismatch error for reduce operations
 */
export type ReduceRankMismatchError<
  Expected extends number,
  Actual extends number,
> = ReduceShapeError<`Reduce pattern expects ${Expected} dimensions but tensor has ${Actual}`>;

/**
 * Create specific composite resolution error for reduce operations
 */
export type ReduceCompositeResolutionError<
  CompositePattern extends string,
  Dimension extends number,
> = ReduceShapeError<`Cannot resolve '${CompositePattern}' from dimension ${Dimension}. Specify axis values: reduce(tensor, pattern, operation, keepDims, {axis: number})`>;

/**
 * Create specific dimension product mismatch error for reduce operations
 */
export type ReduceProductMismatchError<
  CompositePattern extends string,
  Expected extends number,
  Actual extends number,
> = ReduceShapeError<`Composite '${CompositePattern}' expects product ${Expected} but axes give ${Actual}. Check axis values`>;

/**
 * Create specific fractional dimension error for reduce operations
 */
export type ReduceFractionalDimensionError<Pattern extends string> =
  ReduceShapeError<`Reduce pattern '${Pattern}' produces fractional dimensions. Composite axes must divide evenly. Use integer axis values: reduce(tensor, pattern, operation, keepDims, {axis: integer})`>;

// =============================================================================
// Repeat-Specific Error Types
// =============================================================================

/**
 * Repeat parse error - issues with repeat pattern syntax
 *
 * @example
 * type Error = RepeatParseError<"Missing arrow operator '->'">;
 * // Result: "[Repeat ❌] Parse Error: Missing arrow operator '->'"
 */
export type RepeatParseError<Message extends string> = `[Repeat ❌] Parse Error: ${Message}`;

/**
 * Repeat axis error - issues with repeat axis names and usage
 *
 * @example
 * type Error = RepeatAxisError<"New axis 'c' requires explicit size">;
 * // Result: "[Repeat ❌] Axis Error: New axis 'c' requires explicit size"
 */
export type RepeatAxisError<Message extends string> = `[Repeat ❌] Axis Error: ${Message}`;

/**
 * Repeat shape error - issues with repeat dimension compatibility
 *
 * @example
 * type Error = RepeatShapeError<"Cannot resolve '(h w)' in repeat pattern">;
 * // Result: "[Repeat ❌] Shape Error: Cannot resolve '(h w)' in repeat pattern"
 */
export type RepeatShapeError<Message extends string> = `[Repeat ❌] Shape Error: ${Message}`;

// =============================================================================
// Repeat-Specific Error Factory Functions
// =============================================================================

/**
 * Create specific error for missing axis dimensions in repeat operations
 * Repeat operations allow new axes but they must have explicit sizes
 */
export type RepeatMissingAxisError<AxisName extends string> =
  RepeatAxisError<`New axis '${AxisName}' requires explicit size. Specify: repeat(tensor, pattern, {${AxisName}: number})`>;

/**
 * Create specific error for multiple missing axes in repeat operations
 */
export type RepeatMissingAxesError<AxesNames extends readonly string[]> =
  RepeatAxisError<`New axes [${FormatAxesList<AxesNames>}] require explicit sizes. Specify: repeat(tensor, pattern, {${FormatAxisSpecs<AxesNames>}})`>;

/**
 * Format axis specifications for error messages
 * @example FormatAxisSpecs<['w', 'c']> → "w: number, c: number"
 */
type FormatAxisSpecs<T extends readonly string[]> = T extends readonly [
  infer First extends string,
  ...infer Rest extends readonly string[],
]
  ? Rest['length'] extends 0
    ? `${First}: number`
    : `${First}: number, ${FormatAxisSpecs<Rest>}`
  : never;

/**
 * Create specific error for invalid axis sizes in repeat operations
 */
export type RepeatInvalidSizeError<
  AxisName extends string,
  Size extends number,
> = RepeatAxisError<`Invalid size ${Size} for axis '${AxisName}'. Repeat sizes must be positive integers`>;

/**
 * Create specific duplicate axis error for repeat operations
 */
export type RepeatDuplicateAxisError<
  AxisName extends string,
  Side extends 'input' | 'output',
> = RepeatAxisError<`Duplicate axis '${AxisName}' in ${Side}. Each axis can appear at most once per side`>;

/**
 * Create specific rank mismatch error for repeat operations
 */
export type RepeatRankMismatchError<
  Expected extends number,
  Actual extends number,
> = RepeatShapeError<`Repeat pattern expects ${Expected} dimensions but tensor has ${Actual}`>;

/**
 * Create specific composite resolution error for repeat operations
 */
export type RepeatCompositeResolutionError<
  CompositePattern extends string,
  Dimension extends number,
> = RepeatShapeError<`Cannot resolve '${CompositePattern}' from dimension ${Dimension}. Specify axis values: repeat(tensor, pattern, {axis: number})`>;

/**
 * Create specific dimension product mismatch error for repeat operations
 */
export type RepeatProductMismatchError<
  CompositePattern extends string,
  Expected extends number,
  Actual extends number,
> = RepeatShapeError<`Composite '${CompositePattern}' expects product ${Expected} but axes give ${Actual}. Check axis values`>;

/**
 * Create specific fractional dimension error for repeat operations
 */
export type RepeatFractionalDimensionError<Pattern extends string> =
  RepeatShapeError<`Repeat pattern '${Pattern}' produces fractional dimensions. Composite axes must divide evenly. Use integer axis values: repeat(tensor, pattern, {axis: integer})`>;
