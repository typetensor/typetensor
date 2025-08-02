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
export type ParseErrorMessages = {
  MissingArrow: EinopsParseError<"Missing arrow operator '->'. Pattern must be 'input -> output'">;
  EmptyInput: EinopsParseError<"Empty input pattern. Specify input axes before '->'">;
  EmptyOutput: EinopsParseError<"Empty output pattern. Specify output axes after '->'">;
  InvalidSyntax: EinopsParseError<'Invalid pattern syntax. Check parentheses and axis names'>;
};

/**
 * Common axis error messages
 */
export type AxisErrorMessages = {
  DuplicateInput: EinopsAxisError<'Duplicate axes in input. Each axis can appear at most once per side'>;
  DuplicateOutput: EinopsAxisError<'Duplicate axes in output. Each axis can appear at most once per side'>;
  MultipleEllipsisInput: EinopsAxisError<"Multiple ellipsis '...' in input. Only one ellipsis allowed per side">;
  MultipleEllipsisOutput: EinopsAxisError<"Multiple ellipsis '...' in output. Only one ellipsis allowed per side">;
  UnknownAxis: EinopsAxisError<'Unknown axis in output. All output axes must exist in input'>;
};

/**
 * Common shape error messages
 */
export type ShapeErrorMessages = {
  RankMismatch: EinopsShapeError<'Pattern rank does not match tensor dimensions'>;
  CompositeResolution: EinopsShapeError<'Cannot resolve composite pattern. Specify axis dimensions'>;
  DimensionMismatch: EinopsShapeError<'Composite pattern dimension mismatch. Check axis values'>;
  InvalidDecomposition: EinopsShapeError<'Cannot decompose dimension with given axes'>;
};

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
