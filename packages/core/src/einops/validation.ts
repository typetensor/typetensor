/**
 * Type-level validation functions for einops patterns
 *
 * This module provides compile-time validation of einops patterns,
 * ensuring that patterns are well-formed and semantically valid.
 */

import type {
  TypeAxisPattern,
  TypeSimpleAxis,
  TypeCompositeAxis,
  TypeEllipsisAxis,
} from './type-parser';

// =============================================================================
// Axis Collection Utilities
// =============================================================================

/**
 * Collect all simple axis names from a pattern
 */
export type CollectAxisNames<
  Patterns extends readonly TypeAxisPattern[],
  Collected extends readonly string[] = readonly [],
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? CollectAxisNames<Tail, readonly [...Collected, Head['name']]>
      : Collected
    : Head extends TypeCompositeAxis
      ? Tail extends readonly TypeAxisPattern[]
        ? CollectAxisNames<Tail, CollectAxisNames<Head['axes'], Collected>>
        : Collected
      : Tail extends readonly TypeAxisPattern[]
        ? CollectAxisNames<Tail, Collected>
        : Collected
  : Collected;

/**
 * Check if an axis name exists in a list
 */
export type HasAxis<Name extends string, Axes extends readonly string[]> = Name extends Axes[number]
  ? true
  : false;

/**
 * Count occurrences of an axis name
 */
export type CountAxis<
  Name extends string,
  Axes extends readonly string[],
  Count extends number = 0,
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends Name
    ? Tail extends readonly string[]
      ? CountAxis<Name, Tail, Count> // TODO: Increment count
      : Count
    : Tail extends readonly string[]
      ? CountAxis<Name, Tail, Count>
      : Count
  : Count;

// =============================================================================
// Ellipsis Validation
// =============================================================================

/**
 * Count the number of ellipsis patterns
 */
export type CountEllipsis<
  Patterns extends readonly TypeAxisPattern[],
  Count extends number = 0,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeEllipsisAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? CountEllipsis<Tail, Count> // TODO: Increment count
      : 1
    : Head extends TypeCompositeAxis
      ? Tail extends readonly TypeAxisPattern[]
        ? CountEllipsis<Tail, CountEllipsis<Head['axes'], Count>>
        : Count
      : Tail extends readonly TypeAxisPattern[]
        ? CountEllipsis<Tail, Count>
        : Count
  : Count;

/**
 * Validate that at most one ellipsis exists
 */
export type ValidateEllipsisCount<Patterns extends readonly TypeAxisPattern[]> =
  CountEllipsis<Patterns> extends 0 | 1 ? true : false;

// =============================================================================
// Axis Uniqueness Validation
// =============================================================================

/**
 * Check for duplicate axis names
 */
export type HasDuplicateAxes<
  Axes extends readonly string[],
  Seen extends Record<string, true> = {},
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends string
    ? Head extends keyof Seen
      ? true // Found duplicate
      : Tail extends readonly string[]
        ? HasDuplicateAxes<Tail, Seen & Record<Head, true>>
        : false
    : false
  : false;

/**
 * Validate that all axis names are unique
 */
export type ValidateUniqueAxes<Patterns extends readonly TypeAxisPattern[]> =
  HasDuplicateAxes<CollectAxisNames<Patterns>> extends true ? false : true;

// =============================================================================
// Output Axes Validation
// =============================================================================

/**
 * Check if all axes in output exist in input (except singletons)
 */
export type ValidateOutputAxes<
  Output extends readonly TypeAxisPattern[],
  InputAxes extends readonly string[],
> = Output extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? HasAxis<Head['name'], InputAxes> extends true
      ? Tail extends readonly TypeAxisPattern[]
        ? ValidateOutputAxes<Tail, InputAxes>
        : true
      : false // Unknown axis in output
    : Head extends TypeCompositeAxis
      ? ValidateOutputAxes<Head['axes'], InputAxes> extends true
        ? Tail extends readonly TypeAxisPattern[]
          ? ValidateOutputAxes<Tail, InputAxes>
          : true
        : false
      : Tail extends readonly TypeAxisPattern[]
        ? ValidateOutputAxes<Tail, InputAxes> // Skip ellipsis/singleton
        : true
  : true;

// =============================================================================
// Axis Name Validation
// =============================================================================

/**
 * Check if a character is a valid starting character for an axis name
 */
export type IsValidAxisStartChar<C extends string> = C extends
  | 'a'
  | 'b'
  | 'c'
  | 'd'
  | 'e'
  | 'f'
  | 'g'
  | 'h'
  | 'i'
  | 'j'
  | 'k'
  | 'l'
  | 'm'
  | 'n'
  | 'o'
  | 'p'
  | 'q'
  | 'r'
  | 's'
  | 't'
  | 'u'
  | 'v'
  | 'w'
  | 'x'
  | 'y'
  | 'z'
  | 'A'
  | 'B'
  | 'C'
  | 'D'
  | 'E'
  | 'F'
  | 'G'
  | 'H'
  | 'I'
  | 'J'
  | 'K'
  | 'L'
  | 'M'
  | 'N'
  | 'O'
  | 'P'
  | 'Q'
  | 'R'
  | 'S'
  | 'T'
  | 'U'
  | 'V'
  | 'W'
  | 'X'
  | 'Y'
  | 'Z'
  | '_'
  ? true
  : false;

/**
 * Check if a character is valid in an axis name
 */
export type IsValidAxisChar<C extends string> = C extends
  | 'a'
  | 'b'
  | 'c'
  | 'd'
  | 'e'
  | 'f'
  | 'g'
  | 'h'
  | 'i'
  | 'j'
  | 'k'
  | 'l'
  | 'm'
  | 'n'
  | 'o'
  | 'p'
  | 'q'
  | 'r'
  | 's'
  | 't'
  | 'u'
  | 'v'
  | 'w'
  | 'x'
  | 'y'
  | 'z'
  | 'A'
  | 'B'
  | 'C'
  | 'D'
  | 'E'
  | 'F'
  | 'G'
  | 'H'
  | 'I'
  | 'J'
  | 'K'
  | 'L'
  | 'M'
  | 'N'
  | 'O'
  | 'P'
  | 'Q'
  | 'R'
  | 'S'
  | 'T'
  | 'U'
  | 'V'
  | 'W'
  | 'X'
  | 'Y'
  | 'Z'
  | '0'
  | '1'
  | '2'
  | '3'
  | '4'
  | '5'
  | '6'
  | '7'
  | '8'
  | '9'
  | '_'
  ? true
  : false;

/**
 * Validate an axis name follows JavaScript identifier rules
 */
export type ValidateAxisName<Name extends string> = Name extends `${infer First}${infer Rest}`
  ? IsValidAxisStartChar<First> extends true
    ? ValidateAxisNameRest<Rest>
    : false
  : false;

/**
 * Validate the rest of an axis name
 */
type ValidateAxisNameRest<Rest extends string> = Rest extends ''
  ? true
  : Rest extends `${infer Head}${infer Tail}`
    ? IsValidAxisChar<Head> extends true
      ? ValidateAxisNameRest<Tail>
      : false
    : false;

/**
 * Validate all axis names in patterns
 */
export type ValidateAllAxisNames<Patterns extends readonly TypeAxisPattern[]> =
  ValidateAxisNamesInPatterns<Patterns>;

type ValidateAxisNamesInPatterns<Patterns extends readonly TypeAxisPattern[]> =
  Patterns extends readonly [infer Head, ...infer Tail]
    ? Head extends TypeSimpleAxis
      ? ValidateAxisName<Head['name']> extends true
        ? Tail extends readonly TypeAxisPattern[]
          ? ValidateAxisNamesInPatterns<Tail>
          : true
        : false
      : Head extends TypeCompositeAxis
        ? ValidateAxisNamesInPatterns<Head['axes']> extends true
          ? Tail extends readonly TypeAxisPattern[]
            ? ValidateAxisNamesInPatterns<Tail>
            : true
          : false
        : Tail extends readonly TypeAxisPattern[]
          ? ValidateAxisNamesInPatterns<Tail>
          : true
    : true;

// =============================================================================
// Composite Pattern Validation
// =============================================================================

/**
 * Check if a composite pattern is empty
 */
export type IsEmptyComposite<Pattern extends TypeAxisPattern> = Pattern extends TypeCompositeAxis
  ? Pattern['axes'] extends readonly []
    ? true
    : false
  : false;

/**
 * Validate composite patterns are not empty
 */
export type ValidateNonEmptyComposites<Patterns extends readonly TypeAxisPattern[]> =
  Patterns extends readonly [infer Head, ...infer Tail]
    ? Head extends TypeAxisPattern
      ? IsEmptyComposite<Head> extends true
        ? false
        : Head extends TypeCompositeAxis
          ? ValidateNonEmptyComposites<Head['axes']> extends true
            ? Tail extends readonly TypeAxisPattern[]
              ? ValidateNonEmptyComposites<Tail>
              : true
            : false
          : Tail extends readonly TypeAxisPattern[]
            ? ValidateNonEmptyComposites<Tail>
            : true
      : false
    : true;

// =============================================================================
// Main Validation Functions
// =============================================================================

/**
 * Validation result type
 */
export interface ValidationResult<Valid extends boolean, Error extends string = never> {
  readonly valid: Valid;
  readonly error: Valid extends false ? Error : never;
}

/**
 * Validate input patterns
 */
export type ValidateInputPatterns<Patterns extends readonly TypeAxisPattern[]> =
  ValidateEllipsisCount<Patterns> extends false
    ? ValidationResult<false, 'Multiple ellipsis patterns not allowed'>
    : ValidateUniqueAxes<Patterns> extends false
      ? ValidationResult<false, 'Duplicate axis names in input'>
      : ValidateAllAxisNames<Patterns> extends false
        ? ValidationResult<false, 'Invalid axis name'>
        : ValidateNonEmptyComposites<Patterns> extends false
          ? ValidationResult<false, 'Empty composite patterns not allowed'>
          : ValidationResult<true>;

/**
 * Validate output patterns against input
 */
export type ValidateOutputPatterns<
  Output extends readonly TypeAxisPattern[],
  Input extends readonly TypeAxisPattern[],
> =
  ValidateEllipsisCount<Output> extends false
    ? ValidationResult<false, 'Multiple ellipsis patterns not allowed'>
    : ValidateUniqueAxes<Output> extends false
      ? ValidationResult<false, 'Duplicate axis names in output'>
      : ValidateAllAxisNames<Output> extends false
        ? ValidationResult<false, 'Invalid axis name'>
        : ValidateNonEmptyComposites<Output> extends false
          ? ValidationResult<false, 'Empty composite patterns not allowed'>
          : ValidateOutputAxes<Output, CollectAxisNames<Input>> extends false
            ? ValidationResult<false, 'Output contains unknown axes'>
            : ValidationResult<true>;

/**
 * Complete pattern validation
 */
export type ValidatePattern<
  InputPatterns extends readonly TypeAxisPattern[],
  OutputPatterns extends readonly TypeAxisPattern[],
> =
  ValidateInputPatterns<InputPatterns> extends ValidationResult<false, infer InputError>
    ? ValidationResult<false, `Input validation failed: ${InputError}`>
    : ValidateOutputPatterns<OutputPatterns, InputPatterns> extends ValidationResult<
          false,
          infer OutputError
        >
      ? ValidationResult<false, `Output validation failed: ${OutputError}`>
      : ValidationResult<true>;

// =============================================================================
// Utility Types for Error Messages
// =============================================================================

/**
 * Extract unknown axes from output
 */
export type FindUnknownAxes<
  Output extends readonly TypeAxisPattern[],
  InputAxes extends readonly string[],
  Unknown extends readonly string[] = readonly [],
> = Output extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? HasAxis<Head['name'], InputAxes> extends false
      ? Tail extends readonly TypeAxisPattern[]
        ? FindUnknownAxes<Tail, InputAxes, readonly [...Unknown, Head['name']]>
        : readonly [...Unknown, Head['name']]
      : Tail extends readonly TypeAxisPattern[]
        ? FindUnknownAxes<Tail, InputAxes, Unknown>
        : Unknown
    : Head extends TypeCompositeAxis
      ? Tail extends readonly TypeAxisPattern[]
        ? FindUnknownAxes<Tail, InputAxes, FindUnknownAxes<Head['axes'], InputAxes, Unknown>>
        : FindUnknownAxes<Head['axes'], InputAxes, Unknown>
      : Tail extends readonly TypeAxisPattern[]
        ? FindUnknownAxes<Tail, InputAxes, Unknown>
        : Unknown
  : Unknown;

/**
 * Find duplicate axes
 */
export type FindDuplicates<
  Axes extends readonly string[],
  Seen extends Record<string, true> = {},
  Duplicates extends readonly string[] = readonly [],
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends string
    ? Head extends keyof Seen
      ? Tail extends readonly string[]
        ? FindDuplicates<Tail, Seen, readonly [...Duplicates, Head]>
        : readonly [...Duplicates, Head]
      : Tail extends readonly string[]
        ? FindDuplicates<Tail, Seen & Record<Head, true>, Duplicates>
        : Duplicates
    : Duplicates
  : Duplicates;
