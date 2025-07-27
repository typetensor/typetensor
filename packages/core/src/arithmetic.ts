import type { Compare, Add, Divide, Mod } from 'ts-arithmetic';

/**
 * Absolute value of a number
 */
export type Abs<N extends number> = `${N}` extends `-${infer Positive extends number}`
  ? Positive
  : N;

/**
 * Maximum of two numbers
 */
export type Max<A extends number, B extends number> = Compare<A, B> extends 1 ? A : B;

/**
 * Minimum of two numbers
 */
export type Min<A extends number, B extends number> = Compare<A, B> extends -1 ? A : B;

/**
 * Extract integer part from a number
 * IntegerPart<4.5> = 4
 * IntegerPart<4> = 4
 */
export type IntegerPart<N extends number> = `${N}` extends `${infer I extends number}.${string}`
  ? I
  : N;

/**
 * Check if a number is an integer (no decimal part)
 * IsInteger<4> = true
 * IsInteger<4.5> = false
 */
export type IsInteger<N extends number> = `${N}` extends `${number}.${number}` ? false : true;

/**
 * Ceiling division - divides and rounds up
 * Returns the smallest integer greater than or equal to A/B
 * Ceil<10, 3> = 4
 * Ceil<9, 3> = 3
 */
export type Ceil<A extends number, B extends number> =
  Mod<A, B> extends 0 ? Divide<A, B> : Add<IntegerPart<Divide<A, B>>, 1>;

/**
 * Clamp a value between min and max
 */
export type Clamp<Value extends number, MinVal extends number, MaxVal extends number> =
  Compare<Value, MinVal> extends -1 ? MinVal : Compare<Value, MaxVal> extends 1 ? MaxVal : Value;

/**
 * Normalize a potentially negative index to positive
 * Negative indices count from the end: -1 = Dim-1, -2 = Dim-2, etc.
 */
export type NormalizeIndex<Idx extends number, Dim extends number> = `${Idx}` extends `-${string}`
  ? Add<Dim, Idx>
  : Idx;

// =============================================================================
// Helper Types
// =============================================================================

/**
 * Decrement a number type (for recursive operations)
 */
export type Decrement<N extends number> = N extends 0
  ? never
  : N extends 1
    ? 0
    : N extends 2
      ? 1
      : N extends 3
        ? 2
        : N extends 4
          ? 3
          : N extends 5
            ? 4
            : N extends 6
              ? 5
              : N extends 7
                ? 6
                : N extends 8
                  ? 7
                  : number;
