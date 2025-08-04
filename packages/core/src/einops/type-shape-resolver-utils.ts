/**
 * Common utilities for type-level shape resolution
 *
 * This module contains shared types and utilities used by both
 * rearrange and reduce shape resolvers.
 */

import type { Shape } from '../shape/types';
import type {
  TypeAxisPattern,
  TypeSimpleAxis,
  TypeCompositeAxis,
  TypeEllipsisAxis,
  TypeSingletonAxis,
} from './type-parser';
import type { Add, Multiply, Divide, Subtract } from 'ts-arithmetic';

// =============================================================================
// Basic Helper Types
// =============================================================================

/**
 * Map of axis names to their dimension values
 */
export type AxisMap = Record<string, number>;

/**
 * Merge intersection types into a single object type
 */
export type MergeIntersection<T> = T extends object ? { [K in keyof T]: T[K] } : never;

/**
 * Append element to tuple
 */
export type Append<T extends readonly unknown[], E> = readonly [...T, E];

/**
 * Concatenate two tuples
 */
export type Concat<T1 extends readonly unknown[], T2 extends readonly unknown[]> = readonly [
  ...T1,
  ...T2,
];

/**
 * Take N elements from array
 */
export type Take<
  T extends readonly unknown[],
  N extends number,
  Acc extends readonly unknown[] = readonly [],
> = Acc['length'] extends N
  ? Acc
  : T extends readonly [infer Head, ...infer Tail]
    ? Take<Tail, N, readonly [...Acc, Head]>
    : Acc;

/**
 * Drop N elements from array
 */
export type Drop<
  T extends readonly unknown[],
  N extends number,
  Count extends readonly unknown[] = readonly [],
> = Count['length'] extends N
  ? T
  : T extends readonly [unknown, ...infer Tail]
    ? Drop<Tail, N, readonly [...Count, unknown]>
    : readonly [];

// =============================================================================
// Pattern Validation
// =============================================================================

/**
 * Check if an array has duplicate simple axis names
 */
export type HasDuplicateAxisNames<
  Patterns extends readonly TypeAxisPattern[],
  Seen extends string = never,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends Seen
      ? true // Found duplicate
      : Tail extends readonly TypeAxisPattern[]
        ? HasDuplicateAxisNames<Tail, Seen | Head['name']>
        : false
    : Head extends TypeCompositeAxis
      ? HasDuplicateAxisNames<Head['axes']> extends true
        ? true
        : Tail extends readonly TypeAxisPattern[]
          ? HasDuplicateAxisNames<Tail, Seen>
          : false
      : Tail extends readonly TypeAxisPattern[]
        ? HasDuplicateAxisNames<Tail, Seen>
        : false
  : false;

/**
 * Count ellipsis axes in pattern
 */
export type CountEllipsis<
  Patterns extends readonly TypeAxisPattern[],
  Count extends number = 0,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeEllipsisAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? CountEllipsis<Tail, Add<Count, 1>>
      : Add<Count, 1>
    : Head extends TypeCompositeAxis
      ? CountEllipsis<Head['axes']> extends infer InnerCount
        ? InnerCount extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? CountEllipsis<Tail, Add<Count, InnerCount>>
            : Add<Count, InnerCount>
          : never
        : never
      : Tail extends readonly TypeAxisPattern[]
        ? CountEllipsis<Tail, Count>
        : Count
  : Count;

// =============================================================================
// Axis Counting
// =============================================================================

/**
 * Count simple axes in pattern list (for shape consumption)
 */
export type CountSimpleAxes<
  Patterns extends readonly TypeAxisPattern[],
  Count extends number = 0,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? CountSimpleAxes<Tail, Add<Count, 1>>
      : Add<Count, 1>
    : Head extends TypeCompositeAxis
      ? CountSimpleAxes<Head['axes']> extends infer InnerCount
        ? InnerCount extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? CountSimpleAxes<Tail, Add<Count, InnerCount>>
            : Add<Count, InnerCount>
          : never
        : never
      : Tail extends readonly TypeAxisPattern[]
        ? CountSimpleAxes<Tail, Count>
        : Count
  : Count;

/**
 * Count all axes that consume dimensions (simple, composite, singleton)
 * Used for ellipsis dimension calculation
 */
export type CountConsumingAxes<
  Patterns extends readonly TypeAxisPattern[],
  Count extends number = 0,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis | TypeSingletonAxis | TypeCompositeAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? CountConsumingAxes<Tail, Add<Count, 1>>
      : Add<Count, 1>
    : Head extends TypeEllipsisAxis
      ? Tail extends readonly TypeAxisPattern[]
        ? CountConsumingAxes<Tail, Count>
        : Count
      : never
  : Count;

/**
 * Flatten composite axes to get all simple axes
 */
export type FlattenAxes<
  Patterns extends readonly TypeAxisPattern[],
  Result extends readonly TypeSimpleAxis[] = readonly [],
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? FlattenAxes<Tail, readonly [...Result, Head]>
      : never
    : Head extends TypeCompositeAxis
      ? FlattenAxes<Head['axes']> extends infer Flattened
        ? Flattened extends readonly TypeSimpleAxis[]
          ? Tail extends readonly TypeAxisPattern[]
            ? FlattenAxes<Tail, readonly [...Result, ...Flattened]>
            : never
          : never
        : never
      : Tail extends readonly TypeAxisPattern[]
        ? FlattenAxes<Tail, Result>
        : Result
  : Result;

// =============================================================================
// Composite Validation
// =============================================================================

/**
 * Validate that composite axis dimensions match the actual dimension
 */
export type ValidateCompositeProduct<
  Axes extends readonly TypeSimpleAxis[],
  ExpectedProduct extends number,
  ProvidedAxes extends Record<string, number>,
> = ComputeCompositeProduct<Axes, ProvidedAxes> extends ExpectedProduct ? true : false;

/**
 * Compute product of composite axis dimensions
 */
export type ComputeCompositeProduct<
  Axes extends readonly TypeSimpleAxis[],
  ProvidedAxes extends Record<string, number>,
  Product extends number = 1,
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof ProvidedAxes
      ? ProvidedAxes[Head['name']] extends number
        ? Tail extends readonly TypeSimpleAxis[]
          ? ComputeCompositeProduct<
              Tail,
              ProvidedAxes,
              Multiply<Product, ProvidedAxes[Head['name']]>
            >
          : Multiply<Product, ProvidedAxes[Head['name']]>
        : Product // Axis not provided, can't validate
      : Product // Axis not provided, can't validate
    : never
  : Product;

/**
 * Check if all axes are provided
 */
export type AllAxesProvided<
  Axes extends readonly TypeSimpleAxis[],
  ProvidedAxes extends Record<string, number>,
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof ProvidedAxes
      ? Tail extends readonly TypeSimpleAxis[]
        ? AllAxesProvided<Tail, ProvidedAxes>
        : true
      : false
    : never
  : true;

/**
 * Validate all composite axes have correct products
 */
export type ValidateComposites<
  Patterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  ProvidedAxes extends Record<string, number> | undefined,
  CurrentIndex extends number = 0,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeCompositeAxis
    ? ProvidedAxes extends Record<string, number>
      ? CurrentIndex extends keyof InputShape
        ? InputShape[CurrentIndex] extends number
          ? FlattenAxes<Head['axes']> extends infer FlatAxes
            ? FlatAxes extends readonly TypeSimpleAxis[]
              ? // Check if all axes in composite are provided
                AllAxesProvided<FlatAxes, ProvidedAxes> extends true
                ? ValidateCompositeProduct<
                    FlatAxes,
                    InputShape[CurrentIndex],
                    ProvidedAxes
                  > extends true
                  ? Tail extends readonly TypeAxisPattern[]
                    ? ValidateComposites<Tail, InputShape, ProvidedAxes, Add<CurrentIndex, 1>>
                    : true
                  : false // Product mismatch
                : // Not all axes provided - check if this is valid (0 or 1 unknown) or invalid (multiple unknowns)
                  CountUnknownAxes<FlatAxes, ProvidedAxes> extends 0 | 1
                  ? Tail extends readonly TypeAxisPattern[]
                    ? ValidateComposites<Tail, InputShape, ProvidedAxes, Add<CurrentIndex, 1>>
                    : true // Valid partial composite (0 or 1 unknown axes)
                  : false // Invalid partial composite (multiple unknown axes)
              : never
            : never
          : never
        : never
      : Tail extends readonly TypeAxisPattern[]
        ? ValidateComposites<Tail, InputShape, ProvidedAxes, Add<CurrentIndex, 1>>
        : true // No provided axes, skip validation
    : Head extends TypeSimpleAxis | TypeSingletonAxis
      ? Tail extends readonly TypeAxisPattern[]
        ? ValidateComposites<Tail, InputShape, ProvidedAxes, Add<CurrentIndex, 1>>
        : true
      : Head extends TypeEllipsisAxis
        ? true // Skip ellipsis, we already validated count
        : never
  : true;

// =============================================================================
// Axis Mapping
// =============================================================================

/**
 * Build axis map from input pattern and shape
 */
export type BuildAxisMap<
  InputPatterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  ProvidedAxes extends Record<string, number> | undefined = undefined,
  CurrentIndex extends number = 0,
  Map extends AxisMap = {},
> = InputPatterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? CurrentIndex extends keyof InputShape
      ? InputShape[CurrentIndex] extends number
        ? Tail extends readonly TypeAxisPattern[]
          ? BuildAxisMap<
              Tail,
              InputShape,
              ProvidedAxes,
              Add<CurrentIndex, 1>,
              Map & Record<Head['name'], InputShape[CurrentIndex]>
            >
          : never
        : never
      : never
    : Head extends TypeCompositeAxis
      ? FlattenAxes<Head['axes']> extends infer FlatAxes
        ? FlatAxes extends readonly TypeSimpleAxis[]
          ? MapCompositeAxes<FlatAxes, InputShape, CurrentIndex, ProvidedAxes> extends infer CompMap
            ? CompMap extends AxisMap
              ? CountSimpleAxes<Head['axes']> extends infer AxisCount
                ? AxisCount extends number
                  ? Tail extends readonly TypeAxisPattern[]
                    ? BuildAxisMap<
                        Tail,
                        InputShape,
                        ProvidedAxes,
                        Add<CurrentIndex, 1>,
                        Map & CompMap
                      >
                    : never
                  : never
                : never
              : never
            : never
          : never
        : never
      : Head extends TypeEllipsisAxis
        ? Tail extends readonly TypeAxisPattern[]
          ? CountSimpleAxes<Tail> extends infer TailAxesCount
            ? TailAxesCount extends number
              ? InputShape['length'] extends number
                ? // Calculate how many dimensions ellipsis captures
                  Subtract<
                    Subtract<InputShape['length'], CurrentIndex>,
                    TailAxesCount
                  > extends infer EllipsisCount
                  ? EllipsisCount extends number
                    ? // Extract the ellipsis dimensions
                      Take<Drop<InputShape, CurrentIndex>, EllipsisCount> extends infer CapturedDims
                      ? CapturedDims extends Shape
                        ? BuildAxisMap<
                            Tail,
                            Drop<InputShape, Add<CurrentIndex, EllipsisCount>>,
                            ProvidedAxes,
                            0,
                            Map
                          >
                        : never
                      : never
                    : never
                  : never
                : never
              : never
            : never
          : never
        : Head extends TypeSingletonAxis
          ? Tail extends readonly TypeAxisPattern[]
            ? BuildAxisMap<Tail, InputShape, ProvidedAxes, Add<CurrentIndex, 1>, Map>
            : never
          : never
  : MergeIntersection<Map>;

/**
 * Map composite axes with provided dimensions
 */
type MapCompositeAxes<
  Axes extends readonly TypeSimpleAxis[],
  InputShape extends Shape,
  StartIndex extends number,
  ProvidedAxes extends Record<string, number> | undefined,
> = StartIndex extends keyof InputShape
  ? InputShape[StartIndex] extends number
    ? ProvidedAxes extends Record<string, number>
      ? ComputeCompositeAxisMap<Axes, InputShape[StartIndex], ProvidedAxes>
      : // No provided axes, distribute dimension evenly (not implemented)
        never
    : never
  : never;

/**
 * Count unknown axes in a composite pattern
 */
export type CountUnknownAxes<
  Axes extends readonly TypeSimpleAxis[],
  ProvidedAxes extends Record<string, number>,
  Count extends number = 0,
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof ProvidedAxes
      ? Tail extends readonly TypeSimpleAxis[]
        ? CountUnknownAxes<Tail, ProvidedAxes, Count>
        : Count
      : Tail extends readonly TypeSimpleAxis[]
        ? CountUnknownAxes<Tail, ProvidedAxes, Add<Count, 1>>
        : Add<Count, 1>
    : never
  : Count;

/**
 * Compute axis map for composite pattern
 *
 * TYPE-LEVEL DESIGN: Composite Axis Dimension Inference
 *
 * This type computes dimensions for composite patterns at compile time.
 * For pattern "(h w)" with total dimension 20:
 * - If h=4 provided, infers w=5
 * - If w=5 provided, infers h=4
 * - If both provided, validates h*w=20
 * - If neither provided, cannot infer (would need runtime info)
 *
 * The recursion through axes is necessary to build the type-level map.
 * We cannot use loops at type level, only recursion.
 */
export type ComputeCompositeAxisMap<
  Axes extends readonly TypeSimpleAxis[],
  TotalDim extends number,
  ProvidedAxes extends Record<string, number>,
  Result extends AxisMap = {},
  ProcessedAxes extends readonly TypeSimpleAxis[] = readonly [],
  OriginalAxes extends readonly TypeSimpleAxis[] = Axes,
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof ProvidedAxes
      ? ProvidedAxes[Head['name']] extends number
        ? Tail extends readonly TypeSimpleAxis[]
          ? ComputeCompositeAxisMap<
              Tail,
              TotalDim,
              ProvidedAxes,
              Result & Record<Head['name'], ProvidedAxes[Head['name']]>,
              readonly [...ProcessedAxes, Head],
              OriginalAxes
            >
          : never
        : never
      : // Unknown axis - compute from total and known axes
        ComputeUnknownDimension<OriginalAxes, TotalDim, ProvidedAxes> extends infer UnknownDim
        ? UnknownDim extends number
          ? Tail extends readonly TypeSimpleAxis[]
            ? ComputeCompositeAxisMap<
                Tail,
                TotalDim,
                ProvidedAxes,
                Result & Record<Head['name'], UnknownDim>,
                readonly [...ProcessedAxes, Head],
                OriginalAxes
              >
            : Result & Record<Head['name'], UnknownDim>
          : never
        : never
    : never
  : MergeIntersection<Result>;

/**
 * Compute unknown dimension from total and known axes
 *
 * TYPE ALGORITHM: Unknown Dimension Calculation
 *
 * Given total dimension and known axes, compute the unknown:
 * total = known1 * known2 * ... * unknown * ...
 * Therefore: unknown = total / (product of all known)
 *
 * CONSTRAINT: Only ONE unknown axis allowed per composite.
 * Multiple unknowns would have infinite solutions.
 */
export type ComputeUnknownDimension<
  AllAxes extends readonly TypeSimpleAxis[],
  TotalDim extends number,
  ProvidedAxes extends Record<string, number>,
  KnownProduct extends number = 1,
> =
  // Only check for multiple unknowns on the first call (when KnownProduct is 1)
  KnownProduct extends 1
    ? CountUnknownAxes<AllAxes, ProvidedAxes> extends infer UnknownCount
      ? UnknownCount extends number
        ? UnknownCount extends 0 | 1
          ? // Proceed with original logic
            ComputeUnknownDimensionImpl<AllAxes, TotalDim, ProvidedAxes, KnownProduct>
          : never // Multiple unknowns
        : never
      : never
    : ComputeUnknownDimensionImpl<AllAxes, TotalDim, ProvidedAxes, KnownProduct>;

/**
 * Implementation of ComputeUnknownDimension (original logic)
 */
type ComputeUnknownDimensionImpl<
  AllAxes extends readonly TypeSimpleAxis[],
  TotalDim extends number,
  ProvidedAxes extends Record<string, number>,
  KnownProduct extends number = 1,
> = AllAxes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof ProvidedAxes
      ? ProvidedAxes[Head['name']] extends number
        ? Tail extends readonly TypeSimpleAxis[]
          ? ComputeUnknownDimensionImpl<
              Tail,
              TotalDim,
              ProvidedAxes,
              Multiply<KnownProduct, ProvidedAxes[Head['name']]>
            >
          : never
        : never
      : Tail extends readonly TypeSimpleAxis[]
        ? ComputeUnknownDimensionImpl<Tail, TotalDim, ProvidedAxes, KnownProduct>
        : Divide<TotalDim, KnownProduct>
    : never
  : Divide<TotalDim, KnownProduct>;

/**
 * Extract ellipsis dimensions from input pattern and shape
 */
export type ExtractEllipsisDims<
  InputPatterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  CurrentIndex extends number = 0,
> = InputPatterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeEllipsisAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? CountConsumingAxes<Tail> extends infer TailCount
        ? TailCount extends number
          ? InputShape['length'] extends number
            ? // Calculate how many dimensions ellipsis captures
              Subtract<
                Subtract<InputShape['length'], CurrentIndex>,
                TailCount
              > extends infer EllipsisCount
              ? EllipsisCount extends number
                ? Take<Drop<InputShape, CurrentIndex>, EllipsisCount>
                : readonly []
              : readonly []
            : readonly []
          : readonly []
        : readonly []
      : readonly []
    : Head extends TypeSimpleAxis
      ? Tail extends readonly TypeAxisPattern[]
        ? ExtractEllipsisDims<Tail, InputShape, Add<CurrentIndex, 1>>
        : readonly []
      : Head extends TypeSingletonAxis
        ? Tail extends readonly TypeAxisPattern[]
          ? ExtractEllipsisDims<Tail, InputShape, Add<CurrentIndex, 1>>
          : readonly []
        : Head extends TypeCompositeAxis
          ? Tail extends readonly TypeAxisPattern[]
            ? ExtractEllipsisDims<Tail, InputShape, Add<CurrentIndex, 1>>
            : readonly []
          : readonly []
  : readonly [];
