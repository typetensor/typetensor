/**
 * Type-level shape resolver for einops patterns
 *
 * This module provides compile-time shape computation from einops patterns,
 * enabling static type checking for tensor transformations.
 */

import type { Shape, Product } from '../shape/types';
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
import type { Add, Multiply, Divide } from 'ts-arithmetic';

// =============================================================================
// Helper Types
// =============================================================================

/**
 * Get element at index from tuple
 */
type At<T extends readonly unknown[], I extends number> = T[I];

/**
 * Append element to tuple
 */
type Append<T extends readonly unknown[], E> = readonly [...T, E];

/**
 * Concatenate two tuples
 */
type Concat<T1 extends readonly unknown[], T2 extends readonly unknown[]> = readonly [...T1, ...T2];

/**
 * Find index of element in array
 */
type IndexOf<
  T extends readonly unknown[],
  E,
  Acc extends readonly unknown[] = readonly [],
> = T extends readonly [infer Head, ...infer Tail]
  ? Head extends E
    ? Acc['length']
    : IndexOf<Tail, E, readonly [...Acc, unknown]>
  : -1;

/**
 * Take N elements from array
 */
type Take<
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
type Drop<
  T extends readonly unknown[],
  N extends number,
  Count extends readonly unknown[] = readonly [],
> = Count['length'] extends N
  ? T
  : T extends readonly [unknown, ...infer Tail]
    ? Drop<Tail, N, readonly [...Count, unknown]>
    : readonly [];

// =============================================================================
// Axis to Dimension Mapping
// =============================================================================

/**
 * Map of axis names to their dimension values
 */
type AxisMap = Record<string, number>;

/**
 * Merge intersection types into a single object type
 */
export type MergeIntersection<T> = T extends object ? { [K in keyof T]: T[K] } : never;

/**
 * Count simple axes in pattern list (for shape consumption)
 */
type CountSimpleAxes<
  Patterns extends readonly TypeAxisPattern[],
  Count extends number = 0,
> = Patterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Tail extends readonly TypeAxisPattern[]
      ? CountSimpleAxes<Tail, Count extends number ? number : never>
      : number
    : Head extends TypeCompositeAxis
      ? CountSimpleAxes<Head['axes']> extends infer InnerCount
        ? InnerCount extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? CountSimpleAxes<Tail, Count extends number ? number : never>
            : number
          : never
        : never
      : Tail extends readonly TypeAxisPattern[]
        ? CountSimpleAxes<Tail, Count>
        : Count
  : Count;

/**
 * Flatten composite axes to get all simple axes
 */
type FlattenAxes<
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

/**
 * Build axis map from input pattern and shape
 */
type BuildAxisMap<
  InputPatterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  ProvidedAxes extends Record<string, number> | undefined = undefined,
  CurrentIndex extends number = 0,
  Map extends AxisMap = {},
  EllipsisDims extends Shape = readonly [],
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
              Map & { [K in Head['name']]: InputShape[CurrentIndex] }
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
        ? CountSimpleAxes<Tail> extends infer TailAxes
          ? TailAxes extends number
            ? InputShape['length'] extends number
              ? Tail extends readonly TypeAxisPattern[]
                ? BuildAxisMap<
                    Tail,
                    Drop<InputShape, CurrentIndex extends number ? CurrentIndex : 0>,
                    ProvidedAxes,
                    0,
                    Map
                  >
                : never
              : never
            : never
          : never
        : Head extends TypeSingletonAxis
          ? Tail extends readonly TypeAxisPattern[]
            ? BuildAxisMap<Tail, InputShape, ProvidedAxes, CurrentIndex, Map>
            : never
          : never
  : Map;

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
 * Compute axis map for composite pattern
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
              Result & { [K in Head['name']]: ProvidedAxes[Head['name']] },
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
                Result & { [K in Head['name']]: UnknownDim },
                readonly [...ProcessedAxes, Head],
                OriginalAxes
              >
            : Result & { [K in Head['name']]: UnknownDim }
          : never
        : never
    : never
  : MergeIntersection<Result>;

/**
 * Compute unknown dimension from total and known axes
 */
export type ComputeUnknownDimension<
  AllAxes extends readonly TypeSimpleAxis[],
  TotalDim extends number,
  ProvidedAxes extends Record<string, number>,
  KnownProduct extends number = 1,
> = AllAxes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof ProvidedAxes
      ? ProvidedAxes[Head['name']] extends number
        ? Tail extends readonly TypeSimpleAxis[]
          ? ComputeUnknownDimension<
              Tail,
              TotalDim,
              ProvidedAxes,
              Multiply<KnownProduct, ProvidedAxes[Head['name']]>
            >
          : never
        : never
      : Tail extends readonly TypeSimpleAxis[]
        ? ComputeUnknownDimension<Tail, TotalDim, ProvidedAxes, KnownProduct>
        : Divide<TotalDim, KnownProduct>
    : never
  : Divide<TotalDim, KnownProduct>;

// =============================================================================
// Output Shape Computation
// =============================================================================

/**
 * Compute output shape from output patterns and axis map
 */
type ComputeOutputShape<
  OutputPatterns extends readonly TypeAxisPattern[],
  Map extends AxisMap,
  EllipsisDims extends Shape = readonly [],
  Result extends Shape = readonly [],
> = OutputPatterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof Map
      ? Map[Head['name']] extends number
        ? Tail extends readonly TypeAxisPattern[]
          ? ComputeOutputShape<Tail, Map, EllipsisDims, Append<Result, Map[Head['name']]>>
          : never
        : never
      : never
    : Head extends TypeCompositeAxis
      ? ComputeCompositeOutput<Head['axes'], Map> extends infer CompDim
        ? CompDim extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? ComputeOutputShape<Tail, Map, EllipsisDims, Append<Result, CompDim>>
            : never
          : never
        : never
      : Head extends TypeEllipsisAxis
        ? Tail extends readonly TypeAxisPattern[]
          ? ComputeOutputShape<Tail, Map, EllipsisDims, Concat<Result, EllipsisDims>>
          : never
        : Head extends TypeSingletonAxis
          ? Tail extends readonly TypeAxisPattern[]
            ? ComputeOutputShape<Tail, Map, EllipsisDims, Append<Result, 1>>
            : never
          : never
  : Result;

/**
 * Compute dimension for composite axis in output
 */
type ComputeCompositeOutput<
  Axes extends readonly TypeAxisPattern[],
  Map extends AxisMap,
  Product extends number = 1,
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof Map
      ? Map[Head['name']] extends number
        ? Tail extends readonly TypeAxisPattern[]
          ? ComputeCompositeOutput<Tail, Map, Multiply<Product, Map[Head['name']]>>
          : Multiply<Product, Map[Head['name']]>
        : never
      : never
    : Head extends TypeCompositeAxis
      ? ComputeCompositeOutput<Head['axes'], Map> extends infer InnerProd
        ? InnerProd extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? ComputeCompositeOutput<Tail, Map, Multiply<Product, InnerProd>>
            : Multiply<Product, InnerProd>
          : never
        : never
      : never
  : Product;

// =============================================================================
// Main Resolver
// =============================================================================

/**
 * Resolve einops pattern to output shape
 */
export type ResolveEinopsShape<
  Pattern extends string,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined = undefined,
> =
  ParsePattern<Pattern> extends infer ParsedAST
    ? ParsedAST extends TypeEinopsAST
      ? BuildAxisMap<ParsedAST['input'], InputShape, Axes> extends infer AxisMapping
        ? AxisMapping extends AxisMap
          ? ComputeOutputShape<ParsedAST['output'], AxisMapping>
          : never
        : never
      : ParsedAST extends TypeParseError<string>
        ? never
        : never
    : never;
