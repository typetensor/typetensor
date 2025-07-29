/**
 * Type-level shape resolver for einops patterns
 *
 * This module provides compile-time shape computation from einops patterns,
 * enabling static type checking for tensor transformations.
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
import type { Add, Multiply, Divide, Subtract } from 'ts-arithmetic';

// =============================================================================
// Helper Types
// =============================================================================

/**
 * Check if an array has duplicate simple axis names
 */
type HasDuplicateAxisNames<
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
type CountEllipsis<
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

/**
 * Validate that composite axis dimensions match the actual dimension
 */
type ValidateCompositeProduct<
  Axes extends readonly TypeSimpleAxis[],
  ExpectedProduct extends number,
  ProvidedAxes extends Record<string, number>,
> = ComputeCompositeProduct<Axes, ProvidedAxes> extends ExpectedProduct ? true : false;

/**
 * Compute product of composite axis dimensions
 */
type ComputeCompositeProduct<
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
 * Append element to tuple
 */
type Append<T extends readonly unknown[], E> = readonly [...T, E];

/**
 * Concatenate two tuples
 */
type Concat<T1 extends readonly unknown[], T2 extends readonly unknown[]> = readonly [...T1, ...T2];

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
 * Process a simple axis in BuildAxisMap
 */
export type ProcessSimpleAxis<
  AxisName extends string,
  InputShape extends Shape,
  CurrentIndex extends number,
  Map extends AxisMap,
> = CurrentIndex extends keyof InputShape
  ? InputShape[CurrentIndex] extends number
    ? Map & Record<AxisName, InputShape[CurrentIndex]>
    : never
  : never;

/**
 * Process an ellipsis axis in BuildAxisMap
 */
export type ProcessEllipsisAxis<
  Tail extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  CurrentIndex extends number,
> =
  CountSimpleAxes<Tail> extends infer TailAxesCount
    ? TailAxesCount extends number
      ? InputShape['length'] extends number
        ? Subtract<
            Subtract<InputShape['length'], CurrentIndex>,
            TailAxesCount
          > extends infer EllipsisCount
          ? EllipsisCount extends number
            ? {
                remainingPatterns: Tail;
                remainingShape: Drop<InputShape, Add<CurrentIndex, EllipsisCount>>;
                nextIndex: 0;
                capturedDims: Take<Drop<InputShape, CurrentIndex>, EllipsisCount>;
              }
            : never
          : never
        : never
      : never
    : never;

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
  : Map;

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
export type ComputeOutputShape<
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
      : never // Output axis not found in input
    : Head extends TypeCompositeAxis
      ? ComputeCompositeOutput<Head['axes'], Map> extends infer CompDim
        ? CompDim extends number
          ? 0 extends CompDim
            ? never // Invalid composite (includes unknown axis)
            : Tail extends readonly TypeAxisPattern[]
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
      : 0 // Axis not found in map
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
      ? // Validate no duplicate axes in input
        HasDuplicateAxisNames<ParsedAST['input']> extends true
        ? never
        : // Validate at most one ellipsis
          CountEllipsis<ParsedAST['input']> extends infer InputEllipsisCount
          ? InputEllipsisCount extends number
            ? InputEllipsisCount extends 0 | 1
              ? CountEllipsis<ParsedAST['output']> extends infer OutputEllipsisCount
                ? OutputEllipsisCount extends number
                  ? OutputEllipsisCount extends 0 | 1
                    ? BuildAxisMap<ParsedAST['input'], InputShape, Axes> extends infer AxisMapping
                      ? AxisMapping extends AxisMap
                        ? ExtractEllipsisDims<
                            ParsedAST['input'],
                            InputShape
                          > extends infer EllipsisDims
                          ? EllipsisDims extends Shape
                            ? ValidateAndComputeOutput<
                                ParsedAST,
                                AxisMapping,
                                EllipsisDims,
                                InputShape,
                                Axes
                              >
                            : never
                          : never
                        : never
                      : never
                    : never // Multiple ellipsis in output
                  : never
                : never
              : never // Multiple ellipsis in input
            : never
          : never
      : ParsedAST extends TypeParseError<string>
        ? never
        : never
    : never;

/**
 * Validate and compute output shape
 */
type ValidateAndComputeOutput<
  AST extends TypeEinopsAST,
  AxisMapping extends AxisMap,
  EllipsisDims extends Shape,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
> =
  ValidateComposites<AST['input'], InputShape, Axes> extends true
    ? ComputeOutputShape<AST['output'], AxisMapping, EllipsisDims>
    : never;

/**
 * Validate all composite axes have correct products
 */
type ValidateComposites<
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
                : Tail extends readonly TypeAxisPattern[]
                  ? ValidateComposites<Tail, InputShape, ProvidedAxes, Add<CurrentIndex, 1>>
                  : true // Not all axes provided, skip validation
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

/**
 * Check if all axes are provided
 */
type AllAxesProvided<
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
