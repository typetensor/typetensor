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
import type {
  AxisMap,
  Append,
  Concat,
  HasDuplicateAxisNames,
  CountEllipsis,
  ValidateComposites,
  BuildAxisMap,
  ExtractEllipsisDims,
} from './type-shape-resolver-utils';
import type { Multiply } from 'ts-arithmetic';

// =============================================================================
// Rearrange-specific Helper Types
// =============================================================================

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
export type ValidateAndComputeOutput<
  AST extends TypeEinopsAST,
  AxisMapping extends AxisMap,
  EllipsisDims extends Shape,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
> =
  ValidateComposites<AST['input'], InputShape, Axes> extends true
    ? ComputeOutputShape<AST['output'], AxisMapping, EllipsisDims>
    : never;
