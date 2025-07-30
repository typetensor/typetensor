/**
 * Type-level shape resolver for einops reduce patterns
 *
 * This module provides compile-time shape computation for reduce operations,
 * determining the output shape based on which axes are being reduced.
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
  Drop,
  CountConsumingAxes,
  ValidateComposites,
  AxisMap,
} from './type-shape-resolver-utils';
import type { Multiply } from 'ts-arithmetic';
import type { CollectAxisNames } from './validation';
import type { Add, Subtract } from 'ts-arithmetic';

// =============================================================================
// Helper Types for Reduce
// =============================================================================

/**
 * Check if an axis from input is present in output
 */
type IsAxisReduced<
  AxisName extends string,
  OutputPatterns extends readonly TypeAxisPattern[],
> = AxisName extends CollectAxisNames<OutputPatterns>[number] ? false : true;

/**
 * Validate that all output axes exist in input
 */
type ValidateOutputAxesExist<
  OutputAxes extends readonly string[],
  InputAxes extends readonly string[],
> = OutputAxes extends readonly [infer Head, ...infer Tail]
  ? Head extends string
    ? Head extends InputAxes[number]
      ? Tail extends readonly string[]
        ? ValidateOutputAxesExist<Tail, InputAxes>
        : true
      : false // Output axis not in input
    : false
  : true;

// =============================================================================
// Shape Computation for Reduce
// =============================================================================

/**
 * Build output shape for reduce operation
 */
type BuildReduceShape<
  InputPatterns extends readonly TypeAxisPattern[],
  OutputPatterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  AxisMap extends Record<string, number>,
  EllipsisDims extends Shape,
  KeepDims extends boolean,
> = KeepDims extends true
  ? BuildReduceShapeKeepDims<InputPatterns, OutputPatterns, InputShape, AxisMap, EllipsisDims>
  : ComputeOutputShapeFromPatterns<OutputPatterns, AxisMap, EllipsisDims>;

/**
 * Build shape when keepDims is true (preserve input structure with 1s for reduced dims)
 */
type BuildReduceShapeKeepDims<
  InputPatterns extends readonly TypeAxisPattern[],
  OutputPatterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  AxisMap extends Record<string, number>,
  EllipsisDims extends Shape,
  CurrentIndex extends number = 0,
  Result extends Shape = readonly [],
> = InputPatterns extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? IsAxisReduced<Head['name'], OutputPatterns> extends true
      ? Tail extends readonly TypeAxisPattern[]
        ? BuildReduceShapeKeepDims<
            Tail,
            OutputPatterns,
            InputShape,
            AxisMap,
            EllipsisDims,
            Add<CurrentIndex, 1>,
            readonly [...Result, 1]
          >
        : readonly [...Result, 1]
      : CurrentIndex extends keyof InputShape
        ? InputShape[CurrentIndex] extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? BuildReduceShapeKeepDims<
                Tail,
                OutputPatterns,
                InputShape,
                AxisMap,
                EllipsisDims,
                Add<CurrentIndex, 1>,
                readonly [...Result, InputShape[CurrentIndex]]
              >
            : readonly [...Result, InputShape[CurrentIndex]]
          : never
        : never
    : Head extends TypeCompositeAxis
      ? ProcessCompositeForReduceKeepDims<
          Head,
          OutputPatterns,
          InputShape,
          CurrentIndex,
          AxisMap
        > extends infer CompositeResult
        ? CompositeResult extends { dims: Shape; nextIndex: number }
          ? Tail extends readonly TypeAxisPattern[]
            ? BuildReduceShapeKeepDims<
                Tail,
                OutputPatterns,
                InputShape,
                AxisMap,
                EllipsisDims,
                CompositeResult['nextIndex'],
                readonly [...Result, ...CompositeResult['dims']]
              >
            : readonly [...Result, ...CompositeResult['dims']]
          : never
        : never
      : Head extends TypeEllipsisAxis
        ? ProcessEllipsisForReduceKeepDims<
            OutputPatterns,
            EllipsisDims
          > extends infer EllipsisResult
          ? EllipsisResult extends Shape
            ? Tail extends readonly TypeAxisPattern[]
              ? CountConsumingAxes<Tail> extends infer TailCount
                ? TailCount extends number
                  ? InputShape['length'] extends number
                    ? BuildReduceShapeKeepDims<
                        Tail,
                        OutputPatterns,
                        Drop<InputShape, Subtract<InputShape['length'], TailCount>>,
                        AxisMap,
                        EllipsisDims,
                        0,
                        readonly [...Result, ...EllipsisResult]
                      >
                    : never
                  : never
                : never
              : readonly [...Result, ...EllipsisResult]
            : never
          : never
        : Head extends TypeSingletonAxis
          ? Tail extends readonly TypeAxisPattern[]
            ? BuildReduceShapeKeepDims<
                Tail,
                OutputPatterns,
                InputShape,
                AxisMap,
                EllipsisDims,
                Add<CurrentIndex, 1>,
                readonly [...Result, 1]
              >
            : readonly [...Result, 1]
          : never
  : Result;

/**
 * Compute output shape directly from output patterns (for keepDims=false)
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
      ? ComputeCompositeOutputDim<Head['axes'], AxisMap> extends infer CompDim
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
 * Compute dimension for composite axis in output
 */
type ComputeCompositeOutputDim<
  Axes extends readonly TypeAxisPattern[],
  AxisMap extends Record<string, number>,
  Product extends number = 1,
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? Head['name'] extends keyof AxisMap
      ? AxisMap[Head['name']] extends number
        ? Tail extends readonly TypeAxisPattern[]
          ? ComputeCompositeOutputDim<Tail, AxisMap, Multiply<Product, AxisMap[Head['name']]>>
          : Multiply<Product, AxisMap[Head['name']]>
        : never
      : never
    : Head extends TypeCompositeAxis
      ? ComputeCompositeOutputDim<Head['axes'], AxisMap> extends infer InnerProd
        ? InnerProd extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? ComputeCompositeOutputDim<Tail, AxisMap, Multiply<Product, InnerProd>>
            : Multiply<Product, InnerProd>
          : never
        : never
      : never
  : Product;

/**
 * Process composite axis for reduction with keepDims
 */
type ProcessCompositeForReduceKeepDims<
  Composite extends TypeCompositeAxis,
  OutputPatterns extends readonly TypeAxisPattern[],
  InputShape extends Shape,
  CurrentIndex extends number,
  AxisMap extends Record<string, number>,
> = CurrentIndex extends keyof InputShape
  ? InputShape[CurrentIndex] extends number
    ? ComputeCompositeReduceShapeKeepDims<
        Composite['axes'],
        OutputPatterns,
        InputShape[CurrentIndex],
        AxisMap
      > extends infer CompositeDims
      ? CompositeDims extends Shape
        ? { dims: CompositeDims; nextIndex: Add<CurrentIndex, 1> }
        : never
      : never
    : never
  : never;

/**
 * Compute shape for composite pattern in reduction with keepDims
 */
type ComputeCompositeReduceShapeKeepDims<
  Axes extends readonly TypeAxisPattern[],
  OutputPatterns extends readonly TypeAxisPattern[],
  TotalDim extends number,
  AxisMap extends Record<string, number>,
  Result extends Shape = readonly [],
> = Axes extends readonly [infer Head, ...infer Tail]
  ? Head extends TypeSimpleAxis
    ? IsAxisReduced<Head['name'], OutputPatterns> extends true
      ? Tail extends readonly TypeAxisPattern[]
        ? ComputeCompositeReduceShapeKeepDims<
            Tail,
            OutputPatterns,
            TotalDim,
            AxisMap,
            readonly [...Result, 1]
          >
        : readonly [...Result, 1]
      : Head['name'] extends keyof AxisMap
        ? AxisMap[Head['name']] extends number
          ? Tail extends readonly TypeAxisPattern[]
            ? ComputeCompositeReduceShapeKeepDims<
                Tail,
                OutputPatterns,
                TotalDim,
                AxisMap,
                readonly [...Result, AxisMap[Head['name']]]
              >
            : readonly [...Result, AxisMap[Head['name']]]
          : never
        : never
    : never
  : Result;

/**
 * Process ellipsis for reduction with keepDims
 */
type ProcessEllipsisForReduceKeepDims<
  OutputPatterns extends readonly TypeAxisPattern[],
  EllipsisDims extends Shape,
> =
  HasEllipsis<OutputPatterns> extends true
    ? EllipsisDims // Ellipsis preserved in output
    : CreateOnesShape<EllipsisDims['length']>; // All ellipsis dims become 1

/**
 * Check if pattern has ellipsis
 */
type HasEllipsis<Patterns extends readonly TypeAxisPattern[]> = Patterns extends readonly [
  infer Head,
  ...infer Tail,
]
  ? Head extends TypeEllipsisAxis
    ? true
    : Tail extends readonly TypeAxisPattern[]
      ? HasEllipsis<Tail>
      : false
  : false;

/**
 * Create a shape of all 1s with given length
 */
type CreateOnesShape<
  Length extends number,
  Result extends Shape = readonly [],
  Counter extends readonly unknown[] = readonly [],
> = Counter['length'] extends Length
  ? Result
  : CreateOnesShape<Length, readonly [...Result, 1], readonly [...Counter, unknown]>;

// =============================================================================
// Validation for Reduce Patterns
// =============================================================================

/**
 * Validate reduce pattern
 */
export type ValidateReducePattern<AST extends TypeEinopsAST> =
  // Check no duplicate axes
  HasDuplicateAxisNames<AST['input']> extends true
    ? { valid: false; error: 'Duplicate axes in input pattern' }
    : HasDuplicateAxisNames<AST['output']> extends true
      ? { valid: false; error: 'Duplicate axes in output pattern' }
      : // Check at most one ellipsis
        CountEllipsis<AST['input']> extends 0 | 1
        ? CountEllipsis<AST['output']> extends 0 | 1
          ? // Check all output axes exist in input
            ValidateOutputAxesExist<
              CollectAxisNames<AST['output']>,
              CollectAxisNames<AST['input']>
            > extends true
            ? { valid: true } // Allow patterns where nothing is reduced (like einops)
            : { valid: false; error: 'Output contains axes not present in input' }
          : { valid: false; error: 'Multiple ellipsis in output' }
        : { valid: false; error: 'Multiple ellipsis in input' };

// =============================================================================
// Main Resolver for Reduce
// =============================================================================

export type ValidateAndComputeOutput<
  AST extends TypeEinopsAST,
  AxisMapping extends AxisMap,
  EllipsisDims extends Shape,
  InputShape extends Shape,
  Axes extends Record<string, number> | undefined,
  KeepDims extends boolean,
> =
  ValidateComposites<AST['input'], InputShape, Axes> extends true
    ? BuildReduceShape<AST['input'], AST['output'], InputShape, AxisMapping, EllipsisDims, KeepDims>
    : never;

/**
 * Resolve reduce pattern to output shape
 */
export type ResolveReduceShape<
  Pattern extends string,
  InputShape extends Shape,
  KeepDims extends boolean = false,
  Axes extends Record<string, number> | undefined = undefined,
> =
  ParsePattern<Pattern> extends infer ParsedAST
    ? ParsedAST extends TypeEinopsAST
      ? ValidateReducePattern<ParsedAST> extends { valid: true }
        ? BuildAxisMap<ParsedAST['input'], InputShape, Axes> extends infer AxisMapping
          ? AxisMapping extends Record<string, number>
            ? ExtractEllipsisDims<ParsedAST['input'], InputShape> extends infer EllipsisDims
              ? EllipsisDims extends Shape
                ? // Validate composite patterns have correct dimensions
                  ValidateAndComputeOutput<
                    ParsedAST,
                    AxisMapping,
                    EllipsisDims,
                    InputShape,
                    Axes,
                    KeepDims
                  >
                : never
              : never
            : never
          : never
        : ValidateReducePattern<ParsedAST> extends { valid: false; error: infer E }
          ? never & { __error: E }
          : never
      : ParsedAST extends TypeParseError<infer E>
        ? never & { __error: E }
        : never
    : never;
