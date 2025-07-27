/**
 * Broadcasting engine for efficient tensor operations
 *
 * This module implements NumPy-compatible broadcasting rules with optimized
 * execution strategies for different broadcasting patterns.
 */

import type { Shape } from './types.js';
import { RuntimeShape, MAX_TENSOR_SIZE } from './runtime.js';

// =============================================================================
// Broadcasting Strategy Types
// =============================================================================

/**
 * Mutable array-like interface for broadcasting outputs
 * Unlike ArrayLike<number>, this permits element assignment
 */
export interface MutableArrayLike {
  readonly length: number;
  [index: number]: number;
}

/**
 * Different broadcasting patterns that can be optimized
 */
export type BroadcastStrategy =
  | 'scalar' // One operand is scalar (rank 0)
  | 'vector' // One operand is vector, simple expansion
  | 'general'; // Complex broadcasting with multiple dimension differences

/**
 * Broadcasting execution context
 */
export interface BroadcastContext {
  readonly strategy: BroadcastStrategy;
  readonly inputShapes: readonly Shape[];
  readonly outputShape: Shape;
  readonly expansions: readonly BroadcastExpansion[];
}

/**
 * Information about how each input should be expanded
 */
export interface BroadcastExpansion {
  readonly inputIndex: number;
  readonly repeats: readonly number[]; // How many times to repeat each dimension
  readonly strides: readonly number[]; // Effective strides after broadcasting
}

// =============================================================================
// Broadcasting Manager
// =============================================================================

/**
 * Main class for analyzing and executing broadcasting operations
 */
// eslint-disable-next-line @typescript-eslint/no-extraneous-class
export class BroadcastManager {
  /**
   * Analyze the broadcasting pattern for optimal execution
   */
  static analyze(shapes: readonly Shape[]): BroadcastStrategy {
    if (shapes.length < 2) {
      throw new Error('Broadcasting requires at least 2 shapes');
    }

    // Check for scalar broadcasting
    const scalarCount = shapes.filter((shape) => shape.length === 0).length;
    if (scalarCount > 0) {
      return 'scalar';
    }

    // Check for simple vector broadcasting
    if (this.isVectorBroadcast(shapes)) {
      return 'vector';
    }

    return 'general';
  }

  /**
   * Check if two shapes can be broadcast together
   */
  static canBroadcast(shape1: Shape, shape2: Shape): boolean {
    return RuntimeShape.canBroadcast(shape1, shape2);
  }

  /**
   * Compute the broadcasted shape for multiple input shapes
   */
  static broadcastShapes(shapes: readonly Shape[]): Shape {
    if (shapes.length === 0) {
      throw new Error('Cannot broadcast empty array of shapes');
    }

    if (shapes.length === 1) {
      const shape = shapes[0];
      if (!shape) {
        throw new Error('Invalid shape in array');
      }
      return shape;
    }

    // Reduce shapes pairwise
    return shapes.reduce((acc, shape) => RuntimeShape.broadcastShapes(acc, shape));
  }

  /**
   * Create a complete broadcasting context for execution
   */
  static createContext(shapes: readonly Shape[]): BroadcastContext {
    const strategy = this.analyze(shapes);
    const outputShape = this.broadcastShapes(shapes);

    // Validate output size doesn't exceed limits
    const outputSize = RuntimeShape.product(outputShape);
    if (outputSize > MAX_TENSOR_SIZE) {
      throw new Error(
        `Broadcasting would create tensor of size ${outputSize.toString()} which exceeds ` +
          `maximum safe size of ${MAX_TENSOR_SIZE.toString()}. Output shape: [${outputShape.join(', ')}]`,
      );
    }

    const expansions = this.computeExpansions(shapes, outputShape);

    return {
      strategy,
      inputShapes: shapes,
      outputShape,
      expansions,
    };
  }

  /**
   * Execute a broadcast operation with the given context and operation function
   */
  static execute<T extends MutableArrayLike>(
    context: BroadcastContext,
    inputs: readonly ArrayLike<number>[],
    operation: (...values: number[]) => number,
    outputConstructor: new (length: number) => T,
  ): T {
    const { strategy, outputShape } = context;
    const outputSize = RuntimeShape.product(outputShape);
    const output = new outputConstructor(outputSize);

    switch (strategy) {
      case 'scalar':
        return this.executeScalarBroadcast(context, inputs, operation, output);
      case 'vector':
        return this.executeVectorBroadcast(context, inputs, operation, output);
      case 'general':
        return this.executeGeneralBroadcast(context, inputs, operation, output);
      default:
        throw new Error(`Unknown broadcast strategy: ${strategy as unknown as string}`);
    }
  }

  // =============================================================================
  // Private Implementation Methods
  // =============================================================================

  /**
   * Check if this is a simple vector broadcast pattern
   */
  private static isVectorBroadcast(shapes: readonly Shape[]): boolean {
    if (shapes.length !== 2) {
      return false;
    }

    const shape1 = shapes[0];
    const shape2 = shapes[1];

    if (!shape1 || !shape2) {
      return false;
    }

    // One is vector, other is higher dimensional
    return (
      (shape1.length === 1 && shape2.length > 1) ||
      (shape2.length === 1 && shape1.length > 1) ||
      // Both vectors of different or same size (with 1s)
      (shape1.length === 1 && shape2.length === 1)
    );
  }

  /**
   * Compute expansion information for each input
   */
  private static computeExpansions(
    inputShapes: readonly Shape[],
    outputShape: Shape,
  ): BroadcastExpansion[] {
    return inputShapes.map((inputShape, inputIndex) => {
      const expansion = this.computeSingleExpansion(inputShape, outputShape);
      return {
        inputIndex,
        ...expansion,
      };
    });
  }

  /**
   * Compute expansion for a single input shape
   */
  private static computeSingleExpansion(
    inputShape: Shape,
    outputShape: Shape,
  ): Omit<BroadcastExpansion, 'inputIndex'> {
    const outputRank = outputShape.length;
    const inputRank = inputShape.length;

    // Pad input shape with 1s on the left
    const paddedInput = [...Array<number>(outputRank - inputRank).fill(1), ...inputShape];

    const repeats: number[] = [];
    const strides: number[] = [];

    let stride = 1;

    // Compute repeats and strides from right to left
    for (let i = outputRank - 1; i >= 0; i--) {
      // Validate array bounds before access
      if (i >= paddedInput.length || i >= outputShape.length) {
        throw new Error(`Index ${i.toString()} out of bounds for broadcasting computation`);
      }

      const inputDim = paddedInput[i];
      const outputDim = outputShape[i];

      if (inputDim === undefined || outputDim === undefined) {
        throw new Error(`Invalid dimension access at index ${i.toString()}`);
      }

      if (inputDim === 1 && outputDim > 1) {
        repeats[i] = outputDim;
        strides[i] = 0; // Stride 0 means repeat the same element
      } else if (inputDim === outputDim) {
        repeats[i] = 1;
        strides[i] = stride;
      } else {
        throw new Error(
          `Incompatible dimensions: input ${inputDim.toString()}, output ${outputDim.toString()}`,
        );
      }

      if (inputDim > 1) {
        stride *= inputDim;
      }
    }

    return { repeats, strides };
  }

  /**
   * Optimized execution for scalar broadcasting
   */
  private static executeScalarBroadcast<T extends MutableArrayLike>(
    context: BroadcastContext,
    inputs: readonly ArrayLike<number>[],
    operation: (...values: number[]) => number,
    output: T,
  ): T {
    const { expansions } = context;
    const scalarIndex = expansions.findIndex((exp) => {
      const shape = context.inputShapes[exp.inputIndex];
      return shape ? shape.length === 0 : false;
    });

    if (scalarIndex === -1) {
      throw new Error('No scalar found in scalar broadcast');
    }

    const scalarInput = inputs[scalarIndex];
    if (!scalarInput) {
      throw new Error('Scalar input not found');
    }

    const otherIndex = 1 - scalarIndex;
    const otherInput = inputs[otherIndex];
    if (!otherInput) {
      throw new Error('Other input not found');
    }

    const scalarValue = scalarInput[0];
    if (scalarValue === undefined) {
      throw new Error('Scalar value is undefined');
    }

    // Vectorized operation with scalar
    for (let i = 0; i < output.length; i++) {
      const values = inputs.map((_, idx) => {
        if (idx === scalarIndex) {
          return scalarValue;
        }
        const value = otherInput[i];
        if (value === undefined) {
          throw new Error(`Input value at index ${i.toString()} is undefined`);
        }
        return value;
      });
      output[i] = operation(...values);
    }

    return output;
  }

  /**
   * Optimized execution for vector broadcasting
   */
  private static executeVectorBroadcast<T extends MutableArrayLike>(
    context: BroadcastContext,
    inputs: readonly ArrayLike<number>[],
    operation: (...values: number[]) => number,
    output: T,
  ): T {
    const { expansions, outputShape } = context;

    // Create output shape manager for index calculations
    const outputShapeObj = new RuntimeShape(outputShape);

    // Optimized loop for vector broadcasting
    for (let i = 0; i < output.length; i++) {
      const outputIndices = outputShapeObj.unravel(i);

      const values = inputs.map((input, inputIdx) => {
        const expansion = expansions[inputIdx];
        if (!expansion) {
          throw new Error(`No expansion found for input ${inputIdx.toString()}`);
        }

        const inputShape = context.inputShapes[inputIdx];
        if (!inputShape) {
          throw new Error(`No input shape found for input ${inputIdx.toString()}`);
        }

        const inputIndex = this.computeInputIndex(outputIndices, expansion, inputShape);
        const value = input[inputIndex];
        if (value === undefined) {
          throw new Error(`Input value at vector index ${inputIndex.toString()} is undefined`);
        }
        return value;
      });

      output[i] = operation(...values);
    }

    return output;
  }

  /**
   * General broadcasting execution for complex patterns
   */
  private static executeGeneralBroadcast<T extends MutableArrayLike>(
    context: BroadcastContext,
    inputs: readonly ArrayLike<number>[],
    operation: (...values: number[]) => number,
    output: T,
  ): T {
    const { expansions, outputShape } = context;
    const outputShapeObj = new RuntimeShape(outputShape);

    // General case: compute input indices for each output position
    for (let i = 0; i < output.length; i++) {
      const outputIndices = outputShapeObj.unravel(i);

      const values = inputs.map((input, inputIdx) => {
        const expansion = expansions[inputIdx];
        if (!expansion) {
          throw new Error(`No expansion found for input ${inputIdx.toString()}`);
        }

        const inputShape = context.inputShapes[inputIdx];
        if (!inputShape) {
          throw new Error(`No input shape found for input ${inputIdx.toString()}`);
        }

        const inputIndex = this.computeInputIndex(outputIndices, expansion, inputShape);
        const value = input[inputIndex];
        if (value === undefined) {
          throw new Error(`Input value at general index ${inputIndex.toString()} is undefined`);
        }
        return value;
      });

      output[i] = operation(...values);
    }

    return output;
  }

  /**
   * Compute the linear input index from output indices and expansion info
   */
  private static computeInputIndex(
    outputIndices: number[],
    expansion: BroadcastExpansion,
    _inputShape: Shape,
  ): number {
    const { strides } = expansion;
    const outputRank = outputIndices.length;

    let inputIndex = 0;

    // Map output indices to input indices, handling broadcasting
    for (let i = 0; i < outputRank; i++) {
      if (i >= outputIndices.length || i >= strides.length) {
        throw new Error(
          `Index ${i.toString()} out of bounds for dimension ${outputRank.toString()}`,
        );
      }

      const outputIdx = outputIndices[i];
      const stride = strides[i];

      if (outputIdx === undefined) {
        throw new Error(`Output index at ${i.toString()} is undefined`);
      }
      if (stride === undefined) {
        throw new Error(`No stride found for dimension ${i.toString()}`);
      }

      if (stride > 0) {
        inputIndex += outputIdx * stride;
      }
      // If stride is 0, this dimension is broadcast (repeated), so don't advance
    }

    return inputIndex;
  }
}

// =============================================================================
// Specialized Broadcasting Utilities
// =============================================================================

/**
 * High-performance broadcasting for binary operations
 */
// eslint-disable-next-line @typescript-eslint/no-extraneous-class
export class BinaryBroadcaster {
  /**
   * Execute a binary operation with broadcasting
   */
  static execute<T extends MutableArrayLike>(
    shape1: Shape,
    input1: ArrayLike<number>,
    shape2: Shape,
    input2: ArrayLike<number>,
    operation: (a: number, b: number) => number,
    outputConstructor: new (length: number) => T,
  ): { result: T; shape: Shape } {
    // Quick path for identical shapes
    if (RuntimeShape.equals(shape1, shape2)) {
      return this.executeIdenticalShapes(input1, input2, operation, outputConstructor, shape1);
    }

    // Quick path for scalar operations
    if (shape1.length === 0) {
      const scalar = input1[0];
      if (scalar === undefined) {
        throw new Error('Scalar value is undefined');
      }
      return this.executeScalarLeft(scalar, input2, operation, outputConstructor, shape2);
    }
    if (shape2.length === 0) {
      const scalar = input2[0];
      if (scalar === undefined) {
        throw new Error('Scalar value is undefined');
      }
      return this.executeScalarRight(input1, scalar, operation, outputConstructor, shape1);
    }

    // General broadcasting
    const context = BroadcastManager.createContext([shape1, shape2]);
    const result = BroadcastManager.execute(
      context,
      [input1, input2],
      operation,
      outputConstructor,
    );

    return { result, shape: context.outputShape };
  }

  /**
   * Optimized execution for identical shapes
   */
  private static executeIdenticalShapes<T extends MutableArrayLike>(
    input1: ArrayLike<number>,
    input2: ArrayLike<number>,
    operation: (a: number, b: number) => number,
    outputConstructor: new (length: number) => T,
    shape: Shape,
  ): { result: T; shape: Shape } {
    const result = new outputConstructor(input1.length);

    for (let i = 0; i < input1.length; i++) {
      const val1 = input1[i];
      const val2 = input2[i];
      if (val1 === undefined || val2 === undefined) {
        throw new Error(`Input values at index ${i.toString()} are undefined`);
      }
      result[i] = operation(val1, val2);
    }

    return { result, shape };
  }

  /**
   * Optimized execution for left scalar
   */
  private static executeScalarLeft<T extends MutableArrayLike>(
    scalar: number,
    input: ArrayLike<number>,
    operation: (a: number, b: number) => number,
    outputConstructor: new (length: number) => T,
    shape: Shape,
  ): { result: T; shape: Shape } {
    const result = new outputConstructor(input.length);

    for (let i = 0; i < input.length; i++) {
      const value = input[i];
      if (value === undefined) {
        throw new Error(`Input value at index ${i.toString()} is undefined`);
      }
      result[i] = operation(scalar, value);
    }

    return { result, shape };
  }

  /**
   * Optimized execution for right scalar
   */
  private static executeScalarRight<T extends MutableArrayLike>(
    input: ArrayLike<number>,
    scalar: number,
    operation: (a: number, b: number) => number,
    outputConstructor: new (length: number) => T,
    shape: Shape,
  ): { result: T; shape: Shape } {
    const result = new outputConstructor(input.length);

    for (let i = 0; i < input.length; i++) {
      const value = input[i];
      if (value === undefined) {
        throw new Error(`Input value at index ${i.toString()} is undefined`);
      }
      result[i] = operation(value, scalar);
    }

    return { result, shape };
  }
}

// =============================================================================
// Broadcasting for Reduction Operations
// =============================================================================

/**
 * Broadcasting utilities for reduction operations (sum, mean, etc.)
 */
// eslint-disable-next-line @typescript-eslint/no-extraneous-class
export class ReductionBroadcaster {
  /**
   * Compute the shape after reduction along specific axes
   */
  static getReducedShape(inputShape: Shape, axes?: number[], keepDims = false): Shape {
    if (axes === undefined) {
      // Reduce all dimensions
      return keepDims ? (inputShape.map(() => 1) as Shape) : ([] as Shape);
    }

    const inputRank = inputShape.length;
    const normalizedAxes = axes.map((axis) => (axis < 0 ? inputRank + axis : axis));

    // Validate axes
    for (const axis of normalizedAxes) {
      if (axis < 0 || axis >= inputRank) {
        throw new Error(
          `Reduction axis ${axis.toString()} out of bounds for rank ${inputRank.toString()} tensor`,
        );
      }
    }

    if (keepDims) {
      return inputShape.map((dim, idx) => (normalizedAxes.includes(idx) ? 1 : dim)) as Shape;
    } else {
      return inputShape.filter((_, idx) => !normalizedAxes.includes(idx)) as Shape;
    }
  }

  /**
   * Execute a reduction operation
   */
  static reduce<T extends MutableArrayLike>(
    inputShape: Shape,
    input: ArrayLike<number>,
    operation: (accumulator: number, value: number) => number,
    initialValue: number,
    axes?: number[],
    keepDims = false,
    outputConstructor: new (length: number) => T = Float32Array as unknown as new (
      length: number,
    ) => T,
  ): { result: T; shape: Shape } {
    const outputShape = this.getReducedShape(inputShape, axes, keepDims);
    const outputSize = RuntimeShape.product(outputShape);
    const result = new outputConstructor(outputSize);

    if (axes === undefined) {
      // Reduce all elements to scalar
      let acc = initialValue;
      for (let i = 0; i < input.length; i++) {
        const value = input[i];
        if (value === undefined) {
          throw new Error(`Input value at index ${i.toString()} is undefined`);
        }
        acc = operation(acc, value);
      }
      result[0] = acc;
    } else {
      // Reduce along specific axes
      this.executeAxisReduction(
        inputShape,
        input,
        outputShape,
        result,
        axes,
        operation,
        initialValue,
      );
    }

    return { result, shape: outputShape };
  }

  /**
   * Execute reduction along specific axes
   */

  private static executeAxisReduction<T extends MutableArrayLike>(
    inputShape: Shape,
    input: ArrayLike<number>,
    outputShape: Shape,
    output: T,
    axes: number[],
    operation: (acc: number, val: number) => number,
    initialValue: number,
  ): void {
    const inputShapeObj = new RuntimeShape(inputShape);
    const outputShapeObj = new RuntimeShape(outputShape);
    const inputRank = inputShape.length;

    // Normalize axes
    const normalizedAxes = axes.map((axis) => (axis < 0 ? inputRank + axis : axis));

    // Initialize output
    for (let i = 0; i < output.length; i++) {
      output[i] = initialValue;
    }

    // Determine if keepDims was used by comparing ranks
    const keepDims = inputShape.length === outputShape.length;

    // Iterate through all input elements
    for (let inputIdx = 0; inputIdx < input.length; inputIdx++) {
      const inputIndices = inputShapeObj.unravel(inputIdx);

      let outputIndices: number[];
      if (keepDims) {
        // With keepDims, map reduced dimensions to 0, keep others as-is
        outputIndices = inputIndices.map((idx, dimIdx) =>
          normalizedAxes.includes(dimIdx) ? 0 : idx,
        );
      } else {
        // Without keepDims, filter out reduced dimensions
        outputIndices = inputIndices.filter((_, dimIdx) => !normalizedAxes.includes(dimIdx));
      }

      const outputIdx =
        outputIndices.length > 0 && outputShape.length > 0
          ? outputShapeObj.ravel(outputIndices)
          : 0;

      const currentOutput = output[outputIdx];
      const inputValue = input[inputIdx];
      if (currentOutput === undefined || inputValue === undefined) {
        throw new Error(
          `Values at indices ${outputIdx.toString()}/${inputIdx.toString()} are undefined`,
        );
      }
      output[outputIdx] = operation(currentOutput, inputValue);
    }
  }
}

// =============================================================================
// Export convenience functions
// =============================================================================

/**
 * Check if shapes can be broadcast
 */
export function canBroadcast(...shapes: Shape[]): boolean {
  if (shapes.length < 2) {
    return true;
  }

  try {
    BroadcastManager.broadcastShapes(shapes);
    return true;
  } catch {
    return false;
  }
}

/**
 * Compute broadcast shape for multiple inputs
 */
export function broadcastShapes(...shapes: Shape[]): Shape {
  return BroadcastManager.broadcastShapes(shapes);
}

/**
 * Execute binary operation with broadcasting
 */
export function broadcastBinaryOp<T extends MutableArrayLike>(
  shape1: Shape,
  input1: ArrayLike<number>,
  shape2: Shape,
  input2: ArrayLike<number>,
  operation: (a: number, b: number) => number,
  outputConstructor: new (length: number) => T,
): { result: T; shape: Shape };
export function broadcastBinaryOp(
  shape1: Shape,
  input1: ArrayLike<number>,
  shape2: Shape,
  input2: ArrayLike<number>,
  operation: (a: number, b: number) => number,
): { result: Float32Array; shape: Shape };
export function broadcastBinaryOp<T extends MutableArrayLike>(
  shape1: Shape,
  input1: ArrayLike<number>,
  shape2: Shape,
  input2: ArrayLike<number>,
  operation: (a: number, b: number) => number,
  outputConstructor?: new (length: number) => T,
): { result: T | Float32Array; shape: Shape } {
  if (outputConstructor) {
    return BinaryBroadcaster.execute(shape1, input1, shape2, input2, operation, outputConstructor);
  } else {
    return BinaryBroadcaster.execute(shape1, input1, shape2, input2, operation, Float32Array);
  }
}
