/**
 * Repeat operation for einops patterns
 *
 * This module provides the repeat functionality for replicating tensor
 * elements using einops pattern syntax. Unlike reduce, repeat can create
 * new axes and repeat elements along existing axes.
 */

import { parse } from './scanner';
import { type ResolvedPattern } from './axis-resolver';
import type { EinopsAST, AxisPattern, CompositeAxis, SimpleAxis } from './ast';
import {
  isSimpleAxis,
  isCompositeAxis,
  isEllipsisAxis,
  isSingletonAxis,
  getAxisNames,
} from './ast';
import { Tensor, ChainablePromise } from '../tensor/tensor';
import type { AnyStorageTransformation, AnyTensorStorage } from '../storage/layout';
import type { RepeatOp } from '../storage/einops';
import type { ValidRepeatPattern } from './type-shape-resolver-repeat';

// =============================================================================
// Types
// =============================================================================

/**
 * Options for repeat operation
 */
export interface RepeatOptions {
  /**
   * Required axis dimensions for new axes and repetition factors
   * New axes in repeat output require explicit size specifications
   */
  readonly axes?: Record<string, number>;
}

/**
 * Planned tensor operation for repetition
 */
interface RepeatOperation {
  readonly type:
    | 'reshape'
    | 'expand'
    | 'tile'
    | 'identity'
    | 'repeat_with_new_axes'
    | 'composite_repetition'
    | 'create_with_coordinate_mapping';
  readonly params?: {
    readonly shape?: readonly number[];
    readonly reps?: readonly number[];
    readonly targetShape?: readonly number[];
    readonly newAxes?: readonly string[];
    readonly inputPatterns?: readonly AxisPattern[];
    readonly outputPatterns?: readonly AxisPattern[];
    readonly axisDimensions?: Map<string, number>;
  };
}

/**
 * Error thrown during repeat operation
 */
export class RepeatError extends Error {
  constructor(
    message: string,
    public readonly pattern: string,
    public readonly context?: {
      inputShape?: readonly number[];
      outputShape?: readonly number[];
      newAxes?: string[];
      repeatedAxes?: string[];
    },
  ) {
    super(message);
    this.name = 'RepeatError';
  }
}

// =============================================================================
// Main Repeat Function
// =============================================================================

/**
 * Repeat tensor elements according to einops pattern
 *
 * @param tensor - Input tensor to repeat (can be Tensor or ChainablePromise)
 * @param pattern - Einops pattern like "h w -> h w c" or "h w -> (h h2) w"
 * @param axes - Required axis dimensions for new axes and repetition factors
 * @returns ChainablePromise with repeated tensor
 *
 * @example
 * ```typescript
 * // Add new axis (convert grayscale to RGB)
 * const rgb = await repeat(grayscale, 'h w -> h w c', { c: 3 });
 *
 * // Repeat along existing axis (2x upsampling)
 * const upsampled = await repeat(image, 'h w -> (h h2) (w w2)', { h2: 2, w2: 2 });
 *
 * // Add batch dimension
 * const batch = await repeat(image, 'h w c -> batch h w c', { batch: 8 });
 *
 * // Complex repetition pattern
 * const tiled = await repeat(patch, 'h w -> (h h2) (w w2) c', { h2: 3, w2: 3, c: 4 });
 * ```
 */
export function repeat<
  S extends AnyStorageTransformation & { __output: AnyTensorStorage },
  Pattern extends string,
  const Axes extends Record<string, number> | undefined = undefined,
>(
  tensor: Tensor<S> | ChainablePromise<S>,
  pattern: ValidRepeatPattern<Pattern, S['__output']['__shape'], Axes> extends string
    ? ValidRepeatPattern<Pattern, S['__output']['__shape'], Axes> // Show the actual error message
    : Pattern,
  axes?: Axes,
): ChainablePromise<RepeatOp<S['__output'], Pattern, Axes>> {
  return new ChainablePromise((resolve, reject) => {
    void (async () => {
      try {
        // Resolve tensor if it's a ChainablePromise
        const resolvedTensor = tensor instanceof ChainablePromise ? await tensor : tensor;

        if (!(resolvedTensor instanceof Tensor)) {
          throw new Error('Expected a Tensor instance');
        }

        // Step 1: Parse the pattern
        const ast = parse(pattern);

        // Step 2: Validate the pattern follows repeat rules
        validateRepeatPattern(ast, axes);

        // Step 3: Resolve axes against tensor shape
        const resolved = resolveAxes(ast, resolvedTensor.shape, axes);

        // Step 4: Plan operations
        const operations = planRepeatOperations(ast, resolvedTensor.shape, resolved);

        // Step 5: Execute operations
        const result = await executeRepeatOperations(resolvedTensor, operations);

        // Return with correct type
        resolve(result as unknown as Tensor<RepeatOp<S['__output'], Pattern, Axes>>);
      } catch (error) {
        if (error instanceof Error) {
          const inputShape = tensor instanceof ChainablePromise ? undefined : tensor.shape;
          reject(
            new RepeatError(`Failed to repeat tensor: ${error.message}`, pattern, {
              // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-explicit-any
              inputShape: inputShape as any,
            }),
          );
        } else {
          reject(error);
        }
      }
    })();
  });
}

// =============================================================================
// Validation Functions
// =============================================================================

/**
 * Validate that a pattern is valid for repeat operation
 * Rules:
 * 1. New axes in output require explicit sizes in axes parameter
 * 2. No duplicate axes in input or output
 * 3. Repetition factors must be positive integers
 */
function validateRepeatPattern(ast: EinopsAST, providedAxes?: Record<string, number>): void {
  // Check for multiple ellipsis
  const inputEllipsisCount = ast.input.filter((p) => isEllipsisAxis(p)).length;
  const outputEllipsisCount = ast.output.filter((p) => isEllipsisAxis(p)).length;

  if (inputEllipsisCount > 1) {
    throw new RepeatError(
      'Multiple ellipsis (...) in input pattern is not allowed',
      ast.metadata.originalPattern,
    );
  }

  if (outputEllipsisCount > 1) {
    throw new RepeatError(
      'Multiple ellipsis (...) in output pattern is not allowed',
      ast.metadata.originalPattern,
    );
  }

  // Collect all axis names (not singletons)
  const inputAxes = new Set(getAxisNames(ast.input));
  const outputAxes = new Set(getAxisNames(ast.output));

  // Check for duplicate axes in input
  const inputAxisList = getAxisNames(ast.input);
  const inputDuplicates = inputAxisList.filter(
    (axis, index) => inputAxisList.indexOf(axis) !== index,
  );
  if (inputDuplicates.length > 0) {
    throw new RepeatError(
      `Duplicate axes in input pattern: {${[...new Set(inputDuplicates)].join(', ')}}`,
      ast.metadata.originalPattern,
    );
  }

  // Check for duplicate axes in output
  const outputAxisList = getAxisNames(ast.output);
  const outputDuplicates = outputAxisList.filter(
    (axis, index) => outputAxisList.indexOf(axis) !== index,
  );
  if (outputDuplicates.length > 0) {
    throw new RepeatError(
      `Duplicate axes in output pattern: {${[...new Set(outputDuplicates)].join(', ')}}`,
      ast.metadata.originalPattern,
    );
  }

  // Find new axes (axes in output but not in input)
  const newAxes = [...outputAxes].filter((axis) => !inputAxes.has(axis));

  // Check that all new axes have provided sizes
  if (newAxes.length > 0) {
    if (!providedAxes) {
      throw new RepeatError(
        `New axes in output require explicit sizes: {${newAxes.join(', ')}}. ` +
          `Specify: repeat(tensor, pattern, {${newAxes.map((axis) => `${axis}: number`).join(', ')}})`,
        ast.metadata.originalPattern,
      );
    }

    const missingAxes = newAxes.filter((axis) => !(axis in providedAxes));
    if (missingAxes.length > 0) {
      throw new RepeatError(
        `Missing sizes for new axes: {${missingAxes.join(', ')}}. ` +
          `Specify: repeat(tensor, pattern, {${missingAxes.map((axis) => `${axis}: number`).join(', ')}})`,
        ast.metadata.originalPattern,
      );
    }

    // Validate all new axis sizes are positive integers
    for (const axis of newAxes) {
      const size = providedAxes[axis];
      if (size === undefined || size <= 0 || !Number.isInteger(size)) {
        throw new RepeatError(
          `Invalid size for axis '${axis}': ${size}. Repeat sizes must be positive integers`,
          ast.metadata.originalPattern,
        );
      }
    }
  }

  // Validate provided axes that aren't new axes (repetition factors)
  if (providedAxes) {
    for (const [axis, size] of Object.entries(providedAxes)) {
      if (size <= 0 || !Number.isInteger(size)) {
        throw new RepeatError(
          `Invalid size for axis '${axis}': ${size}. Repeat sizes must be positive integers`,
          ast.metadata.originalPattern,
        );
      }
    }
  }
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Custom axis resolution for repeat operations
 * Unlike reduce/rearrange, repeat can create new axes that don't exist in input
 */
function resolveAxes(
  ast: EinopsAST,
  inputShape: readonly number[],
  providedAxes?: Record<string, number>,
): ResolvedPattern {
  // Step 1: Resolve input pattern axes using standard resolver but without output validation
  const axisDimensions = new Map<string, number>();
  let ellipsisDimensions: readonly number[] | undefined;

  // Resolve input pattern to get dimensions of known axes
  const inputResolution = resolveInputPatternOnly(ast.input, inputShape, providedAxes);
  inputResolution.axisDimensions.forEach((value, key) => {
    axisDimensions.set(key, value);
  });
  ellipsisDimensions = inputResolution.ellipsisDimensions;

  // Step 2: Add new axes from providedAxes that appear in output but not input
  const inputAxes = new Set(getAxisNames(ast.input));
  const outputAxes = new Set(getAxisNames(ast.output));
  const newAxes = [...outputAxes].filter((axis) => !inputAxes.has(axis));

  if (providedAxes) {
    for (const axis of newAxes) {
      if (axis in providedAxes) {
        const axisValue = providedAxes[axis];
        if (axisValue !== undefined) {
          axisDimensions.set(axis, axisValue);
        }
      }
    }
  }

  // Step 3: Validate all output axes are now known (including new ones)
  validateAllOutputAxesKnown(ast.output, axisDimensions);

  // Step 4: Compute output shape
  const outputShape = computeOutputShapeFromPatterns(
    ast.output,
    axisDimensions,
    ellipsisDimensions,
  );

  return {
    axisDimensions,
    outputShape,
    ellipsisDimensions,
  };
}

/**
 * Resolve only the input pattern without validating output axes
 */
function resolveInputPatternOnly(
  patterns: readonly AxisPattern[],
  shape: readonly number[],
  providedAxes?: Record<string, number>,
): { axisDimensions: Map<string, number>; ellipsisDimensions?: readonly number[] } {
  const axisDimensions = new Map<string, number>();
  let shapeIndex = 0;
  let ellipsisDimensions: number[] | undefined;

  // Handle empty pattern (scalar)
  if (patterns.length === 0) {
    if (shape.length !== 0) {
      throw new Error(
        `Empty pattern expects scalar (0-dimensional) tensor but got shape [${shape.join(', ')}]`,
      );
    }
    return { axisDimensions };
  }

  for (let i = 0; i < patterns.length; i++) {
    const pattern = patterns[i];
    if (!pattern) continue;

    if (isSimpleAxis(pattern)) {
      // Simple axis: direct mapping
      if (shapeIndex >= shape.length) {
        throw new Error(`Pattern has more axes than tensor dimensions`);
      }

      const actualDim = shape[shapeIndex];
      shapeIndex++;

      // If a value was provided, verify it matches
      if (providedAxes?.[pattern.name] !== undefined) {
        const expectedDim = providedAxes[pattern.name];
        if (actualDim !== expectedDim) {
          throw new Error(`Axis '${pattern.name}' expected ${expectedDim} but got ${actualDim}`);
        }
      }

      if (actualDim !== undefined) {
        axisDimensions.set(pattern.name, actualDim);
      }
    } else if (isCompositeAxis(pattern)) {
      // Composite axis: resolve nested axes
      if (shapeIndex >= shape.length) {
        throw new Error(`Pattern has more axes than tensor dimensions`);
      }
      const compositeDim = shape[shapeIndex];
      if (compositeDim !== undefined) {
        shapeIndex++;
        resolveCompositeAxisCustom(pattern, compositeDim, axisDimensions, providedAxes);
      }
    } else if (isEllipsisAxis(pattern)) {
      // Ellipsis: consume remaining dimensions
      const remainingPatterns = patterns.length - i - 1;
      const remainingDims = shape.length - shapeIndex;

      if (remainingDims < remainingPatterns) {
        throw new Error(`Not enough dimensions for pattern after ellipsis`);
      }

      const ellipsisCount = remainingDims - remainingPatterns;
      ellipsisDimensions = [];
      for (let j = 0; j < ellipsisCount; j++) {
        const dim = shape[shapeIndex + j];
        if (dim !== undefined) {
          ellipsisDimensions.push(dim);
        }
      }
      shapeIndex += ellipsisCount;
    } else if (isSingletonAxis(pattern)) {
      // Singleton: verify dimension is 1
      if (shapeIndex >= shape.length) {
        throw new Error(`Pattern has more axes than tensor dimensions`);
      }
      const dim = shape[shapeIndex];
      if (dim !== 1) {
        throw new Error(`Expected singleton dimension but got ${dim}`);
      }
      shapeIndex++;
    }
  }

  // Verify all dimensions were consumed
  if (shapeIndex < shape.length) {
    throw new Error(`Pattern does not consume all tensor dimensions`);
  }

  return { axisDimensions, ellipsisDimensions: ellipsisDimensions as readonly number[] };
}

/**
 * Custom composite axis resolution for repeat
 */
function resolveCompositeAxisCustom(
  composite: CompositeAxis,
  totalDim: number,
  axisDimensions: Map<string, number>,
  providedAxes?: Record<string, number>,
): void {
  // Flatten nested composites to get all simple axes
  const simpleAxes = flattenCompositeCustom(composite);
  const innerAxes: { axis: SimpleAxis; value?: number }[] = [];
  let knownProduct = 1;
  let unknownCount = 0;

  // Collect known values for all simple axes
  for (const axis of simpleAxes) {
    const value = providedAxes?.[axis.name] ?? axisDimensions.get(axis.name);
    if (value !== undefined) {
      innerAxes.push({ axis, value });
      knownProduct *= value;
    } else {
      innerAxes.push({ axis });
      unknownCount++;
    }
  }

  // Handle different cases
  if (unknownCount === 0) {
    // All axes known: validate product equals total dimension
    if (knownProduct !== totalDim) {
      throw new Error(
        `Composite dimension mismatch: product of axes ${knownProduct} does not equal dimension ${totalDim}`,
      );
    }

    // Set all provided values in axisDimensions
    for (const { axis, value } of innerAxes) {
      if (value !== undefined) {
        axisDimensions.set(axis.name, value);
      }
    }
  } else if (unknownCount === 1) {
    // One unknown: can infer its value
    if (totalDim % knownProduct !== 0) {
      throw new Error(
        `Cannot evenly split dimension ${totalDim} with known product ${knownProduct}`,
      );
    }

    const inferredValue = totalDim / knownProduct;

    // Set the inferred value
    for (const { axis, value } of innerAxes) {
      if (value === undefined) {
        axisDimensions.set(axis.name, inferredValue);
      } else {
        axisDimensions.set(axis.name, value);
      }
    }
  } else {
    // Multiple unknowns: cannot infer
    throw new Error(`Cannot infer multiple unknown dimensions in composite pattern`);
  }
}

/**
 * Flatten a composite axis to extract all simple axes
 */
function flattenCompositeCustom(composite: CompositeAxis): SimpleAxis[] {
  const result: SimpleAxis[] = [];

  for (const axis of composite.axes) {
    if (isSimpleAxis(axis)) {
      result.push(axis);
    } else if (isCompositeAxis(axis)) {
      // Recursively flatten nested composites
      result.push(...flattenCompositeCustom(axis));
    } else {
      throw new Error(`Composite patterns can only contain simple axes or other composites`);
    }
  }

  return result;
}

/**
 * Validate that all output axes are known (including new ones)
 */
function validateAllOutputAxesKnown(
  patterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
): void {
  for (const pattern of patterns) {
    if (isSimpleAxis(pattern)) {
      if (!axisDimensions.has(pattern.name)) {
        throw new Error(`Unknown axis '${pattern.name}' in output pattern`);
      }
    } else if (isCompositeAxis(pattern)) {
      // Use flattening to handle nested composites properly
      const simpleAxes = flattenCompositeCustom(pattern);
      for (const innerAxis of simpleAxes) {
        if (!axisDimensions.has(innerAxis.name)) {
          throw new Error(`Unknown axis '${innerAxis.name}' in output pattern`);
        }
      }
    }
    // Ellipsis and singleton are always valid in output
  }
}

/**
 * Compute output shape from patterns
 */
function computeOutputShapeFromPatterns(
  patterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
  ellipsisDimensions?: readonly number[],
): number[] {
  const outputShape: number[] = [];

  for (const pattern of patterns) {
    if (isSimpleAxis(pattern)) {
      const dim = axisDimensions.get(pattern.name);
      if (dim === undefined) {
        throw new Error(`Internal error: axis '${pattern.name}' not found`);
      }
      outputShape.push(dim);
    } else if (isCompositeAxis(pattern)) {
      // Composite in output: compute product of all simple axes (including nested)
      const simpleAxes = flattenCompositeCustom(pattern);
      let product = 1;
      for (const axis of simpleAxes) {
        const dim = axisDimensions.get(axis.name);
        if (dim === undefined) {
          throw new Error(`Internal error: axis '${axis.name}' not found`);
        }
        product *= dim;
      }
      outputShape.push(product);
    } else if (isEllipsisAxis(pattern)) {
      // Ellipsis: insert all ellipsis dimensions
      if (ellipsisDimensions) {
        outputShape.push(...ellipsisDimensions);
      }
    } else if (isSingletonAxis(pattern)) {
      // Singleton: always 1
      outputShape.push(1);
    }
  }

  return outputShape;
}

/**
 * Compute the shape that includes all axes from the output pattern
 */
function computeExtendedShape(
  outputPatterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
  ellipsisDimensions?: readonly number[],
): number[] {
  const shape: number[] = [];

  for (const pattern of outputPatterns) {
    if (isSimpleAxis(pattern)) {
      const dim = axisDimensions.get(pattern.name);
      if (dim === undefined) {
        throw new Error(`Axis ${pattern.name} not found in dimensions map`);
      }
      shape.push(dim);
    } else if (isCompositeAxis(pattern)) {
      // Compute composite dimension
      const compositeDim = computeCompositeShape(pattern, axisDimensions);
      shape.push(compositeDim);
    } else if (isSingletonAxis(pattern)) {
      shape.push(1);
    } else if (isEllipsisAxis(pattern)) {
      if (ellipsisDimensions) {
        shape.push(...ellipsisDimensions);
      }
    }
  }

  return shape;
}

/**
 * Compute dimension for composite axis
 */
function computeCompositeShape(
  composite: CompositeAxis,
  axisDimensions: Map<string, number>,
): number {
  let product = 1;

  for (const pattern of composite.axes) {
    if (isSimpleAxis(pattern)) {
      const dim = axisDimensions.get(pattern.name);
      if (dim === undefined) {
        throw new Error(`Axis ${pattern.name} not found in dimensions map`);
      }
      product *= dim;
    } else if (isCompositeAxis(pattern)) {
      product *= computeCompositeShape(pattern, axisDimensions);
    }
    // Skip singletons in composites (they contribute factor of 1)
  }

  return product;
}

/**
 * Plan the sequence of operations for repetition
 * Strategy: Use expand() for new singleton dimensions, tile() for repetition
 */
function planRepeatOperations(
  ast: EinopsAST,
  inputShape: readonly number[],
  resolved: ResolvedPattern,
): RepeatOperation[] {
  const operations: RepeatOperation[] = [];

  // Compute final output shape
  const outputShape = computeExtendedShape(
    ast.output,
    resolved.axisDimensions,
    resolved.ellipsisDimensions,
  );

  // Check if this is just an identity operation
  if (arraysEqual(inputShape, outputShape)) {
    operations.push({ type: 'identity' });
    return operations;
  }

  // Strategy depends on the type of repetition needed
  const inputAxes = new Set(getAxisNames(ast.input));
  const outputAxes = new Set(getAxisNames(ast.output));
  const newAxes = [...outputAxes].filter((axis) => !inputAxes.has(axis));

  // Simplified approach: use coordinate mapping for ALL patterns
  // This handles composites, new axes, and mixed patterns uniformly
  operations.push({
    type: 'create_with_coordinate_mapping',
    params: {
      targetShape: outputShape,
      inputPatterns: ast.input,
      outputPatterns: ast.output,
      axisDimensions: resolved.axisDimensions,
      newAxes,
    },
  });

  return operations;
}

/**
 * Execute the planned operations on the tensor
 */
async function executeRepeatOperations<S extends AnyStorageTransformation>(
  tensor: Tensor<S>,
  operations: RepeatOperation[],
): Promise<Tensor<S>> {
  let result = tensor;

  for (const operation of operations) {
    switch (operation.type) {
      case 'identity':
        // No-op
        break;

      case 'reshape':
        if (operation.params?.shape) {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
          result = (await result.reshape(operation.params.shape as any)) as unknown as Tensor<S>;
        }
        break;

      case 'expand':
        if (operation.params?.targetShape) {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
          result = (await result.expand(
            operation.params.targetShape as any,
          )) as unknown as Tensor<S>;
        }
        break;

      case 'tile':
        if (operation.params?.reps) {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
          result = (await result.tile(operation.params.reps as any)) as unknown as Tensor<S>;
        }
        break;

      case 'repeat_with_new_axes':
        if (operation.params?.targetShape) {
          result = (await createRepeatTensorWithNewAxes(
            result,
            operation.params.targetShape,
            operation.params.newAxes || [],
            operation.params.inputPatterns || [],
            operation.params.outputPatterns || [],
            operation.params.axisDimensions,
          )) as unknown as Tensor<S>;
        }
        break;

      case 'composite_repetition':
        if (operation.params?.targetShape) {
          result = (await createCompositeRepetitionTensor(
            result,
            operation.params.targetShape,
            operation.params.inputPatterns || [],
            operation.params.outputPatterns || [],
            operation.params.axisDimensions || new Map(),
          )) as unknown as Tensor<S>;
        }
        break;

      case 'create_with_coordinate_mapping':
        if (operation.params?.targetShape) {
          result = (await createTensorWithCoordinateMapping(
            result,
            operation.params.targetShape,
            operation.params.inputPatterns || [],
            operation.params.outputPatterns || [],
            operation.params.axisDimensions || new Map(),
            operation.params.newAxes || [],
          )) as unknown as Tensor<S>;
        }
        break;

      default:
        throw new Error(`Unknown operation type: ${(operation as any).type}`);
    }
  }

  return result;
}

/**
 * Create tensor using coordinate mapping approach (simplified and fixed)
 */
async function createTensorWithCoordinateMapping<S extends AnyStorageTransformation>(
  inputTensor: Tensor<S>,
  targetShape: readonly number[],
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
  _newAxes: readonly string[],
): Promise<Tensor<S>> {
  // Get the input data
  const inputData = await inputTensor.toArray();
  const inputShape = inputTensor.shape;

  // Create output array with target shape
  const outputArray = createNestedArray(targetShape);

  // Fill the output array using improved coordinate mapping
  fillRepeatArrayImproved(
    inputData,
    outputArray,
    inputShape,
    targetShape,
    inputPatterns,
    outputPatterns,
    axisDimensions,
  );

  // Create new tensor from the filled array
  const { tensor } = await import('../tensor/creation');
  return tensor(outputArray as any, {
    device: inputTensor.device,
    dtype: inputTensor.dtype,
  }) as unknown as Tensor<S>;
}

/**
 * Improved array filling with proper composite pattern handling
 */
function fillRepeatArrayImproved(
  inputData: any,
  outputData: any,
  inputShape: readonly number[],
  _outputShape: readonly number[],
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
): void {
  // Generate all coordinates in the input
  const inputCoords = generateCoordinates(inputShape);

  // For each input coordinate, map it to output coordinates and set values
  for (const inputCoord of inputCoords) {
    const inputValue = getNestedValue(inputData, inputCoord);

    // Map input coordinate to all corresponding output coordinates
    const outputCoords = mapInputToOutputCoordinatesImproved(
      inputCoord,
      inputPatterns,
      outputPatterns,
      axisDimensions,
    );

    // Set the value at all corresponding output coordinates
    for (const outputCoord of outputCoords) {
      setNestedValue(outputData, outputCoord, inputValue);
    }
  }
}

/**
 * Improved coordinate mapping that properly handles composites
 */
function mapInputToOutputCoordinatesImproved(
  inputCoord: readonly number[],
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
): number[][] {
  // Step 1: Extract axis values from input coordinate
  const inputAxisValues = extractAxisValuesFromCoordinate(
    inputCoord,
    inputPatterns,
    axisDimensions,
  );

  // Step 2: Generate all output coordinates using these axis values
  return generateOutputCoordinates(outputPatterns, inputAxisValues, axisDimensions);
}

/**
 * Extract axis name -> value mapping from input coordinate
 */
function extractAxisValuesFromCoordinate(
  coord: readonly number[],
  patterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
): Map<string, number> {
  const axisValues = new Map<string, number>();
  let pos = 0;

  for (const pattern of patterns) {
    if (isSimpleAxis(pattern)) {
      axisValues.set(pattern.name, coord[pos] || 0);
      pos++;
    } else if (isCompositeAxis(pattern)) {
      // For composite patterns, decompose the coordinate
      const compositeDim = coord[pos] || 0;
      const simpleAxes = extractSimpleAxesFromComposite(pattern);

      // Decompose coordinate using actual axis dimensions
      let remainingCoord = compositeDim;
      for (let i = simpleAxes.length - 1; i >= 0; i--) {
        const axis = simpleAxes[i];
        if (axis) {
          const axisSize = axisDimensions.get(axis.name) || 1;
          const axisValue = remainingCoord % axisSize;
          axisValues.set(axis.name, axisValue);
          remainingCoord = Math.floor(remainingCoord / axisSize);
        }
      }
      pos++;
    } else if (isEllipsisAxis(pattern)) {
      // Handle ellipsis - consume remaining dimensions
      const remainingPatterns = patterns.length - patterns.indexOf(pattern) - 1;
      const ellipsisSize = coord.length - pos - remainingPatterns;

      // Store ellipsis dimensions
      for (let i = 0; i < ellipsisSize; i++) {
        axisValues.set(`__ellipsis_${i}`, coord[pos + i] || 0);
      }
      pos += ellipsisSize;
    } else if (isSingletonAxis(pattern)) {
      pos++;
    }
  }

  return axisValues;
}

/**
 * Generate all output coordinates from axis values
 * This is the core function that needs to handle composite patterns correctly
 */
function generateOutputCoordinates(
  patterns: readonly AxisPattern[],
  inputAxisValues: Map<string, number>,
  axisDimensions: Map<string, number>,
): number[][] {
  // Track positions that need expansion (for new axes or repetition)
  const expansions: Array<{ pos: number; values: number[] }> = [];
  const baseCoordinates: number[] = [];

  let pos = 0;
  for (const pattern of patterns) {
    if (isSimpleAxis(pattern)) {
      const axisValue = inputAxisValues.get(pattern.name);
      if (axisValue !== undefined) {
        // Known axis from input - use its value directly
        baseCoordinates.push(axisValue);
      } else {
        // New axis - needs to be expanded to all possible values
        const axisSize = axisDimensions.get(pattern.name) || 1;
        const values: number[] = [];
        for (let i = 0; i < axisSize; i++) {
          values.push(i);
        }
        expansions.push({ pos, values });
        baseCoordinates.push(0); // Placeholder, will be replaced
      }
      pos++;
    } else if (isCompositeAxis(pattern)) {
      // Composite axis like (h h2) - this is the tricky part
      const simpleAxes = extractSimpleAxesFromComposite(pattern);

      // Check which axes are from input vs new (repetition axes)
      const knownAxes: Array<{ axis: SimpleAxis; value: number }> = [];
      const newAxes: Array<{ axis: SimpleAxis; size: number }> = [];

      for (const axis of simpleAxes) {
        if (axis) {
          const inputValue = inputAxisValues.get(axis.name);
          if (inputValue !== undefined) {
            knownAxes.push({ axis, value: inputValue });
          } else {
            const axisSize = axisDimensions.get(axis.name) || 1;
            newAxes.push({ axis, size: axisSize });
          }
        }
      }

      if (knownAxes.length === simpleAxes.length) {
        // All axes are known - compute composite coordinate directly
        let compositeCoord = 0;
        let multiplier = 1;

        for (let i = simpleAxes.length - 1; i >= 0; i--) {
          const axis = simpleAxes[i];
          if (axis) {
            const axisValue = inputAxisValues.get(axis.name) || 0;
            compositeCoord += axisValue * multiplier;
            const axisSize = axisDimensions.get(axis.name) || 1;
            multiplier *= axisSize;
          }
        }
        baseCoordinates.push(compositeCoord);
      } else {
        // Mixed case: some axes known, some new (repetition)
        // For pattern (h h2) where h is known and h2 is new:
        // Generate all possible composite coordinates
        const possibleValues: number[] = [];

        // Generate all combinations of new axes
        function generateCompositeValues(
          axisIndex: number,
          currentValues: Map<string, number>,
        ): void {
          if (axisIndex >= newAxes.length) {
            // All new axes assigned - compute composite coordinate
            let compositeCoord = 0;
            let multiplier = 1;

            for (let i = simpleAxes.length - 1; i >= 0; i--) {
              const axis = simpleAxes[i];
              if (axis) {
                const axisValue =
                  currentValues.get(axis.name) ?? inputAxisValues.get(axis.name) ?? 0;
                compositeCoord += axisValue * multiplier;
                const axisSize = axisDimensions.get(axis.name) || 1;
                multiplier *= axisSize;
              }
            }
            possibleValues.push(compositeCoord);
            return;
          }

          const { axis, size } = newAxes[axisIndex]!;
          for (let i = 0; i < size; i++) {
            const newValues = new Map(currentValues);
            newValues.set(axis.name, i);
            generateCompositeValues(axisIndex + 1, newValues);
          }
        }

        generateCompositeValues(0, new Map());
        expansions.push({ pos, values: possibleValues });
        baseCoordinates.push(0); // Placeholder
      }
      pos++;
    } else if (isEllipsisAxis(pattern)) {
      // Handle ellipsis in output - need to determine how many dimensions it represents
      // Find ellipsis dimensions from input axis values
      let ellipsisCount = 0;
      for (const [key, value] of inputAxisValues) {
        if (key.startsWith('__ellipsis_')) {
          ellipsisCount++;
          baseCoordinates.push(value);
        }
      }
      pos += ellipsisCount;
    } else if (isSingletonAxis(pattern)) {
      baseCoordinates.push(0);
      pos++;
    }
  }

  // If no expansions needed, return the single base coordinate
  if (expansions.length === 0) {
    return [baseCoordinates];
  }

  // Generate all combinations of expansions
  const allCoords: number[][] = [];

  function generateAllCombinations(expansionIndex: number, currentCoord: number[]): void {
    if (expansionIndex >= expansions.length) {
      allCoords.push([...currentCoord]);
      return;
    }

    const { pos: expandPos, values } = expansions[expansionIndex]!;
    for (const value of values) {
      const newCoord = [...currentCoord];
      newCoord[expandPos] = value;
      generateAllCombinations(expansionIndex + 1, newCoord);
    }
  }

  generateAllCombinations(0, baseCoordinates);
  return allCoords;
}

/**
 * Create a new tensor by repeating elements according to repeat pattern with new axes
 */
async function createRepeatTensorWithNewAxes<S extends AnyStorageTransformation>(
  inputTensor: Tensor<S>,
  targetShape: readonly number[],
  newAxes: readonly string[],
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  axisDimensions?: Map<string, number>,
): Promise<Tensor<S>> {
  // Get the input data
  const inputData = await inputTensor.toArray();
  const inputShape = inputTensor.shape;

  // Create output array with target shape
  const outputArray = createNestedArray(targetShape);

  // Fill the output array by mapping input elements to output positions
  fillRepeatArray(
    inputData,
    outputArray,
    inputShape,
    targetShape,
    inputPatterns,
    outputPatterns,
    newAxes,
    axisDimensions,
  );

  // Create new tensor from the filled array
  const { tensor } = await import('../tensor/creation');
  // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
  return tensor(outputArray as any, {
    device: inputTensor.device,
    dtype: inputTensor.dtype,
  }) as unknown as Tensor<S>;
}

/**
 * Create a nested array structure with the given shape, filled with zeros
 */
function createNestedArray(shape: readonly number[]): any {
  if (shape.length === 0) {
    return 0; // Scalar
  }
  if (shape.length === 1) {
    return new Array(shape[0]).fill(0);
  }

  const result = new Array(shape[0]);
  for (let i = 0; i < shape[0]!; i++) {
    result[i] = createNestedArray(shape.slice(1));
  }
  return result;
}

/**
 * Fill the output array by repeating input elements according to the pattern
 * This is the core logic that implements the repeat semantics
 */
function fillRepeatArray(
  inputData: any,
  outputData: any,
  inputShape: readonly number[],
  outputShape: readonly number[],
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  newAxes: readonly string[],
  axisDimensions?: Map<string, number>,
): void {
  // For simple cases like 'h w -> h w c', we need to:
  // 1. For each position in the input (h, w)
  // 2. Replicate that value across all positions in the new axis c

  // Create mapping from input coordinates to output coordinates
  const inputCoords = generateCoordinates(inputShape);

  for (const inputCoord of inputCoords) {
    const inputValue = getNestedValue(inputData, inputCoord);

    // For each input coordinate, determine all output coordinates that should have this value
    const outputCoords = mapInputCoordToOutputCoords(
      inputCoord,
      inputPatterns,
      outputPatterns,
      inputShape,
      outputShape,
      newAxes,
      axisDimensions,
    );

    // Set the value at all corresponding output coordinates
    for (const outputCoord of outputCoords) {
      setNestedValue(outputData, outputCoord, inputValue);
    }
  }
}

/**
 * Generate all coordinate combinations for a given shape
 */
function generateCoordinates(shape: readonly number[]): number[][] {
  if (shape.length === 0) {
    return [[]]; // Single empty coordinate for scalar
  }

  const result: number[][] = [];
  const current: number[] = new Array(shape.length).fill(0);

  function generateRecursive(dim: number): void {
    if (dim === shape.length) {
      result.push([...current]);
      return;
    }

    const size = shape[dim];
    if (size !== undefined) {
      for (let i = 0; i < size; i++) {
        current[dim] = i;
        generateRecursive(dim + 1);
      }
    }
  }

  generateRecursive(0);
  return result;
}

/**
 * Get value from nested array using coordinate
 */
function getNestedValue(data: any, coord: readonly number[]): any {
  let current = data;
  for (const index of coord) {
    current = current[index];
  }
  return current;
}

/**
 * Set value in nested array using coordinate
 */
function setNestedValue(data: any, coord: readonly number[], value: any): void {
  let current = data;
  for (let i = 0; i < coord.length - 1; i++) {
    current = current[coord[i]!];
  }
  current[coord[coord.length - 1]!] = value;
}

/**
 * Map an input coordinate to all output coordinates that should have the same value
 * For patterns with new axes, each input coordinate maps to multiple output coordinates
 */
function mapInputCoordToOutputCoords(
  inputCoord: readonly number[],
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  inputShape: readonly number[],
  outputShape: readonly number[],
  newAxes: readonly string[],
  axisDimensions?: Map<string, number>,
): number[][] {
  // For simple patterns like 'h w -> h w c', this is straightforward:
  // input coord [i, j] maps to output coords [i, j, 0], [i, j, 1], ..., [i, j, c-1]
  // For composite patterns like 'h w -> (h h2) w c', input coord [i, j] maps to
  // output coords [i*h2 + k, j, 0], [i*h2 + k, j, 1], ... for k = 0, 1, ..., h2-1

  // Build comprehensive axis mapping that handles both simple and composite patterns
  const inputAxisCoords = buildAxisCoordinateMapping(
    inputPatterns,
    inputCoord,
    inputShape,
    axisDimensions,
  );
  const outputCoordGenerator = buildOutputCoordinateGenerator(
    outputPatterns,
    outputShape,
    newAxes,
    axisDimensions,
  );

  // Generate all output coordinates for this input coordinate
  return outputCoordGenerator(inputAxisCoords);
}

/**
 * Build mapping from axis names to their coordinate values from input coordinate
 */
function buildAxisCoordinateMapping(
  patterns: readonly AxisPattern[],
  coord: readonly number[],
  shape: readonly number[],
  axisDimensions?: Map<string, number>,
): Map<string, number> {
  const axisCoords = new Map<string, number>();
  let pos = 0;

  for (const pattern of patterns) {
    if (isSimpleAxis(pattern)) {
      axisCoords.set(pattern.name, coord[pos] || 0);
      pos++;
    } else if (isCompositeAxis(pattern)) {
      // For composite patterns, we need to decompose the coordinate
      const compositeCoord = coord[pos] || 0;

      // Extract simple axes from the composite
      const simpleAxes = extractSimpleAxesFromComposite(pattern);

      // Decompose the composite coordinate using actual axis dimensions
      let remainingCoord = compositeCoord;

      // Process axes in reverse order (rightmost varies fastest)
      for (let i = simpleAxes.length - 1; i >= 0; i--) {
        const axis = simpleAxes[i];
        if (axis && axisDimensions) {
          // Use actual axis dimensions from resolution
          const axisSize = axisDimensions.get(axis.name);
          if (axisSize !== undefined) {
            const axisCoord = remainingCoord % axisSize;
            axisCoords.set(axis.name, axisCoord);
            remainingCoord = Math.floor(remainingCoord / axisSize);
          }
        }
      }
      pos++;
    } else if (isEllipsisAxis(pattern)) {
      // Handle ellipsis - consume remaining dimensions
      const remainingPatterns = patterns.length - patterns.indexOf(pattern) - 1;
      const ellipsisSize = shape.length - pos - remainingPatterns;
      pos += ellipsisSize;
    } else if (isSingletonAxis(pattern)) {
      pos++;
    }
  }

  return axisCoords;
}

/**
 * Build a function that generates output coordinates from input axis coordinates
 */
function buildOutputCoordinateGenerator(
  patterns: readonly AxisPattern[],
  shape: readonly number[],
  newAxes: readonly string[],
  axisDimensions?: Map<string, number>,
): (inputAxisCoords: Map<string, number>) => number[][] {
  return (inputAxisCoords: Map<string, number>): number[][] => {
    const outputCoords: number[][] = [];
    const baseCoord: number[] = new Array(shape.length);
    let pos = 0;

    // Build base coordinate by processing output patterns
    const newAxisInfo: Array<{ pos: number; size: number }> = [];

    for (const pattern of patterns) {
      if (isSimpleAxis(pattern)) {
        const axisCoord = inputAxisCoords.get(pattern.name);
        if (axisCoord !== undefined) {
          baseCoord[pos] = axisCoord;
        } else if (newAxes.includes(pattern.name)) {
          // This is a new axis - track it for expansion
          newAxisInfo.push({ pos, size: shape[pos] || 1 });
        }
        pos++;
      } else if (isCompositeAxis(pattern)) {
        // Handle composite patterns in output
        const simpleAxes = extractSimpleAxesFromComposite(pattern);

        // For composite patterns, we need to handle repetition properly
        // For patterns like (h h2), input h=1 maps to output positions [h*h2, h*h2+1, ..., h*h2+h2-1]
        // But we need to generate ALL possible coordinates, not just one

        // For now, set a base coordinate and handle expansion later
        let compositeCoord = 0;
        let multiplier = 1;

        // Process in reverse order (rightmost varies fastest)
        for (let i = simpleAxes.length - 1; i >= 0; i--) {
          const axis = simpleAxes[i];
          if (axis && axisDimensions) {
            const axisCoord = inputAxisCoords.get(axis.name) || 0;
            const axisSize = axisDimensions.get(axis.name) || 1;

            compositeCoord += axisCoord * multiplier;
            multiplier *= axisSize;
          }
        }

        baseCoord[pos] = compositeCoord;
        pos++;
      } else if (isEllipsisAxis(pattern)) {
        // Handle ellipsis in output
        const remainingPatterns = patterns.length - patterns.indexOf(pattern) - 1;
        const ellipsisSize = shape.length - pos - remainingPatterns;

        // Copy ellipsis dimensions from input (or fill with 0 for new ellipsis)
        for (let i = 0; i < ellipsisSize; i++) {
          baseCoord[pos + i] = 0; // Simplified - assume 0 for now
        }
        pos += ellipsisSize;
      } else if (isSingletonAxis(pattern)) {
        baseCoord[pos] = 0;
        pos++;
      }
    }

    // Generate all combinations for new axes
    function generateCombinations(newAxisIndex: number, currentCoord: number[]): void {
      if (newAxisIndex >= newAxisInfo.length) {
        outputCoords.push([...currentCoord]);
        return;
      }

      const { pos: axisPos, size } = newAxisInfo[newAxisIndex]!;

      for (let i = 0; i < size; i++) {
        currentCoord[axisPos] = i;
        generateCombinations(newAxisIndex + 1, currentCoord);
      }
    }

    generateCombinations(0, [...baseCoord]);
    return outputCoords.length > 0 ? outputCoords : [baseCoord];
  };
}

/**
 * Expand tensor to add new axes at specific positions
 */
async function expandTensorWithNewAxes<S extends AnyStorageTransformation>(
  inputTensor: Tensor<S>,
  targetShape: readonly number[],
  outputPatterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
): Promise<Tensor<S>> {
  let result = inputTensor;
  const currentShape = [...result.shape];

  // Build mapping from axis names to positions in target shape
  const axisPositions = new Map<string, number>();
  let pos = 0;

  for (const pattern of outputPatterns) {
    if (isSimpleAxis(pattern)) {
      axisPositions.set(pattern.name, pos);
      pos++;
    } else if (isCompositeAxis(pattern)) {
      pos++; // Composite takes one position
    } else if (isEllipsisAxis(pattern)) {
      // Handle ellipsis
      const remainingPatterns = outputPatterns.length - outputPatterns.indexOf(pattern) - 1;
      const ellipsisSize = targetShape.length - pos - remainingPatterns;
      pos += ellipsisSize;
    } else if (isSingletonAxis(pattern)) {
      pos++;
    }
  }

  // Add new dimensions by unsqueezing at the right positions
  const insertions: Array<{ pos: number; size: number }> = [];

  for (const [axisName, targetPos] of axisPositions) {
    const size = axisDimensions.get(axisName);
    if (size !== undefined && size > 1) {
      // Check if this axis is new (not in current shape)
      let isNewAxis = true;
      let currentPos = 0;

      for (const pattern of outputPatterns) {
        if (isSimpleAxis(pattern) && pattern.name === axisName) {
          if (currentPos < currentShape.length) {
            isNewAxis = false;
          }
          break;
        }
        if (
          !isSimpleAxis(pattern) ||
          !axisDimensions.has(pattern.name) ||
          currentPos < currentShape.length
        ) {
          currentPos++;
        }
      }

      if (isNewAxis) {
        insertions.push({ pos: targetPos, size });
      }
    }
  }

  // Sort insertions by position (reverse order to maintain correct indices)
  insertions.sort((a, b) => b.pos - a.pos);

  // Apply unsqueeze and expand operations
  for (const { pos, size } of insertions) {
    // Unsqueeze to add dimension of size 1
    result = (await result.unsqueeze(pos)) as unknown as Tensor<S>;

    // Expand to the target size
    const expandShape = [...result.shape];
    expandShape[pos] = size;
    result = (await result.expand(expandShape as any)) as unknown as Tensor<S>;
  }

  return result;
}

/**
 * Extract simple axes from a composite pattern
 */
function extractSimpleAxesFromComposite(composite: CompositeAxis): SimpleAxis[] {
  const result: SimpleAxis[] = [];

  for (const axis of composite.axes) {
    if (isSimpleAxis(axis)) {
      result.push(axis);
    } else if (isCompositeAxis(axis)) {
      // Recursively extract from nested composites
      result.push(...extractSimpleAxesFromComposite(axis));
    }
  }

  return result;
}

/**
 * Create tensor with composite repetition patterns like 'w -> (w w2)'
 * This handles cases where existing axes are repeated within composite patterns.
 */
async function createCompositeRepetitionTensor<S extends AnyStorageTransformation>(
  inputTensor: Tensor<S>,
  targetShape: readonly number[],
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
): Promise<Tensor<S>> {
  // Check if this is a mixed case with standalone new axes
  const inputAxisNames = getAxisNames(inputPatterns);
  const outputAxisNames = getAxisNames(outputPatterns);
  const newAxes = outputAxisNames.filter((axis) => !inputAxisNames.includes(axis));

  const hasStandaloneNewAxes = newAxes.some((axis) => {
    for (const pattern of outputPatterns) {
      if (isSimpleAxis(pattern) && pattern.name === axis) {
        return true; // Standalone simple axis
      }
    }
    return false;
  });

  if (hasStandaloneNewAxes) {
    // Mixed case: handle composites first with tensor operations, then add new axes
    // Step 1: Create intermediate pattern without standalone new axes
    const intermediateOutputPatterns = outputPatterns.filter((pattern) => {
      if (isSimpleAxis(pattern) && newAxes.includes(pattern.name)) {
        // Check if this is a truly standalone new axis
        return false;
      }
      return true;
    });

    // Step 2: Handle composite repetition first
    let result = inputTensor;
    if (intermediateOutputPatterns.length > 0) {
      const intermediateShape = computeExtendedShape(intermediateOutputPatterns, axisDimensions);
      const operations = planCompositeRepetitionOperations(
        inputTensor.shape,
        intermediateShape,
        inputPatterns,
        intermediateOutputPatterns,
        axisDimensions,
      );

      // Execute composite operations
      for (const op of operations) {
        switch (op.type) {
          case 'reshape':
            if (op.shape) {
              result = (await result.reshape(op.shape as any)) as unknown as Tensor<S>;
            }
            break;
          case 'tile':
            if (op.reps) {
              result = (await result.tile(op.reps as any)) as unknown as Tensor<S>;
            }
            break;
        }
      }
    }

    // Step 3: Add standalone new axes using unsqueeze + expand
    const standaloneNewAxes = newAxes.filter((axis) => {
      for (const pattern of outputPatterns) {
        if (isSimpleAxis(pattern) && pattern.name === axis) {
          return true;
        }
      }
      return false;
    });

    if (standaloneNewAxes.length > 0) {
      // Add new axes by expanding to target shape
      return expandTensorWithNewAxes(result, targetShape, outputPatterns, axisDimensions);
    }

    return result;
  }

  // Pure composite repetition case
  let result = inputTensor;

  // Plan the operations based on the composite pattern
  const operations = planCompositeRepetitionOperations(
    inputTensor.shape,
    targetShape,
    inputPatterns,
    outputPatterns,
    axisDimensions,
  );

  // Execute the operations
  for (const op of operations) {
    switch (op.type) {
      case 'reshape':
        if (op.shape) {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
          result = (await result.reshape(op.shape as any)) as unknown as Tensor<S>;
        }
        break;
      case 'tile':
        if (op.reps) {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
          result = (await result.tile(op.reps as any)) as unknown as Tensor<S>;
        }
        break;
      case 'flatten':
        if (op.targetShape) {
          // eslint-disable-next-line @typescript-eslint/no-unsafe-argument, @typescript-eslint/no-explicit-any
          result = (await result.reshape(op.targetShape as any)) as unknown as Tensor<S>;
        }
        break;
    }
  }

  return result;
}

/**
 * Plan operations for composite repetition
 */
function planCompositeRepetitionOperations(
  inputShape: readonly number[],
  targetShape: readonly number[],
  inputPatterns: readonly AxisPattern[],
  outputPatterns: readonly AxisPattern[],
  axisDimensions: Map<string, number>,
): Array<{
  type: 'reshape' | 'tile' | 'flatten';
  shape?: readonly number[];
  reps?: readonly number[];
  targetShape?: readonly number[];
}> {
  const operations: Array<{
    type: 'reshape' | 'tile' | 'flatten';
    shape?: readonly number[];
    reps?: readonly number[];
    targetShape?: readonly number[];
  }> = [];

  // Identify which axes need repetition within composites
  const inputAxisNames = getAxisNames(inputPatterns);
  const outputAxisNames = getAxisNames(outputPatterns);
  const newAxes = outputAxisNames.filter((axis) => !inputAxisNames.includes(axis));

  // Check if pattern has both composite repetition and standalone new axes
  // For patterns like 'h w -> (h h2) w c':
  // - (h h2) is composite repetition
  // - c is standalone new axis

  const hasStandaloneNewAxes = newAxes.some((axis) => {
    // Check if this new axis appears outside composites
    for (const pattern of outputPatterns) {
      if (isSimpleAxis(pattern) && pattern.name === axis) {
        return true; // Standalone simple axis
      }
    }
    return false;
  });

  if (hasStandaloneNewAxes) {
    // Mixed case: use general tensor creation logic
    // This will be handled by switching the operation type
    return []; // Return empty - will be handled by different operation type
  }

  // Pure composite repetition case like 'h w -> (h h2) (w w2)'
  // Map input axes to their repetition factors
  const repetitionFactors: number[] = [];

  for (let i = 0; i < inputShape.length; i++) {
    const axisName = inputAxisNames[i];
    if (axisName) {
      // Look for repetition factor in output composites
      let factor = 1;
      for (const pattern of outputPatterns) {
        if (isCompositeAxis(pattern)) {
          const compositeAxes = getAxisNames([pattern]);
          if (compositeAxes.includes(axisName)) {
            // Find the repetition factor axis (new axis in this composite)
            const repetitionAxisName = compositeAxes.find((axis) => axis !== axisName);
            if (repetitionAxisName) {
              factor = axisDimensions.get(repetitionAxisName) || 1;
            }
          }
        }
      }
      repetitionFactors.push(factor);
    } else {
      repetitionFactors.push(1);
    }
  }

  // Check if all factors are integers > 1
  const needsRepetition = repetitionFactors.some((f) => f > 1);
  if (!needsRepetition) {
    return operations;
  }

  // Step 1: Reshape to interleave repetition dimensions
  // [h, w] -> [h, 1, w, 1] (insert singleton for each repetition)
  const intermediateShape: number[] = [];
  for (let i = 0; i < inputShape.length; i++) {
    intermediateShape.push(inputShape[i]!);
    intermediateShape.push(1); // Singleton for repetition
  }

  operations.push({
    type: 'reshape',
    shape: intermediateShape,
  });

  // Step 2: Tile along repetition dimensions
  // [h, 1, w, 1] -> [h, h2, w, w2]
  const tileReps: number[] = [];
  for (let i = 0; i < repetitionFactors.length; i++) {
    tileReps.push(1); // Don't repeat the original dimension
    tileReps.push(repetitionFactors[i]!); // Repeat the singleton
  }

  operations.push({
    type: 'tile',
    reps: tileReps,
  });

  // Step 3: Reshape to merge dimensions
  // [h, h2, w, w2] -> [h*h2, w*w2]
  operations.push({
    type: 'flatten',
    targetShape: [...targetShape],
  });

  return operations;
}

/**
 * Check if two arrays are equal
 */
function arraysEqual(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((val, i) => val === b[i]);
}
