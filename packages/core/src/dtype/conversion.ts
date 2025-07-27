/**
 * Type conversion and validation for DType operations
 *
 * This module provides safe type conversion between different DTypes,
 * including overflow/underflow detection, precision loss tracking,
 * and configurable conversion policies for different use cases.
 */

import type { AnyDType, JSTypeOf, CanSafelyCast } from './types.js';
import { type RuntimeDType } from './runtime.js';

// =============================================================================
// Conversion Options and Policies
// =============================================================================

/**
 * Options for controlling type conversion behavior
 */
export interface ConversionOptions {
  /**
   * Allow conversions that may lose precision (e.g., float64 to int32)
   * @default false
   */
  readonly allowPrecisionLoss?: boolean;

  /**
   * Allow conversions that may overflow/underflow (e.g., int32 to int8)
   * @default false
   */
  readonly allowOverflow?: boolean;

  /**
   * How to handle NaN values during conversion
   * - 'error': Throw an error (default)
   * - 'zero': Convert to zero
   * - 'clamp': Clamp to type bounds
   */
  readonly nanHandling?: 'error' | 'zero' | 'clamp';

  /**
   * How to handle Infinity values during conversion
   * - 'error': Throw an error (default)
   * - 'clamp': Clamp to type bounds
   */
  readonly infinityHandling?: 'error' | 'clamp';

  /**
   * How to handle values that exceed the target type's range
   * - 'error': Throw an error (default)
   * - 'clamp': Clamp to type bounds
   * - 'wrap': Use modular arithmetic (JavaScript default)
   */
  readonly overflowHandling?: 'error' | 'clamp' | 'wrap';
}

/**
 * Default conversion options for strict type safety
 */
export const STRICT_CONVERSION_OPTIONS: Required<ConversionOptions> = {
  allowPrecisionLoss: false,
  allowOverflow: false,
  nanHandling: 'error',
  infinityHandling: 'error',
  overflowHandling: 'error',
} as const;

/**
 * Permissive conversion options for compatibility with JavaScript
 */
export const PERMISSIVE_CONVERSION_OPTIONS: Required<ConversionOptions> = {
  allowPrecisionLoss: true,
  allowOverflow: true,
  nanHandling: 'clamp',
  infinityHandling: 'clamp',
  overflowHandling: 'clamp',
} as const;

// =============================================================================
// Conversion Result Types
// =============================================================================

/**
 * Result of a type conversion operation
 */
export type ConversionResult<T extends AnyDType> =
  | {
      readonly success: true;
      readonly value: JSTypeOf<T>;
      readonly warnings: readonly string[];
    }
  | {
      readonly success: false;
      readonly errors: readonly string[];
    };

/**
 * Information about precision loss during conversion
 */
export interface PrecisionLossInfo {
  readonly originalValue: number | bigint | boolean;
  readonly convertedValue: number | bigint | boolean;
  readonly lossType: 'truncation' | 'rounding' | 'overflow' | 'underflow' | 'range';
  readonly message: string;
}

// =============================================================================
// Core Conversion Functions
// =============================================================================

/**
 * Convert a value from one DType to another with comprehensive validation
 *
 * @example
 * const int32DType = getDType('int32');
 * const float32DType = getDType('float32');
 * const result = convertValue(42, int32DType, float32DType);
 * if (result.success) {
 *   console.log(result.value); // 42 (as number, not changed)
 * }
 */
export function convertValue<From extends AnyDType, To extends AnyDType>(
  value: JSTypeOf<From>,
  fromDType: RuntimeDType<From>,
  toDType: RuntimeDType<To>,
  options: ConversionOptions = {},
): ConversionResult<To> {
  const opts = { ...STRICT_CONVERSION_OPTIONS, ...options };
  const warnings: string[] = [];
  const errors: string[] = [];

  // Same type, no conversion needed
  if (fromDType.name === toDType.name) {
    return { success: true, value: value as JSTypeOf<To>, warnings };
  }

  try {
    // Handle JavaScript type conversions
    if (fromDType.jsType !== toDType.jsType) {
      const jsConversionResult = convertJavaScriptType(value, fromDType, toDType, opts);
      if (!jsConversionResult.success) {
        return jsConversionResult;
      }
      // Accumulate warnings from JS type conversion
      warnings.push(...jsConversionResult.warnings);
      const sameTypeResult = convertSameJSType(jsConversionResult.value, fromDType, toDType, opts);
      if (sameTypeResult.success) {
        warnings.push(...sameTypeResult.warnings);
        return { success: true, value: sameTypeResult.value, warnings };
      }
      return sameTypeResult;
    }

    // Same JavaScript type, handle range and precision
    return convertSameJSType(value, fromDType, toDType, opts);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
    return { success: false, errors };
  }
}

/**
 * Convert between different JavaScript types (number <-> boolean <-> bigint)
 */
function convertJavaScriptType<From extends AnyDType, To extends AnyDType>(
  value: JSTypeOf<From>,
  fromDType: RuntimeDType<From>,
  toDType: RuntimeDType<To>,
  options: Required<ConversionOptions>,
): ConversionResult<To> {
  const warnings: string[] = [];

  // Boolean to number/bigint
  if (fromDType.jsType === 'boolean') {
    const boolValue = value as boolean;
    if (toDType.jsType === 'number') {
      return { success: true, value: (boolValue ? 1 : 0) as JSTypeOf<To>, warnings };
    }
    if (toDType.jsType === 'bigint') {
      return { success: true, value: (boolValue ? 1n : 0n) as JSTypeOf<To>, warnings };
    }
  }

  // Number/bigint to boolean
  if (toDType.jsType === 'boolean') {
    let boolResult: boolean;
    if (fromDType.jsType === 'number') {
      const numValue = value as number;
      if (!Number.isFinite(numValue)) {
        if (options.nanHandling === 'error') {
          return { success: false, errors: [`Cannot convert ${numValue.toString()} to boolean`] };
        }
        // NumPy/PyTorch behavior: NaN and Inf convert to true
        boolResult = true;
        // Don't add warning for NumPy-compatible behavior
      } else {
        boolResult = numValue !== 0;
      }
    } else if (fromDType.jsType === 'bigint') {
      const bigintValue = value as bigint;
      boolResult = bigintValue !== 0n;
    } else {
      return { success: false, errors: [`Cannot convert ${fromDType.jsType} to boolean`] };
    }
    return { success: true, value: boolResult as JSTypeOf<To>, warnings };
  }

  // Number to bigint
  if (fromDType.jsType === 'number' && toDType.jsType === 'bigint') {
    const numValue = value as number;

    if (!Number.isFinite(numValue)) {
      if (options.nanHandling === 'error') {
        return { success: false, errors: [`Cannot convert ${numValue.toString()} to bigint`] };
      }
      // NumPy/PyTorch behavior: NaN → 0, Inf → max, -Inf → min
      let bigintValue: bigint;
      if (Number.isNaN(numValue)) {
        bigintValue = 0n;
      } else if (numValue === Infinity) {
        bigintValue = toDType.maxValue as bigint;
      } else {
        // -Infinity
        bigintValue = toDType.minValue as bigint;
      }
      warnings.push(`Special value ${numValue.toString()} converted to ${bigintValue.toString()}`);
      return { success: true, value: bigintValue as JSTypeOf<To>, warnings };
    }

    if (!Number.isInteger(numValue)) {
      if (!options.allowPrecisionLoss) {
        return {
          success: false,
          errors: [
            `Cannot convert non-integer ${numValue.toString()} to bigint without precision loss`,
          ],
        };
      }
      const truncated = Math.trunc(numValue);
      warnings.push(`Precision loss: ${numValue.toString()} truncated to ${truncated.toString()}`);
      return { success: true, value: BigInt(truncated) as JSTypeOf<To>, warnings };
    }

    return { success: true, value: BigInt(numValue) as JSTypeOf<To>, warnings };
  }

  // Bigint to number
  if (fromDType.jsType === 'bigint' && toDType.jsType === 'number') {
    const bigintValue = value as bigint;

    // Check if the bigint can be safely represented as a number
    if (
      bigintValue > BigInt(Number.MAX_SAFE_INTEGER) ||
      bigintValue < BigInt(Number.MIN_SAFE_INTEGER)
    ) {
      if (!options.allowPrecisionLoss) {
        return {
          success: false,
          errors: [`BigInt ${bigintValue.toString()} exceeds safe integer range for number`],
        };
      }
      warnings.push(
        `Precision loss: BigInt ${bigintValue.toString()} may lose precision as number`,
      );
    }

    return { success: true, value: Number(bigintValue) as JSTypeOf<To>, warnings };
  }

  return { success: false, errors: [`Cannot convert ${fromDType.jsType} to ${toDType.jsType}`] };
}

/**
 * Convert between DTypes with the same JavaScript type
 */
function convertSameJSType<From extends AnyDType, To extends AnyDType>(
  value: unknown,
  fromDType: RuntimeDType<From>,
  toDType: RuntimeDType<To>,
  options: Required<ConversionOptions>,
): ConversionResult<To> {
  const warnings: string[] = [];

  // Handle number types
  if (fromDType.jsType === 'number' && toDType.jsType === 'number') {
    const numValue = value as number;

    // Handle special float values
    if (!Number.isFinite(numValue)) {
      if (toDType.isInteger) {
        if (options.nanHandling === 'error') {
          return {
            success: false,
            errors: [`Cannot convert ${numValue.toString()} to integer type ${toDType.name}`],
          };
        }

        let convertedValue: number;
        if (Number.isNaN(numValue)) {
          convertedValue = options.nanHandling === 'zero' ? 0 : (toDType.minValue as number);
        } else {
          // Infinity
          if (options.infinityHandling === 'error') {
            return {
              success: false,
              errors: [`Cannot convert ${numValue.toString()} to integer type ${toDType.name}`],
            };
          }
          convertedValue =
            numValue > 0 ? (toDType.maxValue as number) : (toDType.minValue as number);
        }

        warnings.push(
          `Special value ${numValue.toString()} converted to ${convertedValue.toString()}`,
        );
        return { success: true, value: convertedValue as JSTypeOf<To>, warnings };
      }

      // Float to float conversion with special values
      return { success: true, value: numValue as JSTypeOf<To>, warnings };
    }

    // Integer type validation and conversion
    if (toDType.isInteger) {
      if (!Number.isInteger(numValue)) {
        if (!options.allowPrecisionLoss) {
          return {
            success: false,
            errors: [
              `Cannot convert non-integer ${numValue.toString()} to ${toDType.name} without precision loss`,
            ],
          };
        }

        const truncated = Math.trunc(numValue);
        warnings.push(
          `Precision loss: ${numValue.toString()} truncated to ${truncated.toString()}`,
        );
        const integerResult = validateAndConvertInteger(truncated, toDType, options);
        if (integerResult.success) {
          return {
            success: true,
            value: integerResult.value,
            warnings: [...warnings, ...integerResult.warnings],
          };
        } else {
          return integerResult;
        }
      }

      const integerResult = validateAndConvertInteger(numValue, toDType, options);
      if (integerResult.success) {
        return {
          success: true,
          value: integerResult.value,
          warnings: [...warnings, ...integerResult.warnings],
        };
      } else {
        return integerResult;
      }
    }

    // Float type conversion
    if (fromDType.isInteger || toDType.name === 'float64') {
      // Integer to float or float32 to float64 - always safe
      return { success: true, value: numValue as JSTypeOf<To>, warnings };
    }

    // Float64 to float32 - check for precision loss
    if (fromDType.name === 'float64' && toDType.name === 'float32') {
      const asFloat32 = Math.fround(numValue);
      if (asFloat32 !== numValue && Number.isFinite(numValue)) {
        if (!options.allowPrecisionLoss) {
          return {
            success: false,
            errors: [`Precision loss converting ${numValue.toString()} from float64 to float32`],
          };
        }
        warnings.push(
          `Precision loss: float64 ${numValue.toString()} rounded to float32 ${asFloat32.toString()}`,
        );
      }
      return { success: true, value: asFloat32 as JSTypeOf<To>, warnings };
    }

    return { success: true, value: numValue as JSTypeOf<To>, warnings };
  }

  // Handle bigint types
  if (fromDType.jsType === 'bigint' && toDType.jsType === 'bigint') {
    const bigintValue = value as bigint;

    // Validate range
    if (bigintValue < toDType.minValue || bigintValue > toDType.maxValue) {
      if (!options.allowOverflow) {
        return {
          success: false,
          errors: [
            `Value ${bigintValue.toString()} out of range for ${toDType.name} [${toDType.minValue.toString()}, ${toDType.maxValue.toString()}]`,
          ],
        };
      }

      let clampedValue: bigint;
      if (options.overflowHandling === 'clamp') {
        clampedValue =
          bigintValue < toDType.minValue
            ? (toDType.minValue as bigint)
            : (toDType.maxValue as bigint);
        warnings.push(
          `Value ${bigintValue.toString()} clamped to ${clampedValue.toString()} for ${toDType.name}`,
        );
      } else {
        // Wrap around (modular arithmetic)
        const range = (toDType.maxValue as bigint) - (toDType.minValue as bigint) + 1n;
        clampedValue =
          ((bigintValue - (toDType.minValue as bigint)) % range) + (toDType.minValue as bigint);
        warnings.push(
          `Value ${bigintValue.toString()} wrapped to ${clampedValue.toString()} for ${toDType.name}`,
        );
      }

      return { success: true, value: clampedValue as JSTypeOf<To>, warnings };
    }

    return { success: true, value: bigintValue as JSTypeOf<To>, warnings };
  }

  // Boolean type (should not reach here in same-JS-type conversion)
  return { success: true, value: value as JSTypeOf<To>, warnings };
}

/**
 * Validate and convert integer values with overflow handling
 */
function validateAndConvertInteger<To extends AnyDType>(
  value: number,
  toDType: RuntimeDType<To>,
  options: Required<ConversionOptions>,
): ConversionResult<To> {
  const warnings: string[] = [];

  if (value < toDType.minValue || value > toDType.maxValue) {
    if (!options.allowOverflow) {
      return {
        success: false,
        errors: [
          `Value ${value.toString()} out of range for ${toDType.name} [${toDType.minValue.toString()}, ${toDType.maxValue.toString()}]`,
        ],
      };
    }

    let convertedValue: number;
    if (options.overflowHandling === 'clamp') {
      convertedValue = Math.max(
        toDType.minValue as number,
        Math.min(toDType.maxValue as number, value),
      );
      warnings.push(
        `Value ${value.toString()} clamped to ${convertedValue.toString()} for ${toDType.name}`,
      );
    } else {
      // JavaScript default overflow behavior (wrap around)
      // This uses the same behavior as TypedArray assignment
      const tempArray = new toDType.typedArrayConstructor(1);
      tempArray[0] = value;
      convertedValue = tempArray[0];
      warnings.push(
        `Value ${value.toString()} wrapped to ${convertedValue.toString()} for ${toDType.name}`,
      );
    }

    return { success: true, value: convertedValue as JSTypeOf<To>, warnings };
  }

  return { success: true, value: value as JSTypeOf<To>, warnings };
}

// =============================================================================
// Batch Conversion Functions
// =============================================================================

/**
 * Convert an array of values from one DType to another
 *
 * @example
 * const values = [1, 2, 3.14];
 * const int32DType = getDType('int32');
 * const float32DType = getDType('float32');
 * const result = convertArray(values, int32DType, float32DType);
 */
export function convertArray<From extends AnyDType, To extends AnyDType>(
  values: readonly JSTypeOf<From>[],
  fromDType: RuntimeDType<From>,
  toDType: RuntimeDType<To>,
  options: ConversionOptions = {},
): ConversionResult<To> & { values?: readonly JSTypeOf<To>[] } {
  const convertedValues: JSTypeOf<To>[] = [];
  const allWarnings: string[] = [];
  const allErrors: string[] = [];

  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (value === undefined) {
      allErrors.push(`Value at index ${i.toString()} is undefined`);
      continue;
    }
    const result = convertValue(value, fromDType, toDType, options);

    if (result.success) {
      convertedValues.push(result.value);
      allWarnings.push(...result.warnings.map((w) => `[${i.toString()}] ${w}`));
    } else {
      allErrors.push(...result.errors.map((e) => `[${i.toString()}] ${e}`));
    }
  }

  if (allErrors.length > 0) {
    return { success: false, errors: allErrors };
  }

  return {
    success: true,
    value: convertedValues as unknown as JSTypeOf<To>, // Type system limitation
    values: convertedValues,
    warnings: allWarnings,
  };
}

// =============================================================================
// Type-Safe Conversion Utilities
// =============================================================================

/**
 * Type-safe conversion that only compiles for safe casts
 *
 * @example
 * const value: Int8 = safeCast(42, getDType('int8'), getDType('int32')); // ✓ Compiles
 * const value2: Int8 = safeCast(42, getDType('float64'), getDType('int8')); // ✗ Compile error
 */
export function safeCast<From extends AnyDType, To extends AnyDType>(
  value: JSTypeOf<From>,
  fromDType: RuntimeDType<From>,
  toDType: RuntimeDType<To>,
): CanSafelyCast<From, To> extends true ? JSTypeOf<To> : never {
  const result = convertValue(value, fromDType, toDType, STRICT_CONVERSION_OPTIONS);

  if (!result.success) {
    throw new Error(`Safe cast failed: ${result.errors.join(', ')}`);
  }

  return result.value as CanSafelyCast<From, To> extends true ? JSTypeOf<To> : never;
}

/**
 * Unsafe conversion that may lose data but provides warnings
 *
 * @example
 * const result = unsafeCast(3.14, getDType('float32'), getDType('int32'));
 * if (result.success) {
 *   console.log(result.value); // 3
 *   console.log(result.warnings); // ["Precision loss: 3.14 truncated to 3"]
 * }
 */
export function unsafeCast<From extends AnyDType, To extends AnyDType>(
  value: JSTypeOf<From>,
  fromDType: RuntimeDType<From>,
  toDType: RuntimeDType<To>,
): ConversionResult<To> {
  return convertValue(value, fromDType, toDType, PERMISSIVE_CONVERSION_OPTIONS);
}

/**
 * Check if a conversion would be lossy without performing it
 *
 * @example
 * const isLossy = wouldBeLossy(3.14, getDType('float32'), getDType('int32')); // true
 * const isNotLossy = wouldBeLossy(42, getDType('int32'), getDType('float32')); // false
 */
export function wouldBeLossy<From extends AnyDType, To extends AnyDType>(
  value: JSTypeOf<From>,
  fromDType: RuntimeDType<From>,
  toDType: RuntimeDType<To>,
): boolean {
  const result = convertValue(value, fromDType, toDType, STRICT_CONVERSION_OPTIONS);
  return !result.success;
}

// =============================================================================
// Conversion Error Classes
// =============================================================================

/**
 * Error thrown when type conversion fails
 */
export class ConversionError extends Error {
  constructor(
    message: string,
    public readonly fromDType: RuntimeDType,
    public readonly toDType: RuntimeDType,
    public readonly value: unknown,
  ) {
    super(message);
    this.name = 'ConversionError';
  }
}

/**
 * Error thrown when precision would be lost in conversion
 */
export class PrecisionLossError extends ConversionError {
  constructor(
    fromDType: RuntimeDType,
    toDType: RuntimeDType,
    value: unknown,
    public readonly precisionLossInfo: PrecisionLossInfo,
  ) {
    super(
      `Precision loss converting ${fromDType.name} to ${toDType.name}: ${precisionLossInfo.message}`,
      fromDType,
      toDType,
      value,
    );
    this.name = 'PrecisionLossError';
  }
}

/**
 * Error thrown when value overflows target type
 */
export class OverflowError extends ConversionError {
  constructor(fromDType: RuntimeDType, toDType: RuntimeDType, value: unknown) {
    super(
      `Value ${String(value)} overflows ${toDType.name} range [${toDType.minValue.toString()}, ${toDType.maxValue.toString()}]`,
      fromDType,
      toDType,
      value,
    );
    this.name = 'OverflowError';
  }
}
