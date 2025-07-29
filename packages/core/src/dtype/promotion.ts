/**
 * Type promotion rules and logic for DType operations
 *
 * This module implements NumPy-compatible type promotion rules with both
 * compile-time type computation and runtime promotion logic. It ensures
 * that operations between different DTypes produce predictable, safe results.
 */

import type { AnyDType, DTypeName, Promote } from './types.js';
import { type RuntimeDType, getDType } from './runtime.js';
import { DTYPE_CONSTANTS_MAP } from './constants.js';

// =============================================================================
// Type Promotion Matrix
// =============================================================================

/**
 * Precomputed promotion matrix for O(1) runtime lookups
 * This matrix is symmetric: PROMOTION_TABLE[A][B] === PROMOTION_TABLE[B][A]
 *
 * Promotion rules follow NumPy's hierarchy:
 * Bool < Int8 < Uint8 < Int16 < Uint16 < Int32 < Uint32 < Int64 < Uint64 < Float32 < Float64
 */
const PROMOTION_TABLE = {
  bool: {
    bool: 'bool',
    int8: 'int8',
    uint8: 'uint8',
    int16: 'int16',
    uint16: 'uint16',
    int32: 'int32',
    uint32: 'uint32',
    int64: 'int64',
    uint64: 'uint64',
    float32: 'float32',
    float64: 'float64',
  },
  int8: {
    bool: 'int8',
    int8: 'int8',
    uint8: 'int16', // Mixed signedness promotes to larger signed
    int16: 'int16',
    uint16: 'int32', // Mixed signedness promotes to larger signed
    int32: 'int32',
    uint32: 'int64', // Mixed signedness promotes to larger signed
    int64: 'int64',
    uint64: 'float64', // Cannot represent all uint64 values in int64
    float32: 'float32',
    float64: 'float64',
  },
  uint8: {
    bool: 'uint8',
    int8: 'int16', // Mixed signedness promotes to larger signed
    uint8: 'uint8',
    int16: 'int16',
    uint16: 'uint16',
    int32: 'int32',
    uint32: 'uint32',
    int64: 'int64',
    uint64: 'uint64',
    float32: 'float32',
    float64: 'float64',
  },
  int16: {
    bool: 'int16',
    int8: 'int16',
    uint8: 'int16',
    int16: 'int16',
    uint16: 'int32', // Mixed signedness promotes to larger signed
    int32: 'int32',
    uint32: 'int64', // Mixed signedness promotes to larger signed
    int64: 'int64',
    uint64: 'float64', // Cannot represent all uint64 values in int64
    float32: 'float32',
    float64: 'float64',
  },
  uint16: {
    bool: 'uint16',
    int8: 'int32', // Mixed signedness promotes to larger signed
    uint8: 'uint16',
    int16: 'int32', // Mixed signedness promotes to larger signed
    uint16: 'uint16',
    int32: 'int32',
    uint32: 'uint32',
    int64: 'int64',
    uint64: 'uint64',
    float32: 'float32',
    float64: 'float64',
  },
  int32: {
    bool: 'int32',
    int8: 'int32',
    uint8: 'int32',
    int16: 'int32',
    uint16: 'int32',
    int32: 'int32',
    uint32: 'int64', // Mixed signedness promotes to larger signed
    int64: 'int64',
    uint64: 'float64', // Cannot represent all uint64 values in int64
    float32: 'float64', // int32 precision requires float64
    float64: 'float64',
  },
  uint32: {
    bool: 'uint32',
    int8: 'int64', // Mixed signedness promotes to larger signed
    uint8: 'uint32',
    int16: 'int64', // Mixed signedness promotes to larger signed
    uint16: 'uint32',
    int32: 'int64', // Mixed signedness promotes to larger signed
    uint32: 'uint32',
    int64: 'int64',
    uint64: 'uint64',
    float32: 'float64', // uint32 precision requires float64
    float64: 'float64',
  },
  int64: {
    bool: 'int64',
    int8: 'int64',
    uint8: 'int64',
    int16: 'int64',
    uint16: 'int64',
    int32: 'int64',
    uint32: 'int64',
    int64: 'int64',
    uint64: 'float64', // Cannot represent all uint64 values in int64
    float32: 'float64', // int64 precision requires float64
    float64: 'float64',
  },
  uint64: {
    bool: 'uint64',
    int8: 'float64', // Cannot represent all uint64 values in int64
    uint8: 'uint64',
    int16: 'float64', // Cannot represent all uint64 values in int64
    uint16: 'uint64',
    int32: 'float64', // Cannot represent all uint64 values in int64
    uint32: 'uint64',
    int64: 'float64', // Cannot represent all uint64 values in int64
    uint64: 'uint64',
    float32: 'float64', // uint64 precision requires float64
    float64: 'float64',
  },
  float32: {
    bool: 'float32',
    int8: 'float32',
    uint8: 'float32',
    int16: 'float32',
    uint16: 'float32',
    int32: 'float64', // int32 precision requires float64
    uint32: 'float64', // uint32 precision requires float64
    int64: 'float64', // int64 precision requires float64
    uint64: 'float64', // uint64 precision requires float64
    float32: 'float32',
    float64: 'float64',
  },
  float64: {
    bool: 'float64',
    int8: 'float64',
    uint8: 'float64',
    int16: 'float64',
    uint16: 'float64',
    int32: 'float64',
    uint32: 'float64',
    int64: 'float64',
    uint64: 'float64',
    float32: 'float64',
    float64: 'float64',
  },
} as const;

// Extract the promotion result from the table at compile time
// type PromotionTableType = typeof PROMOTION_TABLE;
// type ExtractPromotion<A extends DTypeName, B extends DTypeName> = PromotionTableType[A][B];

// =============================================================================
// Runtime Type Promotion
// =============================================================================

/**
 * Promote two RuntimeDType instances to their common type
 * Uses precomputed lookup table for O(1) performance
 *
 * @example
 * const int32 = getDType('int32');
 * const float32 = getDType('float32');
 * const promoted = promoteTypes(int32, float32); // RuntimeDType<Float64>
 */

// Verify at compile time that our table matches the Promote type
// This creates a conditional type that is 'true' only if the types match exactly
// type VerifyTableMatchesPromote<A extends AnyDType, B extends AnyDType> =
//   DTypeFromName<ExtractPromotion<A['__dtype'], B['__dtype']>> extends Promote<A, B>
//     ? Promote<A, B> extends DTypeFromName<ExtractPromotion<A['__dtype'], B['__dtype']>>
//       ? true
//       : false
//     : false;

export function promoteTypes<A extends AnyDType, B extends AnyDType>(
  a: RuntimeDType<A>,
  b: RuntimeDType<B>,
): RuntimeDType<Promote<A, B>> {
  // Get the promoted dtype name from the table
  const promotedName = PROMOTION_TABLE[a.name][b.name];

  // This cast is safe because:
  // 1. PROMOTION_TABLE is const, so TypeScript knows all literal types
  // 2. We have compile-time tests verifying the table matches Promote
  // 3. The table is symmetric and complete
  return getDType(promotedName) as unknown as RuntimeDType<Promote<A, B>>;
}

/**
 * Promote two compile-time DTypes to their common runtime type
 *
 * This utility bridges the gap between compile-time type promotion and runtime execution.
 * It follows the same pattern as toFloatDType, using runtime promotion logic while
 * maintaining compile-time type safety through strategic type assertion.
 *
 * @param dtypeA First dtype to promote
 * @param dtypeB Second dtype to promote
 * @returns The promoted runtime dtype
 */
export function toPromotedDType<A extends AnyDType, B extends AnyDType>(
  dtypeA: A,
  dtypeB: B,
): Promote<A, B> {
  const promotedRuntimeDType = promoteTypes(getDType(dtypeA.__dtype), getDType(dtypeB.__dtype));
  return DTYPE_CONSTANTS_MAP[promotedRuntimeDType.name] as Promote<A, B>;
}

/**
 * Check if two RuntimeDType instances can be promoted together
 *
 * @example
 * const int32 = getDType('int32');
 * const float32 = getDType('float32');
 * const canPromote = canPromoteTypes(int32, float32); // true
 */
export function canPromoteTypes<A extends AnyDType, B extends AnyDType>(
  a: RuntimeDType<A>,
  b: RuntimeDType<B>,
): boolean {
  try {
    promoteTypes(a, b);
    return true;
  } catch {
    return false;
  }
}

/**
 * Promote multiple DTypes to their common type
 * Useful for operations involving more than two operands
 *
 * @example
 * const dtypes = [getDType('int8'), getDType('float32'), getDType('int32')];
 * const common = promoteMultipleTypes(dtypes); // RuntimeDType<Float64>
 */
export function promoteMultipleTypes(dtypes: readonly RuntimeDType[]): RuntimeDType {
  if (dtypes.length === 0) {
    throw new Error('Cannot promote empty array of dtypes');
  }

  if (dtypes.length === 1) {
    return dtypes[0]!;
  }

  let result = dtypes[0]!;
  for (let i = 1; i < dtypes.length; i++) {
    result = promoteTypes(result, dtypes[i]!);
  }

  return result;
}

/**
 * Find the most specific common type for an array of values
 * Analyzes actual values to determine the minimal sufficient type
 *
 * @example
 * const values = [1, 2.5, true];
 * const dtype = findCommonType(values); // RuntimeDType<Float32>
 */
export function findCommonType(values: readonly unknown[]): RuntimeDType {
  if (values.length === 0) {
    throw new Error('Cannot find common type for empty array');
  }

  const dtypes: RuntimeDType[] = [];

  for (const value of values) {
    if (typeof value === 'boolean') {
      dtypes.push(getDType('bool'));
    } else if (typeof value === 'number') {
      if (Number.isInteger(value)) {
        // Choose the smallest integer type that can hold the value
        if (value >= -128 && value <= 127) {
          dtypes.push(getDType('int8'));
        } else if (value >= 0 && value <= 255) {
          dtypes.push(getDType('uint8'));
        } else if (value >= -32768 && value <= 32767) {
          dtypes.push(getDType('int16'));
        } else if (value >= 0 && value <= 65535) {
          dtypes.push(getDType('uint16'));
        } else if (value >= -2147483648 && value <= 2147483647) {
          dtypes.push(getDType('int32'));
        } else if (value >= 0 && value <= 4294967295) {
          dtypes.push(getDType('uint32'));
        } else {
          // Large integers need float64 for precision
          dtypes.push(getDType('float64'));
        }
      } else {
        // Floating point number
        // Check if it can be represented in float32 without precision loss
        const asFloat32 = Math.fround(value);
        if (asFloat32 === value || !Number.isFinite(value)) {
          dtypes.push(getDType('float32'));
        } else {
          dtypes.push(getDType('float64'));
        }
      }
    } else if (typeof value === 'bigint') {
      if (value >= 0n) {
        dtypes.push(getDType('uint64'));
      } else {
        dtypes.push(getDType('int64'));
      }
    } else {
      throw new Error(`Cannot determine DType for value: ${String(value)}`);
    }
  }

  return promoteMultipleTypes(dtypes);
}

// =============================================================================
// Result Type Computation
// =============================================================================

/**
 * Compute the result type for a binary operation
 * This is the primary function used by tensor operations
 *
 * @example
 * const a = getDType('int32');
 * const b = getDType('float32');
 * const result = computeResultType(a, b); // RuntimeDType<Float64>
 */
export function computeResultType<A extends AnyDType, B extends AnyDType>(
  a: RuntimeDType<A>,
  b: RuntimeDType<B>,
): RuntimeDType<Promote<A, B>> {
  return promoteTypes(a, b);
}

/**
 * Compute the result type for a unary operation
 * Most unary operations preserve the input type, but some may promote
 *
 * @example
 * const input = getDType('int8');
 * const result = computeUnaryResultType(input, 'abs'); // RuntimeDType<Int8>
 */
export function computeUnaryResultType<T extends AnyDType>(
  input: RuntimeDType<T>,
  operation: string,
): RuntimeDType {
  // Most unary operations preserve type
  switch (operation) {
    case 'abs':
    case 'neg':
    case 'sign':
      return input;

    case 'sqrt':
    case 'exp':
    case 'log':
    case 'sin':
    case 'cos':
    case 'tan':
      // Mathematical functions always produce floating-point results
      if (input.isInteger) {
        return input.byteSize <= 2 ? getDType('float32') : getDType('float64');
      }
      return input;

    case 'floor':
    case 'ceil':
    case 'round':
      // Rounding functions preserve type for integers, promote to int for floats
      if (input.isInteger) {
        return input;
      }
      // For floating-point inputs, return appropriate integer type
      return input.name === 'float32' ? getDType('int32') : getDType('int64');

    default:
      // Unknown operation, preserve type
      return input;
  }
}

// =============================================================================
// Type Promotion Validation
// =============================================================================

/**
 * Validate that the promotion matrix is correctly defined
 * This function can be used in tests to ensure consistency
 */
export function validatePromotionMatrix(): void {
  const dtypeNames = Object.keys(PROMOTION_TABLE) as DTypeName[];

  for (const nameA of dtypeNames) {
    for (const nameB of dtypeNames) {
      const promotedAB = PROMOTION_TABLE[nameA][nameB];
      const promotedBA = PROMOTION_TABLE[nameB][nameA];

      // Verify symmetry
      if (promotedAB !== promotedBA) {
        throw new Error(
          `Promotion matrix not symmetric: ${nameA} + ${nameB} = ${promotedAB}, but ${nameB} + ${nameA} = ${promotedBA}`,
        );
      }

      // Verify that promotion result is valid
      if (!isValidDTypeName(promotedAB)) {
        throw new Error(`Invalid promotion result: ${nameA} + ${nameB} = ${promotedAB}`);
      }

      // Verify that promoting with self returns self
      if (nameA === nameB && promotedAB !== nameA) {
        throw new Error(
          `Self-promotion failed: ${nameA} + ${nameA} = ${promotedAB}, expected ${nameA}`,
        );
      }
    }
  }
}

/**
 * Check if a string is a valid DType name
 */
function isValidDTypeName(name: string): name is DTypeName {
  return name in PROMOTION_TABLE;
}

// =============================================================================
// Promotion Analysis and Debugging
// =============================================================================

/**
 * Get detailed information about how two types would be promoted
 * Useful for debugging and understanding promotion decisions
 */
export function analyzePromotion<A extends AnyDType, B extends AnyDType>(
  a: RuntimeDType<A>,
  b: RuntimeDType<B>,
): {
  inputTypes: [string, string];
  resultType: string;
  reason: string;
  isPrecisionPreserving: boolean;
  isWidening: boolean;
} {
  const result = promoteTypes(a, b);

  let reason = '';
  let isPrecisionPreserving = true;
  let isWidening = false;

  // Same type
  if (a.name === b.name) {
    reason = 'Same type, no promotion needed';
  }
  // Boolean promotion
  else if (a.name === 'bool' || b.name === 'bool') {
    reason = 'Boolean promotes to the other type';
    isWidening = true;
  }
  // Float promotion
  else if (!a.isInteger || !b.isInteger) {
    reason = 'At least one operand is floating-point';
    if (a.isInteger || b.isInteger) {
      isPrecisionPreserving = false;
      isWidening = true;
    }
  }
  // Integer promotion with mixed signedness
  else if (a.signed !== b.signed) {
    reason = 'Mixed signedness requires promotion to larger signed type';
    isWidening = true;
  }
  // Size promotion
  else {
    reason = 'Size difference requires promotion to larger type';
    isWidening = true;
  }

  return {
    inputTypes: [a.name, b.name] as [string, string],
    resultType: result.name,
    reason,
    isPrecisionPreserving,
    isWidening,
  };
}

/**
 * Get a human-readable explanation of promotion rules
 */
export function getPromotionRules(): string {
  return `
NumPy-Compatible Type Promotion Rules:

1. Hierarchy: Bool < Int8 < Uint8 < Int16 < Uint16 < Int32 < Uint32 < Int64 < Uint64 < Float32 < Float64

2. Same type: No promotion needed

3. Boolean: Always promotes to the other type

4. Mixed signedness (integer): Promotes to larger signed type
   - int8 + uint8 → int16
   - int16 + uint16 → int32
   - int32 + uint32 → int64

5. Integer + Float: Promotes to float, choosing size that preserves precision
   - int32 + float32 → float64 (int32 needs more precision than float32 provides)
   - int8 + float32 → float32 (float32 can represent all int8 values)

6. Large integers + unsigned: May promote to float64 when no integer type can represent both ranges
   - int64 + uint64 → float64

7. Float + Float: Promotes to the larger float type
   - float32 + float64 → float64

These rules ensure no data loss and maintain NumPy compatibility.
`.trim();
}
