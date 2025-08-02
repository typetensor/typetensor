/**
 * Minimal error handling for WASM backend
 * 
 * Since arena-based memory management handles most safety concerns,
 * we only need basic error classes for operational issues.
 */

/**
 * Base WASM error class
 */
export class WASMError extends Error {
  constructor(message: string, public readonly context?: Record<string, unknown>) {
    super(message);
    this.name = 'WASMError';
  }
}

/**
 * Operation execution error
 */
export class WASMOperationError extends WASMError {
  constructor(operation: string, reason: string, context?: Record<string, unknown>) {
    super(`WASM operation '${operation}' failed: ${reason}`, context);
    this.name = 'WASMOperationError';
  }
}

/**
 * Invalid state error (device not initialized, etc.)
 */
export class WASMInvalidStateError extends WASMError {
  constructor(operation: string, currentState: string, expectedState: string, context?: Record<string, unknown>) {
    super(`Cannot ${operation} in state '${currentState}', expected '${expectedState}'`, context);
    this.name = 'WASMInvalidStateError';
  }
}

/**
 * Bounds/size validation error
 */
export class WASMBoundsError extends WASMError {
  constructor(operation: string, value: number, bounds: { min?: number; max?: number }, context?: Record<string, unknown>) {
    const boundsStr = `min=${bounds.min ?? 'none'}, max=${bounds.max ?? 'none'}`;
    super(`${operation} value ${value} outside bounds (${boundsStr})`, context);
    this.name = 'WASMBoundsError';
  }
}