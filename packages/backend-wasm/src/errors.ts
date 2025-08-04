/**
 * Minimal error handling for WASM backend
 *
 * Since arena-based memory management handles most safety concerns,
 * we only need basic error classes for operational issues.
 */

/**
 * Base WASM error class with error categories and context
 */
export class WASMError extends Error {
  public readonly code: string;
  public readonly category: 'operational' | 'initialization' | 'memory' | 'bounds';

  constructor(
    message: string,
    code: string,
    category: 'operational' | 'initialization' | 'memory' | 'bounds',
    public readonly context?: Record<string, unknown>,
  ) {
    super(message);
    this.name = 'WASMError';
    this.code = code;
    this.category = category;
  }

  getFormattedMessage(): string {
    let formatted = `${this.name}: ${this.message}`;
    if (this.context && Object.keys(this.context).length > 0) {
      formatted += '\nContext:\n';
      for (const [key, value] of Object.entries(this.context)) {
        formatted += `  ${key}: ${value}\n`;
      }
    }
    return formatted;
  }
}

/**
 * Operation execution error
 */
export class WASMOperationError extends WASMError {
  constructor(operation: string, reason: string, context?: Record<string, unknown>) {
    super(
      `WASM operation '${operation}' failed: ${reason}`,
      'OPERATION_FAILED',
      'operational',
      context,
    );
    this.name = 'WASMOperationError';
  }
}

/**
 * Invalid state error (device not initialized, etc.)
 */
export class WASMInvalidStateError extends WASMError {
  constructor(
    operation: string,
    currentState: string,
    expectedState: string,
    context?: Record<string, unknown>,
  ) {
    super(
      `Cannot ${operation} in state '${currentState}', expected '${expectedState}'`,
      'INVALID_STATE',
      'initialization',
      context,
    );
    this.name = 'WASMInvalidStateError';
  }
}

/**
 * Bounds/size validation error
 */
export class WASMBoundsError extends WASMError {
  constructor(
    operation: string,
    value: number,
    bounds: { min?: number; max?: number },
    context?: Record<string, unknown>,
  ) {
    const boundsStr = `min=${bounds.min ?? 'none'}, max=${bounds.max ?? 'none'}`;
    super(
      `Bounds check failed for ${operation}: value ${value} outside bounds (${boundsStr})`,
      'BOUNDS_CHECK_FAILED',
      'operational',
      context,
    );
    this.name = 'WASMBoundsError';
  }
}

/**
 * Memory allocation error
 */
export class WASMAllocationError extends WASMError {
  constructor(requestedSize: number, reason: string, context?: Record<string, unknown>) {
    super(`Allocation failed for ${requestedSize} bytes: ${reason}`, 'ALLOCATION_FAILED', 'memory', context);
    this.name = 'WASMAllocationError';
  }
}

/**
 * Memory limit exceeded error
 */
export class WASMMemoryLimitError extends WASMError {
  constructor(requestedSize: number, availableSize: number, context?: Record<string, unknown>) {
    super(
      `Memory limit exceeded: requested ${requestedSize} bytes, available ${availableSize} bytes`,
      'MEMORY_LIMIT_EXCEEDED',
      'memory',
      context,
    );
    this.name = 'WASMMemoryLimitError';
  }
}
