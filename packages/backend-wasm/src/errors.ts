/**
 * Custom error classes for WASM backend error handling
 * 
 * Error Classification:
 * - Critical Errors: Must throw and halt execution (memory corruption, invalid states)
 * - Cleanup Errors: Should log but not throw (GC cleanup, disposal during shutdown)  
 * - Operational Errors: Should throw with recovery hints (allocation failures, out of bounds)
 */

/**
 * Base class for all WASM backend errors
 */
export abstract class WASMError extends Error {
  abstract readonly category: 'critical' | 'cleanup' | 'operational';
  abstract readonly code: string;
  
  constructor(message: string, public readonly context?: Record<string, unknown>) {
    super(message);
    this.name = this.constructor.name;
    
    // Maintain proper stack trace for V8
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, this.constructor);
    }
  }

  /**
   * Get formatted error message with context
   */
  getFormattedMessage(): string {
    let message = this.message;
    
    if (this.context && Object.keys(this.context).length > 0) {
      const contextStr = Object.entries(this.context)
        .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
        .join(', ');
      message += ` (Context: ${contextStr})`;
    }
    
    return message;
  }
}

/**
 * Critical errors that indicate serious problems requiring immediate attention
 */
export abstract class WASMCriticalError extends WASMError {
  readonly category = 'critical' as const;
}

/**
 * Memory corruption or invalid state detected
 */
export class WASMMemoryCorruptionError extends WASMCriticalError {
  readonly code = 'MEMORY_CORRUPTION';
  
  constructor(message: string, context?: Record<string, unknown>) {
    super(`Memory corruption detected: ${message}`, context);
  }
}

/**
 * Buffer handle is in an invalid state
 */
export class WASMInvalidHandleError extends WASMCriticalError {
  readonly code = 'INVALID_HANDLE';
  
  constructor(handleId: string | number, operation: string, context?: Record<string, unknown>) {
    super(
      `Invalid buffer handle ${handleId} for operation '${operation}'`,
      { handleId, operation, ...context }
    );
  }
}

/**
 * Device is in an invalid state for the requested operation
 */
export class WASMInvalidStateError extends WASMCriticalError {
  readonly code = 'INVALID_STATE';
  
  constructor(operation: string, expectedState: string, actualState: string, context?: Record<string, unknown>) {
    super(
      `Cannot perform '${operation}': expected state '${expectedState}', but device is in state '${actualState}'`,
      { operation, expectedState, actualState, ...context }
    );
  }
}

/**
 * Operational errors that can potentially be recovered from
 */
export abstract class WASMOperationalError extends WASMError {
  readonly category = 'operational' as const;
}

/**
 * Memory allocation failed
 */
export class WASMAllocationError extends WASMOperationalError {
  readonly code = 'ALLOCATION_FAILED';
  
  constructor(
    requestedSize: number, 
    reason: string, 
    suggestions?: string[], 
    context?: Record<string, unknown>
  ) {
    let message = `Failed to allocate ${requestedSize} bytes: ${reason}`;
    
    if (suggestions && suggestions.length > 0) {
      message += `\n\nSuggestions:\n${suggestions.map(s => `  - ${s}`).join('\n')}`;
    }
    
    super(message, { requestedSize, reason, suggestions, ...context });
  }
}

/**
 * Memory limit exceeded
 */
export class WASMMemoryLimitError extends WASMOperationalError {
  readonly code = 'MEMORY_LIMIT_EXCEEDED';
  
  constructor(
    currentUsage: number, 
    limit: number, 
    requestedSize: number, 
    context?: Record<string, unknown>
  ) {
    const message = `Memory limit exceeded: ${currentUsage + requestedSize} bytes would exceed limit of ${limit} bytes`;
    const suggestions = [
      'Try disposing unused buffers with device.disposeData()',
      'Consider calling device.performIntensiveCleanup() to compact memory pools',
      'Reduce the size of your allocation or increase memory limits'
    ];
    
    super(message, { currentUsage, limit, requestedSize, suggestions, ...context });
  }
}

/**
 * Operation failed due to bounds checking
 */
export class WASMBoundsError extends WASMOperationalError {
  readonly code = 'BOUNDS_CHECK_FAILED';
  
  constructor(
    operation: string,
    index: number | string,
    bounds: { min?: number; max?: number; size?: number },
    context?: Record<string, unknown>
  ) {
    let message = `Bounds check failed for ${operation}: index ${index}`;
    
    if (bounds.size !== undefined) {
      message += ` is out of bounds for size ${bounds.size}`;
    } else if (bounds.min !== undefined && bounds.max !== undefined) {
      message += ` is not in range [${bounds.min}, ${bounds.max}]`;
    } else if (bounds.max !== undefined) {
      message += ` exceeds maximum ${bounds.max}`;
    } else if (bounds.min !== undefined) {
      message += ` is below minimum ${bounds.min}`;
    }
    
    super(message, { operation, index, bounds, ...context });
  }
}

/**
 * WASM operation execution failed
 */
export class WASMOperationError extends WASMOperationalError {
  readonly code = 'OPERATION_FAILED';
  
  constructor(
    operation: string,
    reason: string,
    inputs?: Record<string, unknown>,
    context?: Record<string, unknown>
  ) {
    super(
      `WASM operation '${operation}' failed: ${reason}`,
      { operation, reason, inputs, ...context }
    );
  }
}

/**
 * Cleanup errors that should be logged but not thrown
 */
export abstract class WASMCleanupError extends WASMError {
  readonly category = 'cleanup' as const;
}

/**
 * Buffer cleanup failed during disposal
 */
export class WASMCleanupBufferError extends WASMCleanupError {
  readonly code = 'CLEANUP_BUFFER_FAILED';
  
  constructor(bufferId: string, reason: string, context?: Record<string, unknown>) {
    super(
      `Failed to cleanup buffer ${bufferId}: ${reason}. This may indicate the buffer was already disposed or the WASM module is shutting down.`,
      { bufferId, reason, ...context }
    );
  }
}

/**
 * Finalization registry cleanup failed
 */
export class WASMCleanupFinalizationError extends WASMCleanupError {
  readonly code = 'CLEANUP_FINALIZATION_FAILED';
  
  constructor(reason: string, context?: Record<string, unknown>) {
    super(
      `Finalization registry cleanup failed: ${reason}. This is typically harmless but may indicate resource leaks.`,
      { reason, ...context }
    );
  }
}

/**
 * Error handler utility functions
 */
export class WASMErrorHandler {
  /**
   * Handle error based on its category
   * - Critical errors: always throw
   * - Operational errors: always throw  
   * - Cleanup errors: log warning and continue
   */
  static handle(error: WASMError): never | void {
    if (error.category === 'cleanup') {
      console.warn(`[WASM Cleanup Warning] ${error.getFormattedMessage()}`);
      return;
    }
    
    // Critical and operational errors should be thrown
    throw error;
  }
  
  /**
   * Create appropriate error for buffer release failure
   */
  static createBufferReleaseError(
    bufferId: string, 
    wasmResult: boolean,
    context?: Record<string, unknown>
  ): WASMError {
    if (!wasmResult) {
      // If WASM says release failed, this could be critical or cleanup depending on context
      return new WASMCleanupBufferError(
        bufferId,
        'WASM release_buffer returned false',
        context
      );
    }
    
    return new WASMInvalidHandleError(bufferId, 'release', context);
  }
  
  /**
   * Create appropriate error for allocation failure
   */
  static createAllocationError(
    size: number,
    wasmError: Error,
    currentStats?: { totalAllocated: number; limit?: number }
  ): WASMError {
    const errorMessage = wasmError.message;
    
    // Check for specific WASM error patterns
    if (errorMessage.includes('Out of bounds memory access')) {
      const suggestions = [
        'The WASM memory heap may be exhausted',
        'Try disposing unused buffers to free memory',
        'Consider reducing allocation size or increasing WASM memory limits'
      ];
      return new WASMAllocationError(size, 'WASM memory exhausted', suggestions, {
        wasmError: errorMessage,
        currentStats
      });
    }
    
    if (currentStats?.limit && currentStats.totalAllocated + size > currentStats.limit) {
      return new WASMMemoryLimitError(
        currentStats.totalAllocated,
        currentStats.limit,
        size,
        { wasmError: errorMessage }
      );
    }
    
    return new WASMAllocationError(size, errorMessage, undefined, {
      wasmError: errorMessage,
      currentStats
    });
  }
}