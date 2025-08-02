/**
 * Defensive wrapper for WASM operations to handle "Unreachable code" errors
 * 
 * This module provides a robust error handling layer that can detect and recover
 * from WASM memory corruption and panic states.
 */

import type { WasmOperationDispatcher, WasmBufferHandle } from './types/wasm-bindings';
import { WASMErrorHandler, WASMInvalidStateError } from './errors';

export interface WASMOperationOptions {
  maxRetries?: number;
  retryDelay?: number;
  fallbackEnabled?: boolean;
}

export interface WASMRecoveryStats {
  totalOperations: number;
  failedOperations: number;
  recoveredOperations: number;
  unrecoverableErrors: number;
  lastRecoveryTime?: number;
}

/**
 * Defensive wrapper that provides error recovery for WASM operations
 */
export class WASMDefensiveWrapper {
  private operationDispatcher: WasmOperationDispatcher;
  private stats: WASMRecoveryStats = {
    totalOperations: 0,
    failedOperations: 0,
    recoveredOperations: 0,
    unrecoverableErrors: 0
  };
  private isRecovering = false;
  private lastHealthCheck = 0;
  private healthCheckInterval = 5000; // 5 seconds

  constructor(operationDispatcher: WasmOperationDispatcher) {
    this.operationDispatcher = operationDispatcher;
  }

  /**
   * Execute a WASM operation with defensive error handling
   */
  async executeWithDefense<T>(
    operation: () => T,
    operationName: string,
    options: WASMOperationOptions = {}
  ): Promise<T> {
    const { 
      maxRetries = 3, 
      retryDelay = 100, 
      fallbackEnabled = true 
    } = options;

    this.stats.totalOperations++;

    // Pre-flight health check for critical operations
    if (this.shouldPerformHealthCheck()) {
      await this.performHealthCheck();
    }

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return operation();
      } catch (error) {
        this.stats.failedOperations++;
        
        if (this.isUnrecoverableError(error)) {
          this.stats.unrecoverableErrors++;
          throw this.wrapUnrecoverableError(error, operationName);
        }

        if (this.isWASMPanicError(error)) {
          console.warn(`WASM panic detected in ${operationName} (attempt ${attempt + 1}/${maxRetries + 1}):`, error);
          
          if (attempt < maxRetries) {
            await this.attemptRecovery(operationName, retryDelay * (attempt + 1));
            continue;
          } else if (fallbackEnabled) {
            return await this.attemptFallback(operationName, error);
          }
        }

        // Re-throw other errors immediately
        throw error;
      }
    }

    // Should never reach here, but typescript requires it
    throw new WASMInvalidStateError(
      operationName,
      'unknown',
      'exhausted_retries',
      { maxRetries, operationName }
    );
  }

  /**
   * Check if the error indicates a WASM panic (unreachable code)
   */
  private isWASMPanicError(error: any): boolean {
    if (!error) return false;
    
    const errorMessage = error.message || String(error);
    return (
      errorMessage.includes('Unreachable code should not be executed') ||
      errorMessage.includes('unreachable') ||
      errorMessage.includes('wasm trap') ||
      errorMessage.includes('RuntimeError')
    );
  }

  /**
   * Check if error is unrecoverable
   */
  private isUnrecoverableError(error: any): boolean {
    if (!error) return false;
    
    const errorMessage = error.message || String(error);
    return (
      errorMessage.includes('out of memory') ||
      errorMessage.includes('stack overflow') ||
      errorMessage.includes('invalid module')
    );
  }

  /**
   * Wrap unrecoverable errors with helpful context
   */
  private wrapUnrecoverableError(error: any, operationName: string): Error {
    return new WASMInvalidStateError(
      operationName,
      'corrupted',
      'unrecoverable',
      {
        originalError: error instanceof Error ? error.message : String(error),
        suggestions: [
          'Restart the WASM device',
          'Reduce memory pressure',
          'Break operations into smaller chunks'
        ],
        stats: this.stats
      }
    );
  }

  /**
   * Attempt to recover from WASM panic
   */
  private async attemptRecovery(operationName: string, delay: number): Promise<void> {
    if (this.isRecovering) {
      // Already recovering, just wait
      await this.sleep(delay);
      return;
    }

    this.isRecovering = true;
    this.stats.lastRecoveryTime = Date.now();

    try {
      console.log(`Attempting WASM recovery for ${operationName}...`);
      
      // Strategy 1: Force garbage collection
      if (typeof global !== 'undefined' && global.gc) {
        global.gc();
      }
      
      // Strategy 2: Small delay to let WASM settle
      await this.sleep(delay);
      
      // Strategy 3: Try to perform a simple operation to test recovery
      try {
        const testStats = this.operationDispatcher.get_memory_stats();
        console.log('WASM recovery successful, memory stats:', {
          allocated: testStats.total_allocated_bytes,
          buffers: testStats.active_buffers
        });
        this.stats.recoveredOperations++;
      } catch (testError) {
        console.warn('WASM recovery test failed:', testError);
        throw testError;
      }
      
    } finally {
      this.isRecovering = false;
    }
  }

  /**
   * Attempt fallback operation when WASM fails
   */
  private async attemptFallback<T>(operationName: string, originalError: any): Promise<T> {
    console.warn(`Attempting fallback for ${operationName} after WASM failure`);
    
    // For now, we don't have fallback implementations for most operations
    // This is where we could implement JavaScript fallbacks in the future
    
    throw new WASMInvalidStateError(
      operationName,
      'panic',
      'no_fallback',
      {
        originalError: originalError instanceof Error ? originalError.message : String(originalError),
        message: 'WASM operation failed and no fallback implementation is available',
        suggestions: [
          'Try reducing the operation size',
          'Restart the WASM device',
          'Use a different backend if available'
        ]
      }
    );
  }

  /**
   * Perform health check on WASM operations
   */
  private async performHealthCheck(): Promise<void> {
    this.lastHealthCheck = Date.now();
    
    try {
      // Simple health check: get memory stats
      const stats = this.operationDispatcher.get_memory_stats();
      
      // Check for concerning memory patterns and trigger cleanup
      if (stats.active_buffers > 10000) {
        console.warn('High buffer count detected:', stats.active_buffers);
        
        // Aggressive cleanup for high buffer counts
        if (stats.active_buffers > 50000) {
          console.log('Triggering intensive cleanup due to high buffer count');
          try {
            this.operationDispatcher.intensive_cleanup();
          } catch (cleanupError) {
            console.warn('Intensive cleanup failed:', cleanupError);
          }
        }
      }
      
      if (stats.total_allocated_bytes > 1024 * 1024 * 1024) { // > 1GB
        console.warn('High memory usage detected:', stats.total_allocated_bytes);
        
        // Aggressive cleanup for high memory usage
        if (stats.total_allocated_bytes > 2 * 1024 * 1024 * 1024) { // > 2GB
          console.log('Triggering intensive cleanup due to high memory usage');
          try {
            this.operationDispatcher.intensive_cleanup();
            
            // Also trigger JavaScript GC if available
            if (typeof global !== 'undefined' && global.gc) {
              global.gc();
            }
          } catch (cleanupError) {
            console.warn('Intensive cleanup failed:', cleanupError);
          }
        }
      }
      
    } catch (error) {
      console.warn('WASM health check failed:', error);
      // Don't throw here - just log the warning
    }
  }

  /**
   * Check if we should perform a health check
   */
  private shouldPerformHealthCheck(): boolean {
    const timeSinceLastCheck = Date.now() - this.lastHealthCheck;
    
    // More frequent checks if we've had recent failures
    if (this.stats.failedOperations > 10) {
      return timeSinceLastCheck > (this.healthCheckInterval / 2); // Every 2.5 seconds
    }
    
    // Normal frequency
    return timeSinceLastCheck > this.healthCheckInterval;
  }

  /**
   * Get recovery statistics
   */
  getStats(): WASMRecoveryStats {
    return { ...this.stats };
  }

  /**
   * Reset statistics
   */
  resetStats(): void {
    this.stats = {
      totalOperations: 0,
      failedOperations: 0,
      recoveredOperations: 0,
      unrecoverableErrors: 0
    };
  }

  /**
   * Simple sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Create wrapper methods for common WASM operations
 */
export function createDefensiveWASMOperations(operationDispatcher: WasmOperationDispatcher) {
  const wrapper = new WASMDefensiveWrapper(operationDispatcher);

  return {
    wrapper,
    
    // Defensive buffer operations
    createEmptyBuffer: async (size: number) => 
      wrapper.executeWithDefense(
        () => operationDispatcher.create_empty_buffer(size),
        'create_empty_buffer'
      ),
    
    createBufferWithData: async (data: Uint8Array) =>
      wrapper.executeWithDefense(
        () => operationDispatcher.create_buffer_with_js_data(data),
        'create_buffer_with_js_data'
      ),
    
    copyBufferToJs: async (handle: WasmBufferHandle) =>
      wrapper.executeWithDefense(
        () => operationDispatcher.copy_buffer_to_js(handle),
        'copy_buffer_to_js'
      ),
    
    releaseBuffer: async (handle: WasmBufferHandle) =>
      wrapper.executeWithDefense(
        () => operationDispatcher.release_buffer(handle),
        'release_buffer',
        { fallbackEnabled: false } // Buffer release should not have fallbacks
      ),
    
    getMemoryStats: async () =>
      wrapper.executeWithDefense(
        () => operationDispatcher.get_memory_stats(),
        'get_memory_stats'
      ),
    
    intensiveCleanup: async () =>
      wrapper.executeWithDefense(
        () => operationDispatcher.intensive_cleanup(),
        'intensive_cleanup'
      )
  };
}