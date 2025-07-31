/**
 * Test utilities for ensuring test independence and WASM module isolation
 */

import { resetWASMModule } from './loader';
import { WASMErrorHandler, WASMCleanupError } from './errors';

/**
 * Resets the WASM module to ensure test isolation.
 * Call this in afterAll() hooks in each test file.
 */
export function resetWASMForTests(): void {
  try {
    // Reset the cached WASM module so next test file gets a fresh instance
    resetWASMModule();
  } catch (error) {
    // Don't fail tests if reset fails - handle as cleanup error
    class WASMTestCleanupError extends WASMCleanupError {
      readonly code = 'TEST_CLEANUP_FAILED';
      
      constructor(reason: string, context?: Record<string, unknown>) {
        super(
          `Test cleanup failed: ${reason}. This may affect test isolation but should not fail the test.`,
          context
        );
      }
    }
    
    const cleanupError = new WASMTestCleanupError(
      error instanceof Error ? error.message : String(error),
      { operation: 'resetWASMModule', testContext: true }
    );
    WASMErrorHandler.handle(cleanupError);
  }
}

/**
 * Setup function to call in beforeAll() hooks
 */
export function setupWASMTests(): void {
  // Ensure we start with a clean slate
  resetWASMForTests();
}