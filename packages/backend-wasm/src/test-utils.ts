/**
 * Test utilities for ensuring test independence and WASM module isolation
 */

import { resetWASMModule } from './loader';

/**
 * Resets the WASM module to ensure test isolation.
 * Call this in afterAll() hooks in each test file.
 */
export function resetWASMForTests(): void {
  try {
    // Reset the cached WASM module so next test file gets a fresh instance
    resetWASMModule();
  } catch (error) {
    // Don't fail tests if reset fails - just log it
    console.warn('Failed to reset WASM module between tests:', error);
  }
}

/**
 * Setup function to call in beforeAll() hooks
 */
export function setupWASMTests(): void {
  // Ensure we start with a clean slate
  resetWASMForTests();
}