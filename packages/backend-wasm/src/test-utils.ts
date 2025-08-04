/**
 * Test utilities for arena-based WASM backend
 *
 * With arena memory management, test isolation is automatic
 * and no manual cleanup is required.
 */

/**
 * Setup function to call in beforeAll() hooks
 * No-op since arena handles all memory management
 */
export function setupWASMTests(): void {
  // Arena-based memory management provides automatic test isolation
}

/**
 * Cleanup function to call in afterAll() hooks
 * No-op since arena handles all memory management
 */
export function resetWASMForTests(): void {
  // Arena automatically provides clean slate for each test
}
