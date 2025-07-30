/**
 * WebAssembly backend for TypeTensor
 *
 * High-performance tensor operations compiled to WebAssembly with SIMD optimizations.
 * This backend provides near-native performance for tensor computations in browsers and Node.js.
 */

export { WASMDevice } from './device';
export { WASMDeviceData } from './data';
export { loadWASMModule } from './loader';
export * from './types';

// Export singleton instance for convenience
import { WASMDevice } from './device';

// Create and export a singleton WASM device instance
let _wasmDevice: WASMDevice | null = null;

/**
 * Get or create the singleton WASM device instance
 * 
 * @returns Promise resolving to the WASM device
 */
export async function getWASMDevice(): Promise<WASMDevice> {
  if (!_wasmDevice) {
    _wasmDevice = await WASMDevice.create();
  }
  return _wasmDevice;
}

/**
 * Create a new WASM device instance
 * 
 * @returns Promise resolving to a new WASM device
 */
export async function createWASMDevice(): Promise<WASMDevice> {
  return await WASMDevice.create();
}

// For backward compatibility
export class WASMBackend {
  static async create(): Promise<WASMDevice> {
    return await getWASMDevice();
  }
}
