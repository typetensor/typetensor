/**
 * Minimal WASM Backend exports
 *
 * This file exports only the essential components with zero abstraction layers.
 */

export { WASMDevice } from './device';
export { WASMTensorData, createWASMTensorData } from './data';
export type {
  WasmExecutor,
  WasmTensor,
  WasmTensorMeta,
  WasmMemoryStats,
  WasmDType,
  WasmOperation,
  WASMCapabilities,
  WASMMemoryStats,
  WASMLoadOptions,
  OperationName,
  DTypeName,
} from './types';
export { OPS, DTYPES } from './types';

// Device factory functions
import { WASMDevice } from './device';

let _wasmDevice: WASMDevice | null = null;

/**
 * Get shared WASM device instance
 */
export async function getWASMDevice(): Promise<WASMDevice> {
  if (!_wasmDevice) {
    _wasmDevice = await WASMDevice.create();
  }
  return _wasmDevice;
}

/**
 * Create new WASM device instance
 */
export async function createWASMDevice(): Promise<WASMDevice> {
  return await WASMDevice.create();
}

/**
 * Backward compatibility alias
 */
export class WASMBackend {
  static async create(): Promise<WASMDevice> {
    return await getWASMDevice();
  }
}

/**
 * Test utilities - for test compatibility
 */
export function resetWASMForTests(): void {
  // Arena automatically resets between test runs - no manual cleanup needed
  _wasmDevice = null;
}
