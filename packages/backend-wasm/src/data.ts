/**
 * Minimal DeviceData wrapper for WasmTensor
 * 
 * This provides the simplest possible DeviceData implementation 
 * that wraps WasmTensor with no additional abstractions.
 */

import type { DeviceData, Device } from '@typetensor/core';
import type { WasmTensor } from './types';

/**
 * Minimal wrapper around WasmTensor that implements DeviceData interface
 * 
 * Arena-based memory management in Rust handles all cleanup automatically,
 * so this wrapper is extremely simple with no manual memory management.
 */
export class WASMTensorData implements DeviceData {
  readonly id: string;
  readonly device: Device;
  
  constructor(
    device: Device,
    readonly wasmTensor: WasmTensor
  ) {
    this.device = device;
    // Use tensor metadata for unique ID
    this.id = `wasm-tensor-${wasmTensor.meta.dtype}-${wasmTensor.meta.size}`;
  }

  get byteLength(): number {
    return this.wasmTensor.byte_size;
  }

  clone(): WASMTensorData {
    // Arena handles cloning safety - just return same reference
    // Rust's arena system ensures memory safety through its ownership model
    return new WASMTensorData(this.device, this.wasmTensor);
  }
}

/**
 * Factory function to create WASMTensorData instances
 */
export function createWASMTensorData(
  device: Device,
  wasmTensor: WasmTensor
): WASMTensorData {
  return new WASMTensorData(device, wasmTensor);
}