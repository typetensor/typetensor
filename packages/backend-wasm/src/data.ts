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
 *
 * Supports view metadata for zero-copy operations like reshape and transpose.
 */
export class WASMTensorData implements DeviceData {
  readonly id: string;
  readonly device: Device;
  readonly viewMetadata?: {
    shape: readonly number[];
    strides: readonly number[];
    offset: number;
    dtype: { __byteSize: number };
  };

  constructor(
    device: Device,
    readonly wasmTensor: WasmTensor,
    viewMetadata?: {
      shape: readonly number[];
      strides: readonly number[];
      offset: number;
      dtype: { __byteSize: number };
    },
  ) {
    this.device = device;
    // Use tensor metadata for unique ID
    this.id = `wasm-tensor-${wasmTensor.meta.dtype}-${wasmTensor.meta.size}`;
    this.viewMetadata = viewMetadata;
  }

  get byteLength(): number {
    return this.wasmTensor.byte_size;
  }

  clone(): WASMTensorData {
    // Arena handles cloning safety - just return same reference
    // Rust's arena system ensures memory safety through its ownership model
    return new WASMTensorData(this.device, this.wasmTensor, this.viewMetadata);
  }
}

/**
 * Factory function to create WASMTensorData instances
 */
export function createWASMTensorData(
  device: Device,
  wasmTensor: WasmTensor,
  viewMetadata?: {
    shape: readonly number[];
    strides: readonly number[];
    offset: number;
    dtype: { __byteSize: number };
  },
): WASMTensorData {
  return new WASMTensorData(device, wasmTensor, viewMetadata);
}
