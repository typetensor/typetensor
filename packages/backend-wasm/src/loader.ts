import type { WASMModule, WASMLoadOptions } from './types';

import init, * as wasmBindings from '../pkg/typetensor_wasm.js';
let wasmModule: WASMModule | null = null;
let wasmPromise: Promise<WASMModule> | null = null;

export async function loadWASMModule(options: WASMLoadOptions = {}): Promise<WASMModule> {
  if (wasmPromise) {
    return wasmPromise;
  }
  
  if (wasmModule) {
    return wasmModule;
  }

  
  wasmPromise = loadWASMModuleInternal(options);
  
  try {
    wasmModule = await wasmPromise;
    return wasmModule;
  } catch (error) {
    wasmPromise = null;
    throw error;
  }
}

async function loadWASMModuleInternal(_options: WASMLoadOptions): Promise<WASMModule> {


  try {
    const wasmInitResult = await init();
    const module: WASMModule = {
      memory: wasmInitResult?.memory || new WebAssembly.Memory({ initial: 256 }),
      greet: wasmBindings.greet,
      get_version: wasmBindings.get_version,
      WasmOperationDispatcher: wasmBindings.WasmOperationDispatcher,
      WasmMemoryManager: wasmBindings.WasmMemoryManager,
      WasmBufferHandle: wasmBindings.WasmBufferHandle,
      WasmTensorMeta: wasmBindings.WasmTensorMeta,
      has_simd_support: wasmBindings.has_simd_support,
      has_shared_memory_support: wasmBindings.has_shared_memory_support,
      get_optimal_thread_count: wasmBindings.get_optimal_thread_count,
    };

    return module;
  } catch (error) {
    throw new Error(`Failed to initialize WASM module: ${error}`);
  }
}

export function getLoadedWASMModule(): WASMModule {
  if (!wasmModule) {
    throw new Error('WASM module not loaded. Call loadWASMModule() first.');
  }
  return wasmModule;
}

export function isWASMModuleLoaded(): boolean {
  return wasmModule !== null;
}

export function resetWASMModule(): void {
  wasmModule = null;
  wasmPromise = null;
}