// Type definitions for WASM bindings
// These provide strong typing for the WASM module interface

export interface WasmBufferHandle {
  id(): number;
  size(): number;
  ptr(): number;
  is_initialized(): boolean;
  clone_handle(): WasmBufferHandle;
}

export interface WasmTensorMeta {
  constructor(
    dtype: number,
    shape: number[],
    strides: number[],
    size: number,
    offset: number
  ): WasmTensorMeta;
  dtype(): number;
  shape(): number[];
  strides(): number[];
  size(): number;
  offset(): number;
  byte_size(): number;
}

export interface WasmMemoryStats {
  total_allocated_bytes: number;
  active_buffers: number;
  get_pool_summary(): string;
}

export interface WasmOperationDispatcher {
  create_buffer_with_js_data(data: Uint8Array): WasmBufferHandle;
  create_empty_buffer(size: number): WasmBufferHandle;
  release_buffer(handle: WasmBufferHandle): boolean;
  copy_buffer_to_js(handle: WasmBufferHandle): Uint8Array;
  get_buffer_view_info(handle: WasmBufferHandle): Uint32Array;
  get_memory_stats(): WasmMemoryStats;
  compact_pools(): void;
  intensive_cleanup(): void;
  clone_buffer_handle(handle: WasmBufferHandle): WasmBufferHandle;
  
  execute_unary(
    op: number,
    input: WasmBufferHandle,
    inputMeta: WasmTensorMeta,
    outputMeta: WasmTensorMeta,
    output: WasmBufferHandle
  ): WasmBufferHandle;
  
  execute_binary(
    op: number,
    inputA: WasmBufferHandle,
    inputB: WasmBufferHandle,
    inputMetaA: WasmTensorMeta,
    inputMetaB: WasmTensorMeta,
    outputMeta: WasmTensorMeta,
    output: WasmBufferHandle
  ): WasmBufferHandle;
  
  execute_reduction(
    op: number,
    input: WasmBufferHandle,
    inputMeta: WasmTensorMeta,
    outputMeta: WasmTensorMeta,
    axes: number[] | null,
    keepDims: boolean,
    output: WasmBufferHandle
  ): WasmBufferHandle;
  
  execute_softmax(
    op: number,
    input: WasmBufferHandle,
    inputMeta: WasmTensorMeta,
    outputMeta: WasmTensorMeta,
    axis: number,
    output: WasmBufferHandle
  ): WasmBufferHandle;
}

export interface WasmMemoryManager {
  constructor(): WasmMemoryManager;
  get_memory_stats(): WasmMemoryStats;
}

export interface WasmModule {
  memory: WebAssembly.Memory;
  greet(): void;
  get_version(): string;
  WasmOperationDispatcher: new () => WasmOperationDispatcher;
  WasmMemoryManager: new () => WasmMemoryManager;
  WasmBufferHandle: WasmBufferHandle;
  WasmTensorMeta: new (
    dtype: number,
    shape: number[],
    strides: number[],
    size: number,
    offset: number
  ) => WasmTensorMeta;
  has_simd_support(): boolean;
  has_shared_memory_support(): boolean;
  get_optimal_thread_count(): number;
}