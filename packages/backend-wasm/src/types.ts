/**
 * Minimal type definitions for WASM backend - Direct interface to Rust
 *
 * This file provides the simplest possible interface to the WasmExecutor
 * with zero abstraction layers.
 */

// Re-export the exact WASM types from the generated bindings
export type {
  WasmExecutor,
  WasmTensor,
  WasmTensorMeta,
  WasmMemoryStats,
  WasmDType,
  WasmOperation,
  PatternCacheStats,
} from '../wasm/pkg/typetensor_wasm';

// Operation mapping - Exact match to WasmOperation enum in Rust
export const OPS = {
  // Creation
  create: 0,

  // Unary operations (implemented in unary.rs)
  neg: 1,
  abs: 2,
  sin: 3,
  cos: 4,
  exp: 5,
  log: 6,
  sqrt: 7,
  square: 8,

  // Binary operations (implemented in binary.rs)
  add: 10,
  sub: 11,
  mul: 12,
  div: 13,

  // View operations (implemented in view.rs)
  reshape: 20,
  view: 21,
  slice: 22,
  flatten: 23,
  permute: 24,
  transpose: 25,
  squeeze: 26,
  unsqueeze: 27,
  expand: 28,
  tile: 29,

  // Matrix operations (implemented in matmul.rs)
  matmul: 30,

  // Softmax operations (implemented in softmax.rs)
  softmax: 40,
  log_softmax: 41,

  // Reduction operations (implemented in reduction.rs)
  sum: 50,
  mean: 51,
  max: 52,
  min: 53,
  prod: 54,

  // Einops operations
  rearrange: 60,
  reduce: 61,
} as const;

// Data type mapping - matches WasmDType enum exactly
export const DTYPES = {
  bool: 0,
  int8: 1,
  uint8: 2,
  int16: 3,
  uint16: 4,
  int32: 5,
  uint32: 6,
  float32: 7,
  float64: 8,
  bigint64: 9,
  biguint64: 10,
  // Aliases for common dtype names
  int64: 9, // alias for bigint64
  uint64: 10, // alias for biguint64
} as const;

// Type helpers for operation and dtype mapping
export type OperationName = keyof typeof OPS;
export type DTypeName = keyof typeof DTYPES;

// WASM module interface for loading
export interface WASMModule {
  memory: WebAssembly.Memory;
  WasmExecutor: typeof import('../wasm/pkg/typetensor_wasm').WasmExecutor;
}

// Loading options (minimal)
export interface WASMLoadOptions {
  debug?: boolean;
}

// Device capabilities
export interface WASMCapabilities {
  simd: boolean;
  sharedMemory: boolean;
  optimalThreadCount: number;
  availableMemory: number;
  version: string;
}

// Simplified memory stats for backward compatibility
export interface WASMMemoryStats {
  totalAllocated: number;
  activeBuffers: number;
  poolSummary: string;
}
