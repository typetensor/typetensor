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

// Operation mapping - ONLY operations actually implemented in Rust backend
export const OPS = {
  // Unary operations (implemented in unary.rs)
  neg: 1, abs: 2, sin: 3, cos: 4, exp: 5, log: 6, sqrt: 7, square: 8,
  
  // Binary operations (implemented in binary.rs)
  add: 10, sub: 11, mul: 12, div: 13,
  
  // Matrix operations (implemented in matmul.rs)
  matmul: 30,
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
  memoryConfig?: {
    maxMemory?: number;
    compactThreshold?: number;
    autoCompact?: boolean;
  };
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