/**
 * TypeScript type definitions for the WASM backend
 * 
 * Provides type mappings between TypeTensor core types and WASM backend types.
 */

import type { AnyDType } from '@typetensor/core';

/**
 * Mapping between TypeTensor dtypes and WASM dtype enum values
 */
export const DTYPE_MAPPING: Record<string, number> = {
  'int8': 0,
  'uint8': 1,
  'int16': 2,
  'uint16': 3,
  'int32': 4,
  'uint32': 5,
  'float32': 6,
  'float64': 7,
  'bigint64': 8,
  'biguint64': 9,
} as const;

/**
 * Reverse mapping from WASM dtype enum to TypeTensor dtype names
 */
export const REVERSE_DTYPE_MAPPING: Record<number, string> = Object.fromEntries(
  Object.entries(DTYPE_MAPPING).map(([k, v]) => [v, k])
);

/**
 * Operation type mapping between TypeTensor and WASM backend
 */
export const OPERATION_MAPPING: Record<string, number> = {
  // Creation
  'create': 0,
  
  // Unary operations
  'neg': 1,
  'abs': 2,
  'sin': 3,
  'cos': 4,
  'exp': 5,
  'log': 6,
  'sqrt': 7,
  'square': 8,
  
  // Binary operations
  'add': 10,
  'sub': 11,
  'mul': 12,
  'div': 13,
  
  // View operations
  'reshape': 20,
  'view': 21,
  'slice': 22,
  'flatten': 23,
  'permute': 24,
  'transpose': 25,
  'squeeze': 26,
  'unsqueeze': 27,
  'expand': 28,
  'tile': 29,
  
  // Matrix operations
  'matmul': 30,
  
  // Activation functions
  'softmax': 40,
  'log_softmax': 41,
  
  // Reduction operations
  'sum': 50,
  'mean': 51,
  'max': 52,
  'min': 53,
  'prod': 54,
  
  // Einops operations
  'rearrange': 60,
  'reduce': 61,
} as const;

/**
 * Convert TypeTensor dtype to WASM dtype enum value
 */
export function dtypeToWasm(dtype: AnyDType): number {
  // Use __dtype property
  const dtypeName = (dtype as any).__dtype || (dtype as any).__name || 'unknown';
  const wasmValue = DTYPE_MAPPING[dtypeName];
  if (wasmValue === undefined) {
    throw new Error(`Unsupported dtype: ${dtypeName}. Available: ${Object.keys(DTYPE_MAPPING).join(', ')}`);
  }
  return wasmValue;
}

/**
 * Convert operation name to WASM operation enum value
 */
export function operationToWasm(operation: string): number {
  const wasmValue = OPERATION_MAPPING[operation];
  if (wasmValue === undefined) {
    throw new Error(`Unsupported operation: ${operation}`);
  }
  return wasmValue;
}

/**
 * WASM module interface definition
 * This will be augmented with the actual WASM bindings when the module is loaded
 */
export interface WASMModule {
  // Memory management
  memory: WebAssembly.Memory;
  
  // Core functions (will be bound from Rust)
  greet(): void;
  get_version(): string;
  
  // Operation dispatcher
  WasmOperationDispatcher: any;
  WasmMemoryManager: any;
  WasmBufferHandle: any;
  WasmTensorMeta: any;
  
  // Utility functions
  has_simd_support(): boolean;
  has_shared_memory_support(): boolean;
  get_optimal_thread_count(): number;
}

/**
 * WASM loading options
 */
export interface WASMLoadOptions {
  /** Enable debug logging */
  debug?: boolean;
}

/**
 * WASM backend capabilities
 */
export interface WASMCapabilities {
  /** SIMD instructions available */
  simd: boolean;
  
  /** SharedArrayBuffer available for threading */
  sharedMemory: boolean;
  
  /** Optimal number of threads */
  optimalThreadCount: number;
  
  /** Available memory in bytes */
  availableMemory: number;
  
  /** WASM module version */
  version: string;
}

/**
 * Memory statistics from WASM backend
 */
export interface WASMMemoryStats {
  /** Total allocated memory in bytes */
  totalAllocated: number;
  
  /** Number of active buffers currently in use */
  activeBuffers: number;
  
  /** Pool details summary as a string */
  poolSummary?: string;
}