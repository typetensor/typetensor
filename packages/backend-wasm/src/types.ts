
import type { AnyDType } from '@typetensor/core';

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

export const REVERSE_DTYPE_MAPPING: Record<number, string> = Object.fromEntries(
  Object.entries(DTYPE_MAPPING).map(([k, v]) => [v, k])
);

export const OPERATION_MAPPING: Record<string, number> = {
  'create': 0,
  'neg': 1,
  'abs': 2,
  'sin': 3,
  'cos': 4,
  'exp': 5,
  'log': 6,
  'sqrt': 7,
  'square': 8,
  'add': 10,
  'sub': 11,
  'mul': 12,
  'div': 13,
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
  'matmul': 30,
  'softmax': 40,
  'log_softmax': 41,
  'sum': 50,
  'mean': 51,
  'max': 52,
  'min': 53,
  'prod': 54,
  'rearrange': 60,
  'reduce': 61,
} as const;

export function dtypeToWasm(dtype: AnyDType): number {
  const dtypeName = (dtype as any).__dtype || (dtype as any).__name || 'unknown';
  const wasmValue = DTYPE_MAPPING[dtypeName];
  if (wasmValue === undefined) {
    throw new Error(`Unsupported dtype: ${dtypeName}. Available: ${Object.keys(DTYPE_MAPPING).join(', ')}`);
  }
  return wasmValue;
}

export function operationToWasm(operation: string): number {
  const wasmValue = OPERATION_MAPPING[operation];
  if (wasmValue === undefined) {
    throw new Error(`Unsupported operation: ${operation}`);
  }
  return wasmValue;
}

export interface WASMModule {
  memory: WebAssembly.Memory;
  greet(): void;
  get_version(): string;
  WasmOperationDispatcher: any; // Defined in wasm-bindings.d.ts
  WasmMemoryManager: any; // Defined in wasm-bindings.d.ts
  WasmBufferHandle: any; // Defined in wasm-bindings.d.ts
  WasmTensorMeta: any; // Defined in wasm-bindings.d.ts
  has_simd_support(): boolean;
  has_shared_memory_support(): boolean;
  get_optimal_thread_count(): number;
}

export interface WASMLoadOptions {
  debug?: boolean;
  memoryConfig?: WASMMemoryConfig;
}

export interface WASMMemoryConfig {
  /** Maximum memory limit in bytes. Default: 512MB */
  maxMemory?: number;
  /** Compact when memory usage exceeds this threshold (0-1). Default: 0.8 */
  compactThreshold?: number;
  /** Enable automatic compaction. Default: true */
  autoCompact?: boolean;
}

export interface WASMCapabilities {
  simd: boolean;
  sharedMemory: boolean;
  optimalThreadCount: number;
  availableMemory: number;
  version: string;
}

export interface WASMMemoryStats {
  totalAllocated: number;
  activeBuffers: number;
  poolSummary?: string;
}