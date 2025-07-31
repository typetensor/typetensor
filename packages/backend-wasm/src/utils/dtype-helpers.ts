/**
 * Helper functions for working with DTypes in the WASM backend
 */

import type { AnyDType } from '@typetensor/core';
import type { RuntimeDType } from '@typetensor/core/dtype';

/**
 * Get the byte size from a dtype object
 * Handles both compile-time DType interface and runtime RuntimeDType instances
 */
export function getDTypeByteSize(dtype: AnyDType | RuntimeDType | any): number {
  // Handle RuntimeDType instances
  if (dtype && typeof dtype === 'object' && 'byteSize' in dtype) {
    return dtype.byteSize;
  }
  
  // Handle compile-time DType interface
  if (dtype && typeof dtype === 'object' && '__byteSize' in dtype) {
    return dtype.__byteSize;
  }
  
  // Fallback to name-based lookup
  const dtypeName = dtype.__dtype || dtype.__name || dtype.name;
  switch (dtypeName) {
    case 'bool':
    case 'int8':
    case 'uint8':
      return 1;
    case 'int16':
    case 'uint16':
      return 2;
    case 'int32':
    case 'uint32':
    case 'float32':
      return 4;
    case 'int64':
    case 'uint64':
    case 'float64':
      return 8;
    default:
      throw new Error(`Unknown dtype for byte size: ${dtypeName}`);
  }
}

/**
 * Get the dtype name from various dtype representations
 */
export function getDTypeName(dtype: AnyDType | RuntimeDType | any): string {
  // Handle RuntimeDType instances
  if (dtype && typeof dtype === 'object' && 'name' in dtype) {
    return dtype.name;
  }
  
  // Handle compile-time DType interface
  if (dtype && typeof dtype === 'object' && '__dtype' in dtype) {
    return dtype.__dtype;
  }
  
  // Handle alternative property name
  if (dtype && typeof dtype === 'object' && '__name' in dtype) {
    return dtype.__name;
  }
  
  throw new Error(`Cannot extract dtype name from: ${JSON.stringify(dtype)}`);
}