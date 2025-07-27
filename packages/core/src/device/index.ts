/**
 * Device module exports
 *
 * @module device
 *
 * This module provides the minimal device abstraction for tensor backends.
 * The interfaces are intentionally minimal to allow maximum flexibility
 * for backend implementations while maintaining type safety.
 */

export type { Device, DeviceData } from './types';
