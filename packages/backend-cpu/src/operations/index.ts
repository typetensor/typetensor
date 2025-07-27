/**
 * Operation dispatcher for CPU device
 *
 * Routes tensor operations to their specific implementations.
 */

import type { Device, DeviceData, AnyStorageTransformation } from '@typetensor/core';
import { executeUnaryOp } from './unary';
import { executeBinaryOp } from './binary';
import { executeViewOp } from './view';

/**
 * Execute a tensor operation on the CPU device
 *
 * @param device - The CPU device instance
 * @param op - Operation descriptor from the storage layer
 * @param inputs - Input tensor data
 * @param output - Optional pre-allocated output buffer
 * @returns Result tensor data
 */
export async function executeOperation(
  device: Device,
  op: AnyStorageTransformation,
  inputs: DeviceData[],
  output?: DeviceData,
): Promise<DeviceData> {
  switch (op.__op) {
    // Creation operation - just return the output or create new data
    case 'create': {
      if (output) {
        return output;
      }
      // Calculate size from output metadata
      const size = op.__output.__size * op.__output.__dtype.__byteSize;
      return device.createData(size);
    }

    // Unary operations
    case 'neg':
    case 'abs':
    case 'sin':
    case 'cos':
    case 'exp':
    case 'log':
    case 'sqrt':
    case 'square': {
      if (inputs.length !== 1) {
        throw new Error(
          `Unary operation ${op.__op} requires exactly 1 input, got ${inputs.length}`,
        );
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeUnaryOp(device, op, input, output);
    }

    // Binary operations
    case 'add':
    case 'sub':
    case 'mul':
    case 'div': {
      if (inputs.length !== 2) {
        throw new Error(
          `Binary operation ${op.__op} requires exactly 2 inputs, got ${inputs.length}`,
        );
      }
      const inputA = inputs[0];
      const inputB = inputs[1];
      if (!inputA || !inputB) {
        throw new Error('One or more inputs are undefined');
      }
      return executeBinaryOp(device, op, inputA, inputB, output);
    }

    // View operations - these don't create new data, just return the input
    case 'reshape':
    case 'flatten':
    case 'view': {
      if (inputs.length !== 1) {
        throw new Error(`View operation ${op.__op} requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeViewOp(device, op, input);
    }

    default:
      throw new Error(`Unsupported operation: ${op.__op}`);
  }
}
