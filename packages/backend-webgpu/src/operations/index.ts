/**
 * Operation dispatcher for WebGPU device
 */

import type { Device, DeviceData, AnyStorageTransformation } from '@typetensor/core';
import { assertExhaustiveSwitch } from '@typetensor/core';
import { executeUnaryOp } from './unary';
import { executeBinaryOp } from './binary';
import { executeViewOp, executeSliceOp, executeExpandOp, executeTileOp } from './view';
import { executeMatmulOp } from './matmul';
import { executeSoftmaxOp, executeLogSoftmaxOp } from './softmax';
import {
  executeSumOp,
  executeMeanOp,
  executeMaxOp,
  executeMinOp,
  executeProdOp,
} from './reduction';
import { executeRearrangeOp, executeReduceOp } from './einops';
import { WebGPUDevice } from '../device';

/**
 * Execute a tensor operation on the WebGPU device
 */
export async function executeOperation(
  device: Device,
  op: AnyStorageTransformation,
  inputs: DeviceData[],
  output?: DeviceData,
): Promise<DeviceData> {
  const webgpuDevice = device as WebGPUDevice;

  switch (op.__op) {
    // Creation operation
    case 'create': {
      if (output) {
        return output;
      }
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
      return executeUnaryOp(webgpuDevice, op, input, output);
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
      return executeBinaryOp(webgpuDevice, op, inputA, inputB, output);
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
      return executeViewOp(webgpuDevice, op, input);
    }

    // Slice operation
    case 'slice': {
      if (inputs.length !== 1) {
        throw new Error(`Slice operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeSliceOp(
        webgpuDevice,
        op as AnyStorageTransformation & { __op: 'slice' },
        input,
        output,
      );
    }

    // Transpose and permute operations
    case 'transpose':
    case 'permute':
    case 'squeeze':
    case 'unsqueeze': {
      if (inputs.length !== 1) {
        throw new Error(`${op.__op} operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      // These are view operations - return the same data
      return input;
    }

    // Expand operation
    case 'expand': {
      if (inputs.length !== 1) {
        throw new Error(`Expand operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeExpandOp(
        webgpuDevice,
        op as AnyStorageTransformation & { __op: 'expand' },
        input,
        output,
      );
    }

    // Tile operation
    case 'tile': {
      if (inputs.length !== 1) {
        throw new Error(`Tile operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeTileOp(
        webgpuDevice,
        op as AnyStorageTransformation & { __op: 'tile' },
        input,
        output,
      );
    }

    // Matrix multiplication
    case 'matmul': {
      if (inputs.length !== 2) {
        throw new Error(`Matrix multiplication requires exactly 2 inputs, got ${inputs.length}`);
      }
      const inputA = inputs[0];
      const inputB = inputs[1];
      if (!inputA || !inputB) {
        throw new Error('One or more inputs are undefined');
      }
      return executeMatmulOp(webgpuDevice, op, inputA, inputB, output);
    }

    // Softmax operations
    case 'softmax': {
      if (inputs.length !== 1) {
        throw new Error(`Softmax operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeSoftmaxOp(
        webgpuDevice,
        op as AnyStorageTransformation & { __softmaxAxis: number },
        input,
        output,
      );
    }

    case 'log_softmax': {
      if (inputs.length !== 1) {
        throw new Error(`Log-softmax operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeLogSoftmaxOp(
        webgpuDevice,
        op as AnyStorageTransformation & { __logSoftmaxAxis: number },
        input,
        output,
      );
    }

    // Reduction operations
    case 'sum': {
      if (inputs.length !== 1) {
        throw new Error(`Sum operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeSumOp(
        webgpuDevice,
        op as AnyStorageTransformation & {
          __sumAxes: readonly number[] | undefined;
          __keepDims: boolean;
        },
        input,
        output,
      );
    }

    case 'mean': {
      if (inputs.length !== 1) {
        throw new Error(`Mean operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeMeanOp(
        webgpuDevice,
        op as AnyStorageTransformation & {
          __meanAxes: readonly number[] | undefined;
          __keepDims: boolean;
        },
        input,
        output,
      );
    }

    case 'max': {
      if (inputs.length !== 1) {
        throw new Error(`Max operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeMaxOp(
        webgpuDevice,
        op as AnyStorageTransformation & {
          __maxAxes: readonly number[] | undefined;
          __keepDims: boolean;
        },
        input,
        output,
      );
    }

    case 'min': {
      if (inputs.length !== 1) {
        throw new Error(`Min operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeMinOp(
        webgpuDevice,
        op as AnyStorageTransformation & {
          __minAxes: readonly number[] | undefined;
          __keepDims: boolean;
        },
        input,
        output,
      );
    }

    case 'prod': {
      if (inputs.length !== 1) {
        throw new Error(`Product operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeProdOp(
        webgpuDevice,
        op as AnyStorageTransformation & {
          __prodAxes: readonly number[] | undefined;
          __keepDims: boolean;
        },
        input,
        output,
      );
    }

    // Einops operations
    case 'rearrange': {
      if (inputs.length !== 1) {
        throw new Error(`Rearrange operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeRearrangeOp(
        webgpuDevice,
        op as AnyStorageTransformation & { __op: 'rearrange' },
        input,
        output,
      );
    }

    case 'reduce': {
      if (inputs.length !== 1) {
        throw new Error(`Reduce operation requires exactly 1 input, got ${inputs.length}`);
      }
      const input = inputs[0];
      if (!input) {
        throw new Error('Input is undefined');
      }
      return executeReduceOp(
        webgpuDevice,
        op as AnyStorageTransformation & { __op: 'reduce' },
        input,
        output,
      );
    }

    default:
      // This will cause a TypeScript compile error if any operation case is missing
      return assertExhaustiveSwitch(op.__op as never);
  }
}