/**
 * Test generators for tensor property assertions
 * 
 * These generators test tensor metadata properties like shape, dtype, device,
 * strides, and other structural information.
 */

import type { Device } from '@typetensor/core';
import { tensor, zeros, float32, int32, bool, int64 } from '@typetensor/core';

/**
 * Generates tests for tensor property access and validation
 * 
 * @param device - Device instance to test against
 * @param testFramework - Test framework object with describe/it/expect functions
 */
export function generateTensorPropertyTests(
  device: Device,
  testFramework: {
    describe: (name: string, fn: () => void) => void;
    it: (name: string, fn: () => void | Promise<void>) => void;
    expect: (actual: unknown) => {
      toBe: (expected: unknown) => void;
      toEqual: (expected: unknown) => void;
      toBeGreaterThan: (expected: number) => void;
      toBeTruthy: () => void;
      toBeFalsy: () => void;
      toBeInstanceOf?: (constructor: any) => void;
    };
  }
) {
  const { describe, it, expect } = testFramework;

  describe(`Tensor Properties Tests (${device.type}:${device.id})`, () => {
    
    describe('shape properties', () => {
      it('should report correct shape for scalars', async () => {
        // PyTorch: scalar = torch.tensor(42.0)
        // scalar.shape = torch.Size([]), scalar.ndim = 0, scalar.numel() = 1
        const scalar = await tensor(42, { device, dtype: float32 });
        expect(scalar.shape).toEqual([]);
        expect(scalar.ndim).toBe(0);
        expect(scalar.size).toBe(1);
      });

      it('should report correct shape for vectors', async () => {
        // PyTorch: vector = torch.tensor([1, 2, 3, 4, 5])
        // vector.shape = torch.Size([5]), vector.ndim = 1, vector.numel() = 5
        const vector = await tensor([1, 2, 3, 4, 5] as const, { device, dtype: float32 });
        expect(vector.shape).toEqual([5]);
        expect(vector.ndim).toBe(1);
        expect(vector.size).toBe(5);
      });

      it('should report correct shape for matrices', async () => {
        // PyTorch: matrix = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        // matrix.shape = torch.Size([4, 3]), matrix.ndim = 2, matrix.numel() = 12
        const matrix = await tensor([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
          [10, 11, 12]
        ] as const, { device, dtype: float32 });
        
        expect(matrix.shape).toEqual([4, 3]);
        expect(matrix.ndim).toBe(2);
        expect(matrix.size).toBe(12);
      });

      it('should report correct shape for higher dimensional tensors', async () => {
        // PyTorch: tensor4d = torch.zeros(2, 3, 4, 5)
        // tensor4d.shape = torch.Size([2, 3, 4, 5]), tensor4d.ndim = 4, tensor4d.numel() = 120
        const tensor4d = await zeros([2, 3, 4, 5] as const, { device, dtype: float32 });
        expect(tensor4d.shape).toEqual([2, 3, 4, 5]);
        expect(tensor4d.ndim).toBe(4);
        expect(tensor4d.size).toBe(120);
      });
    });

    describe('dtype properties', () => {
      it('should preserve dtype for different numeric types', async () => {
        // PyTorch: torch.tensor(3.14)
        // Output: tensor(3.1400), dtype: torch.float32
        const float32Tensor = await tensor(3.14, { device, dtype: float32 });
        expect(float32Tensor.dtype).toBe(float32);
        expect(float32Tensor.dtype.__dtype).toBe('float32');

        // PyTorch: torch.tensor(42, dtype=torch.int32)
        // Output: tensor(42, dtype=torch.int32)
        const int32Tensor = await tensor(42, { device, dtype: int32 });
        expect(int32Tensor.dtype).toBe(int32);
        expect(int32Tensor.dtype.__dtype).toBe('int32');

        // PyTorch: torch.tensor(123, dtype=torch.int64)
        // Output: tensor(123), dtype: torch.int64
        const int64Tensor = await tensor(123n, { device, dtype: int64 });
        expect(int64Tensor.dtype).toBe(int64);
        expect(int64Tensor.dtype.__dtype).toBe('int64');
      });

      it('should preserve dtype for boolean tensors', async () => {
        // PyTorch: torch.tensor([True, False, True])
        // Output: tensor([ True, False,  True]), dtype: torch.bool
        const boolTensor = await tensor([true, false, true] as const, { device, dtype: bool });
        expect(boolTensor.dtype).toBe(bool);
        expect(boolTensor.dtype.__dtype).toBe('bool');
      });
    });

    describe('device properties', () => {
      it('should reference the correct device', async () => {
        const tensor1 = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        expect(tensor1.device).toBe(device);
        expect(tensor1.device.id).toBe(device.id);
        expect(tensor1.device.type).toBe(device.type);
      });
    });

    describe('storage properties', () => {
      it('should compute correct strides for row-major layout', async () => {
        // PyTorch: matrix = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        // matrix.stride() = (3, 1)  # row-major/C-order
        // matrix.is_contiguous() = True
        const matrix = await tensor([
          [1, 2, 3],
          [4, 5, 6]
        ] as const, { device, dtype: float32 });
        
        expect(Array.isArray(matrix.strides)).toBe(true);
        expect(matrix.strides.length).toBe(2);
        expect(matrix.strides[0]).toBe(3); // stride for rows
        expect(matrix.strides[1]).toBe(1); // stride for columns
      });

      it('should compute correct strides for 3D tensors', async () => {
        // PyTorch: tensor3d = torch.zeros(2, 3, 4)
        // tensor3d.stride() = (12, 4, 1)
        // Shape [2, 3, 4] -> strides [3*4, 4, 1] = [12, 4, 1]
        const tensor3d = await zeros([2, 3, 4] as const, { device, dtype: float32 });
        
        expect(Array.isArray(tensor3d.strides)).toBe(true);
        expect(tensor3d.strides.length).toBe(3);
        expect(tensor3d.strides[0]).toBe(12); // 3*4 = 12
        expect(tensor3d.strides[1]).toBe(4);  // 4
        expect(tensor3d.strides[2]).toBe(1);  // 1
      });

      it('should have zero offset for new tensors', async () => {
        // PyTorch: tensor1 = torch.tensor([1, 2, 3])
        // tensor1.storage_offset() = 0  # New tensors start at offset 0
        const tensor1 = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        
        // New tensors should start at offset 0
        expect(typeof tensor1.storage.__offset).toBe('number');
        expect(tensor1.storage.__offset).toBe(0);
      });
    });

    describe('memory properties', () => {
      it('should create tensors with correct structure for different dtypes', async () => {
        // float32 matrix
        const float32Matrix = await zeros([10, 10] as const, { device, dtype: float32 });
        expect(float32Matrix.size).toBe(100);
        expect(float32Matrix.dtype).toBe(float32);
        
        // int32 vector
        const int32Vector = await zeros([50] as const, { device, dtype: int32 });
        expect(int32Vector.size).toBe(50);
        expect(int32Vector.dtype).toBe(int32);
        
        // bool vector
        const boolVector = await zeros([8] as const, { device, dtype: bool });
        expect(boolVector.size).toBe(8);
        expect(boolVector.dtype).toBe(bool);
      });

      it('should maintain device association', async () => {
        const tensor1 = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        
        // Tensor should be associated with the correct device
        expect(tensor1.device).toBe(device);
        expect(tensor1.device.id).toBe(device.id);
        expect(tensor1.device.type).toBe(device.type);
        
        // Tensor should have correct structural properties
        expect(tensor1.size).toBe(3);
        expect(tensor1.ndim).toBe(1);
        expect(tensor1.shape).toEqual([3]);
      });
    });

    describe('equality and comparison', () => {
      it('should create independent tensor instances', async () => {
        const tensor1 = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        const tensor2 = await tensor([1, 2, 3] as const, { device, dtype: float32 });
        
        // Different tensor instances should not be the same object
        expect(tensor1 === tensor2).toBeFalsy();
        
        // But should have equivalent properties
        expect(tensor1.shape).toEqual(tensor2.shape);
        expect(tensor1.ndim).toBe(tensor2.ndim);
        expect(tensor1.size).toBe(tensor2.size);
        expect(tensor1.dtype).toBe(tensor2.dtype);
        expect(tensor1.device).toBe(tensor2.device);
        expect(tensor1.strides).toEqual(tensor2.strides);
        
        // And should have the same data when extracted
        const data1 = await tensor1.toArray();
        const data2 = await tensor2.toArray();
        expect(data1).toEqual(data2);
      });

      it('should provide consistent property access', async () => {
        const tensor1 = await tensor([[1, 2], [3, 4]] as const, { device, dtype: float32 });
        
        // Multiple accesses should return identical values (stable references)
        const shape1 = tensor1.shape;
        const shape2 = tensor1.shape;
        expect(shape1).toEqual(shape2);
        
        const strides1 = tensor1.strides;
        const strides2 = tensor1.strides;
        expect(strides1).toEqual(strides2);
        
        // Properties should be immutable references
        expect(tensor1.ndim).toBe(2);
        expect(tensor1.size).toBe(4);
        expect(tensor1.dtype).toBe(float32);
        expect(tensor1.device).toBe(device);
      });
    });

    describe('edge cases', () => {
      it('should handle empty tensors', async () => {
        // PyTorch: empty1d = torch.tensor([])
        // empty1d.shape = torch.Size([0]), empty1d.numel() = 0
        const empty1d = await tensor([] as const, { device, dtype: float32 });
        expect(empty1d.shape).toEqual([0]);
        expect(empty1d.ndim).toBe(1);
        expect(empty1d.size).toBe(0);

        // PyTorch: empty2d = torch.zeros(0, 5)
        // empty2d.shape = torch.Size([0, 5]), empty2d.numel() = 0
        const empty2d = await zeros([0, 5] as const, { device, dtype: float32 });
        expect(empty2d.shape).toEqual([0, 5]);
        expect(empty2d.ndim).toBe(2);
        expect(empty2d.size).toBe(0);
      });

      it('should handle large tensors', async () => {
        // PyTorch: large = torch.zeros(100, 100)
        // large.shape = torch.Size([100, 100]), large.numel() = 10000
        const large = await zeros([100, 100] as const, { device, dtype: float32 });
        expect(large.shape).toEqual([100, 100]);
        expect(large.ndim).toBe(2);
        expect(large.size).toBe(10000);
        expect(large.dtype).toBe(float32);
        expect(large.device).toBe(device);
      });
    });
  });
}