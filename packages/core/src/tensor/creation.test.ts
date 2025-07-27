/**
 * Runtime tests for tensor creation functions
 *
 * These tests focus on core functionality that doesn't require backend operations:
 * - Shape inference
 * - Property validation
 * - Error handling
 * - Type conversions
 */

import { describe, it, expect } from 'bun:test';
import { tensor, zeros, ones, eye } from './creation';
import { float32, int32, int8, uint8, int64, bool } from '../dtype/constants';
import type { Int32, Int8, Uint8, Int64, Bool } from '../dtype/types';
import { Tensor } from './tensor';
import type { CreateOp, TensorStorage } from '../storage/layout';
import type { ReshapeOp } from '../storage/view';
import { expectTypeOf } from 'expect-type';
import type { Device } from '../device';

const mockDevice = {
  id: 'mock',
  type: 'mock',
  execute: async () => {},
  createData: () => ({
    device: mockDevice,
    byteLength: 0,
  }),
  disposeData: () => {},
  readData: async () => new ArrayBuffer(0),
  writeData: async () => {},
  __sliceIndices: [],
  __strides: [],
  __shape: [],
  __dtype: float32,
  __layout: {
    c_contiguous: true,
    f_contiguous: false,
  },
} as unknown as Device;

describe('tensor() shape inference', () => {
  it('should infer scalar shape', async () => {
    const scalar = await tensor(42 as const, { device: mockDevice, dtype: float32 });
    expectTypeOf(scalar).toEqualTypeOf<
      Tensor<CreateOp<TensorStorage<typeof float32, readonly []>>>
    >();
    expect(scalar.shape).toEqual([]);
    expect(scalar.ndim).toBe(0);
    expect(scalar.size).toBe(1);
    expect(scalar.dtype).toBe(float32);
  });

  it('should infer 1D shape', async () => {
    const vec = await tensor([1, 2, 3] as const, { device: mockDevice, dtype: float32 });
    expectTypeOf(vec).toEqualTypeOf<
      Tensor<CreateOp<TensorStorage<typeof float32, readonly [3]>>>
    >();
    expect(vec.shape).toEqual([3]);
    expect(vec.ndim).toBe(1);
    expect(vec.size).toBe(3);
    expect(vec.dtype).toBe(float32);
  });

  it('should infer 2D shape', async () => {
    const mat = await tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: mockDevice, dtype: float32 },
    );
    expectTypeOf(mat).toEqualTypeOf<
      Tensor<CreateOp<TensorStorage<typeof float32, readonly [2, 2]>>>
    >();
    expect(mat.shape).toEqual([2, 2]);
    expect(mat.ndim).toBe(2);
    expect(mat.size).toBe(4);
    expect(mat.dtype).toBe(float32);
  });

  it('should infer 3D shape', async () => {
    const t3d = await tensor(
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
      { device: mockDevice, dtype: float32 },
    );
    expectTypeOf(t3d).toEqualTypeOf<
      Tensor<CreateOp<TensorStorage<typeof float32, readonly [2, 2, 2]>>>
    >();
    expect(t3d.shape).toEqual([2, 2, 2]);
    expect(t3d.ndim).toBe(3);
    expect(t3d.size).toBe(8);
    expect(t3d.dtype).toBe(float32);
  });

  it('should infer deeply nested shapes', async () => {
    const deep = await tensor([[[[1]]]], { device: mockDevice, dtype: float32 });
    expectTypeOf(deep).toEqualTypeOf<
      Tensor<CreateOp<TensorStorage<typeof float32, readonly [1, 1, 1, 1]>>>
    >();
    expect(deep.shape).toEqual([1, 1, 1, 1]);
    expect(deep.ndim).toBe(4);
    expect(deep.dtype).toBe(float32);
  });
});

describe('tensor() dtype handling', () => {
  it('should use default dtype (float32)', async () => {
    const t = await tensor([1, 2, 3], { device: mockDevice, dtype: float32 });
    expectTypeOf(t).toEqualTypeOf<Tensor<CreateOp<TensorStorage<typeof float32, readonly [3]>>>>();
    expect(t.dtype).toBe(float32);
  });

  it('should respect explicit dtype', async () => {
    const intTensor = await tensor([1.5, 2.7, 3.9] as const, { device: mockDevice, dtype: int32 });
    expectTypeOf(intTensor).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Int32, readonly [3]>>>>();
    expect(intTensor.dtype).toBe(int32);
  });

  it('should handle different numeric dtypes', async () => {
    const int8Tensor = await tensor([1, 2, 3] as const, { device: mockDevice, dtype: int8 });
    expectTypeOf(int8Tensor).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Int8, readonly [3]>>>>();
    expect(int8Tensor.dtype).toBe(int8);

    const uint8Tensor = await tensor([250, 251, 252] as const, {
      device: mockDevice,
      dtype: uint8,
    });
    expectTypeOf(uint8Tensor).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Uint8, readonly [3]>>>>();
    expect(uint8Tensor.dtype).toBe(uint8);
  });

  it('should handle bigint dtype', async () => {
    const bigintTensor = await tensor([1n, 2n, 3n], { device: mockDevice, dtype: int64 });
    expectTypeOf(bigintTensor).toEqualTypeOf<
      Tensor<CreateOp<TensorStorage<Int64, readonly [3]>>>
    >();
    expect(bigintTensor.dtype).toBe(int64);
  });

  it('should handle boolean dtype', async () => {
    const boolTensor = await tensor([true, false, true] as const, {
      device: mockDevice,
      dtype: bool,
    });
    expectTypeOf(boolTensor).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Bool, readonly [3]>>>>();
    expect(boolTensor.dtype).toBe(bool);
  });
});

describe('tensor() property validation', () => {
  it('should compute correct strides', async () => {
    // 1D
    const t1 = await tensor([1, 2, 3], { device: mockDevice, dtype: float32 });
    expect(t1.strides).toEqual([1]);

    // 2D (row-major)
    const t2 = await tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { device: mockDevice, dtype: float32 },
    );
    expect(t2.strides).toEqual([3, 1]);

    // 3D
    const t3 = await tensor(
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
      { device: mockDevice, dtype: float32 },
    );
    expect(t3.strides).toEqual([4, 2, 1]);
  });

  it('should set correct layout flags', async () => {
    const t = await tensor(
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      { device: mockDevice, dtype: float32 },
    );

    expect(t.layout.c_contiguous).toBe(true);
    expect(t.layout.f_contiguous).toBe(false);
    expect(t.layout.is_view).toBe(false);
    expect(t.layout.writeable).toBe(true);
    expect(t.layout.aligned).toBe(true);
  });
});

describe('tensor() error handling', () => {
  it('should throw on inconsistent nested array dimensions', async () => {
    await expect(
      tensor([[1, 2], [3]], { device: mockDevice, dtype: float32 }), // Inconsistent row lengths
    ).rejects.toThrow(/Inconsistent array dimensions/);
  });

  it('should create zero-sized tensor from empty arrays', async () => {
    const emptyTensor = await tensor([], { device: mockDevice, dtype: float32 });
    expect(emptyTensor.shape).toEqual([0]);
    expect(emptyTensor.size).toBe(0);
    expect(emptyTensor.dtype).toBe(float32);
  });
});

describe('zeros() shape specification', () => {
  it('should create tensors with specified shape', async () => {
    const z1 = await zeros([2, 3] as const, { device: mockDevice, dtype: float32 });
    expect(z1.shape).toEqual([2, 3]);
    expect(z1.size).toBe(6);

    const z2 = await zeros([5] as const, { device: mockDevice, dtype: float32 });
    expect(z2.shape).toEqual([5]);
    expect(z2.size).toBe(5);

    const z3 = await zeros([] as const, { device: mockDevice, dtype: float32 });
    expect(z3.shape).toEqual([]);
    expect(z3.size).toBe(1);
  });

  it('should respect dtype parameter', async () => {
    const z = await zeros([2, 2] as const, { device: mockDevice, dtype: int8 });
    expect(z.dtype).toBe(int8);
  });

  it('should handle zero-sized dimensions', async () => {
    const z = await zeros([0, 3] as const, { device: mockDevice, dtype: float32 });
    expect(z.shape).toEqual([0, 3]);
    expect(z.size).toBe(0);
  });

  it('should compute correct strides', async () => {
    const z = await zeros([2, 3, 4] as const, { device: mockDevice, dtype: float32 });
    expect(z.strides).toEqual([12, 4, 1]);
  });
});

describe('ones() shape specification', () => {
  it('should create tensors with specified shape', async () => {
    const o = await ones([2, 3] as const, { device: mockDevice, dtype: float32 });
    expect(o.shape).toEqual([2, 3]);
    expect(o.dtype).toBe(float32);
  });

  it('should handle different dtypes', async () => {
    const intOnes = await ones([2, 2] as const, { device: mockDevice, dtype: int32 });
    expectTypeOf(intOnes).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Int32, readonly [2, 2]>>>>();
    expect(intOnes.dtype).toBe(int32);

    const bigintOnes = await ones([3] as const, { device: mockDevice, dtype: int64 });
    expectTypeOf(bigintOnes).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Int64, readonly [3]>>>>();
    expect(bigintOnes.dtype).toBe(int64);

    const boolOnes = await ones([2, 2] as const, { device: mockDevice, dtype: bool });
    expectTypeOf(boolOnes).toEqualTypeOf<Tensor<CreateOp<TensorStorage<Bool, readonly [2, 2]>>>>();
    expect(boolOnes.dtype).toBe(bool);
  });

  it('should create high-dimensional tensors', async () => {
    const o = await ones([2, 3, 4, 5] as const, { device: mockDevice, dtype: float32 });
    expect(o.shape).toEqual([2, 3, 4, 5]);
    expect(o.ndim).toBe(4);
    expect(o.size).toBe(120);
  });
});

describe('eye() shape specification', () => {
  it('should create square matrices', async () => {
    const I3 = await eye(3, { device: mockDevice, dtype: float32 });
    expect(I3.shape).toEqual([3, 3]);
    expect(I3.size).toBe(9);

    const I1 = await eye(1, { device: mockDevice, dtype: float32 });
    expect(I1.shape).toEqual([1, 1]);
    expect(I1.size).toBe(1);

    const I5 = await eye(5, { device: mockDevice, dtype: float32 });
    expect(I5.shape).toEqual([5, 5]);
    expect(I5.size).toBe(25);
  });

  it('should respect dtype parameter', async () => {
    const I = await eye(3, { device: mockDevice, dtype: int32 });
    expect(I.dtype).toBe(int32);
  });

  it('should handle zero-sized identity matrix', async () => {
    const I = await eye(0, { device: mockDevice, dtype: float32 });
    expect(I.shape).toEqual([0, 0]);
    expect(I.size).toBe(0);
  });

  it('should compute correct strides', async () => {
    const I = await eye(4, { device: mockDevice, dtype: float32 });
    expect(I.strides).toEqual([4, 1]);
  });
});

describe('tensor string representation', () => {
  it('should format scalar tensors', async () => {
    const scalar = await tensor(42, { device: mockDevice, dtype: float32 });
    expect(scalar.toString()).toBe('Tensor(shape=scalar [], dtype=float32, device=mock)');
  });

  it('should format vector tensors', async () => {
    const vec = await tensor([1, 2, 3], { device: mockDevice, dtype: float32 });
    expect(vec.toString()).toBe('Tensor(shape=[3], dtype=float32, device=mock)');
  });

  it('should format matrix tensors with different dtypes', async () => {
    const mat = await tensor(
      [
        [1, 2],
        [3, 4],
      ],
      { device: mockDevice, dtype: int32 },
    );
    expect(mat.toString()).toBe('Tensor(shape=[2, 2], dtype=int32, device=mock)');
  });
});

describe('view operations', () => {
  it('should create views efficiently', async () => {
    const t = await tensor([1, 2, 3, 4, 5, 6] as const, { device: mockDevice, dtype: float32 });
    expectTypeOf(t).toEqualTypeOf<Tensor<CreateOp<TensorStorage<typeof float32, readonly [6]>>>>();

    const start = performance.now();
    const view = t.reshape([2, 3] as const);
    const duration = performance.now() - start;

    // Type assertion for reshaped tensor
    expectTypeOf(view).toMatchTypeOf<
      Tensor<ReshapeOp<TensorStorage<typeof float32, readonly [6]>, readonly [2, 3]>>
    >();

    // Reshape should be very fast as it's just a view
    expect(duration).toBeLessThan(1); // Less than 1ms
    expect(view.layout.is_view).toBe(true);
    expect(view.shape).toEqual([2, 3]);
  });

  it('should validate reshape constraints', async () => {
    const t = await tensor([1, 2, 3, 4, 5, 6] as const, { device: mockDevice, dtype: float32 });

    // Valid reshapes
    expect(() => t.reshape([2, 3] as const)).not.toThrow();
    expect(() => t.reshape([3, 2] as const)).not.toThrow();
    expect(() => t.reshape([6] as const)).not.toThrow();

    // Invalid reshapes
    // @ts-expect-error - Type error
    expect(() => t.reshape([2, 2] as const)).toThrow(/different number of elements/);
    // @ts-expect-error - Type error
    expect(() => t.reshape([3, 3] as const)).toThrow(/different number of elements/);
  });
});
