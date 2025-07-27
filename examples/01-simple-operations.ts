/**
 * TypeTensor: Type-Safe Tensor Operations in TypeScript
 *
 * This example demonstrates how TypeTensor uses TypeScript's type system
 * to catch shape mismatches and invalid operations at compile time.
 *
 * Run with: bun run examples/type-safe-tensors.ts
 */

import { tensor, zeros, ones, eye, float32, int32 } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';

async function main(): Promise<void> {
  console.log('TypeTensor: Type-Safe Tensor Operations\n');
  console.log('='.repeat(50));

  // ============================================================================
  // 1. Basic Tensor Creation with Shape Inference
  // ============================================================================
  console.log('\n1. Shape Inference from Data:');

  // TypeScript infers shape [3] from the array
  const vector = await tensor([1, 2, 3] as const, { device: cpu, dtype: float32 });
  console.log(await vector.format());

  // TypeScript infers shape [2, 3] from nested arrays
  const matrix = await tensor(
    [
      [1, 2, 3],
      [4, 5, 6],
    ] as const,
    { device: cpu, dtype: float32 },
  );
  console.log(await matrix.format());

  // ============================================================================
  // 2. Type-Safe Operations
  // ============================================================================
  console.log('\n\n2. Type-Safe Arithmetic Operations:');

  // Addition with compatible shapes
  const a = await tensor([1, 2, 3] as const, { device: cpu, dtype: float32 });
  const b = await tensor([10, 20, 30] as const, { device: cpu, dtype: float32 });
  const sum = await a.add(b);
  console.log(await sum.format());

  // Broadcasting: adding a scalar to a vector
  const scalar = await tensor(100, { device: cpu, dtype: float32 });
  const broadcasted = await a.add(scalar);
  console.log(await broadcasted.format());

  // ============================================================================
  // 3. Shape Safety with Reshaping
  // ============================================================================
  console.log('\n\n3. Compile-Time Shape Validation:');

  const original = await tensor([1, 2, 3, 4, 5, 6] as const, { device: cpu, dtype: float32 });

  // Valid reshape: 6 elements → [2, 3]
  const reshaped = original.reshape([2, 3] as const);
  console.log(await reshaped.format());

  // Valid reshape: 6 elements → [3, 2]
  const reshaped2 = original.reshape([3, 2] as const);
  console.log(await reshaped2.format());

  // Using view with dimension inference (-1)
  const viewed = original.view([2, -1] as const);
  console.log(await viewed.format());

  // ============================================================================
  // 4. What TypeScript Prevents (Commented Examples)
  // ============================================================================
  console.log('\n\n4. TypeScript Prevents These Errors at Compile Time:');
  console.log('(Uncomment to see TypeScript errors)');

  // ❌ Invalid reshape: wrong number of elements
  // const invalid = original.reshape([2, 2] as const);
  // TypeScript Error: Cannot reshape: 6 ≠ 4 elements

  // ❌ Shape mismatch in operations (without broadcasting)
  // const vec3 = await tensor([1, 2, 3] as const, { device: cpu });
  // const vec4 = await tensor([1, 2, 3, 4] as const, { device: cpu });
  // const invalid = await vec3.add(vec4);
  // TypeScript Error: Shapes [3] and [4] cannot broadcast

  // ❌ Forgot 'as const' for shape inference
  // const noConst = await tensor([[1, 2], [3, 4]], { device: cpu });
  // TypeScript Error: Shape must use 'as const'

  // ============================================================================
  // 5. Practical Example: Matrix Operations
  // ============================================================================
  console.log('\n\n5. Practical Example - Image Brightness Adjustment:');

  // Simulate a small 3x3 grayscale image
  const image = await tensor(
    [
      [100, 150, 200],
      [120, 180, 210],
      [140, 160, 190],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Original image values:');
  console.log(await image.format());

  // Increase brightness by 20%
  const brightness_factor = await tensor(1.2, { device: cpu, dtype: float32 });
  const brightened = await image.mul(brightness_factor);

  console.log('\nBrightened image (×1.2):');
  console.log(await brightened.format());

  // Apply gamma correction
  const gamma_corrected = await (
    await brightened.div(await tensor(255, { device: cpu, dtype: float32 }))
  ).sqrt();
  const final_image = await gamma_corrected.mul(await tensor(255, { device: cpu, dtype: float32 }));

  console.log('\nAfter gamma correction:');
  console.log(await final_image.format());

  // ============================================================================
  // 6. Special Tensor Creation
  // ============================================================================
  console.log('\n\n6. Creating Special Tensors:');

  const zeros_matrix = await zeros([2, 3], { device: cpu, dtype: float32 });
  console.log(await zeros_matrix.format());

  const ones_vector = await ones([4], { device: cpu, dtype: int32 });
  console.log(await ones_vector.format());

  const identity = await eye(3, { device: cpu, dtype: float32 });
  console.log(await identity.format());

  // ============================================================================
  // 7. 3D and Higher Dimensional Tensors
  // ============================================================================
  console.log('\n\n7. 3D and Higher Dimensional Tensors:');

  // 3D tensor (like a batch of images)
  const tensor_3d = await tensor(
    [
      [[1, 2], [3, 4], [5, 6]],
      [[7, 8], [9, 10], [11, 12]],
    ] as const,
    { device: cpu, dtype: float32 },
  );
  console.log('3D tensor shape [2, 3, 2]:');
  console.log(await tensor_3d.format());

  // 4D tensor (like a batch of RGB images)
  const tensor_4d = await tensor(
    [
      [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
      ],
      [
        [[13, 14, 15], [16, 17, 18]],
        [[19, 20, 21], [22, 23, 24]],
      ],
    ] as const,
    { device: cpu, dtype: float32 },
  );
  console.log('\n4D tensor shape [2, 2, 2, 3]:');
  console.log(await tensor_4d.format());

  // ============================================================================
  // Summary
  // ============================================================================
  console.log('\n' + '='.repeat(50));
  console.log('Key TypeTensor Features Demonstrated:');
  console.log('✓ Compile-time shape validation');
  console.log('✓ Type-safe broadcasting');
  console.log('✓ Invalid operations caught before runtime');
  console.log('✓ Intuitive API with full TypeScript support');
}

// Run the example
main().catch(console.error);
