import { cpu } from '@typetensor/backend-cpu';
import { tensor, float32 } from '@typetensor/core';

async function main() {
  // Create a simple tensor with [2, 3]
  const t = await tensor(
    [
      [1, 2, 3],
      [4, 5, 6],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  // Attempt to reshape to [3, 3]
  await t.reshape([3, 3] as const);
  //              ^ error: [TypeTensor ❌] Cannot reshape: 6 ≠ 9 elements
  // Error is automatically detected at compile time for invalid reshapes!

  // Attempt to add two tensors with different shapes that cannot be broadcasted
  const a = await tensor([1, 2, 3] as const, { device: cpu, dtype: float32 });
  const b = await tensor([10, 20, 30, 40] as const, { device: cpu, dtype: float32 });
  const sum = await a.add(b);
  //                      ^ error: [TypeTensor ❌] Cannot add tensors with shapes [3] and [4]. Shapes must be compatible for broadcasting.
  // Error is automatically detected at compile time for invalid broadcasts!
  console.log(await sum.format());
}

main();
