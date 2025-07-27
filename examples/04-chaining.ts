import { cpu } from '@typetensor/backend-cpu';
import { tensor, float32 } from '@typetensor/core';

async function main() {
  // Sequential operations on different lines works!
  const t = await tensor([1, 2, 3, 4, 5, 6] as const, { device: cpu, dtype: float32 });
  const reshaped = await t.reshape([2, 3] as const);
  const transposed = await reshaped.transpose();
  const flattened = await transposed.flatten();
  console.log(await flattened.format());

  // Chaining with .then() works!
  const flattened2 = await tensor([1, 2, 3, 4, 5, 6] as const, { device: cpu, dtype: float32 })
    .then((t) => t.reshape([2, 3] as const))
    .then((t) => t.transpose())
    .then((t) => t.flatten());
  console.log(await flattened2.format());

  // Chaining awaitables with a single await works!
  const flattened3 = await (
    await tensor([1, 2, 3, 4, 5, 6] as const, { device: cpu, dtype: float32 })
  )
    .reshape([2, 3] as const)
    .transpose()
    .flatten();
  console.log(await flattened3.format());
}

main();
