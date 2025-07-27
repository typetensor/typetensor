/**
 * TypeTensor: View Operations (Reshape, View, Slice)
 *
 * This example demonstrates tensor view operations that create different
 * views of the same data with compile-time shape validation.
 *
 * Run with: bun run examples/03-view-operations.ts
 */

import { tensor, float32, zeros } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';

async function main(): Promise<void> {
  console.log('TypeTensor: View Operations Demo\n');
  console.log('='.repeat(50));

  // =============================================================================
  // Reshape Operations
  // =============================================================================
  console.log('\n1. RESHAPE OPERATIONS');
  console.log('-'.repeat(30));

  // Create a 2x3 matrix
  const matrix = await tensor(
    [
      [1, 2, 3],
      [4, 5, 6],
    ] as const,
    {
      dtype: float32,
      device: cpu,
    },
  );
  console.log('Original 2x3 matrix:', await matrix.toArray());
  console.log('Shape:', matrix.shape);

  // Reshape to 3x2
  const reshaped1 = matrix.reshape([3, 2] as const);
  console.log('\nReshaped to 3x2:', await reshaped1.toArray());

  // Reshape to 1D (flatten)
  const flattened = matrix.flatten();
  console.log('\nFlattened to 1D:', await flattened.toArray());

  // Reshape to 6x1 column vector
  const column = matrix.reshape([6, 1] as const);
  console.log('\nReshaped to 6x1:', await column.toArray());

  // =============================================================================
  // View Operations with Dimension Inference
  // =============================================================================
  console.log('\n\n2. VIEW OPERATIONS (with -1 dimension inference)');
  console.log('-'.repeat(30));

  // Create a 2x3x4 tensor
  const tensor3d = await zeros([2, 3, 4] as const, {
    dtype: float32,
    device: cpu,
  });
  console.log('3D tensor shape:', tensor3d.shape);

  // View with inferred dimension
  const view1 = tensor3d.view([-1, 6] as const); // Shape: [4, 6]
  console.log('\nView with [-1, 6] shape:', view1.shape);

  const view2 = tensor3d.view([6, -1] as const); // Shape: [6, 4]
  console.log('View with [6, -1] shape:', view2.shape);

  const view3 = tensor3d.view([2, -1] as const); // Shape: [2, 12]
  console.log('View with [2, -1] shape:', view3.shape);

  // =============================================================================
  // Slice Operations
  // =============================================================================
  console.log('\n\n3. SLICE OPERATIONS');
  console.log('-'.repeat(30));

  // Create a 3x4 matrix for slicing examples
  const data = await tensor(
    [
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ] as const,
    {
      dtype: float32,
      device: cpu,
    },
  );
  console.log('Original 3x4 matrix:', await data.toArray());

  // Integer indexing - select a single row
  console.log('\n--- Integer Indexing ---');
  const row0 = await data.slice([0]); // First row
  console.log('slice([0]) - First row:', await row0.toArray());
  console.log('Shape:', row0.shape); // [4]

  const row2 = await data.slice([2]); // Last row
  console.log('\nslice([2]) - Last row:', await row2.toArray());
  console.log('Shape:', row2.shape); // [4]

  // Column selection with null
  console.log('\n--- Column Selection ---');
  const col1 = await data.slice([null, 1]); // Second column
  console.log('slice([null, 1]) - Second column:', await col1.toArray());
  console.log('Shape:', col1.shape); // [3]

  // Slice ranges
  console.log('\n--- Slice Ranges ---');
  const rows01 = await data.slice([{ start: 0, stop: 2 }]); // First 2 rows
  console.log('slice([{start: 0, stop: 2}]) - First 2 rows:', await rows01.toArray());
  console.log('Shape:', rows01.shape); // [2, 4]

  const cols12 = await data.slice([null, { start: 1, stop: 3 }]); // Columns 1-2
  console.log('\nslice([null, {start: 1, stop: 3}]) - Columns 1-2:', await cols12.toArray());
  console.log('Shape:', cols12.shape); // [3, 2]

  // Combined slicing
  const submatrix = await data.slice([
    { start: 0, stop: 2 },
    { start: 1, stop: 3 },
  ]);
  console.log(
    '\nslice([{start: 0, stop: 2}, {start: 1, stop: 3}]) - Submatrix:',
    await submatrix.toArray(),
  );
  console.log('Shape:', submatrix.shape); // [2, 2]

  // Step slicing
  console.log('\n--- Step Slicing ---');
  const everyOther = await data.slice([null, { step: 2 }]); // Every other column
  console.log('slice([null, {step: 2}]) - Every other column:', await everyOther.toArray());
  console.log('Shape:', everyOther.shape); // [3, 2]

  // Negative indices
  console.log('\n--- Negative Indices ---');
  const lastRow = await data.slice([-1]); // Last row using negative index
  console.log('slice([-1]) - Last row:', await lastRow.toArray());
  console.log('Shape:', lastRow.shape); // [4]

  // =============================================================================
  // Complex Example: Combining Operations
  // =============================================================================
  console.log('\n\n4. COMBINING VIEW OPERATIONS');
  console.log('-'.repeat(30));

  // Create a larger tensor
  const large = await tensor(
    [
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      [
        [7, 8],
        [9, 10],
        [11, 12],
      ],
    ] as const,
    {
      dtype: float32,
      device: cpu,
    },
  );
  console.log('Original 2x3x2 tensor:', await large.toArray());

  // Slice then reshape
  const sliced = await large.slice([0]); // Get first 3x2 matrix
  console.log('\nAfter slice([0]):', await sliced.toArray());
  console.log('Shape:', sliced.shape); // [3, 2]

  const reshaped = sliced.reshape([2, 3] as const);
  console.log('\nAfter reshape([2, 3]):', await reshaped.toArray());
  console.log('Shape:', reshaped.shape); // [2, 3]

  console.log('\n✅ All view operations completed successfully!');
}

// Run the example
main().catch(console.error);
