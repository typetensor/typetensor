/**
 * TypeTensor: View Operations (Reshape, View, Slice, Transpose, Permute)
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

  // =============================================================================
  // Transpose Operations
  // =============================================================================
  console.log('\n\n5. TRANSPOSE OPERATIONS');
  console.log('-'.repeat(30));

  // Create a 2D matrix for transpose
  const matrix2d = await tensor(
    [
      [1, 2, 3],
      [4, 5, 6],
    ] as const,
    {
      dtype: float32,
      device: cpu,
    },
  );
  console.log('Original 2x3 matrix:', await matrix2d.toArray());
  console.log('Shape:', matrix2d.shape);

  // Transpose the matrix
  const transposed2d = matrix2d.transpose();
  console.log('\nTransposed matrix:', await transposed2d.toArray());
  console.log('Shape:', transposed2d.shape); // [3, 2]

  // Using T property (shorthand for transpose)
  const transposedT = matrix2d.T;
  console.log('\nUsing .T property:', await transposedT.toArray());

  // Transpose 3D tensor (swaps last two dimensions)
  const tensor3dForTranspose = await tensor(
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
  console.log('\nOriginal 2x3x2 tensor:', await tensor3dForTranspose.toArray());
  console.log('Shape:', tensor3dForTranspose.shape);

  const transposed3d = tensor3dForTranspose.transpose();
  console.log('\nTransposed 3D tensor:', await transposed3d.toArray());
  console.log('Shape:', transposed3d.shape); // [2, 2, 3]

  // =============================================================================
  // Permute Operations
  // =============================================================================
  console.log('\n\n6. PERMUTE OPERATIONS');
  console.log('-'.repeat(30));

  // Create a 3D tensor for permutation examples
  const tensor3dForPermute = await tensor(
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
  console.log('Original tensor shape:', tensor3dForPermute.shape); // [2, 3, 2]

  // Permute to [3, 2, 2] - move dimension 1 to front
  const permuted1 = tensor3dForPermute.permute([1, 0, 2] as const);
  console.log('\nPermute([1, 0, 2]):', await permuted1.toArray());
  console.log('Shape:', permuted1.shape); // [3, 2, 2]

  // Permute to [2, 2, 3] - move last dimension to middle
  const permuted2 = tensor3dForPermute.permute([0, 2, 1] as const);
  console.log('\nPermute([0, 2, 1]):', await permuted2.toArray());
  console.log('Shape:', permuted2.shape); // [2, 2, 3]

  // Permute to [2, 2, 3] - reverse all dimensions
  const permuted3 = tensor3dForPermute.permute([2, 1, 0] as const);
  console.log('\nPermute([2, 1, 0]):', await permuted3.toArray());
  console.log('Shape:', permuted3.shape); // [2, 3, 2]

  // Example: NHWC to NCHW conversion (common in deep learning)
  // Create a simulated image tensor [batch=1, height=4, width=4, channels=3]
  const nhwc = await zeros([1, 4, 4, 3] as const, {
    dtype: float32,
    device: cpu,
  });
  console.log('\nNHWC tensor shape (batch, height, width, channels):', nhwc.shape);

  // Convert to NCHW format
  const nchw = nhwc.permute([0, 3, 1, 2] as const);
  console.log('NCHW tensor shape (batch, channels, height, width):', nchw.shape); // [1, 3, 4, 4]

  // Identity permutation (no change)
  const identity = tensor3dForPermute.permute([0, 1, 2] as const);
  console.log('\nIdentity permutation shape:', identity.shape); // [2, 3, 2]

  console.log('\n✅ All view operations completed successfully!');
}

// Run the example
main().catch(console.error);
