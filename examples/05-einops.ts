/**
 * TypeTensor Einops: Simple Rearrange Examples
 *
 * This example demonstrates how to use the rearrange function with
 * TypeTensor tensors to perform common tensor manipulations using
 * Einstein notation patterns.
 *
 * Run with: bun run examples/05-einops-rearrange.ts
 */

import { tensor, float32 } from '@typetensor/core';
import { rearrange } from '@typetensor/core';
import { cpu } from '@typetensor/backend-cpu';

async function main(): Promise<void> {
  console.log('TypeTensor Einops: Rearrange Examples\n');
  console.log('='.repeat(50));

  // ============================================================================
  // 1. Simple Transpose: Swap dimensions
  // ============================================================================
  console.log('\n1. Simple Transpose:');

  const matrix = await tensor(
    [
      [1, 2, 3],
      [4, 5, 6],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Original matrix [2 x 3]:');
  console.log(await matrix.format());

  // Transpose using einops pattern
  const transposed = rearrange(matrix, 'height width -> width height');

  console.log('\nTransposed [3 x 2]:');
  console.log(await transposed.format());

  // ============================================================================
  // 2. Reorder Dimensions: Change from CHW to HWC format
  // ============================================================================
  console.log('\n\n2. Image Format Conversion:');

  // Create a small "image" in channels-first format (CHW)
  const image_chw = await tensor(
    [
      [
        [1, 2],
        [3, 4],
      ], // Red channel
      [
        [5, 6],
        [7, 8],
      ], // Green channel
      [
        [9, 10],
        [11, 12],
      ], // Blue channel
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Channels-first (CHW) format [3 x 2 x 2]:');
  console.log(await image_chw.format());

  // Convert to channels-last format (HWC)
  const image_hwc = rearrange(image_chw, 'channels height width -> height width channels');

  console.log('\nChannels-last (HWC) format [2 x 2 x 3]:');
  console.log(await image_hwc.format());

  // ============================================================================
  // 3. Batch Processing: Add batch dimension
  // ============================================================================
  console.log('\n\n3. Add Batch Dimension:');

  const single_image = await tensor(
    [
      [1, 2, 3],
      [4, 5, 6],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Single image [2 x 3]:');
  console.log(await single_image.format());

  // Add batch dimension at the beginning
  const batched = rearrange(single_image, 'height width -> 1 height width');

  console.log('\nWith batch dimension [1 x 2 x 3]:');
  console.log(await batched.format());

  // ============================================================================
  // 4. Flatten Spatial Dimensions
  // ============================================================================
  console.log('\n\n4. Flatten for Fully Connected Layer:');

  const feature_map = await tensor(
    [
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
      [
        [9, 10],
        [11, 12],
      ],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Feature map [3 x 2 x 2] (channels x height x width):');
  console.log(await feature_map.format());

  // Flatten spatial dimensions while keeping channels
  const flattened = rearrange(feature_map, 'channels height width -> channels (height width)');

  console.log('\nFlattened [3 x 4] (channels x flattened_spatial):');
  console.log(await flattened.format());

  // ============================================================================
  // 5. Split Composite Dimensions
  // ============================================================================
  console.log('\n\n5. Split Flattened Dimension:');

  const flat_tensor = await tensor(
    [
      [1, 2, 3, 4, 5, 6],
      [7, 8, 9, 10, 11, 12],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Flattened tensor [2 x 6]:');
  console.log(await flat_tensor.format());

  // Split the second dimension into 2 x 3
  const reshaped = rearrange(flat_tensor, 'batch (height width) -> batch height width', {
    height: 2, // width will be inferred as 3
  });

  console.log('\nReshaped to [2 x 2 x 3]:');
  console.log(await reshaped.format());

  // ============================================================================
  // 6. Practical Example: Prepare for Multi-Head Attention
  // ============================================================================
  console.log('\n\n6. Multi-Head Attention Preparation:');

  // Simulated output from a linear layer (batch x seq x embed)
  const embeddings = await tensor(
    [
      [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
      ],
      [
        [17, 18, 19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30, 31, 32],
      ],
    ] as const,
    { device: cpu, dtype: float32 },
  );

  console.log('Token embeddings [2 x 2 x 8] (batch x seq x embed):');
  console.log(await embeddings.format());

  // Split embedding dimension into heads and head_dim
  const multi_head = rearrange(embeddings, 'batch seq (heads dim) -> batch heads seq dim', {
    heads: 4, // 4 heads, each with dimension 2
  });

  console.log('\nMulti-head format [2 x 4 x 2 x 2] (batch x heads x seq x dim):');
  console.log(await multi_head.format());

  // ============================================================================
  // Summary
  // ============================================================================
  console.log('\n' + '='.repeat(50));
  console.log('Einops patterns make tensor operations intuitive:');
  console.log('✓ "h w -> w h" for transpose');
  console.log('✓ "c h w -> h w c" for format conversion');
  console.log('✓ "h w -> 1 h w" to add dimensions');
  console.log('✓ "(h w) c -> h w c" to split dimensions');
  console.log('✓ Named axes make code self-documenting');
}

// Run the example
main().catch(console.error);
