/**
 * Type tests for tile operation
 *
 * These tests validate that our type-level tile operations work correctly
 * at compile time.
 */

import { expectTypeOf } from 'expect-type';
import type { TileShape } from './types';

// =============================================================================
// TileShape Tests
// =============================================================================

// Basic tiling
{
  expectTypeOf<TileShape<[3], [2]>>().toEqualTypeOf<readonly [6]>(); // 3 * 2
  expectTypeOf<TileShape<[2, 3], [2, 3]>>().toEqualTypeOf<readonly [4, 9]>(); // [2*2, 3*3]
  expectTypeOf<TileShape<[5, 2, 3], [2, 3, 4]>>().toEqualTypeOf<readonly [10, 6, 12]>();
}

// Identity tiling (reps of 1)
{
  expectTypeOf<TileShape<[2, 3, 4], [1, 1, 1]>>().toEqualTypeOf<readonly [2, 3, 4]>();
  expectTypeOf<TileShape<[5], [1]>>().toEqualTypeOf<readonly [5]>();
  expectTypeOf<TileShape<[10, 20], [1, 1]>>().toEqualTypeOf<readonly [10, 20]>();
}

// Fewer reps than dimensions (tiles rightmost)
{
  expectTypeOf<TileShape<[2, 3, 4], [2]>>().toEqualTypeOf<readonly [2, 3, 8]>(); // Only last dim
  expectTypeOf<TileShape<[2, 3, 4], [3, 2]>>().toEqualTypeOf<readonly [2, 9, 8]>(); // Last two dims
  expectTypeOf<TileShape<[5, 4, 3, 2], [2]>>().toEqualTypeOf<readonly [5, 4, 3, 4]>();
}

// More reps than dimensions (adds new dims)
{
  expectTypeOf<TileShape<[3], [2, 3]>>().toEqualTypeOf<readonly [2, 9]>(); // [2, 3*3]
  expectTypeOf<TileShape<[2, 3], [2, 3, 4]>>().toEqualTypeOf<readonly [2, 6, 12]>(); // [2, 2*3, 3*4]
  expectTypeOf<TileShape<[5], [2, 3, 4, 5]>>().toEqualTypeOf<readonly [2, 3, 4, 25]>();
}

// Scalar tiling
{
  expectTypeOf<TileShape<[], [3, 4]>>().toEqualTypeOf<readonly [3, 4]>();
  expectTypeOf<TileShape<[], [10]>>().toEqualTypeOf<readonly [10]>();
  expectTypeOf<TileShape<[], []>>().toEqualTypeOf<readonly []>();
}

// Empty reps
{
  expectTypeOf<TileShape<[2, 3], []>>().toEqualTypeOf<readonly [2, 3]>();
  expectTypeOf<TileShape<[5, 4, 3], []>>().toEqualTypeOf<readonly [5, 4, 3]>();
}

// Complex patterns
{
  expectTypeOf<TileShape<[1, 2, 3], [2, 1, 4]>>().toEqualTypeOf<readonly [2, 2, 12]>(); // [1*2, 2*1, 3*4]
  expectTypeOf<TileShape<[2, 1], [3, 2, 5]>>().toEqualTypeOf<readonly [3, 4, 5]>(); // [3, 2*2, 1*5]
}

// Zero repetitions
{
  expectTypeOf<TileShape<[2, 3], [0, 2]>>().toEqualTypeOf<readonly [0, 6]>(); // [2*0, 3*2]
  expectTypeOf<TileShape<[5], [0]>>().toEqualTypeOf<readonly [0]>();
  expectTypeOf<TileShape<[2, 3, 4], [1, 0, 1]>>().toEqualTypeOf<readonly [2, 0, 4]>();
}

// Neural network patterns
{
  // Repeat along batch dimension
  expectTypeOf<TileShape<[1, 768], [32, 1]>>().toEqualTypeOf<readonly [32, 768]>();
  
  // Tile attention patterns
  expectTypeOf<TileShape<[1, 128, 128], [32, 1, 1]>>().toEqualTypeOf<readonly [32, 128, 128]>();
  
  // Repeat sequence patterns
  expectTypeOf<TileShape<[10, 512], [1, 2]>>().toEqualTypeOf<readonly [10, 1024]>();
}

// Edge cases
{
  // Very large repetitions
  expectTypeOf<TileShape<[2], [100]>>().toEqualTypeOf<readonly [200]>();
  expectTypeOf<TileShape<[1, 1], [100, 100]>>().toEqualTypeOf<readonly [100, 100]>();
  
  // Mixed with zeros
  expectTypeOf<TileShape<[0, 5], [2, 3]>>().toEqualTypeOf<readonly [0, 15]>();
  expectTypeOf<TileShape<[3, 0], [4, 5]>>().toEqualTypeOf<readonly [12, 0]>();
}