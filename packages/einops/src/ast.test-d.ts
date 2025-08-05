/**
 * Type tests for the einops AST types
 *
 * These tests validate that our AST types work correctly at compile time
 * using expect-type, ensuring proper type discrimination and readonly constraints.
 */

import { expectTypeOf } from 'expect-type';
import type {
  AxisPattern,
  SimpleAxis,
  CompositeAxis,
  EllipsisAxis,
  SingletonAxis,
  EinopsAST,
  ASTMetadata,
} from './ast';
import type { Position } from './types';

// =============================================================================
// Core AST Pattern Type Tests
// =============================================================================

{
  // Test SimpleAxis type structure
  const simpleAxis: SimpleAxis = {
    type: 'simple',
    name: 'batch',
    position: { start: 0, end: 5 },
  };

  expectTypeOf<typeof simpleAxis.type>().toEqualTypeOf<'simple'>();
  expectTypeOf<typeof simpleAxis.name>().toEqualTypeOf<string>();
  expectTypeOf<typeof simpleAxis.position>().toEqualTypeOf<Position>();
  expectTypeOf<typeof simpleAxis>().toMatchTypeOf<AxisPattern>();
}

{
  // Test CompositeAxis type structure
  const compositeAxis: CompositeAxis = {
    type: 'composite',
    axes: [
      { type: 'simple', name: 'h', position: { start: 1, end: 2 } },
      { type: 'simple', name: 'w', position: { start: 3, end: 4 } },
    ],
    position: { start: 0, end: 5 },
  };

  expectTypeOf<typeof compositeAxis.type>().toEqualTypeOf<'composite'>();
  expectTypeOf<typeof compositeAxis.axes>().toEqualTypeOf<readonly AxisPattern[]>();
  expectTypeOf<typeof compositeAxis.position>().toEqualTypeOf<Position>();
  expectTypeOf<typeof compositeAxis>().toMatchTypeOf<AxisPattern>();
}

{
  // Test EllipsisAxis type structure
  const ellipsisAxis: EllipsisAxis = {
    type: 'ellipsis',
    position: { start: 0, end: 3 },
  };

  expectTypeOf<typeof ellipsisAxis.type>().toEqualTypeOf<'ellipsis'>();
  expectTypeOf<typeof ellipsisAxis.position>().toEqualTypeOf<Position>();
  expectTypeOf<typeof ellipsisAxis>().toMatchTypeOf<AxisPattern>();
}

{
  // Test SingletonAxis type structure
  const singletonAxis: SingletonAxis = {
    type: 'singleton',
    position: { start: 0, end: 1 },
  };

  expectTypeOf<typeof singletonAxis.type>().toEqualTypeOf<'singleton'>();
  expectTypeOf<typeof singletonAxis.position>().toEqualTypeOf<Position>();
  expectTypeOf<typeof singletonAxis>().toMatchTypeOf<AxisPattern>();
}

// =============================================================================
// AxisPattern Union Type Tests
// =============================================================================

{
  // Test that all pattern types are assignable to AxisPattern
  const simpleAxis: SimpleAxis = {
    type: 'simple',
    name: 'test',
    position: { start: 0, end: 4 },
  };

  const compositeAxis: CompositeAxis = {
    type: 'composite',
    axes: [],
    position: { start: 0, end: 2 },
  };

  const ellipsisAxis: EllipsisAxis = {
    type: 'ellipsis',
    position: { start: 0, end: 3 },
  };

  const singletonAxis: SingletonAxis = {
    type: 'singleton',
    position: { start: 0, end: 1 },
  };

  expectTypeOf<typeof simpleAxis>().toMatchTypeOf<AxisPattern>();
  expectTypeOf<typeof compositeAxis>().toMatchTypeOf<AxisPattern>();
  expectTypeOf<typeof ellipsisAxis>().toMatchTypeOf<AxisPattern>();
  expectTypeOf<typeof singletonAxis>().toMatchTypeOf<AxisPattern>();
}

// =============================================================================
// ASTMetadata Type Tests
// =============================================================================

{
  const metadata: ASTMetadata = {
    originalPattern: 'h w -> w h',
    arrowPosition: { start: 4, end: 6 },
    inputTokenCount: 3,
    outputTokenCount: 3,
  };

  expectTypeOf<typeof metadata.originalPattern>().toEqualTypeOf<string>();
  expectTypeOf<typeof metadata.arrowPosition>().toEqualTypeOf<Position>();
  expectTypeOf<typeof metadata.inputTokenCount>().toEqualTypeOf<number>();
  expectTypeOf<typeof metadata.outputTokenCount>().toEqualTypeOf<number>();
}

// =============================================================================
// EinopsAST Type Tests
// =============================================================================

{
  const ast: EinopsAST = {
    input: [
      { type: 'simple', name: 'h', position: { start: 0, end: 1 } },
      { type: 'simple', name: 'w', position: { start: 2, end: 3 } },
    ],
    output: [
      { type: 'simple', name: 'w', position: { start: 7, end: 8 } },
      { type: 'simple', name: 'h', position: { start: 9, end: 10 } },
    ],
    metadata: {
      originalPattern: 'h w -> w h',
      arrowPosition: { start: 4, end: 6 },
      inputTokenCount: 3,
      outputTokenCount: 3,
    },
  };

  expectTypeOf<typeof ast.input>().toEqualTypeOf<readonly AxisPattern[]>();
  expectTypeOf<typeof ast.output>().toEqualTypeOf<readonly AxisPattern[]>();
  expectTypeOf<typeof ast.metadata>().toEqualTypeOf<ASTMetadata>();
}

// =============================================================================
// Readonly Constraint Tests
// =============================================================================

{
  // Test that all properties are readonly
  const ast: EinopsAST = {
    input: [],
    output: [],
    metadata: {
      originalPattern: 'test',
      arrowPosition: { start: 0, end: 2 },
      inputTokenCount: 0,
      outputTokenCount: 0,
    },
  };

  // These should all be readonly arrays/objects
  expectTypeOf<typeof ast.input>().toEqualTypeOf<readonly AxisPattern[]>();
  expectTypeOf<typeof ast.output>().toEqualTypeOf<readonly AxisPattern[]>();

  // Test that AxisPattern union type has correct discriminated properties
  expectTypeOf<AxisPattern>().toHaveProperty('type').toEqualTypeOf<'simple' | 'composite' | 'ellipsis' | 'singleton'>();
  expectTypeOf<AxisPattern>().toHaveProperty('position').toEqualTypeOf<Position>();
}

// =============================================================================
// Complex Pattern Type Tests
// =============================================================================

{
  // Test nested composite patterns
  const nestedComposite: CompositeAxis = {
    type: 'composite',
    axes: [
      {
        type: 'composite',
        axes: [
          { type: 'simple', name: 'a', position: { start: 2, end: 3 } },
          { type: 'simple', name: 'b', position: { start: 4, end: 5 } },
        ],
        position: { start: 1, end: 6 },
      },
      { type: 'simple', name: 'c', position: { start: 7, end: 8 } },
    ],
    position: { start: 0, end: 9 },
  };

  expectTypeOf<typeof nestedComposite>().toMatchTypeOf<AxisPattern>();
  expectTypeOf<(typeof nestedComposite.axes)[0]>().toMatchTypeOf<AxisPattern>();
}

{
  // Test mixed pattern types in composite
  const mixedComposite: CompositeAxis = {
    type: 'composite',
    axes: [
      { type: 'simple', name: 'batch', position: { start: 1, end: 6 } },
      { type: 'ellipsis', position: { start: 7, end: 10 } },
      { type: 'singleton', position: { start: 11, end: 12 } },
    ],
    position: { start: 0, end: 13 },
  };

  expectTypeOf<typeof mixedComposite>().toMatchTypeOf<AxisPattern>();
  expectTypeOf<typeof mixedComposite.axes>().toEqualTypeOf<readonly AxisPattern[]>();
}

// =============================================================================
// Type Inference Tests
// =============================================================================

{
  // Test type inference from pattern arrays
  const patterns: readonly AxisPattern[] = [
    { type: 'simple', name: 'h', position: { start: 0, end: 1 } },
    { type: 'ellipsis', position: { start: 2, end: 5 } },
    { type: 'singleton', position: { start: 6, end: 7 } },
  ];

  // Should be able to filter and narrow types
  const simplePatterns = patterns.filter((p): p is SimpleAxis => p.type === 'simple');
  expectTypeOf<typeof simplePatterns>().toEqualTypeOf<SimpleAxis[]>();

  const ellipsisPatterns = patterns.filter((p): p is EllipsisAxis => p.type === 'ellipsis');
  expectTypeOf<typeof ellipsisPatterns>().toEqualTypeOf<EllipsisAxis[]>();

  const singletonPatterns = patterns.filter((p): p is SingletonAxis => p.type === 'singleton');
  expectTypeOf<typeof singletonPatterns>().toEqualTypeOf<SingletonAxis[]>();

  const compositePatterns = patterns.filter((p): p is CompositeAxis => p.type === 'composite');
  expectTypeOf<typeof compositePatterns>().toEqualTypeOf<CompositeAxis[]>();
}
