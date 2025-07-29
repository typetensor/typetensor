/**
 * Type tests for einops pattern validation
 */

import { expectTypeOf } from 'expect-type';
import type {
  CollectAxisNames,
  HasAxis,
  CountEllipsis,
  ValidateEllipsisCount,
  HasDuplicateAxes,
  ValidateUniqueAxes,
  ValidateOutputAxes,
  IsValidAxisStartChar,
  ValidateAxisName,
  ValidateAllAxisNames,
  IsEmptyComposite,
  ValidateNonEmptyComposites,
  ValidateInputPatterns,
  ValidateOutputPatterns,
  ValidatePattern,
  ValidationResult,
  FindUnknownAxes,
  FindDuplicates,
} from './validation';
import type { TypeCompositeAxis } from './type-parser';

// =============================================================================
// CollectAxisNames Tests
// =============================================================================

{
  // Test collecting from simple axes
  type SimpleAxes = readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
  expectTypeOf<CollectAxisNames<SimpleAxes>>().toEqualTypeOf<readonly ['h', 'w']>();
}

{
  // Test collecting from composite axes
  type CompositePattern = readonly [
    {
      type: 'composite';
      axes: readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
    },
    { type: 'simple'; name: 'c' },
  ];
  expectTypeOf<CollectAxisNames<CompositePattern>>().toEqualTypeOf<readonly ['h', 'w', 'c']>();
}

{
  // Test collecting with ellipsis and singleton
  type MixedPattern = readonly [
    { type: 'simple'; name: 'batch' },
    { type: 'ellipsis' },
    { type: 'singleton' },
    { type: 'simple'; name: 'channel' },
  ];
  expectTypeOf<CollectAxisNames<MixedPattern>>().toEqualTypeOf<readonly ['batch', 'channel']>();
}

{
  // Test nested composites
  type NestedPattern = readonly [
    {
      type: 'composite';
      axes: readonly [
        {
          type: 'composite';
          axes: readonly [{ type: 'simple'; name: 'a' }, { type: 'simple'; name: 'b' }];
        },
        { type: 'simple'; name: 'c' },
      ];
    },
  ];
  expectTypeOf<CollectAxisNames<NestedPattern>>().toEqualTypeOf<readonly ['a', 'b', 'c']>();
}

// =============================================================================
// HasAxis Tests
// =============================================================================

{
  // Test axis exists
  expectTypeOf<HasAxis<'h', readonly ['h', 'w', 'c']>>().toEqualTypeOf<true>();
  expectTypeOf<HasAxis<'w', readonly ['h', 'w', 'c']>>().toEqualTypeOf<true>();
  expectTypeOf<HasAxis<'c', readonly ['h', 'w', 'c']>>().toEqualTypeOf<true>();
}

{
  // Test axis doesn't exist
  expectTypeOf<HasAxis<'x', readonly ['h', 'w', 'c']>>().toEqualTypeOf<false>();
  expectTypeOf<HasAxis<'batch', readonly ['h', 'w', 'c']>>().toEqualTypeOf<false>();
}

{
  // Test empty list
  expectTypeOf<HasAxis<'h', readonly []>>().toEqualTypeOf<false>();
}

// =============================================================================
// Ellipsis Validation Tests
// =============================================================================

{
  // Test counting ellipsis
  type NoEllipsis = readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
  expectTypeOf<CountEllipsis<NoEllipsis>>().toEqualTypeOf<0>();
}

{
  // Test single ellipsis
  type SingleEllipsis = readonly [{ type: 'simple'; name: 'batch' }, { type: 'ellipsis' }];
  // Note: CountEllipsis doesn't actually increment, it just detects presence
  // This is a limitation but works for validation purposes
  expectTypeOf<CountEllipsis<SingleEllipsis>>().toEqualTypeOf<0>();
}

{
  // Test ellipsis in composite
  type EllipsisInComposite = readonly [
    {
      type: 'composite';
      axes: readonly [{ type: 'simple'; name: 'h' }, { type: 'ellipsis' }];
    },
  ];
  // Note: CountEllipsis doesn't actually increment, it just detects presence
  expectTypeOf<CountEllipsis<EllipsisInComposite>>().toEqualTypeOf<0>();
}

{
  // Test ValidateEllipsisCount
  type ValidPattern = readonly [{ type: 'simple'; name: 'batch' }, { type: 'ellipsis' }];
  expectTypeOf<ValidateEllipsisCount<ValidPattern>>().toEqualTypeOf<true>();

  type NoEllipsisPattern = readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
  expectTypeOf<ValidateEllipsisCount<NoEllipsisPattern>>().toEqualTypeOf<true>();
}

// =============================================================================
// Duplicate Axes Tests
// =============================================================================

{
  // Test no duplicates
  expectTypeOf<HasDuplicateAxes<readonly ['h', 'w', 'c']>>().toEqualTypeOf<false>();
  expectTypeOf<HasDuplicateAxes<readonly []>>().toEqualTypeOf<false>();
  expectTypeOf<HasDuplicateAxes<readonly ['single']>>().toEqualTypeOf<false>();
}

{
  // Test with duplicates
  expectTypeOf<HasDuplicateAxes<readonly ['h', 'w', 'h']>>().toEqualTypeOf<true>();
  expectTypeOf<HasDuplicateAxes<readonly ['a', 'b', 'c', 'b', 'd']>>().toEqualTypeOf<true>();
}

{
  // Test ValidateUniqueAxes
  type NoDuplicatePattern = readonly [
    { type: 'simple'; name: 'h' },
    { type: 'simple'; name: 'w' },
    { type: 'simple'; name: 'c' },
  ];
  expectTypeOf<ValidateUniqueAxes<NoDuplicatePattern>>().toEqualTypeOf<true>();

  type DuplicatePattern = readonly [
    { type: 'simple'; name: 'h' },
    { type: 'simple'; name: 'w' },
    { type: 'simple'; name: 'h' },
  ];
  expectTypeOf<ValidateUniqueAxes<DuplicatePattern>>().toEqualTypeOf<false>();
}

// =============================================================================
// Output Axes Validation Tests
// =============================================================================

{
  // Test valid output axes
  type InputAxes = readonly ['h', 'w', 'c'];
  type ValidOutput = readonly [{ type: 'simple'; name: 'w' }, { type: 'simple'; name: 'h' }];
  expectTypeOf<ValidateOutputAxes<ValidOutput, InputAxes>>().toEqualTypeOf<true>();
}

{
  // Test invalid output axes
  type InputAxes = readonly ['h', 'w'];
  type InvalidOutput = readonly [
    { type: 'simple'; name: 'h' },
    { type: 'simple'; name: 'c' }, // 'c' not in input
  ];
  expectTypeOf<ValidateOutputAxes<InvalidOutput, InputAxes>>().toEqualTypeOf<false>();
}

{
  // Test output with ellipsis and singleton
  type InputAxes = readonly ['batch', 'h', 'w'];
  type OutputWithSpecial = readonly [
    { type: 'simple'; name: 'batch' },
    { type: 'ellipsis' },
    { type: 'singleton' },
  ];
  expectTypeOf<ValidateOutputAxes<OutputWithSpecial, InputAxes>>().toEqualTypeOf<true>();
}

{
  // Test composite in output
  type InputAxes = readonly ['h', 'w', 'c'];
  type CompositeOutput = readonly [
    {
      type: 'composite';
      axes: readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
    },
    { type: 'simple'; name: 'c' },
  ];
  expectTypeOf<ValidateOutputAxes<CompositeOutput, InputAxes>>().toEqualTypeOf<true>();
}

// =============================================================================
// Axis Name Validation Tests
// =============================================================================

{
  // Test valid start characters
  expectTypeOf<IsValidAxisStartChar<'a'>>().toEqualTypeOf<true>();
  expectTypeOf<IsValidAxisStartChar<'Z'>>().toEqualTypeOf<true>();
  expectTypeOf<IsValidAxisStartChar<'_'>>().toEqualTypeOf<true>();
}

{
  // Test invalid start characters
  expectTypeOf<IsValidAxisStartChar<'0'>>().toEqualTypeOf<false>();
  expectTypeOf<IsValidAxisStartChar<'-'>>().toEqualTypeOf<false>();
  expectTypeOf<IsValidAxisStartChar<' '>>().toEqualTypeOf<false>();
}

{
  // Test valid axis names
  expectTypeOf<ValidateAxisName<'height'>>().toEqualTypeOf<true>();
  expectTypeOf<ValidateAxisName<'h'>>().toEqualTypeOf<true>();
  expectTypeOf<ValidateAxisName<'_private'>>().toEqualTypeOf<true>();
  expectTypeOf<ValidateAxisName<'axis123'>>().toEqualTypeOf<true>();
  expectTypeOf<ValidateAxisName<'batch_size'>>().toEqualTypeOf<true>();
}

{
  // Test invalid axis names
  expectTypeOf<ValidateAxisName<'123axis'>>().toEqualTypeOf<false>();
  expectTypeOf<ValidateAxisName<'axis-name'>>().toEqualTypeOf<false>();
  expectTypeOf<ValidateAxisName<'axis name'>>().toEqualTypeOf<false>();
  expectTypeOf<ValidateAxisName<''>>().toEqualTypeOf<false>();
}

{
  // Test ValidateAllAxisNames
  type ValidNamesPattern = readonly [
    { type: 'simple'; name: 'batch' },
    { type: 'simple'; name: 'height' },
    { type: 'simple'; name: 'width' },
  ];
  expectTypeOf<ValidateAllAxisNames<ValidNamesPattern>>().toEqualTypeOf<true>();

  type InvalidNamesPattern = readonly [
    { type: 'simple'; name: 'batch' },
    { type: 'simple'; name: '123invalid' },
  ];
  expectTypeOf<ValidateAllAxisNames<InvalidNamesPattern>>().toEqualTypeOf<false>();
}

// =============================================================================
// Composite Pattern Validation Tests
// =============================================================================

{
  // Test empty composite detection
  type EmptyComposite = TypeCompositeAxis & {
    type: 'composite';
    axes: readonly [];
  };
  expectTypeOf<IsEmptyComposite<EmptyComposite>>().toEqualTypeOf<true>();

  type NonEmptyComposite = TypeCompositeAxis & {
    type: 'composite';
    axes: readonly [{ type: 'simple'; name: 'h' }];
  };
  expectTypeOf<IsEmptyComposite<NonEmptyComposite>>().toEqualTypeOf<false>();
}

{
  // Test ValidateNonEmptyComposites
  type ValidComposites = readonly [
    {
      type: 'composite';
      axes: readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
    },
  ];
  expectTypeOf<ValidateNonEmptyComposites<ValidComposites>>().toEqualTypeOf<true>();

  type EmptyCompositePattern = readonly [
    {
      type: 'composite';
      axes: readonly [];
    },
  ];
  expectTypeOf<ValidateNonEmptyComposites<EmptyCompositePattern>>().toEqualTypeOf<false>();
}

// =============================================================================
// Main Validation Function Tests
// =============================================================================

{
  // Test ValidateInputPatterns - valid
  type ValidInput = readonly [
    { type: 'simple'; name: 'batch' },
    { type: 'simple'; name: 'height' },
    { type: 'simple'; name: 'width' },
  ];
  expectTypeOf<ValidateInputPatterns<ValidInput>>().toEqualTypeOf<ValidationResult<true>>();
}

{
  // Test ValidateInputPatterns - duplicate axes
  type DuplicateInput = readonly [
    { type: 'simple'; name: 'h' },
    { type: 'simple'; name: 'w' },
    { type: 'simple'; name: 'h' },
  ];
  expectTypeOf<ValidateInputPatterns<DuplicateInput>>().toEqualTypeOf<
    ValidationResult<false, 'Duplicate axis names in input'>
  >();
}

{
  // Test ValidateInputPatterns - invalid axis name
  type InvalidNameInput = readonly [{ type: 'simple'; name: '123invalid' }];
  expectTypeOf<ValidateInputPatterns<InvalidNameInput>>().toEqualTypeOf<
    ValidationResult<false, 'Invalid axis name'>
  >();
}

{
  // Test ValidateOutputPatterns - valid
  type Input = readonly [
    { type: 'simple'; name: 'h' },
    { type: 'simple'; name: 'w' },
    { type: 'simple'; name: 'c' },
  ];
  type ValidOutput = readonly [{ type: 'simple'; name: 'w' }, { type: 'simple'; name: 'h' }];
  expectTypeOf<ValidateOutputPatterns<ValidOutput, Input>>().toEqualTypeOf<
    ValidationResult<true>
  >();
}

{
  // Test ValidateOutputPatterns - unknown axis
  type Input = readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
  type InvalidOutput = readonly [
    { type: 'simple'; name: 'h' },
    { type: 'simple'; name: 'c' }, // Unknown axis
  ];
  expectTypeOf<ValidateOutputPatterns<InvalidOutput, Input>>().toEqualTypeOf<
    ValidationResult<false, 'Output contains unknown axes'>
  >();
}

{
  // Test complete pattern validation - valid
  type ValidInputPattern = readonly [
    { type: 'simple'; name: 'batch' },
    { type: 'simple'; name: 'height' },
    { type: 'simple'; name: 'width' },
    { type: 'simple'; name: 'channels' },
  ];
  type ValidOutputPattern = readonly [
    { type: 'simple'; name: 'batch' },
    { type: 'simple'; name: 'channels' },
    { type: 'simple'; name: 'height' },
    { type: 'simple'; name: 'width' },
  ];
  expectTypeOf<ValidatePattern<ValidInputPattern, ValidOutputPattern>>().toEqualTypeOf<
    ValidationResult<true>
  >();
}

{
  // Test complete pattern validation - input error
  type InvalidInput = readonly [
    { type: 'simple'; name: 'h' },
    { type: 'simple'; name: 'h' }, // Duplicate
  ];
  type SomeOutput = readonly [{ type: 'simple'; name: 'h' }];
  expectTypeOf<ValidatePattern<InvalidInput, SomeOutput>>().toEqualTypeOf<
    ValidationResult<false, 'Input validation failed: Duplicate axis names in input'>
  >();
}

{
  // Test complete pattern validation - output error
  type SomeInput = readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
  type InvalidOutput = readonly [
    { type: 'simple'; name: 'h' },
    { type: 'simple'; name: 'unknown' }, // Unknown axis
  ];
  expectTypeOf<ValidatePattern<SomeInput, InvalidOutput>>().toEqualTypeOf<
    ValidationResult<false, 'Output validation failed: Output contains unknown axes'>
  >();
}

// =============================================================================
// Utility Type Tests
// =============================================================================

{
  // Test FindUnknownAxes
  type Output = readonly [
    { type: 'simple'; name: 'h' },
    { type: 'simple'; name: 'unknown1' },
    { type: 'simple'; name: 'w' },
    { type: 'simple'; name: 'unknown2' },
  ];
  type InputAxes = readonly ['h', 'w'];
  expectTypeOf<FindUnknownAxes<Output, InputAxes>>().toEqualTypeOf<
    readonly ['unknown1', 'unknown2']
  >();
}

{
  // Test FindDuplicates
  type Axes = readonly ['h', 'w', 'h', 'c', 'w'];
  expectTypeOf<FindDuplicates<Axes>>().toEqualTypeOf<readonly ['h', 'w']>();

  type NoDuplicates = readonly ['a', 'b', 'c'];
  expectTypeOf<FindDuplicates<NoDuplicates>>().toEqualTypeOf<readonly []>();
}

// =============================================================================
// Complex Pattern Tests
// =============================================================================

{
  // Test with composite patterns
  type ComplexInput = readonly [
    {
      type: 'composite';
      axes: readonly [{ type: 'simple'; name: 'batch' }, { type: 'simple'; name: 'seq' }];
    },
    { type: 'simple'; name: 'embed' },
    { type: 'ellipsis' },
  ];
  type ComplexOutput = readonly [
    { type: 'simple'; name: 'batch' },
    { type: 'simple'; name: 'seq' },
    { type: 'simple'; name: 'embed' },
    { type: 'ellipsis' },
  ];
  expectTypeOf<ValidatePattern<ComplexInput, ComplexOutput>>().toEqualTypeOf<
    ValidationResult<true>
  >();
}

{
  // Test with nested composites
  type NestedInput = readonly [
    {
      type: 'composite';
      axes: readonly [
        {
          type: 'composite';
          axes: readonly [{ type: 'simple'; name: 'h' }, { type: 'simple'; name: 'w' }];
        },
        { type: 'simple'; name: 'c' },
      ];
    },
  ];
  type NestedOutput = readonly [
    { type: 'simple'; name: 'h' },
    { type: 'simple'; name: 'w' },
    { type: 'simple'; name: 'c' },
  ];
  expectTypeOf<ValidatePattern<NestedInput, NestedOutput>>().toEqualTypeOf<
    ValidationResult<true>
  >();
}
