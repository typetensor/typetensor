/**
 * Integration tests for type-parser and validation
 */

import { expectTypeOf } from 'expect-type';
import type { ParsePattern, TypeParseError } from './type-parser';
import type { ValidatePattern, ValidationResult } from './validation';

// =============================================================================
// Valid Patterns with Validation
// =============================================================================

{
  // Test simple transpose pattern
  type Pattern = ParsePattern<'h w -> w h'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<true>>();
}

{
  // Test pattern with ellipsis
  type Pattern = ParsePattern<'batch ... height -> height batch ...'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<true>>();
}

{
  // Test composite pattern
  type Pattern = ParsePattern<'(h w) c -> h w c'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<true>>();
}

// =============================================================================
// Invalid Patterns with Validation
// =============================================================================

{
  // Test unknown axis in output
  type Pattern = ParsePattern<'h w -> h w c'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<false, 'Output validation failed: Output contains unknown axes'>>();
}

{
  // Test duplicate axes in input
  type Pattern = ParsePattern<'h h w -> h w'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<false, 'Input validation failed: Duplicate axis names in input'>>();
}

{
  // Test duplicate axes in output
  type Pattern = ParsePattern<'h w c -> h h c'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<false, 'Output validation failed: Duplicate axis names in output'>>();
}

// =============================================================================
// Parse Errors Should Not Reach Validation
// =============================================================================

{
  // Test parse error - missing arrow
  type Pattern = ParsePattern<'h w c'>;
  expectTypeOf<Pattern>().toMatchTypeOf<TypeParseError<string>>();
}

{
  // Test parse error - unbalanced parentheses
  type Pattern = ParsePattern<'(h w -> h w'>;
  expectTypeOf<Pattern>().toMatchTypeOf<TypeParseError<string>>();
}

{
  // Test parse error - empty axis
  type Pattern = ParsePattern<' -> h'>;
  expectTypeOf<Pattern>().toMatchTypeOf<TypeParseError<string>>();
}

// =============================================================================
// Complex Valid Patterns
// =============================================================================

{
  // Test image transpose pattern
  type Pattern = ParsePattern<'batch height width channels -> batch channels height width'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<true>>();
}

{
  // Test attention pattern
  type Pattern = ParsePattern<'(batch seq) heads dims -> batch seq heads dims'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<true>>();
}

{
  // Test pattern with all features
  type Pattern = ParsePattern<'(batch seq) embed ... 1 -> batch seq embed ... 1'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<true>>();
}

// =============================================================================
// Edge Cases
// =============================================================================

{
  // Test single axis pattern
  type Pattern = ParsePattern<'a -> a'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<true>>();
}

{
  // Test ellipsis only pattern
  type Pattern = ParsePattern<'... -> ...'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<true>>();
}

{
  // Test singleton addition
  type Pattern = ParsePattern<'h w -> h w 1'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<true>>();
}

// =============================================================================
// Composite Validation
// =============================================================================

{
  // Test empty composite (should fail)
  type Pattern = ParsePattern<'() -> h'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<false, 'Input validation failed: Empty composite patterns not allowed'>>();
}

{
  // Test output with unknown composite axis
  type Pattern = ParsePattern<'h w -> (h c)'>;
  type Valid = Pattern extends TypeParseError<string> ? false : ValidatePattern<Pattern['input'], Pattern['output']>;
  
  expectTypeOf<Valid>().toEqualTypeOf<ValidationResult<false, 'Output validation failed: Output contains unknown axes'>>();
}