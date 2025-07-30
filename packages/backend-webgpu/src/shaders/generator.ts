/**
 * WGSL shader code generator for tensor operations
 */

// import type { DType } from '@typetensor/core';
import { getWGSLType } from '../utils';

export interface ShaderGeneratorOptions {
  workgroupSize?: number;
  useVec4?: boolean;
}

/**
 * Base class for shader generation
 */
export abstract class ShaderGenerator {
  protected workgroupSize: number;
  protected useVec4: boolean;

  constructor(options: ShaderGeneratorOptions = {}) {
    this.workgroupSize = options.workgroupSize ?? 64;
    this.useVec4 = options.useVec4 ?? false;
  }

  /**
   * Generate complete shader code
   */
  abstract generate(
    inputs: ReadonlyArray<any>,
    output: any,
    params?: Record<string, any>,
  ): string;

  /**
   * Generate buffer declarations
   */
  protected generateBufferDeclarations(
    inputs: ReadonlyArray<any>,
    output: any,
  ): string {
    let code = '';

    // Input buffers
    inputs.forEach((input, i) => {
      const wgslType = getWGSLType(input.__dtype);
      code += `@group(0) @binding(${i}) var<storage, read> input${i}: array<${wgslType}>;\n`;
    });

    // Output buffer
    const outputWgslType = getWGSLType(output.__dtype);
    const outputBinding = inputs.length;
    code += `@group(0) @binding(${outputBinding}) var<storage, read_write> output: array<${outputWgslType}>;\n`;

    return code;
  }

  /**
   * Generate shape constants
   */
  protected generateShapeConstants(
    inputs: ReadonlyArray<any>,
    output: any,
  ): string {
    let code = '\n// Shape constants\n';

    // Input shapes
    inputs.forEach((input, i) => {
      code += `const input${i}_shape = vec${input.__shape.length}<u32>(${input.__shape.join(', ')});\n`;
      code += `const input${i}_strides = vec${input.__strides.length}<u32>(${input.__strides.join(', ')});\n`;
    });

    // Output shape
    code += `const output_shape = vec${output.__shape.length}<u32>(${output.__shape.join(', ')});\n`;
    code += `const output_strides = vec${output.__strides.length}<u32>(${output.__strides.join(', ')});\n`;
    code += `const output_size = ${output.__size}u;\n`;

    return code;
  }

  /**
   * Generate index calculation functions
   */
  protected generateIndexFunctions(): string {
    return `
// Convert linear index to multi-dimensional indices
fn index_to_indices(index: u32, shape: vec2<u32>, strides: vec2<u32>) -> vec2<u32> {
    let i0 = index / strides.x;
    let i1 = (index % strides.x) / strides.y;
    return vec2<u32>(i0, i1);
}

fn index_to_indices3(index: u32, shape: vec3<u32>, strides: vec3<u32>) -> vec3<u32> {
    let i0 = index / strides.x;
    let i1 = (index % strides.x) / strides.y;
    let i2 = (index % strides.y) / strides.z;
    return vec3<u32>(i0, i1, i2);
}

fn index_to_indices4(index: u32, shape: vec4<u32>, strides: vec4<u32>) -> vec4<u32> {
    let i0 = index / strides.x;
    let i1 = (index % strides.x) / strides.y;
    let i2 = (index % strides.y) / strides.z;
    let i3 = (index % strides.z) / strides.w;
    return vec4<u32>(i0, i1, i2, i3);
}

// Convert multi-dimensional indices to linear index
fn indices_to_index(indices: vec2<u32>, strides: vec2<u32>) -> u32 {
    return indices.x * strides.x + indices.y * strides.y;
}

fn indices_to_index3(indices: vec3<u32>, strides: vec3<u32>) -> u32 {
    return indices.x * strides.x + indices.y * strides.y + indices.z * strides.z;
}

fn indices_to_index4(indices: vec4<u32>, strides: vec4<u32>) -> u32 {
    return indices.x * strides.x + indices.y * strides.y + indices.z * strides.z + indices.w * strides.w;
}`;
  }

  /**
   * Generate compute shader entry point
   */
  protected generateEntryPoint(body: string): string {
    return `
@compute @workgroup_size(${this.workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= output_size) {
        return;
    }
    
${body}
}`;
  }

  /**
   * Generate broadcasting logic for binary operations
   */
  protected generateBroadcastingLogic(
    shape1: ReadonlyArray<number>,
    shape2: ReadonlyArray<number>,
    outputShape: ReadonlyArray<number>,
  ): string {
    // Pad shapes to same length
    const maxRank = Math.max(shape1.length, shape2.length, outputShape.length);
    const paddedShape1 = this.padShape(shape1, maxRank);
    const paddedShape2 = this.padShape(shape2, maxRank);

    let code = '\n// Broadcasting logic\n';
    code += `fn broadcast_indices(output_indices: vec${maxRank}<u32>) -> vec2<u32> {\n`;
    code += '    var idx1 = 0u;\n';
    code += '    var idx2 = 0u;\n';

    // Generate index calculation considering broadcasting
    for (let i = 0; i < maxRank; i++) {
      const dim1 = paddedShape1[i];
      const dim2 = paddedShape2[i];
      
      if (dim1 === 1) {
        code += `    // Dimension ${i}: broadcast input1\n`;
      } else {
        code += `    idx1 = idx1 * ${dim1}u + output_indices[${i}];\n`;
      }

      if (dim2 === 1) {
        code += `    // Dimension ${i}: broadcast input2\n`;
      } else {
        code += `    idx2 = idx2 * ${dim2}u + output_indices[${i}];\n`;
      }
    }

    code += '    return vec2<u32>(idx1, idx2);\n';
    code += '}\n';

    return code;
  }

  /**
   * Pad shape to target length
   */
  private padShape(shape: ReadonlyArray<number>, targetLength: number): number[] {
    const padded = new Array(targetLength).fill(1);
    const offset = targetLength - shape.length;
    for (let i = 0; i < shape.length; i++) {
      padded[offset + i] = shape[i];
    }
    return padded;
  }
}