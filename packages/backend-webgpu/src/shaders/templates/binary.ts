/**
 * WGSL shader generator for binary operations
 */

import { ShaderGenerator } from '../generator';
import { getWGSLType } from '../../utils';

export class BinaryShaderGenerator extends ShaderGenerator {
  generate(
    inputs: ReadonlyArray<any>,
    output: any,
    params?: { operation: string },
  ): string {
    if (inputs.length !== 2) {
      throw new Error('Binary operations require exactly two inputs');
    }

    const inputA = inputs[0]!;
    const inputB = inputs[1]!;
    const operation = params?.operation || 'add';
    const wgslType = getWGSLType(output.__dtype);

    // Check if broadcasting is needed
    const needsBroadcasting = !this.shapesEqual(inputA.__shape, inputB.__shape) ||
                             !this.shapesEqual(inputA.__shape, output.__shape);

    if (needsBroadcasting) {
      return this.generateBroadcastingShader(inputA, inputB, output, operation, wgslType);
    } else {
      return this.generateSimpleShader(inputA, inputB, output, operation, wgslType);
    }
  }

  private generateSimpleShader(
    inputA: any,
    inputB: any,
    output: any,
    operation: string,
    wgslType: string,
  ): string {
    // Generate buffer declarations
    const bufferDecls = this.generateBufferDeclarations([inputA, inputB], output);

    // Generate operation-specific function
    const opFunction = this.generateOperationFunction(operation, wgslType);

    // Generate main shader
    const shader = `
${bufferDecls}

${opFunction}

@compute @workgroup_size(${this.workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds check
    if (index >= arrayLength(&output)) {
        return;
    }
    
    // Apply binary operation
    let a = input0[index];
    let b = input1[index];
    output[index] = binary_op(a, b);
}`;

    return shader;
  }

  private generateBroadcastingShader(
    inputA: any,
    inputB: any,
    output: any,
    operation: string,
    wgslType: string,
  ): string {
    // Generate buffer declarations
    const bufferDecls = this.generateBufferDeclarations([inputA, inputB], output);

    // Generate shape constants
    const shapeConstants = this.generateShapeConstants([inputA, inputB], output);

    // Generate index functions
    const indexFunctions = this.generateIndexFunctions();

    // Generate broadcasting logic
    const broadcastingLogic = this.generateBroadcastingIndexCalculation(
      inputA.__shape,
      inputB.__shape,
      output.__shape,
    );

    // Generate operation-specific function
    const opFunction = this.generateOperationFunction(operation, wgslType);

    // Generate main shader
    const shader = `
${bufferDecls}
${shapeConstants}
${indexFunctions}
${broadcastingLogic}
${opFunction}

@compute @workgroup_size(${this.workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_index = global_id.x;
    
    // Bounds check
    if (output_index >= output_size) {
        return;
    }
    
    // Calculate indices for broadcasting
    let indices = calculate_broadcast_indices(output_index);
    
    // Get values from inputs
    let a = input0[indices.x];
    let b = input1[indices.y];
    
    // Apply binary operation
    output[output_index] = binary_op(a, b);
}`;

    return shader;
  }

  private generateOperationFunction(operation: string, wgslType: string): string {
    switch (operation) {
      case 'add':
        return `fn binary_op(a: ${wgslType}, b: ${wgslType}) -> ${wgslType} { return a + b; }`;
      
      case 'sub':
        return `fn binary_op(a: ${wgslType}, b: ${wgslType}) -> ${wgslType} { return a - b; }`;
      
      case 'mul':
        return `fn binary_op(a: ${wgslType}, b: ${wgslType}) -> ${wgslType} { return a * b; }`;
      
      case 'div':
        return `fn binary_op(a: ${wgslType}, b: ${wgslType}) -> ${wgslType} { return a / b; }`;
      
      default:
        throw new Error(`Unsupported binary operation: ${operation}`);
    }
  }

  private generateBroadcastingIndexCalculation(
    shapeA: ReadonlyArray<number>,
    shapeB: ReadonlyArray<number>,
    outputShape: ReadonlyArray<number>,
  ): string {
    const rank = outputShape.length;
    const rankA = shapeA.length;
    const rankB = shapeB.length;

    let code = `
// Calculate input indices from output index with broadcasting
fn calculate_broadcast_indices(output_index: u32) -> vec2<u32> {
    var output_coords: array<u32, ${rank}>;
    var temp_index = output_index;
    
    // Convert linear index to coordinates
    for (var i = ${rank - 1}; i >= 0; i--) {
        output_coords[i] = temp_index % output_shape[i];
        temp_index = temp_index / output_shape[i];
    }
    
    // Calculate index for input A
    var index_a = 0u;`;

    if (rankA > 0) {
      code += `
    var stride_a = 1u;
    for (var i = ${rankA - 1}; i >= 0; i--) {
        let coord_idx = i + ${rank - rankA};
        let dim_a = input0_shape[i];
        if (dim_a > 1u) {
            index_a += output_coords[coord_idx] * stride_a;
        }
        stride_a *= dim_a;
    }`;
    }

    code += `
    
    // Calculate index for input B
    var index_b = 0u;`;

    if (rankB > 0) {
      code += `
    var stride_b = 1u;
    for (var i = ${rankB - 1}; i >= 0; i--) {
        let coord_idx = i + ${rank - rankB};
        let dim_b = input1_shape[i];
        if (dim_b > 1u) {
            index_b += output_coords[coord_idx] * stride_b;
        }
        stride_b *= dim_b;
    }`;
    }

    code += `
    
    return vec2<u32>(index_a, index_b);
}`;

    return code;
  }

  private shapesEqual(shape1: ReadonlyArray<number>, shape2: ReadonlyArray<number>): boolean {
    if (shape1.length !== shape2.length) return false;
    for (let i = 0; i < shape1.length; i++) {
      if (shape1[i] !== shape2[i]) return false;
    }
    return true;
  }
}