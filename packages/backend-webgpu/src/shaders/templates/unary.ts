/**
 * WGSL shader generator for unary operations
 */

import { ShaderGenerator } from '../generator';
import { getWGSLType } from '../../utils';

export class UnaryShaderGenerator extends ShaderGenerator {
  generate(
    inputs: ReadonlyArray<any>,
    output: any,
    params?: { operation: string },
  ): string {
    if (inputs.length !== 1) {
      throw new Error('Unary operations require exactly one input');
    }

    // const input = inputs[0]!;
    const operation = params?.operation || 'neg';
    const wgslType = getWGSLType(output.__dtype);

    // Generate buffer declarations
    const bufferDecls = this.generateBufferDeclarations(inputs, output);

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
    
    // Apply unary operation
    let input_value = input0[index];
    output[index] = unary_op(input_value);
}`;

    return shader;
  }

  private generateOperationFunction(operation: string, wgslType: string): string {
    switch (operation) {
      case 'neg':
        return `fn unary_op(x: ${wgslType}) -> ${wgslType} { return -x; }`;
      
      case 'abs':
        return `fn unary_op(x: ${wgslType}) -> ${wgslType} { return abs(x); }`;
      
      case 'square':
        return `fn unary_op(x: ${wgslType}) -> ${wgslType} { return x * x; }`;
      
      case 'sqrt':
        return `fn unary_op(x: ${wgslType}) -> ${wgslType} { return sqrt(x); }`;
      
      case 'exp':
        return `fn unary_op(x: ${wgslType}) -> ${wgslType} { return exp(x); }`;
      
      case 'log':
        return `fn unary_op(x: ${wgslType}) -> ${wgslType} { return log(x); }`;
      
      case 'sin':
        return `fn unary_op(x: ${wgslType}) -> ${wgslType} { return sin(x); }`;
      
      case 'cos':
        return `fn unary_op(x: ${wgslType}) -> ${wgslType} { return cos(x); }`;
      
      default:
        throw new Error(`Unsupported unary operation: ${operation}`);
    }
  }
}