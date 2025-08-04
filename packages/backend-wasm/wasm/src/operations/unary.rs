/*!
 * Unary operations implementation for WebAssembly backend
 * 
 * Provides optimized implementations of element-wise unary operations
 * with SIMD support where available, as well as dispatching to specialized
 * operation modules for view, reduction, and softmax operations.
 */

use crate::types::{WasmOperation, WasmTensorMeta, WasmDType, WasmResult, WasmError};
use crate::memory::WasmTensor;
use crate::arena::TempArena;
use crate::simd::{float32, float64};
use crate::operations::{view, reduction, softmax};

// Use micromath for fast approximations when available
#[cfg(target_arch = "wasm32")]
use micromath::F32Ext as _;

// Import our fast math functions
use crate::fast_math::{fast_sin_f32, fast_cos_f32};

/// Execute a unary operation
pub fn execute_unary_op(
    operation: WasmOperation,
    input: &WasmTensor,
    output: &WasmTensor,
    arena: &TempArena,
) -> WasmResult<()> {
    // Dispatch to specialized operation modules based on operation type
    match operation {
        // View operations
        WasmOperation::Reshape | WasmOperation::View | WasmOperation::Slice | WasmOperation::Flatten |
        WasmOperation::Permute | WasmOperation::Transpose | WasmOperation::Squeeze | 
        WasmOperation::Unsqueeze | WasmOperation::Expand | WasmOperation::Tile => {
            view::execute_view_op(operation, input, output, arena)
        }
        
        // Reduction operations
        WasmOperation::Sum | WasmOperation::Mean | WasmOperation::Max | WasmOperation::Min | WasmOperation::Prod => {
            reduction::execute_reduction_op(operation, input, output, arena)
        }
        
        // Softmax operations
        WasmOperation::Softmax | WasmOperation::LogSoftmax => {
            // For now, use default axis (last dimension = -1)
            softmax::execute_softmax_op(operation, input, output, arena, Some(-1))
        }
        
        // Element-wise unary operations (original implementation)
        WasmOperation::Neg | WasmOperation::Abs | WasmOperation::Sin | WasmOperation::Cos |
        WasmOperation::Exp | WasmOperation::Log | WasmOperation::Sqrt | WasmOperation::Square => {
            execute_elementwise_unary_op(operation, input, output, arena)
        }
        
        _ => Err(WasmError::NotImplemented),
    }
}

/// Execute element-wise unary operation (original unary implementation)
fn execute_elementwise_unary_op(
    operation: WasmOperation,
    input: &WasmTensor,
    output: &WasmTensor,
    arena: &TempArena,
) -> WasmResult<()> {
    let input_ptr = input.get_read_ptr(arena);
    
    // SAFETY: Casting *const u8 to *mut u8 for output tensor
    // This is safe because:
    // 1. Each tensor has unique, non-overlapping memory allocated by the bump allocator
    // 2. The output tensor is semantically intended to be written to
    // 3. We only have immutable arena access (&TempArena) because operations don't
    //    need to allocate new memory, just access existing allocations
    // 4. No aliasing occurs - input and output tensors have different memory regions
    // 5. The arena's bump allocator ensures memory validity and proper alignment
    let output_ptr = output.get_read_ptr(arena) as *mut u8;
    
    let input_meta = input.metadata();
    let size = input_meta.size();

    match input_meta.dtype() {
        WasmDType::Float32 => {
            let input_slice = unsafe { std::slice::from_raw_parts(input_ptr as *const f32, size) };
            let output_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, size) };
            execute_unary_f32(operation, input_slice, output_slice)?;
        }
        WasmDType::Float64 => {
            let input_slice = unsafe { std::slice::from_raw_parts(input_ptr as *const f64, size) };
            let output_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f64, size) };
            execute_unary_f64(operation, input_slice, output_slice)?;
        }
        WasmDType::Int32 => {
            let input_slice = unsafe { std::slice::from_raw_parts(input_ptr as *const i32, size) };
            let output_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut i32, size) };
            execute_unary_i32(operation, input_slice, output_slice)?;
        }
        WasmDType::Int8 => {
            let input_slice = unsafe { std::slice::from_raw_parts(input_ptr as *const i8, size) };
            let output_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut i8, size) };
            execute_unary_i8(operation, input_slice, output_slice)?;
        }
        _ => return Err(WasmError::NotImplemented),
    }

    Ok(())
}

/// Execute unary operation on f32 arrays
fn execute_unary_f32(
    operation: WasmOperation,
    input: &[f32],
    output: &mut [f32],
) -> WasmResult<()> {
    match operation {
        WasmOperation::Neg => {
            // Use SIMD-optimized negation
            float32::simd_neg(input, output);
        }
        WasmOperation::Abs => {
            // Use SIMD-optimized absolute value
            float32::simd_abs(input, output);
        }
        WasmOperation::Sqrt => {
            // Use libm for accurate sqrt (we can optimize later with tolerance testing)
            for (i, &val) in input.iter().enumerate() {
                output[i] = libm::sqrtf(val);
            }
        }
        WasmOperation::Sin => {
            // Use fast lookup table-based implementation
            for (i, &val) in input.iter().enumerate() {
                output[i] = fast_sin_f32(val);
            }
        }
        WasmOperation::Cos => {
            // Use fast lookup table-based implementation
            for (i, &val) in input.iter().enumerate() {
                output[i] = fast_cos_f32(val);
            }
        }
        WasmOperation::Exp => {
            // Use libm for accurate exp (we can optimize later with tolerance testing)
            for (i, &val) in input.iter().enumerate() {
                output[i] = libm::expf(val);
            }
        }
        WasmOperation::Log => {
            // Use libm for accurate log (we can optimize later with tolerance testing)
            for (i, &val) in input.iter().enumerate() {
                output[i] = if val > 0.0 { 
                    libm::logf(val) 
                } else if val == 0.0 {
                    f32::NEG_INFINITY
                } else {
                    f32::NAN
                };
            }
        }
        WasmOperation::Square => {
            // Use SIMD multiplication for squaring
            float32::simd_mul(input, input, output);
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    Ok(())
}

/// Execute unary operation on f64 arrays
fn execute_unary_f64(
    operation: WasmOperation,
    input: &[f64],
    output: &mut [f64],
) -> WasmResult<()> {
    match operation {
        WasmOperation::Neg => {
            // Use SIMD-optimized negation
            float64::simd_neg(input, output);
        }
        WasmOperation::Abs => {
            // Use SIMD-optimized absolute value
            float64::simd_abs(input, output);
        }
        WasmOperation::Sqrt => {
            // Use SIMD-optimized square root
            float64::simd_sqrt(input, output);
        }
        WasmOperation::Sin => {
            // Keep scalar implementation for complex math functions
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.sin();
            }
        }
        WasmOperation::Cos => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.cos();
            }
        }
        WasmOperation::Exp => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.exp();
            }
        }
        WasmOperation::Log => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = if val > 0.0 { val.ln() } else { f64::NAN };
            }
        }
        WasmOperation::Square => {
            // Implement efficient squaring - we'll need to create a helper for this
            for (i, &val) in input.iter().enumerate() {
                output[i] = val * val;
            }
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    Ok(())
}

/// Execute unary operation on i32 arrays
fn execute_unary_i32(
    operation: WasmOperation,
    input: &[i32],
    output: &mut [i32],
) -> WasmResult<()> {
    match operation {
        WasmOperation::Neg => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.wrapping_neg();
            }
        }
        WasmOperation::Abs => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.abs();
            }
        }
        WasmOperation::Square => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.wrapping_mul(val);
            }
        }
        // Math functions not supported for integer types
        WasmOperation::Sin | WasmOperation::Cos | WasmOperation::Exp | 
        WasmOperation::Log | WasmOperation::Sqrt => {
            return Err(WasmError::InvalidOperation);
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    Ok(())
}

/// Execute unary operation on i8 arrays
fn execute_unary_i8(
    operation: WasmOperation,
    input: &[i8],
    output: &mut [i8],
) -> WasmResult<()> {
    match operation {
        WasmOperation::Neg => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.wrapping_neg();
            }
        }
        WasmOperation::Abs => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.abs();
            }
        }
        WasmOperation::Square => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.wrapping_mul(val);
            }
        }
        // Math functions not supported for integer types
        WasmOperation::Sin | WasmOperation::Cos | WasmOperation::Exp | 
        WasmOperation::Log | WasmOperation::Sqrt => {
            return Err(WasmError::InvalidOperation);
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unary_neg_f32() {
        let input = vec![1.0f32, -2.0, 3.0, -4.0];
        let mut output = vec![0.0f32; 4];
        
        execute_unary_f32(WasmOperation::Neg, &input, &mut output).unwrap();
        
        assert_eq!(output, vec![-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn test_unary_abs_f32() {
        let input = vec![1.0f32, -2.0, 3.0, -4.0];
        let mut output = vec![0.0f32; 4];
        
        execute_unary_f32(WasmOperation::Abs, &input, &mut output).unwrap();
        
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_unary_square_f32() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        
        execute_unary_f32(WasmOperation::Square, &input, &mut output).unwrap();
        
        assert_eq!(output, vec![1.0, 4.0, 9.0, 16.0]);
    }
}