/*!
 * Unary operations implementation for WebAssembly backend
 * 
 * Provides optimized implementations of element-wise unary operations
 * with SIMD support where available.
 */

use crate::types::{WasmOperation, WasmTensorMeta, WasmDType, WasmResult, WasmError};
use crate::memory::{WasmMemoryManager, WasmBufferHandle};
use crate::simd::{float32, float64};

/// Execute a unary operation
pub fn execute_unary_op(
    memory_manager: &mut WasmMemoryManager,
    operation: WasmOperation,
    input: &WasmBufferHandle,
    input_meta: &WasmTensorMeta,
    output: &WasmBufferHandle,
    _output_meta: &WasmTensorMeta,
) -> WasmResult<()> {
    let input_ptr = memory_manager.get_read_ptr(input);
    let output_ptr = memory_manager.get_write_ptr(output);
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
            // Use SIMD-optimized square root
            float32::simd_sqrt(input, output);
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
                output[i] = if val > 0.0 { val.ln() } else { f32::NAN };
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
    use crate::memory::WasmMemoryManager;
    use crate::types::{WasmDType, WasmTensorMeta};

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