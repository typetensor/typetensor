/*!
 * Unary operations implementation for WebAssembly backend
 * 
 * Provides optimized implementations of element-wise unary operations
 * with SIMD support where available.
 */

use crate::types::{WasmOperation, WasmTensorMeta, WasmDType, WasmResult, WasmError};
use crate::memory::{WasmMemoryManager, WasmBufferHandle};

/// Execute a unary operation
pub fn execute_unary_op(
    memory_manager: &mut WasmMemoryManager,
    operation: WasmOperation,
    input: &WasmBufferHandle,
    input_meta: &WasmTensorMeta,
    output: &WasmBufferHandle,
    output_meta: &WasmTensorMeta,
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
            // SIMD optimization opportunity here
            for (i, &val) in input.iter().enumerate() {
                output[i] = -val;
            }
        }
        WasmOperation::Abs => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.abs();
            }
        }
        WasmOperation::Sin => {
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
        WasmOperation::Sqrt => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = if val >= 0.0 { val.sqrt() } else { f32::NAN };
            }
        }
        WasmOperation::Square => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val * val;
            }
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
            for (i, &val) in input.iter().enumerate() {
                output[i] = -val;
            }
        }
        WasmOperation::Abs => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = val.abs();
            }
        }
        WasmOperation::Sin => {
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
        WasmOperation::Sqrt => {
            for (i, &val) in input.iter().enumerate() {
                output[i] = if val >= 0.0 { val.sqrt() } else { f64::NAN };
            }
        }
        WasmOperation::Square => {
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

#[cfg(feature = "simd128")]
mod simd {
    use super::*;
    use std::arch::wasm32::*;

    /// SIMD-optimized negation for f32 arrays
    pub fn neg_f32_simd(input: &[f32], output: &mut [f32]) {
        const SIMD_WIDTH: usize = 4;
        let chunks = input.len() / SIMD_WIDTH;
        
        for i in 0..chunks {
            let base_idx = i * SIMD_WIDTH;
            let input_vec = v128_load(&input[base_idx] as *const f32 as *const v128);
            let neg_vec = f32x4_neg(input_vec);
            v128_store(&mut output[base_idx] as *mut f32 as *mut v128, neg_vec);
        }
        
        // Handle remaining elements
        let remainder_start = chunks * SIMD_WIDTH;
        for i in remainder_start..input.len() {
            output[i] = -input[i];
        }
    }

    /// SIMD-optimized absolute value for f32 arrays
    pub fn abs_f32_simd(input: &[f32], output: &mut [f32]) {
        const SIMD_WIDTH: usize = 4;
        let chunks = input.len() / SIMD_WIDTH;
        
        for i in 0..chunks {
            let base_idx = i * SIMD_WIDTH;
            let input_vec = v128_load(&input[base_idx] as *const f32 as *const v128);
            let abs_vec = f32x4_abs(input_vec);
            v128_store(&mut output[base_idx] as *mut f32 as *mut v128, abs_vec);
        }
        
        // Handle remaining elements
        let remainder_start = chunks * SIMD_WIDTH;
        for i in remainder_start..input.len() {
            output[i] = input[i].abs();
        }
    }
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