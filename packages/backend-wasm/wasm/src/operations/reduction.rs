/*!
 * Reduction operations implementation for WebAssembly backend
 * 
 * Provides optimized implementations of tensor reduction operations
 * such as sum, mean, max, min, and product along specified axes.
 */

use crate::types::{WasmOperation, WasmTensorMeta, WasmDType, WasmResult, WasmError};
use crate::memory::{WasmMemoryManager, WasmBufferHandle};

/// Execute a reduction operation
pub fn execute_reduction_op(
    memory_manager: &mut WasmMemoryManager,
    operation: WasmOperation,
    input: &WasmBufferHandle,
    input_meta: &WasmTensorMeta,
    output: &WasmBufferHandle,
    output_meta: &WasmTensorMeta,
) -> WasmResult<()> {
    let input_ptr = memory_manager.get_read_ptr(input);
    let output_ptr = memory_manager.get_write_ptr(output);
    
    let input_shape = input_meta.shape();
    let input_strides = input_meta.strides();
    
    match input_meta.dtype() {
        WasmDType::Float32 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const f32, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_meta.size()) 
            };
            execute_reduction_f32(
                operation, input_slice, output_slice, 
                &input_shape, &input_strides
            )?;
        }
        WasmDType::Float64 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const f64, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut f64, output_meta.size()) 
            };
            execute_reduction_f64(
                operation, input_slice, output_slice, 
                &input_shape, &input_strides
            )?;
        }
        WasmDType::Int32 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const i32, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut i32, output_meta.size()) 
            };
            execute_reduction_i32(
                operation, input_slice, output_slice, 
                &input_shape, &input_strides
            )?;
        }
        _ => return Err(WasmError::NotImplemented),
    }
    
    Ok(())
}

/// Execute reduction for f32 arrays
fn execute_reduction_f32(
    operation: WasmOperation,
    input: &[f32],
    output: &mut [f32],
    input_shape: &[usize],
    input_strides: &[usize],
) -> WasmResult<()> {
    // For simplicity, implement full tensor reduction first
    // TODO: Add support for axis-specific reductions
    
    match operation {
        WasmOperation::Sum => {
            let mut sum = 0.0f32;
            for &val in input.iter() {
                sum += val;
            }
            output[0] = sum;
        }
        WasmOperation::Mean => {
            let mut sum = 0.0f32;
            for &val in input.iter() {
                sum += val;
            }
            output[0] = sum / (input.len() as f32);
        }
        WasmOperation::Max => {
            let mut max_val = f32::NEG_INFINITY;
            for &val in input.iter() {
                if val > max_val || val.is_nan() {
                    max_val = val;
                }
            }
            output[0] = max_val;
        }
        WasmOperation::Min => {
            let mut min_val = f32::INFINITY;
            for &val in input.iter() {
                if val < min_val || val.is_nan() {
                    min_val = val;
                }
            }
            output[0] = min_val;
        }
        WasmOperation::Prod => {
            let mut prod = 1.0f32;
            for &val in input.iter() {
                prod *= val;
            }
            output[0] = prod;
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    Ok(())
}

/// Execute reduction for f64 arrays
fn execute_reduction_f64(
    operation: WasmOperation,
    input: &[f64],
    output: &mut [f64],
    input_shape: &[usize],
    input_strides: &[usize],
) -> WasmResult<()> {
    match operation {
        WasmOperation::Sum => {
            let mut sum = 0.0f64;
            for &val in input.iter() {
                sum += val;
            }
            output[0] = sum;
        }
        WasmOperation::Mean => {
            let mut sum = 0.0f64;
            for &val in input.iter() {
                sum += val;
            }
            output[0] = sum / (input.len() as f64);
        }
        WasmOperation::Max => {
            let mut max_val = f64::NEG_INFINITY;
            for &val in input.iter() {
                if val > max_val || val.is_nan() {
                    max_val = val;
                }
            }
            output[0] = max_val;
        }
        WasmOperation::Min => {
            let mut min_val = f64::INFINITY;
            for &val in input.iter() {
                if val < min_val || val.is_nan() {
                    min_val = val;
                }
            }
            output[0] = min_val;
        }
        WasmOperation::Prod => {
            let mut prod = 1.0f64;
            for &val in input.iter() {
                prod *= val;
            }
            output[0] = prod;
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    Ok(())
}

/// Execute reduction for i32 arrays
fn execute_reduction_i32(
    operation: WasmOperation,
    input: &[i32],
    output: &mut [i32],
    input_shape: &[usize],
    input_strides: &[usize],
) -> WasmResult<()> {
    match operation {
        WasmOperation::Sum => {
            let mut sum = 0i64; // Use i64 to prevent overflow
            for &val in input.iter() {
                sum += val as i64;
            }
            output[0] = sum.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
        WasmOperation::Mean => {
            let mut sum = 0i64;
            for &val in input.iter() {
                sum += val as i64;
            }
            let mean = sum / (input.len() as i64);
            output[0] = mean.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
        WasmOperation::Max => {
            let mut max_val = i32::MIN;
            for &val in input.iter() {
                if val > max_val {
                    max_val = val;
                }
            }
            output[0] = max_val;
        }
        WasmOperation::Min => {
            let mut min_val = i32::MAX;
            for &val in input.iter() {
                if val < min_val {
                    min_val = val;
                }
            }
            output[0] = min_val;
        }
        WasmOperation::Prod => {
            let mut prod = 1i64; // Use i64 to prevent overflow
            for &val in input.iter() {
                prod = prod.saturating_mul(val as i64);
                // Early exit if overflow would occur
                if prod.abs() > i32::MAX as i64 {
                    break;
                }
            }
            output[0] = prod.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    Ok(())
}

/// Axis-specific reduction for f32 (future implementation)
#[allow(dead_code)]
fn execute_axis_reduction_f32(
    operation: WasmOperation,
    input: &[f32],
    output: &mut [f32],
    input_shape: &[usize],
    input_strides: &[usize],
    axis: usize,
    keep_dims: bool,
) -> WasmResult<()> {
    // TODO: Implement axis-specific reductions
    // This would involve:
    // 1. Computing output shape based on reduced axis
    // 2. Iterating over non-reduced dimensions
    // 3. Accumulating values along the reduction axis
    // 4. Handling keep_dims flag for output shape
    
    Err(WasmError::NotImplemented)
}

/// Optimized sum reduction using Kahan summation for better numerical stability
fn kahan_sum_f32(input: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut c = 0.0f32; // Running compensation for lost low-order bits
    
    for &val in input.iter() {
        let y = val - c;        // So far, so good: c is zero
        let t = sum + y;        // Alas, sum is big, y small, so low-order digits of y are lost
        c = (t - sum) - y;      // (t - sum) cancels the high-order part of y; subtracting y recovers negative (low part of y)
        sum = t;                // Algebraically, c should always be zero. Beware overly-clever optimizing compilers!
    }
    
    sum
}

/// Optimized sum reduction using Kahan summation for f64
fn kahan_sum_f64(input: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    let mut c = 0.0f64;
    
    for &val in input.iter() {
        let y = val - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sum_f32() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 1];
        
        execute_reduction_f32(
            WasmOperation::Sum, &input, &mut output, &[5], &[1]
        ).unwrap();
        
        assert_eq!(output[0], 15.0);
    }

    #[test]
    fn test_mean_f32() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 1];
        
        execute_reduction_f32(
            WasmOperation::Mean, &input, &mut output, &[5], &[1]
        ).unwrap();
        
        assert_eq!(output[0], 3.0);
    }

    #[test]
    fn test_max_f32() {
        let input = vec![1.0f32, 5.0, 2.0, 8.0, 3.0];
        let mut output = vec![0.0f32; 1];
        
        execute_reduction_f32(
            WasmOperation::Max, &input, &mut output, &[5], &[1]
        ).unwrap();
        
        assert_eq!(output[0], 8.0);
    }

    #[test]
    fn test_min_f32() {
        let input = vec![5.0f32, 1.0, 8.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 1];
        
        execute_reduction_f32(
            WasmOperation::Min, &input, &mut output, &[5], &[1]
        ).unwrap();
        
        assert_eq!(output[0], 1.0);
    }

    #[test]
    fn test_prod_f32() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 1];
        
        execute_reduction_f32(
            WasmOperation::Prod, &input, &mut output, &[4], &[1]
        ).unwrap();
        
        assert_eq!(output[0], 24.0);
    }

    #[test]
    fn test_kahan_sum() {
        // Use values that demonstrate Kahan summation's benefit within f32 precision
        let input = vec![1.0f32, 1e7, 1.0, -1e7]; 
        let result = kahan_sum_f32(&input);
        assert_relative_eq!(result, 2.0, epsilon = 1e-5);
        
        // Another test case showing accumulated error reduction
        let mut many_small = vec![0.1f32; 10_000];
        let kahan_result = kahan_sum_f32(&many_small);
        let expected = 1000.0f32;
        
        // Kahan sum should be close to the expected value
        assert_relative_eq!(kahan_result, expected, epsilon = 0.1);
    }
}