/*!
 * Reduction operations implementation for WebAssembly backend
 * 
 * Provides optimized implementations of tensor reduction operations
 * such as sum, mean, max, min, and product along specified axes.
 */

use crate::types::{WasmOperation, WasmTensorMeta, WasmDType, WasmResult, WasmError};
use crate::memory::{WasmMemoryManager, WasmBufferHandle};

/// Execute a reduction operation (legacy - no axis support)
pub fn execute_reduction_op(
    operation: WasmOperation,
    input: &WasmBufferHandle,
    input_meta: &WasmTensorMeta,
    output: &WasmBufferHandle,
    output_meta: &WasmTensorMeta,
) -> WasmResult<()> {
    // Call new version with None axes (reduce all)
    execute_reduction_op_with_axes(
        operation,
        input,
        input_meta,
        output,
        output_meta,
        None,
        false,
    )
}

/// Execute a reduction operation with axis support
pub fn execute_reduction_op_with_axes(
    operation: WasmOperation,
    input: &WasmBufferHandle,
    input_meta: &WasmTensorMeta,
    output: &WasmBufferHandle,
    output_meta: &WasmTensorMeta,
    axes: Option<&[usize]>,
    keep_dims: bool,
) -> WasmResult<()> {
    let input_ptr = input.get_read_ptr();
    let output_ptr = output.ptr() as *mut u8; // Cast to pointer
    
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
                &input_shape, &input_strides, axes, keep_dims
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
                &input_shape, &input_strides, axes, keep_dims
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
                &input_shape, &input_strides, axes, keep_dims
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
    axes: Option<&[usize]>,
    _keep_dims: bool,
) -> WasmResult<()> {
    // Handle different reduction scenarios
    match axes {
        None => {
            // Reduce all dimensions to scalar
            execute_full_reduction_f32(operation, input, output)
        }
        Some(reduction_axes) if reduction_axes.is_empty() => {
            // Empty axes also means reduce all
            execute_full_reduction_f32(operation, input, output)
        }
        Some(reduction_axes) => {
            // Reduce along specific axes
            execute_axis_reduction_f32(
                operation,
                input,
                output,
                input_shape,
                input_strides,
                reduction_axes,
            )
        }
    }
}

/// Execute full reduction (all dimensions to scalar)
fn execute_full_reduction_f32(
    operation: WasmOperation,
    input: &[f32],
    output: &mut [f32],
) -> WasmResult<()> {
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
    axes: Option<&[usize]>,
    _keep_dims: bool,
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
    axes: Option<&[usize]>,
    _keep_dims: bool,
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

/// Execute reduction along specific axes
fn execute_axis_reduction_f32(
    operation: WasmOperation,
    input: &[f32],
    output: &mut [f32],
    input_shape: &[usize],
    input_strides: &[usize],
    axes: &[usize],
) -> WasmResult<()> {
    // Compute output shape by removing reduced dimensions
    let ndim = input_shape.len();
    let mut output_shape = Vec::new();
    let mut output_strides = Vec::new();
    let mut is_reduced_axis = vec![false; ndim];
    
    // Mark axes to reduce
    for &axis in axes {
        if axis >= ndim {
            return Err(WasmError::InvalidInput);
        }
        is_reduced_axis[axis] = true;
    }
    
    // Build output shape (skip reduced axes)
    for i in 0..ndim {
        if !is_reduced_axis[i] {
            output_shape.push(input_shape[i]);
        }
    }
    
    // Compute output strides
    if !output_shape.is_empty() {
        output_strides = vec![1; output_shape.len()];
        for i in (0..output_shape.len()-1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }
    }
    
    
    // Initialize output
    match operation {
        WasmOperation::Sum | WasmOperation::Mean => {
            for val in output.iter_mut() {
                *val = 0.0;
            }
        }
        WasmOperation::Max => {
            for val in output.iter_mut() {
                *val = f32::NEG_INFINITY;
            }
        }
        WasmOperation::Min => {
            for val in output.iter_mut() {
                *val = f32::INFINITY;
            }
        }
        WasmOperation::Prod => {
            for val in output.iter_mut() {
                *val = 1.0;
            }
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    
    // Iterate through all input elements
    let total_elements = input_shape.iter().product::<usize>();
    for flat_idx in 0..total_elements {
        // Convert flat index to multi-dimensional indices
        let mut indices = vec![0; ndim];
        let mut remaining = flat_idx;
        for i in (0..ndim).rev() {
            indices[i] = remaining % input_shape[i];
            remaining /= input_shape[i];
        }
        
        // Compute input offset using strides
        let mut input_offset = 0;
        for i in 0..ndim {
            input_offset += indices[i] * input_strides[i];
        }
        
        // Compute output offset (skip reduced dimensions)
        let mut output_offset = 0;
        let mut out_idx = 0;
        for i in 0..ndim {
            if !is_reduced_axis[i] {
                if out_idx < output_strides.len() {
                    output_offset += indices[i] * output_strides[out_idx];
                }
                out_idx += 1;
            }
        }
        
        // Apply reduction operation
        let input_val = input[input_offset];
        match operation {
            WasmOperation::Sum | WasmOperation::Mean => {
                output[output_offset] += input_val;
            }
            WasmOperation::Max => {
                if input_val > output[output_offset] || input_val.is_nan() {
                    output[output_offset] = input_val;
                }
            }
            WasmOperation::Min => {
                if input_val < output[output_offset] || input_val.is_nan() {
                    output[output_offset] = input_val;
                }
            }
            WasmOperation::Prod => {
                output[output_offset] *= input_val;
            }
            _ => return Err(WasmError::InvalidOperation),
        }
    }
    
    // For mean, divide by the number of elements reduced
    if operation == WasmOperation::Mean {
        let reduced_size = axes.iter()
            .map(|&axis| input_shape[axis])
            .product::<usize>() as f32;
        
        for val in output.iter_mut() {
            *val /= reduced_size;
        }
    }
    
    Ok(())
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
            WasmOperation::Sum, &input, &mut output, &[5], &[1], None, false
        ).unwrap();
        
        assert_eq!(output[0], 15.0);
    }

    #[test]
    fn test_mean_f32() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 1];
        
        execute_reduction_f32(
            WasmOperation::Mean, &input, &mut output, &[5], &[1], None, false
        ).unwrap();
        
        assert_eq!(output[0], 3.0);
    }

    #[test]
    fn test_max_f32() {
        let input = vec![1.0f32, 5.0, 2.0, 8.0, 3.0];
        let mut output = vec![0.0f32; 1];
        
        execute_reduction_f32(
            WasmOperation::Max, &input, &mut output, &[5], &[1], None, false
        ).unwrap();
        
        assert_eq!(output[0], 8.0);
    }

    #[test]
    fn test_min_f32() {
        let input = vec![5.0f32, 1.0, 8.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 1];
        
        execute_reduction_f32(
            WasmOperation::Min, &input, &mut output, &[5], &[1], None, false
        ).unwrap();
        
        assert_eq!(output[0], 1.0);
    }

    #[test]
    fn test_prod_f32() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 1];
        
        execute_reduction_f32(
            WasmOperation::Prod, &input, &mut output, &[4], &[1], None, false
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