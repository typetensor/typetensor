/*!
 * Softmax and LogSoftmax operations - cleaner implementation
 */

use crate::types::{WasmOperation, WasmTensorMeta, WasmDType, WasmResult, WasmError};
use crate::memory::{WasmMemoryManager, WasmBufferHandle};

/// Execute softmax or log_softmax operation
pub fn execute_softmax_op(
    memory_manager: &mut WasmMemoryManager,
    operation: WasmOperation,
    input: &WasmBufferHandle,
    input_meta: &WasmTensorMeta,
    output: &WasmBufferHandle,
    output_meta: &WasmTensorMeta,
    axis: Option<i32>,
) -> WasmResult<()> {
    let input_ptr = memory_manager.get_read_ptr(input);
    let output_ptr = memory_manager.get_write_ptr(output);
    
    let shape = input_meta.shape();
    let strides = input_meta.strides();
    let ndim = shape.len();
    
    // Determine the axis for softmax (default to last dimension)
    let axis = match axis {
        Some(a) => {
            let normalized = if a < 0 { 
                (ndim as i32 + a) as usize 
            } else { 
                a as usize 
            };
            if normalized >= ndim {
                return Err(WasmError::InvalidInput);
            }
            normalized
        }
        None => ndim - 1, // Default to last dimension
    };
    
    match input_meta.dtype() {
        WasmDType::Float32 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const f32, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_meta.size()) 
            };
            execute_softmax_f32(
                operation, input_slice, output_slice, 
                &shape, &strides, axis
            )?;
        }
        WasmDType::Float64 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const f64, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut f64, output_meta.size()) 
            };
            execute_softmax_f64(
                operation, input_slice, output_slice, 
                &shape, &strides, axis
            )?;
        }
        _ => return Err(WasmError::InvalidDType),
    }
    
    Ok(())
}

/// Execute softmax for f32 arrays with correct axis handling
fn execute_softmax_f32(
    operation: WasmOperation,
    input: &[f32],
    output: &mut [f32],
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> WasmResult<()> {
    let ndim = shape.len();
    let axis_size = shape[axis];
    
    // For a 2D matrix with shape [2, 2]:
    // axis=0 means we apply softmax down each column (2 softmaxes, each of size 2)
    // axis=1 means we apply softmax across each row (2 softmaxes, each of size 2)
    
    // Debug
    #[cfg(debug_assertions)]
    {
        println!("Softmax debug: shape={:?}, strides={:?}, axis={}", shape, strides, axis);
    }
    
    // Calculate the total number of elements
    let total_elements = shape.iter().product::<usize>();
    
    // Create an iterator over all positions
    for flat_idx in 0..total_elements {
        // Convert flat index to coordinates
        let mut coords = vec![0; ndim];
        let mut temp = flat_idx;
        for i in (0..ndim).rev() {
            coords[i] = temp % shape[i];
            temp /= shape[i];
        }
        
        // Only process if this is the first element along the softmax axis
        if coords[axis] != 0 {
            continue;
        }
        
        // Step 1: Find max value for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..axis_size {
            coords[axis] = i;
            let idx = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum::<usize>();
            if input[idx] > max_val {
                max_val = input[idx];
            }
        }
        
        // Step 2: Compute exp(x - max) and sum
        let mut sum = 0.0f32;
        let mut exp_vals = vec![0.0f32; axis_size];
        for i in 0..axis_size {
            coords[axis] = i;
            let idx = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum::<usize>();
            exp_vals[i] = (input[idx] - max_val).exp();
            sum += exp_vals[i];
        }
        
        // Step 3: Normalize and apply operation
        for i in 0..axis_size {
            coords[axis] = i;
            let idx = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum::<usize>();
            
            match operation {
                WasmOperation::Softmax => {
                    output[idx] = exp_vals[i] / sum;
                }
                WasmOperation::LogSoftmax => {
                    output[idx] = input[idx] - max_val - sum.ln();
                }
                _ => return Err(WasmError::InvalidOperation),
            }
        }
    }
    
    Ok(())
}

/// Execute softmax for f64 arrays
fn execute_softmax_f64(
    operation: WasmOperation,
    input: &[f64],
    output: &mut [f64],
    shape: &[usize],
    strides: &[usize],
    axis: usize,
) -> WasmResult<()> {
    let ndim = shape.len();
    let axis_size = shape[axis];
    
    let total_elements = shape.iter().product::<usize>();
    
    for flat_idx in 0..total_elements {
        let mut coords = vec![0; ndim];
        let mut temp = flat_idx;
        for i in (0..ndim).rev() {
            coords[i] = temp % shape[i];
            temp /= shape[i];
        }
        
        if coords[axis] != 0 {
            continue;
        }
        
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..axis_size {
            coords[axis] = i;
            let idx = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum::<usize>();
            if input[idx] > max_val {
                max_val = input[idx];
            }
        }
        
        let mut sum = 0.0f64;
        let mut exp_vals = vec![0.0f64; axis_size];
        for i in 0..axis_size {
            coords[axis] = i;
            let idx = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum::<usize>();
            exp_vals[i] = (input[idx] - max_val).exp();
            sum += exp_vals[i];
        }
        
        for i in 0..axis_size {
            coords[axis] = i;
            let idx = coords.iter().zip(strides.iter()).map(|(c, s)| c * s).sum::<usize>();
            
            match operation {
                WasmOperation::Softmax => {
                    output[idx] = exp_vals[i] / sum;
                }
                WasmOperation::LogSoftmax => {
                    output[idx] = input[idx] - max_val - sum.ln();
                }
                _ => return Err(WasmError::InvalidOperation),
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_softmax_2d_axis0() {
        // Input: [[1, 2], [3, 4]]
        // Expected for axis=0: [[0.1192, 0.1192], [0.8808, 0.8808]]
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        
        execute_softmax_f32(
            WasmOperation::Softmax, &input, &mut output, 
            &[2, 2], &[2, 1], 0
        ).unwrap();
        
        // Check column sums = 1
        assert_relative_eq!(output[0] + output[2], 1.0, epsilon = 1e-5);
        assert_relative_eq!(output[1] + output[3], 1.0, epsilon = 1e-5);
        
        // Check expected values
        assert_relative_eq!(output[0], 0.11920292, epsilon = 1e-5);
        assert_relative_eq!(output[1], 0.11920292, epsilon = 1e-5);
        assert_relative_eq!(output[2], 0.8807971, epsilon = 1e-5);
        assert_relative_eq!(output[3], 0.8807971, epsilon = 1e-5);
    }

    #[test]
    fn test_softmax_2d_axis1() {
        // Input: [[1, 2], [3, 4]]
        // Expected for axis=1: [[0.2689, 0.7311], [0.2689, 0.7311]]
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        
        execute_softmax_f32(
            WasmOperation::Softmax, &input, &mut output, 
            &[2, 2], &[2, 1], 1
        ).unwrap();
        
        // Check row sums = 1
        assert_relative_eq!(output[0] + output[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(output[2] + output[3], 1.0, epsilon = 1e-5);
        
        // Check expected values
        assert_relative_eq!(output[0], 0.26894142, epsilon = 1e-5);
        assert_relative_eq!(output[1], 0.73105858, epsilon = 1e-5);
        assert_relative_eq!(output[2], 0.26894142, epsilon = 1e-5);
        assert_relative_eq!(output[3], 0.73105858, epsilon = 1e-5);
    }
}