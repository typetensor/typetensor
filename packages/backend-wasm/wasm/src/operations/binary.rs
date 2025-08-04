/*!
 * Binary operations implementation for WebAssembly backend
 * 
 * Provides optimized implementations of element-wise binary operations
 * with broadcasting support and SIMD optimizations.
 */

use crate::types::{WasmOperation, WasmTensorMeta, WasmDType, WasmResult, WasmError};
use crate::memory::WasmTensor;
use crate::arena::TempArena;
use crate::simd::float32;

/// Execute a binary operation
pub fn execute_binary_op(
    operation: WasmOperation,
    input_a: &WasmTensor,
    input_b: &WasmTensor,
    output: &WasmTensor,
    arena: &TempArena,
) -> WasmResult<()> {
    let input_a_ptr = input_a.get_read_ptr(arena);
    let input_b_ptr = input_b.get_read_ptr(arena);
    
    // SAFETY: Safe pointer cast - see MEMORY_SAFETY.md for detailed explanation
    // Each tensor has unique memory from bump allocator, no aliasing occurs
    let output_ptr = output.get_read_ptr(arena) as *mut u8;
    
    let input_meta_a = input_a.metadata();
    let input_meta_b = input_b.metadata();
    let output_meta = output.metadata();

    // Check if we can use fast path (same size, no broadcasting)
    let same_size = input_meta_a.size() == input_meta_b.size() && 
                   input_meta_a.size() == output_meta.size();
    
    if same_size && input_meta_a.dtype() == input_meta_b.dtype() {
        // Fast path: same shapes and types
        execute_binary_fast(
            operation,
            input_meta_a.dtype(),
            input_a_ptr,
            input_b_ptr,
            output_ptr,
            input_meta_a.size(),
        )?;
    } else {
        // Slow path: handle broadcasting and type conversion
        execute_binary_broadcast(
            operation,
            input_meta_a,
            input_meta_b,
            output_meta,
            input_a_ptr,
            input_b_ptr,
            output_ptr,
        )?;
    }

    Ok(())
}

/// Fast path for binary operations (same shapes, no broadcasting)
fn execute_binary_fast(
    operation: WasmOperation,
    dtype: WasmDType,
    input_a_ptr: *const u8,
    input_b_ptr: *const u8,
    output_ptr: *mut u8,
    size: usize,
) -> WasmResult<()> {
    match dtype {
        WasmDType::Float32 => {
            let a_slice = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f32, size) };
            let b_slice = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f32, size) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, size) };
            
            execute_binary_f32_fast(operation, a_slice, b_slice, out_slice)?;
        }
        WasmDType::Float64 => {
            let a_slice = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f64, size) };
            let b_slice = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f64, size) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f64, size) };
            execute_binary_f64_fast(operation, a_slice, b_slice, out_slice)?;
        }
        WasmDType::Int32 => {
            let a_slice = unsafe { std::slice::from_raw_parts(input_a_ptr as *const i32, size) };
            let b_slice = unsafe { std::slice::from_raw_parts(input_b_ptr as *const i32, size) };
            let out_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut i32, size) };
            execute_binary_i32_fast(operation, a_slice, b_slice, out_slice)?;
        }
        _ => return Err(WasmError::NotImplemented),
    }
    Ok(())
}

/// Fast binary operations for f32 arrays
fn execute_binary_f32_fast(
    operation: WasmOperation,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) -> WasmResult<()> {
    match operation {
        WasmOperation::Add => {
            // Use SIMD-optimized addition
            float32::simd_add(a, b, output);
        }
        WasmOperation::Mul => {
            // Use SIMD-optimized multiplication
            float32::simd_mul(a, b, output);
        }
        WasmOperation::Sub => {
            // Use SIMD-optimized subtraction
            float32::simd_sub(a, b, output);
        }
        WasmOperation::Div => {
            // Use SIMD-optimized division
            float32::simd_div(a, b, output);
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    Ok(())
}

/// Fast binary operations for f64 arrays
fn execute_binary_f64_fast(
    operation: WasmOperation,
    a: &[f64],
    b: &[f64],
    output: &mut [f64],
) -> WasmResult<()> {
    match operation {
        WasmOperation::Add => {
            for ((out, &a_val), &b_val) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
                *out = a_val + b_val;
            }
        }
        WasmOperation::Sub => {
            for ((out, &a_val), &b_val) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
                *out = a_val - b_val;
            }
        }
        WasmOperation::Mul => {
            for ((out, &a_val), &b_val) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
                *out = a_val * b_val;
            }
        }
        WasmOperation::Div => {
            for ((out, &a_val), &b_val) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
                *out = crate::utils::safe_div_f64(a_val, b_val);
            }
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    Ok(())
}

/// Fast binary operations for i32 arrays
fn execute_binary_i32_fast(
    operation: WasmOperation,
    a: &[i32],
    b: &[i32],
    output: &mut [i32],
) -> WasmResult<()> {
    match operation {
        WasmOperation::Add => {
            for ((out, &a_val), &b_val) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
                *out = a_val.wrapping_add(b_val);
            }
        }
        WasmOperation::Sub => {
            for ((out, &a_val), &b_val) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
                *out = a_val.wrapping_sub(b_val);
            }
        }
        WasmOperation::Mul => {
            for ((out, &a_val), &b_val) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
                *out = a_val.wrapping_mul(b_val);
            }
        }
        WasmOperation::Div => {
            for ((out, &a_val), &b_val) in output.iter_mut().zip(a.iter()).zip(b.iter()) {
                *out = if b_val == 0 {
                    if a_val > 0 { i32::MAX } else { i32::MIN }
                } else {
                    a_val / b_val
                };
            }
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    Ok(())
}

/// Slow path for binary operations with broadcasting
fn execute_binary_broadcast(
    operation: WasmOperation,
    input_meta_a: &WasmTensorMeta,
    input_meta_b: &WasmTensorMeta,
    output_meta: &WasmTensorMeta,
    input_a_ptr: *const u8,
    input_b_ptr: *const u8,
    output_ptr: *mut u8,
) -> WasmResult<()> {
    let shape_a = input_meta_a.shape();
    let shape_b = input_meta_b.shape();
    let output_shape = output_meta.shape();

    // Check if input_b is a scalar (single element)
    if input_meta_b.size() == 1 {
        execute_binary_scalar_broadcast(
            operation,
            input_meta_a.dtype(),
            input_a_ptr,
            input_b_ptr,
            output_ptr,
            input_meta_a.size(),
        )?;
        return Ok(());
    } else if input_meta_a.size() == 1 {
        // input_a is scalar
        execute_binary_scalar_broadcast_reverse(
            operation,
            input_meta_b.dtype(),
            input_a_ptr,
            input_b_ptr,
            output_ptr,
            input_meta_b.size(),
        )?;
        return Ok(());
    }
    
    // Try basic broadcasting patterns first
    if try_basic_broadcasting(
        operation,
        input_meta_a,
        input_meta_b,
        output_meta,
        input_a_ptr,
        input_b_ptr,
        output_ptr,
    ).is_ok() {
        return Ok(());
    }
    
    // Fall back to general broadcasting
    execute_general_broadcast(
        operation,
        input_meta_a,
        input_meta_b,
        output_meta,
        input_a_ptr,
        input_b_ptr,
        output_ptr,
    )
}

/// Element-wise binary operation for tensors of the same size
fn execute_binary_elementwise(
    operation: WasmOperation,
    dtype: WasmDType,
    input_a_ptr: *const u8,
    input_b_ptr: *const u8,
    output_ptr: *mut u8,
    size: usize,
) -> WasmResult<()> {
    match dtype {
        WasmDType::Float32 => {
            let input_a = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f32, size) };
            let input_b = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f32, size) };
            let output = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, size) };
            
            execute_binary_f32_fast(operation, input_a, input_b, output)?;
        }
        WasmDType::Float64 => {
            let input_a = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f64, size) };
            let input_b = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f64, size) };
            let output = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f64, size) };
            
            execute_binary_f64_fast(operation, input_a, input_b, output)?;
        }
        _ => return Err(WasmError::InvalidOperation),
    }
    
    Ok(())
}

/// Binary operation with scalar broadcasting (tensor op scalar)
fn execute_binary_scalar_broadcast(
    operation: WasmOperation,
    dtype: WasmDType,
    tensor_ptr: *const u8,
    scalar_ptr: *const u8,
    output_ptr: *mut u8,
    size: usize,
) -> WasmResult<()> {
    match dtype {
        WasmDType::Float32 => {
            let tensor_slice = unsafe { std::slice::from_raw_parts(tensor_ptr as *const f32, size) };
            let scalar_val = unsafe { *(scalar_ptr as *const f32) };
            let output_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, size) };
            
            match operation {
                WasmOperation::Add => {
                    for (out, &tensor_val) in output_slice.iter_mut().zip(tensor_slice.iter()) {
                        *out = tensor_val + scalar_val;
                    }
                }
                WasmOperation::Sub => {
                    for (out, &tensor_val) in output_slice.iter_mut().zip(tensor_slice.iter()) {
                        *out = tensor_val - scalar_val;
                    }
                }
                WasmOperation::Mul => {
                    for (out, &tensor_val) in output_slice.iter_mut().zip(tensor_slice.iter()) {
                        *out = tensor_val * scalar_val;
                    }
                }
                WasmOperation::Div => {
                    for (out, &tensor_val) in output_slice.iter_mut().zip(tensor_slice.iter()) {
                        *out = crate::utils::safe_div_f32(tensor_val, scalar_val);
                    }
                }
                _ => return Err(WasmError::InvalidOperation),
            }
        }
        _ => return Err(WasmError::NotImplemented),
    }
    Ok(())
}

/// Binary operation with scalar broadcasting (scalar op tensor)
fn execute_binary_scalar_broadcast_reverse(
    operation: WasmOperation,
    dtype: WasmDType,
    scalar_ptr: *const u8,
    tensor_ptr: *const u8,
    output_ptr: *mut u8,
    size: usize,
) -> WasmResult<()> {
    match dtype {
        WasmDType::Float32 => {
            let scalar_val = unsafe { *(scalar_ptr as *const f32) };
            let tensor_slice = unsafe { std::slice::from_raw_parts(tensor_ptr as *const f32, size) };
            let output_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, size) };
            
            match operation {
                WasmOperation::Add => {
                    for (out, &tensor_val) in output_slice.iter_mut().zip(tensor_slice.iter()) {
                        *out = scalar_val + tensor_val;
                    }
                }
                WasmOperation::Sub => {
                    for (out, &tensor_val) in output_slice.iter_mut().zip(tensor_slice.iter()) {
                        *out = scalar_val - tensor_val;
                    }
                }
                WasmOperation::Mul => {
                    for (out, &tensor_val) in output_slice.iter_mut().zip(tensor_slice.iter()) {
                        *out = scalar_val * tensor_val;
                    }
                }
                WasmOperation::Div => {
                    for (out, &tensor_val) in output_slice.iter_mut().zip(tensor_slice.iter()) {
                        *out = crate::utils::safe_div_f32(scalar_val, tensor_val);
                    }
                }
                _ => return Err(WasmError::InvalidOperation),
            }
        }
        _ => return Err(WasmError::NotImplemented),
    }
    Ok(())
}


/// Try basic broadcasting patterns
fn try_basic_broadcasting(
    operation: WasmOperation,
    input_meta_a: &WasmTensorMeta,
    input_meta_b: &WasmTensorMeta,
    output_meta: &WasmTensorMeta,
    input_a_ptr: *const u8,
    input_b_ptr: *const u8,
    output_ptr: *mut u8,
) -> WasmResult<()> {
    // Check if we can use one of the optimized patterns
    execute_binary_broadcast_basic(
        operation,
        input_meta_a,
        input_meta_b,
        output_meta,
        input_a_ptr,
        input_b_ptr,
        output_ptr,
    )
}

/// General broadcasting following NumPy rules
fn execute_general_broadcast(
    operation: WasmOperation,
    input_meta_a: &WasmTensorMeta,
    input_meta_b: &WasmTensorMeta,
    output_meta: &WasmTensorMeta,
    input_a_ptr: *const u8,
    input_b_ptr: *const u8,
    output_ptr: *mut u8,
) -> WasmResult<()> {
    // Only handle float32 for now
    if input_meta_a.dtype() != WasmDType::Float32 || input_meta_b.dtype() != WasmDType::Float32 {
        return Err(WasmError::NotImplemented);
    }
    
    let shape_a = input_meta_a.shape();
    let shape_b = input_meta_b.shape();
    let strides_a = input_meta_a.strides();
    let strides_b = input_meta_b.strides();
    let output_shape = output_meta.shape();
    let output_size = output_meta.size();
    
    // Get typed slices
    let a_slice = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f32, input_meta_a.size()) };
    let b_slice = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f32, input_meta_b.size()) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_size) };
    
    // Iterate through all output positions
    for out_idx in 0..output_size {
        // Convert flat index to multi-dimensional indices
        let mut out_indices = vec![0; output_shape.len()];
        let mut remaining = out_idx;
        for i in (0..output_shape.len()).rev() {
            out_indices[i] = remaining % output_shape[i];
            remaining /= output_shape[i];
        }
        
        // Map output indices to input indices for both tensors
        let a_idx = compute_broadcast_index(&out_indices, &shape_a, &strides_a, &output_shape);
        let b_idx = compute_broadcast_index(&out_indices, &shape_b, &strides_b, &output_shape);
        
        // Get values and perform operation
        let a_val = a_slice[a_idx];
        let b_val = b_slice[b_idx];
        
        out_slice[out_idx] = match operation {
            WasmOperation::Add => a_val + b_val,
            WasmOperation::Sub => a_val - b_val,
            WasmOperation::Mul => a_val * b_val,
            WasmOperation::Div => crate::utils::safe_div_f32(a_val, b_val),
            _ => return Err(WasmError::InvalidOperation),
        };
    }
    
    Ok(())
}

/// Compute the index in a tensor given output indices and broadcasting rules
fn compute_broadcast_index(
    out_indices: &[usize],
    tensor_shape: &[usize],
    tensor_strides: &[usize],
    output_shape: &[usize],
) -> usize {
    let mut index = 0;
    let ndim_out = output_shape.len();
    let ndim_tensor = tensor_shape.len();
    
    // Broadcasting aligns dimensions from the right
    let offset = ndim_out.saturating_sub(ndim_tensor);
    
    for i in 0..ndim_tensor {
        let out_dim = i + offset;
        let out_idx = out_indices[out_dim];
        
        // Apply broadcasting rule: if dimension is 1, use index 0
        let tensor_idx = if tensor_shape[i] == 1 {
            0
        } else {
            out_idx
        };
        
        index += tensor_idx * tensor_strides[i];
    }
    
    index
}

/// Basic broadcasting for common cases (vector-matrix)
fn execute_binary_broadcast_basic(
    operation: WasmOperation,
    input_meta_a: &WasmTensorMeta,
    input_meta_b: &WasmTensorMeta,
    output_meta: &WasmTensorMeta,
    input_a_ptr: *const u8,
    input_b_ptr: *const u8,
    output_ptr: *mut u8,
) -> WasmResult<()> {
    let shape_a = input_meta_a.shape();
    let shape_b = input_meta_b.shape();
    let _output_shape = output_meta.shape();
    
    // Only handle float32 for now
    if input_meta_a.dtype() != WasmDType::Float32 || input_meta_b.dtype() != WasmDType::Float32 {
        return Err(WasmError::NotImplemented);
    }
    
    // Case 1: Vector [n] + Matrix [m, n] -> broadcast vector across rows
    if shape_a.len() == 1 && shape_b.len() == 2 && shape_a[0] == shape_b[1] {
        let rows = shape_b[0];
        let cols = shape_b[1];
        
        let a_slice = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f32, cols) };
        let b_slice = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f32, rows * cols) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, rows * cols) };
        
        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let a_val = a_slice[col];
                let b_val = b_slice[idx];
                
                out_slice[idx] = match operation {
                    WasmOperation::Add => a_val + b_val,
                    WasmOperation::Sub => a_val - b_val,
                    WasmOperation::Mul => a_val * b_val,
                    WasmOperation::Div => crate::utils::safe_div_f32(a_val, b_val),
                    _ => return Err(WasmError::InvalidOperation),
                };
            }
        }
        Ok(())
    }
    // Case 2: Matrix [m, n] + Vector [n] -> broadcast vector across rows
    else if shape_a.len() == 2 && shape_b.len() == 1 && shape_a[1] == shape_b[0] {
        let rows = shape_a[0];
        let cols = shape_a[1];
        
        let a_slice = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f32, rows * cols) };
        let b_slice = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f32, cols) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, rows * cols) };
        
        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let a_val = a_slice[idx];
                let b_val = b_slice[col];
                
                out_slice[idx] = match operation {
                    WasmOperation::Add => a_val + b_val,
                    WasmOperation::Sub => a_val - b_val,
                    WasmOperation::Mul => a_val * b_val,
                    WasmOperation::Div => crate::utils::safe_div_f32(a_val, b_val),
                    _ => return Err(WasmError::InvalidOperation),
                };
            }
        }
        Ok(())
    }
    // Case 3: Vector [m] (as [m, 1]) + Matrix [m, n] -> broadcast vector across columns
    else if shape_a.len() == 1 && shape_b.len() == 2 && shape_a[0] == shape_b[0] {
        let rows = shape_b[0];
        let cols = shape_b[1];
        
        let a_slice = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f32, rows) };
        let b_slice = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f32, rows * cols) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, rows * cols) };
        
        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let a_val = a_slice[row];
                let b_val = b_slice[idx];
                
                out_slice[idx] = match operation {
                    WasmOperation::Add => a_val + b_val,
                    WasmOperation::Sub => a_val - b_val,
                    WasmOperation::Mul => a_val * b_val,
                    WasmOperation::Div => crate::utils::safe_div_f32(a_val, b_val),
                    _ => return Err(WasmError::InvalidOperation),
                };
            }
        }
        Ok(())
    }
    // Case 4: Matrix [m, n] + Vector [m] (as [m, 1]) -> broadcast vector across columns
    else if shape_a.len() == 2 && shape_b.len() == 1 && shape_a[0] == shape_b[0] {
        let rows = shape_a[0];
        let cols = shape_a[1];
        
        let a_slice = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f32, rows * cols) };
        let b_slice = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f32, rows) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, rows * cols) };
        
        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let a_val = a_slice[idx];
                let b_val = b_slice[row];
                
                out_slice[idx] = match operation {
                    WasmOperation::Add => a_val + b_val,
                    WasmOperation::Sub => a_val - b_val,
                    WasmOperation::Mul => a_val * b_val,
                    WasmOperation::Div => crate::utils::safe_div_f32(a_val, b_val),
                    _ => return Err(WasmError::InvalidOperation),
                };
            }
        }
        Ok(())
    }
    // Case 5: Column vector [m, 1] + Matrix [m, n] -> broadcast column vector across columns
    else if shape_a.len() == 2 && shape_b.len() == 2 && shape_a[0] == shape_b[0] && shape_a[1] == 1 {
        let rows = shape_b[0];
        let cols = shape_b[1];
        
        let a_slice = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f32, rows) };
        let b_slice = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f32, rows * cols) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, rows * cols) };
        
        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let a_val = a_slice[row];  // Column vector has one value per row
                let b_val = b_slice[idx];
                
                out_slice[idx] = match operation {
                    WasmOperation::Add => a_val + b_val,
                    WasmOperation::Sub => a_val - b_val,
                    WasmOperation::Mul => a_val * b_val,
                    WasmOperation::Div => crate::utils::safe_div_f32(a_val, b_val),
                    _ => return Err(WasmError::InvalidOperation),
                };
            }
        }
        Ok(())
    }
    // Case 6: Matrix [m, n] + Column vector [m, 1] -> broadcast column vector across columns
    else if shape_a.len() == 2 && shape_b.len() == 2 && shape_a[0] == shape_b[0] && shape_b[1] == 1 {
        let rows = shape_a[0];
        let cols = shape_a[1];
        
        let a_slice = unsafe { std::slice::from_raw_parts(input_a_ptr as *const f32, rows * cols) };
        let b_slice = unsafe { std::slice::from_raw_parts(input_b_ptr as *const f32, rows) };
        let out_slice = unsafe { std::slice::from_raw_parts_mut(output_ptr as *mut f32, rows * cols) };
        
        for row in 0..rows {
            for col in 0..cols {
                let idx = row * cols + col;
                let a_val = a_slice[idx];
                let b_val = b_slice[row];  // Column vector has one value per row
                
                out_slice[idx] = match operation {
                    WasmOperation::Add => a_val + b_val,
                    WasmOperation::Sub => a_val - b_val,
                    WasmOperation::Mul => a_val * b_val,
                    WasmOperation::Div => crate::utils::safe_div_f32(a_val, b_val),
                    _ => return Err(WasmError::InvalidOperation),
                };
            }
        }
        Ok(())
    }
    else {
        // Full broadcasting not implemented yet
        return Err(WasmError::NotImplemented);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![0.5f32, 1.5, 2.5, 3.5];
        let mut output = vec![0.0f32; 4];
        
        execute_binary_f32_fast(WasmOperation::Add, &a, &b, &mut output).unwrap();
        
        assert_eq!(output, vec![1.5, 3.5, 5.5, 7.5]);
    }

    #[test]
    fn test_binary_mul_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];
        
        execute_binary_f32_fast(WasmOperation::Mul, &a, &b, &mut output).unwrap();
        
        assert_eq!(output, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_binary_div_f32() {
        let a = vec![4.0f32, 6.0, 8.0, 10.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 4];
        
        execute_binary_f32_fast(WasmOperation::Div, &a, &b, &mut output).unwrap();
        
        assert_eq!(output, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_execute_binary_elementwise() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut output = vec![0.0f32; 4];
        
        // Test element-wise addition
        execute_binary_elementwise(
            WasmOperation::Add,
            WasmDType::Float32,
            a.as_ptr() as *const u8,
            b.as_ptr() as *const u8,
            output.as_mut_ptr() as *mut u8,
            4,
        ).unwrap();
        
        assert_eq!(output, vec![6.0, 8.0, 10.0, 12.0]);
    }
}