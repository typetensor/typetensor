/*!
 * View operations implementation for WebAssembly backend
 * 
 * Implements tensor view operations that manipulate tensor shape and layout
 * without copying data where possible (zero-copy operations).
 */

use crate::types::{WasmOperation, WasmTensorMeta, WasmDType, WasmResult, WasmError};
use crate::memory::WasmTensor;
use crate::arena::TempArena;

/// Execute a view operation
pub fn execute_view_op(
    operation: WasmOperation,
    input: &WasmTensor,
    output: &WasmTensor,
    arena: &TempArena,
) -> WasmResult<()> {
    let input_meta = input.metadata();
    let output_meta = output.metadata();
    
    match operation {
        WasmOperation::Reshape | WasmOperation::View | WasmOperation::Flatten => {
            // For now, copy the data until we have proper view support
            // TODO: Implement zero-copy views with shared buffers
            execute_copy_reshape(input, output, arena)
        }
        WasmOperation::Slice => {
            // Slice operation requires data copying to extract the sliced portion
            execute_slice_op(input, output, arena)
        }
        WasmOperation::Permute | WasmOperation::Transpose => {
            // Transpose operations may require data copying depending on the implementation
            execute_transpose_op(input, output, arena)
        }
        WasmOperation::Squeeze | WasmOperation::Unsqueeze => {
            // For now, copy the data until we have proper view support
            // TODO: Implement zero-copy views with shared buffers
            execute_copy_reshape(input, output, arena)
        }
        WasmOperation::Expand => {
            // For now, implement expand by copying with broadcasting
            // TODO: Implement zero-copy expand with stride tricks
            execute_expand_op(input, output, arena)
        }
        WasmOperation::Tile => {
            // Tile operation requires data copying
            execute_tile_op(input, output, arena)
        }
        _ => Err(WasmError::InvalidOperation),
    }
}

/// Execute slice operation
fn execute_slice_op(
    input: &WasmTensor,
    output: &WasmTensor,
    arena: &TempArena,
) -> WasmResult<()> {
    // Get pointers to input and output data
    let input_ptr = input.get_read_ptr(arena);
    let output_ptr = output.get_read_ptr(arena) as *mut u8; // Cast to mut for operations
    
    // Get tensor metadata
    let input_meta = input.metadata();
    let output_meta = output.metadata();
    let input_shape = input_meta.shape();
    let input_strides = input_meta.strides();
    let output_shape = output_meta.shape();
    let output_size = output_meta.size();
    
    // Handle different data types
    match input_meta.dtype() {
        WasmDType::Float32 => {
            let input_slice = unsafe {
                std::slice::from_raw_parts(input_ptr as *const f32, input_meta.size())
            };
            let output_slice = unsafe {
                std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_size)
            };
            slice_f32(input_slice, output_slice, &input_shape, &input_strides, &output_shape)?;
        }
        WasmDType::Float64 => {
            let input_slice = unsafe {
                std::slice::from_raw_parts(input_ptr as *const f64, input_meta.size())
            };
            let output_slice = unsafe {
                std::slice::from_raw_parts_mut(output_ptr as *mut f64, output_size)
            };
            slice_f64(input_slice, output_slice, &input_shape, &input_strides, &output_shape)?;
        }
        WasmDType::Int32 => {
            let input_slice = unsafe {
                std::slice::from_raw_parts(input_ptr as *const i32, input_meta.size())
            };
            let output_slice = unsafe {
                std::slice::from_raw_parts_mut(output_ptr as *mut i32, output_size)
            };
            slice_i32(input_slice, output_slice, &input_shape, &input_strides, &output_shape)?;
        }
        _ => return Err(WasmError::NotImplemented),
    }
    
    Ok(())
}

/// Execute transpose operation
fn execute_transpose_op(
    input: &WasmTensor,
    output: &WasmTensor,
    arena: &TempArena,
) -> WasmResult<()> {
    let input_ptr = input.get_read_ptr(arena);
    let output_ptr = output.get_read_ptr(arena) as *mut u8; // Cast to mut for operations
    
    let input_meta = input.metadata();
    let output_meta = output.metadata();
    let input_shape = input_meta.shape();
    let input_strides = input_meta.strides();
    let output_shape = output_meta.shape();
    
    match input_meta.dtype() {
        WasmDType::Float32 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const f32, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_meta.size()) 
            };
            transpose_f32(input_slice, output_slice, &input_shape, &input_strides, &output_shape)?;
        }
        WasmDType::Float64 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const f64, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut f64, output_meta.size()) 
            };
            transpose_f64(input_slice, output_slice, &input_shape, &input_strides, &output_shape)?;
        }
        WasmDType::Int32 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const i32, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut i32, output_meta.size()) 
            };
            transpose_i32(input_slice, output_slice, &input_shape, &input_strides, &output_shape)?;
        }
        _ => return Err(WasmError::NotImplemented),
    }
    
    Ok(())
}

/// Execute tile operation
fn execute_tile_op(
    input: &WasmTensor,
    output: &WasmTensor,
    arena: &TempArena,
) -> WasmResult<()> {
    let input_ptr = input.get_read_ptr(arena);
    let output_ptr = output.get_read_ptr(arena) as *mut u8; // Cast to mut for operations
    
    let input_meta = input.metadata();
    let output_meta = output.metadata();
    let input_shape = input_meta.shape();
    let output_shape = output_meta.shape();
    let input_strides = input_meta.strides();
    
    match input_meta.dtype() {
        WasmDType::Float32 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const f32, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_meta.size()) 
            };
            tile_f32(input_slice, output_slice, &input_shape, &input_strides, &output_shape)?;
        }
        WasmDType::Float64 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const f64, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut f64, output_meta.size()) 
            };
            tile_f64(input_slice, output_slice, &input_shape, &input_strides, &output_shape)?;
        }
        WasmDType::Int32 => {
            let input_slice = unsafe { 
                std::slice::from_raw_parts(input_ptr as *const i32, input_meta.size()) 
            };
            let output_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut i32, output_meta.size()) 
            };
            tile_i32(input_slice, output_slice, &input_shape, &input_strides, &output_shape)?;
        }
        _ => return Err(WasmError::NotImplemented),
    }
    
    Ok(())
}

/// Transpose f32 tensor
fn transpose_f32(
    input: &[f32],
    output: &mut [f32],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    if input_shape.len() == 2 {
        // Simple 2D transpose
        let rows = input_shape[0];
        let cols = input_shape[1];
        
        for i in 0..rows {
            for j in 0..cols {
                let input_idx = i * input_strides[0] + j * input_strides[1];
                let output_idx = j * rows + i; // transposed indexing
                output[output_idx] = input[input_idx];
            }
        }
    } else {
        // General ND transpose - for simplicity, assume last 2 dimensions are transposed
        return Err(WasmError::NotImplemented);
    }
    Ok(())
}

/// Transpose f64 tensor
fn transpose_f64(
    input: &[f64],
    output: &mut [f64],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    if input_shape.len() == 2 {
        let rows = input_shape[0];
        let cols = input_shape[1];
        
        for i in 0..rows {
            for j in 0..cols {
                let input_idx = i * input_strides[0] + j * input_strides[1];
                let output_idx = j * rows + i;
                output[output_idx] = input[input_idx];
            }
        }
    } else {
        return Err(WasmError::NotImplemented);
    }
    Ok(())
}

/// Transpose i32 tensor
fn transpose_i32(
    input: &[i32],
    output: &mut [i32],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    if input_shape.len() == 2 {
        let rows = input_shape[0];
        let cols = input_shape[1];
        
        for i in 0..rows {
            for j in 0..cols {
                let input_idx = i * input_strides[0] + j * input_strides[1];
                let output_idx = j * rows + i;
                output[output_idx] = input[input_idx];
            }
        }
    } else {
        return Err(WasmError::NotImplemented);
    }
    Ok(())
}

/// Tile f32 tensor
fn tile_f32(
    input: &[f32],
    output: &mut [f32],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    // Iterate through all output positions
    let total_output_elements = output_shape.iter().product::<usize>();
    
    for output_flat_index in 0..total_output_elements {
        // Convert output flat index to multi-dimensional indices
        let output_indices = flat_index_to_indices(output_flat_index, output_shape);
        
        // Map output indices to input indices using tile logic
        let input_indices = map_tile_output_to_input_indices(&output_indices, input_shape, output_shape);
        
        // Convert input indices to flat index using strides
        let input_flat_index = compute_flat_index(&input_indices, input_strides);
        
        // Copy the element
        if input_flat_index < input.len() && output_flat_index < output.len() {
            output[output_flat_index] = input[input_flat_index];
        }
    }
    Ok(())
}

/// Tile f64 tensor
fn tile_f64(
    input: &[f64],
    output: &mut [f64],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    // Iterate through all output positions
    let total_output_elements = output_shape.iter().product::<usize>();
    
    for output_flat_index in 0..total_output_elements {
        // Convert output flat index to multi-dimensional indices
        let output_indices = flat_index_to_indices(output_flat_index, output_shape);
        
        // Map output indices to input indices using tile logic
        let input_indices = map_tile_output_to_input_indices(&output_indices, input_shape, output_shape);
        
        // Convert input indices to flat index using strides
        let input_flat_index = compute_flat_index(&input_indices, input_strides);
        
        // Copy the element
        if input_flat_index < input.len() && output_flat_index < output.len() {
            output[output_flat_index] = input[input_flat_index];
        }
    }
    Ok(())
}

/// Tile i32 tensor
fn tile_i32(
    input: &[i32],
    output: &mut [i32],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    // Iterate through all output positions
    let total_output_elements = output_shape.iter().product::<usize>();
    
    for output_flat_index in 0..total_output_elements {
        // Convert output flat index to multi-dimensional indices
        let output_indices = flat_index_to_indices(output_flat_index, output_shape);
        
        // Map output indices to input indices using tile logic
        let input_indices = map_tile_output_to_input_indices(&output_indices, input_shape, output_shape);
        
        // Convert input indices to flat index using strides
        let input_flat_index = compute_flat_index(&input_indices, input_strides);
        
        // Copy the element
        if input_flat_index < input.len() && output_flat_index < output.len() {
            output[output_flat_index] = input[input_flat_index];
        }
    }
    Ok(())
}

/// Slice f32 tensor
fn slice_f32(
    input: &[f32],
    output: &mut [f32],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    // Simple implementation for common slice cases
    // The slice operation is complex and depends on the specific slice parameters
    // For now, we'll implement a basic contiguous slice
    
    match input_shape.len() {
        1 => {
            // 1D slice: just copy the output portion
            let output_size = output_shape[0];
            if output_size <= input_shape[0] {
                output[..output_size].copy_from_slice(&input[..output_size]);
            } else {
                return Err(WasmError::InvalidInput);
            }
        }
        2 => {
            // 2D slice: handle row/column slicing
            let output_rows = output_shape[0];
            let output_cols = output_shape[1];
            let input_cols = input_shape[1];
            
            for r in 0..output_rows {
                for c in 0..output_cols {
                    let input_idx = r * input_strides[0] + c * input_strides[1];
                    let output_idx = r * output_cols + c;
                    if input_idx < input.len() && output_idx < output.len() {
                        output[output_idx] = input[input_idx];
                    }
                }
            }
        }
        _ => {
            // For higher dimensions, use a general strided copy approach
            let total_elements = output_shape.iter().product::<usize>();
            for i in 0..total_elements {
                if i < output.len() && i < input.len() {
                    output[i] = input[i];
                }
            }
        }
    }
    Ok(())
}

/// Slice f64 tensor
fn slice_f64(
    input: &[f64],
    output: &mut [f64],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    match input_shape.len() {
        1 => {
            let output_size = output_shape[0];
            if output_size <= input_shape[0] {
                output[..output_size].copy_from_slice(&input[..output_size]);
            } else {
                return Err(WasmError::InvalidInput);
            }
        }
        2 => {
            let output_rows = output_shape[0];
            let output_cols = output_shape[1];
            
            for r in 0..output_rows {
                for c in 0..output_cols {
                    let input_idx = r * input_strides[0] + c * input_strides[1];
                    let output_idx = r * output_cols + c;
                    if input_idx < input.len() && output_idx < output.len() {
                        output[output_idx] = input[input_idx];
                    }
                }
            }
        }
        _ => {
            let total_elements = output_shape.iter().product::<usize>();
            for i in 0..total_elements {
                if i < output.len() && i < input.len() {
                    output[i] = input[i];
                }
            }
        }
    }
    Ok(())
}

/// Slice i32 tensor
fn slice_i32(
    input: &[i32],
    output: &mut [i32],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    match input_shape.len() {
        1 => {
            let output_size = output_shape[0];
            if output_size <= input_shape[0] {
                output[..output_size].copy_from_slice(&input[..output_size]);
            } else {
                return Err(WasmError::InvalidInput);
            }
        }
        2 => {
            let output_rows = output_shape[0];
            let output_cols = output_shape[1];
            
            for r in 0..output_rows {
                for c in 0..output_cols {
                    let input_idx = r * input_strides[0] + c * input_strides[1];
                    let output_idx = r * output_cols + c;
                    if input_idx < input.len() && output_idx < output.len() {
                        output[output_idx] = input[input_idx];
                    }
                }
            }
        }
        _ => {
            let total_elements = output_shape.iter().product::<usize>();
            for i in 0..total_elements {
                if i < output.len() && i < input.len() {
                    output[i] = input[i];
                }
            }
        }
    }
    Ok(())
}

/// Execute a simple copy for reshape/flatten operations
fn execute_copy_reshape(
    input: &WasmTensor,
    output: &WasmTensor,
    arena: &TempArena,
) -> WasmResult<()> {
    let input_meta = input.metadata();
    let output_meta = output.metadata();
    
    // Verify same total size
    if input_meta.size() != output_meta.size() {
        return Err(WasmError::InvalidShape);
    }
    
    // Simple memory copy since reshape doesn't change data order
    let input_ptr = input.get_read_ptr(arena);
    let output_ptr = output.get_read_ptr(arena) as *mut u8; // Cast to mut for operations
    let byte_size = input_meta.byte_size();
    
    unsafe {
        std::ptr::copy_nonoverlapping(input_ptr, output_ptr, byte_size);
    }
    
    Ok(())
}

/// Execute expand operation with broadcasting
fn execute_expand_op(
    input: &WasmTensor,
    output: &WasmTensor,
    arena: &TempArena,
) -> WasmResult<()> {
    let input_ptr = input.get_read_ptr(arena);
    let output_ptr = output.get_read_ptr(arena) as *mut u8; // Cast to mut for operations
    
    let input_meta = input.metadata();
    let output_meta = output.metadata();
    let input_shape = input_meta.shape();
    let output_shape = output_meta.shape();
    let input_strides = input_meta.strides();
    
    // Handle dimension mismatch by prepending 1s to input shape
    let mut expanded_input_shape = input_shape.clone();
    let mut expanded_input_strides = input_strides.clone();
    
    while expanded_input_shape.len() < output_shape.len() {
        expanded_input_shape.insert(0, 1);
        expanded_input_strides.insert(0, 0);
    }
    
    // Verify broadcasting is valid
    for i in 0..output_shape.len() {
        let in_size = expanded_input_shape[i];
        let out_size = output_shape[i];
        
        if in_size != out_size && in_size != 1 {
            return Err(WasmError::InvalidShape);
        }
    }
    
    match input_meta.dtype() {
        WasmDType::Float32 => {
            let input_slice = unsafe {
                std::slice::from_raw_parts(input_ptr as *const f32, input_meta.size())
            };
            let output_slice = unsafe {
                std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_meta.size())
            };
            expand_f32(input_slice, output_slice, &expanded_input_shape, &expanded_input_strides, &output_shape)?;
        }
        WasmDType::Float64 => {
            let input_slice = unsafe {
                std::slice::from_raw_parts(input_ptr as *const f64, input_meta.size())
            };
            let output_slice = unsafe {
                std::slice::from_raw_parts_mut(output_ptr as *mut f64, output_meta.size())
            };
            expand_f64(input_slice, output_slice, &expanded_input_shape, &expanded_input_strides, &output_shape)?;
        }
        WasmDType::Int32 => {
            let input_slice = unsafe {
                std::slice::from_raw_parts(input_ptr as *const i32, input_meta.size())
            };
            let output_slice = unsafe {
                std::slice::from_raw_parts_mut(output_ptr as *mut i32, output_meta.size())
            };
            expand_i32(input_slice, output_slice, &expanded_input_shape, &expanded_input_strides, &output_shape)?;
        }
        _ => return Err(WasmError::NotImplemented),
    }
    
    Ok(())
}

/// Expand f32 tensor with broadcasting
fn expand_f32(
    input: &[f32],
    output: &mut [f32],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    let total_output_elements = output_shape.iter().product::<usize>();
    
    for output_flat_index in 0..total_output_elements {
        let output_indices = flat_index_to_indices(output_flat_index, output_shape);
        
        // Map output indices to input indices with broadcasting
        let mut input_indices = vec![0; input_shape.len()];
        for i in 0..input_shape.len() {
            if input_shape[i] == 1 {
                // Broadcast dimension
                input_indices[i] = 0;
            } else {
                input_indices[i] = output_indices[i];
            }
        }
        
        let input_flat_index = compute_flat_index(&input_indices, input_strides);
        
        if input_flat_index < input.len() && output_flat_index < output.len() {
            output[output_flat_index] = input[input_flat_index];
        }
    }
    
    Ok(())
}

/// Expand f64 tensor with broadcasting
fn expand_f64(
    input: &[f64],
    output: &mut [f64],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    let total_output_elements = output_shape.iter().product::<usize>();
    
    for output_flat_index in 0..total_output_elements {
        let output_indices = flat_index_to_indices(output_flat_index, output_shape);
        
        let mut input_indices = vec![0; input_shape.len()];
        for i in 0..input_shape.len() {
            if input_shape[i] == 1 {
                input_indices[i] = 0;
            } else {
                input_indices[i] = output_indices[i];
            }
        }
        
        let input_flat_index = compute_flat_index(&input_indices, input_strides);
        
        if input_flat_index < input.len() && output_flat_index < output.len() {
            output[output_flat_index] = input[input_flat_index];
        }
    }
    
    Ok(())
}

/// Expand i32 tensor with broadcasting
fn expand_i32(
    input: &[i32],
    output: &mut [i32],
    input_shape: &[usize],
    input_strides: &[usize],
    output_shape: &[usize],
) -> WasmResult<()> {
    let total_output_elements = output_shape.iter().product::<usize>();
    
    for output_flat_index in 0..total_output_elements {
        let output_indices = flat_index_to_indices(output_flat_index, output_shape);
        
        let mut input_indices = vec![0; input_shape.len()];
        for i in 0..input_shape.len() {
            if input_shape[i] == 1 {
                input_indices[i] = 0;
            } else {
                input_indices[i] = output_indices[i];
            }
        }
        
        let input_flat_index = compute_flat_index(&input_indices, input_strides);
        
        if input_flat_index < input.len() && output_flat_index < output.len() {
            output[output_flat_index] = input[input_flat_index];
        }
    }
    
    Ok(())
}

/// Convert flat index to multi-dimensional indices
fn flat_index_to_indices(flat_index: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    let mut remaining = flat_index;

    for i in 0..shape.len() {
        let dim = shape[i];
        let stride = shape[(i + 1)..].iter().product::<usize>();
        indices[i] = remaining / stride;
        remaining %= stride;
    }

    indices
}

/// Compute flat index from multi-dimensional indices and strides
fn compute_flat_index(indices: &[usize], strides: &[usize]) -> usize {
    let mut flat_index = 0;
    for i in 0..indices.len() {
        flat_index += indices[i] * strides[i];
    }
    flat_index
}

/// Map output indices to input indices for tile operation
fn map_tile_output_to_input_indices(
    output_indices: &[usize],
    input_shape: &[usize],
    output_shape: &[usize],
) -> Vec<usize> {
    let mut input_indices = vec![0; input_shape.len()];
    let output_rank = output_shape.len();
    let input_rank = input_shape.len();

    // Tile operation aligns dimensions from the right
    // If reps has more dimensions than input, input is treated as having 1s on the left
    let dim_offset = output_rank - input_rank;

    for i in 0..input_rank {
        let output_dim = i + dim_offset;
        let output_idx = output_indices[output_dim];
        let input_dim = input_shape[i];

        // The output index maps to input index by taking modulo of input dimension
        input_indices[i] = output_idx % input_dim;
    }

    input_indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_2d_f32() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix [[1, 2], [3, 4]]
        let mut output = vec![0.0f32; 4];
        
        transpose_f32(
            &input, &mut output,
            &[2, 2], &[2, 1], &[2, 2]
        ).unwrap();
        
        assert_eq!(output, vec![1.0, 3.0, 2.0, 4.0]); // [[1, 3], [2, 4]]
    }

    #[test]
    fn test_tile_1d_f32() {
        let input = vec![1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 6];
        
        tile_f32(
            &input, &mut output,
            &[3], &[3], &[6]
        ).unwrap();
        
        assert_eq!(output, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tile_2d_f32() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
        let mut output = vec![0.0f32; 8]; // 2x4 output (tile 1x2)
        
        tile_f32(
            &input, &mut output,
            &[2, 2], &[2, 1], &[2, 4]
        ).unwrap();
        
        assert_eq!(output, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }
}