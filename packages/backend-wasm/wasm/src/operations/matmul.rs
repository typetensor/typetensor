/*!
 * Matrix multiplication implementation for WebAssembly backend
 * 
 * Provides optimized matrix multiplication with support for:
 * - Different input dimensions (1D×1D, 1D×2D, 2D×1D, 2D×2D, ND×ND)
 * - Batched operations
 * - SIMD optimizations where available
 */

use crate::types::{WasmTensorMeta, WasmDType, WasmResult, WasmError};
use crate::memory::{WasmMemoryManager, WasmBufferHandle};
use microgemm::{MatRef, MatMut, PackSizes, Kernel, kernels::GenericKernel8x8};

/// Execute matrix multiplication operation
pub fn execute_matmul_op(
    _operation: crate::types::WasmOperation,
    input_a: &WasmBufferHandle,
    input_b: &WasmBufferHandle,
    input_meta_a: &WasmTensorMeta,
    input_meta_b: &WasmTensorMeta,
    output: &WasmBufferHandle,
    output_meta: &WasmTensorMeta,
) -> WasmResult<()> {
    let input_a_ptr = input_a.get_read_ptr();
    let input_b_ptr = input_b.get_read_ptr();
    let output_ptr = output.ptr() as *mut u8; // Cast to pointer

    let shape_a = input_meta_a.shape();
    let shape_b = input_meta_b.shape();
    let shape_out = output_meta.shape();

    let rank_a = shape_a.len();
    let rank_b = shape_b.len();

    match input_meta_a.dtype() {
        WasmDType::Float32 => {
            let a_slice = unsafe { 
                std::slice::from_raw_parts(input_a_ptr as *const f32, input_meta_a.size()) 
            };
            let b_slice = unsafe { 
                std::slice::from_raw_parts(input_b_ptr as *const f32, input_meta_b.size()) 
            };
            let out_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut f32, output_meta.size()) 
            };

            execute_matmul_f32(
                a_slice, b_slice, out_slice,
                &shape_a, &shape_b, &shape_out,
                &input_meta_a.strides(), &input_meta_b.strides(),
                rank_a, rank_b,
            )?;
        }
        WasmDType::Float64 => {
            let a_slice = unsafe { 
                std::slice::from_raw_parts(input_a_ptr as *const f64, input_meta_a.size()) 
            };
            let b_slice = unsafe { 
                std::slice::from_raw_parts(input_b_ptr as *const f64, input_meta_b.size()) 
            };
            let out_slice = unsafe { 
                std::slice::from_raw_parts_mut(output_ptr as *mut f64, output_meta.size()) 
            };

            execute_matmul_f64(
                a_slice, b_slice, out_slice,
                &shape_a, &shape_b, &shape_out,
                &input_meta_a.strides(), &input_meta_b.strides(),
                rank_a, rank_b,
            )?;
        }
        _ => return Err(WasmError::NotImplemented),
    }

    Ok(())
}

/// Matrix multiplication for f32 arrays
fn execute_matmul_f32(
    a: &[f32], b: &[f32], out: &mut [f32],
    shape_a: &[usize], shape_b: &[usize], shape_out: &[usize],
    strides_a: &[usize], strides_b: &[usize],
    rank_a: usize, rank_b: usize,
) -> WasmResult<()> {
    match (rank_a, rank_b) {
        (1, 1) => {
            // 1D × 1D → scalar (dot product)
            let n = shape_a[0];
            let mut sum = 0.0f32;
            for i in 0..n {
                sum += a[i] * b[i];
            }
            out[0] = sum;
        }
        (1, 2) => {
            // 1D × 2D → 1D (vector-matrix multiply)
            let k = shape_a[0]; // vector length
            let n = shape_b[1]; // matrix columns
            
            for j in 0..n {
                let mut sum = 0.0f32;
                for i in 0..k {
                    let a_idx = i * strides_a[0];
                    let b_idx = i * strides_b[0] + j * strides_b[1];
                    sum += a[a_idx] * b[b_idx];
                }
                out[j] = sum;
            }
        }
        (2, 1) => {
            // 2D × 1D → 1D (matrix-vector multiply)
            let m = shape_a[0]; // matrix rows
            let k = shape_a[1]; // matrix columns / vector length
            
            for i in 0..m {
                let mut sum = 0.0f32;
                for j in 0..k {
                    let a_idx = i * strides_a[0] + j * strides_a[1];
                    let b_idx = j * strides_b[0];
                    sum += a[a_idx] * b[b_idx];
                }
                out[i] = sum;
            }
        }
        (2, 2) => {
            // 2D × 2D → 2D (matrix-matrix multiply)
            let m = shape_a[0]; // A rows
            let k = shape_a[1]; // A cols / B rows
            let n = shape_b[1]; // B cols
            
            execute_gemm_f32(a, b, out, m, k, n, strides_a, strides_b);
        }
        _ => {
            // ND × ND → ND (batched matrix multiply)
            execute_batched_matmul_f32(
                a, b, out, shape_a, shape_b, shape_out, strides_a, strides_b
            )?;
        }
    }
    Ok(())
}

/// Matrix multiplication for f64 arrays
fn execute_matmul_f64(
    a: &[f64], b: &[f64], out: &mut [f64],
    shape_a: &[usize], shape_b: &[usize], shape_out: &[usize],
    strides_a: &[usize], strides_b: &[usize],
    rank_a: usize, rank_b: usize,
) -> WasmResult<()> {
    match (rank_a, rank_b) {
        (1, 1) => {
            // 1D × 1D → scalar (dot product)
            let n = shape_a[0];
            let mut sum = 0.0f64;
            for i in 0..n {
                sum += a[i] * b[i];
            }
            out[0] = sum;
        }
        (1, 2) => {
            // 1D × 2D → 1D (vector-matrix multiply)
            let k = shape_a[0]; // vector length
            let n = shape_b[1]; // matrix columns
            
            for j in 0..n {
                let mut sum = 0.0f64;
                for i in 0..k {
                    let a_idx = i * strides_a[0];
                    let b_idx = i * strides_b[0] + j * strides_b[1];
                    sum += a[a_idx] * b[b_idx];
                }
                out[j] = sum;
            }
        }
        (2, 1) => {
            // 2D × 1D → 1D (matrix-vector multiply)
            let m = shape_a[0]; // matrix rows
            let k = shape_a[1]; // matrix columns / vector length
            
            for i in 0..m {
                let mut sum = 0.0f64;
                for j in 0..k {
                    let a_idx = i * strides_a[0] + j * strides_a[1];
                    let b_idx = j * strides_b[0];
                    sum += a[a_idx] * b[b_idx];
                }
                out[i] = sum;
            }
        }
        (2, 2) => {
            // 2D × 2D → 2D (matrix-matrix multiply)
            let m = shape_a[0]; // A rows
            let k = shape_a[1]; // A cols / B rows
            let n = shape_b[1]; // B cols
            
            execute_gemm_f64(a, b, out, m, k, n, strides_a, strides_b);
        }
        _ => {
            // ND × ND → ND (batched matrix multiply)
            execute_batched_matmul_f64(
                a, b, out, shape_a, shape_b, shape_out, strides_a, strides_b
            )?;
        }
    }
    Ok(())
}

/// Optimized GEMM (General Matrix Multiply) for f32 using microgemm
fn execute_gemm_f32(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, k: usize, n: usize,
    strides_a: &[usize], strides_b: &[usize],
) {
    // Use microgemm for high-performance matrix multiplication
    let kernel = GenericKernel8x8::<f32>::new();
    
    // Check if we can use contiguous layout for better performance
    let a_contiguous = strides_a[0] == k && strides_a[1] == 1;
    let b_contiguous = strides_b[0] == n && strides_b[1] == 1;
    
    if a_contiguous && b_contiguous {
        // Fast path: contiguous matrices
        execute_gemm_contiguous_f32(&kernel, a, b, c, m, k, n);
    } else {
        // Slow path: strided matrices - fall back to naive implementation for now
        // TODO: Implement proper strided matrix handling with microgemm
        execute_gemm_naive_f32(a, b, c, m, k, n, strides_a, strides_b);
    }
}

/// Fast contiguous matrix multiplication using microgemm
fn execute_gemm_contiguous_f32(
    kernel: &GenericKernel8x8<f32>,
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, k: usize, n: usize,
) {
    // Create matrix wrappers using row-major layout
    let matrix_a = MatRef::row_major(m, k, a);
    let matrix_b = MatRef::row_major(k, n, b);
    let mut matrix_c = MatMut::row_major(m, n, c);
    
    // Configure optimal blocking parameters for WASM
    // Dimensions must be divisible by kernel dimensions (mr=8, nr=8 for GenericKernel8x8)
    let mr = kernel.mr();
    let nr = kernel.nr();
    
    let pack_sizes = PackSizes {
        mc: ((256.min(m) + mr - 1) / mr) * mr, // Round up to multiple of mr
        kc: 128.min(k), // Inner dimension blocking
        nc: ((256.min(n) + nr - 1) / nr) * nr, // Round up to multiple of nr
    };
    
    // Calculate required packing buffer size
    let packing_buf_size = pack_sizes.buf_len();
    let mut packing_buf = vec![0.0f32; packing_buf_size];
    
    // Perform optimized matrix multiplication: C = A * B (alpha=1.0, beta=0.0)
    kernel.gemm(
        1.0,           // alpha (scalar for A*B)
        matrix_a,      // matrix A
        matrix_b,      // matrix B  
        0.0,           // beta (scalar for existing C)
        &mut matrix_c, // output matrix C
        pack_sizes,    // blocking configuration
        &mut packing_buf, // temporary packing buffer
    );
}

/// Fallback naive implementation for strided matrices
fn execute_gemm_naive_f32(
    a: &[f32], b: &[f32], c: &mut [f32],
    m: usize, k: usize, n: usize,
    strides_a: &[usize], strides_b: &[usize],
) {
    let stride_a_row = strides_a[0];
    let stride_a_col = strides_a[1];
    let stride_b_row = strides_b[0];
    let stride_b_col = strides_b[1];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_idx = i * stride_a_row + p * stride_a_col;
                let b_idx = p * stride_b_row + j * stride_b_col;
                sum += a[a_idx] * b[b_idx];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Optimized GEMM (General Matrix Multiply) for f64 using microgemm
fn execute_gemm_f64(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, k: usize, n: usize,
    strides_a: &[usize], strides_b: &[usize],
) {
    // Use microgemm for high-performance matrix multiplication
    let kernel = GenericKernel8x8::<f64>::new();
    
    // Check if we can use contiguous layout for better performance
    let a_contiguous = strides_a[0] == k && strides_a[1] == 1;
    let b_contiguous = strides_b[0] == n && strides_b[1] == 1;
    
    if a_contiguous && b_contiguous {
        // Fast path: contiguous matrices
        execute_gemm_contiguous_f64(&kernel, a, b, c, m, k, n);
    } else {
        // Slow path: strided matrices - fall back to naive implementation for now
        // TODO: Implement proper strided matrix handling with microgemm
        execute_gemm_naive_f64(a, b, c, m, k, n, strides_a, strides_b);
    }
}

/// Fast contiguous matrix multiplication using microgemm for f64
fn execute_gemm_contiguous_f64(
    kernel: &GenericKernel8x8<f64>,
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, k: usize, n: usize,
) {
    // Create matrix wrappers using row-major layout
    let matrix_a = MatRef::row_major(m, k, a);
    let matrix_b = MatRef::row_major(k, n, b);
    let mut matrix_c = MatMut::row_major(m, n, c);
    
    // Configure optimal blocking parameters for WASM
    // Dimensions must be divisible by kernel dimensions (mr=8, nr=8 for GenericKernel8x8)
    let mr = kernel.mr();
    let nr = kernel.nr();
    
    let pack_sizes = PackSizes {
        mc: ((256.min(m) + mr - 1) / mr) * mr, // Round up to multiple of mr
        kc: 128.min(k), // Inner dimension blocking
        nc: ((256.min(n) + nr - 1) / nr) * nr, // Round up to multiple of nr
    };
    
    // Calculate required packing buffer size
    let packing_buf_size = pack_sizes.buf_len();
    let mut packing_buf = vec![0.0f64; packing_buf_size];
    
    // Perform optimized matrix multiplication: C = A * B (alpha=1.0, beta=0.0)
    kernel.gemm(
        1.0,           // alpha (scalar for A*B)
        matrix_a,      // matrix A
        matrix_b,      // matrix B  
        0.0,           // beta (scalar for existing C)
        &mut matrix_c, // output matrix C
        pack_sizes,    // blocking configuration
        &mut packing_buf, // temporary packing buffer
    );
}

/// Fallback naive implementation for strided matrices (f64)
fn execute_gemm_naive_f64(
    a: &[f64], b: &[f64], c: &mut [f64],
    m: usize, k: usize, n: usize,
    strides_a: &[usize], strides_b: &[usize],
) {
    let stride_a_row = strides_a[0];
    let stride_a_col = strides_a[1];
    let stride_b_row = strides_b[0];
    let stride_b_col = strides_b[1];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                let a_idx = i * stride_a_row + p * stride_a_col;
                let b_idx = p * stride_b_row + j * stride_b_col;
                sum += a[a_idx] * b[b_idx];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Batched matrix multiplication for f32
fn execute_batched_matmul_f32(
    a: &[f32], b: &[f32], out: &mut [f32],
    shape_a: &[usize], shape_b: &[usize], shape_out: &[usize],
    strides_a: &[usize], strides_b: &[usize],
) -> WasmResult<()> {
    let rank_a = shape_a.len();
    let rank_b = shape_b.len();
    let rank_out = shape_out.len();

    // Extract matrix dimensions
    let m = shape_a[rank_a - 2];
    let k = shape_a[rank_a - 1];
    let n = shape_b[rank_b - 1];

    // Compute batch size
    let mut batch_size = 1;
    for i in 0..(rank_out - 2) {
        batch_size *= shape_out[i];
    }

    // For batched matmul, calculate the batch strides properly
    for batch in 0..batch_size {
        // Compute batch indices
        let mut batch_indices = vec![0; rank_out - 2];
        let mut temp = batch;
        for i in (0..(rank_out - 2)).rev() {
            batch_indices[i] = temp % shape_out[i];
            temp /= shape_out[i];
        }
        
        // Compute batch offsets
        let mut offset_a = 0;
        let mut offset_b = 0;
        
        for i in 0..batch_indices.len() {
            let batch_idx = batch_indices[i];

            // Map to input A
            if i < rank_a - 2 && shape_a[i] > 1 {
                offset_a += batch_idx * strides_a[i];
            }

            // Map to input B
            if i < rank_b - 2 && shape_b[i] > 1 {
                offset_b += batch_idx * strides_b[i];
            }
        }

        // Perform matrix multiplication for this batch
        let base_out_idx = batch * m * n;
        let stride_a_row = strides_a[rank_a - 2];
        let stride_a_col = strides_a[rank_a - 1];
        let stride_b_row = strides_b[rank_b - 2];
        let stride_b_col = strides_b[rank_b - 1];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    let a_idx = offset_a + i * stride_a_row + p * stride_a_col;
                    let b_idx = offset_b + p * stride_b_row + j * stride_b_col;
                    sum += a[a_idx] * b[b_idx];
                }
                out[base_out_idx + i * n + j] = sum;
            }
        }
    }

    Ok(())
}

/// Batched matrix multiplication for f64
fn execute_batched_matmul_f64(
    a: &[f64], b: &[f64], out: &mut [f64],
    shape_a: &[usize], shape_b: &[usize], shape_out: &[usize],
    strides_a: &[usize], strides_b: &[usize],
) -> WasmResult<()> {
    // Similar implementation to f32 version
    let rank_a = shape_a.len();
    let rank_b = shape_b.len();
    let rank_out = shape_out.len();

    let m = shape_a[rank_a - 2];
    let k = shape_a[rank_a - 1];
    let n = shape_b[rank_b - 1];

    let mut batch_size = 1;
    for i in 0..(rank_out - 2) {
        batch_size *= shape_out[i];
    }

    for batch in 0..batch_size {
        // Compute batch indices
        let mut batch_indices = vec![0; rank_out - 2];
        let mut temp = batch;
        for i in (0..(rank_out - 2)).rev() {
            batch_indices[i] = temp % shape_out[i];
            temp /= shape_out[i];
        }
        
        // Compute batch offsets
        let mut offset_a = 0;
        let mut offset_b = 0;
        
        for i in 0..batch_indices.len() {
            let batch_idx = batch_indices[i];

            // Map to input A
            if i < rank_a - 2 && shape_a[i] > 1 {
                offset_a += batch_idx * strides_a[i];
            }

            // Map to input B
            if i < rank_b - 2 && shape_b[i] > 1 {
                offset_b += batch_idx * strides_b[i];
            }
        }

        let base_out_idx = batch * m * n;
        let stride_a_row = strides_a[rank_a - 2];
        let stride_a_col = strides_a[rank_a - 1];
        let stride_b_row = strides_b[rank_b - 2];
        let stride_b_col = strides_b[rank_b - 1];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    let a_idx = offset_a + i * stride_a_row + p * stride_a_col;
                    let b_idx = offset_b + p * stride_b_row + j * stride_b_col;
                    sum += a[a_idx] * b[b_idx];
                }
                out[base_out_idx + i * n + j] = sum;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let mut out = vec![0.0f32; 1];
        
        execute_matmul_f32(
            &a, &b, &mut out,
            &[3], &[3], &[1],
            &[1], &[1],
            1, 1,
        ).unwrap();
        
        assert_eq!(out[0], 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_matrix_vector_multiply() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0f32, 6.0]; // 2x1 vector
        let mut out = vec![0.0f32; 2];
        
        execute_matmul_f32(
            &a, &b, &mut out,
            &[2, 2], &[2], &[2],
            &[2, 1], &[1],
            2, 1,
        ).unwrap();
        
        assert_eq!(out, vec![17.0, 39.0]); // [1*5+2*6, 3*5+4*6] = [17, 39]
    }

    #[test]
    fn test_matrix_matrix_multiply() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut out = vec![0.0f32; 4];
        
        execute_matmul_f32(
            &a, &b, &mut out,
            &[2, 2], &[2, 2], &[2, 2],
            &[2, 1], &[2, 1],
            2, 2,
        ).unwrap();
        
        assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]); // [[19, 22], [43, 50]]
    }
}