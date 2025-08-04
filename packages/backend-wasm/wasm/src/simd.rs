/// SIMD-optimized operations for different data types
pub mod float32 {
    use std::arch::wasm32::*;
    
    /// Check if a pointer is properly aligned for SIMD operations
    #[inline]
    fn is_simd_aligned(ptr: *const f32) -> bool {
        (ptr as usize) % 16 == 0
    }
    
    /// Get the optimal chunk size based on array length and alignment
    #[inline]
    fn get_optimal_chunk_size(len: usize) -> usize {
        // Use larger chunks for better instruction-level parallelism
        if len >= 1024 {
            8 // Process 8 SIMD vectors at once (32 elements)
        } else if len >= 256 {
            4 // Process 4 SIMD vectors at once (16 elements)
        } else {
            1 // Process 1 SIMD vector at a time (4 elements)
        }
    }
    
    /// Software prefetch hint for better cache performance
    #[inline]
    unsafe fn prefetch_read(ptr: *const f32, elements_ahead: usize) {
        // Prefetch cache lines ahead for better memory performance
        let prefetch_ptr = ptr.add(elements_ahead);
        // Use a volatile read to hint the prefetch without affecting program logic
        std::ptr::read_volatile(prefetch_ptr);
    }
    /// SIMD-optimized negation for f32 arrays
    #[inline]
    pub fn simd_neg(input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        
        #[cfg(target_feature = "simd128")]
        {
            simd_neg_impl(input, output);
        }
        
        #[cfg(not(target_feature = "simd128"))]
        {
            scalar_neg_impl(input, output);
        }
    }
    
    #[cfg(target_feature = "simd128")]
    #[inline]
    fn simd_neg_impl(input: &[f32], output: &mut [f32]) {
        let len = input.len();
        let input_ptr = input.as_ptr();
        let output_ptr = output.as_mut_ptr();
        
        let chunk_size = get_optimal_chunk_size(len);
        let simd_width = 4;
        let unroll_size = chunk_size * simd_width;
        
        let unrolled_chunks = len / unroll_size;
        let remaining = len % unroll_size;
        
        unsafe {
            // Process multiple SIMD vectors at once (loop unrolling)
            for i in 0..unrolled_chunks {
                let base_idx = i * unroll_size;
                let input_base = input_ptr.add(base_idx);
                let output_base = output_ptr.add(base_idx);
                
                // Unroll the loop based on chunk size
                match chunk_size {
                    8 => {
                        // Process 8 SIMD vectors (32 elements) at once
                        let v0 = v128_load(input_base.add(0) as *const v128);
                        let v1 = v128_load(input_base.add(4) as *const v128);
                        let v2 = v128_load(input_base.add(8) as *const v128);
                        let v3 = v128_load(input_base.add(12) as *const v128);
                        let v4 = v128_load(input_base.add(16) as *const v128);
                        let v5 = v128_load(input_base.add(20) as *const v128);
                        let v6 = v128_load(input_base.add(24) as *const v128);
                        let v7 = v128_load(input_base.add(28) as *const v128);
                        
                        v128_store(output_base.add(0) as *mut v128, f32x4_neg(v0));
                        v128_store(output_base.add(4) as *mut v128, f32x4_neg(v1));
                        v128_store(output_base.add(8) as *mut v128, f32x4_neg(v2));
                        v128_store(output_base.add(12) as *mut v128, f32x4_neg(v3));
                        v128_store(output_base.add(16) as *mut v128, f32x4_neg(v4));
                        v128_store(output_base.add(20) as *mut v128, f32x4_neg(v5));
                        v128_store(output_base.add(24) as *mut v128, f32x4_neg(v6));
                        v128_store(output_base.add(28) as *mut v128, f32x4_neg(v7));
                    }
                    4 => {
                        // Process 4 SIMD vectors (16 elements) at once
                        let v0 = v128_load(input_base.add(0) as *const v128);
                        let v1 = v128_load(input_base.add(4) as *const v128);
                        let v2 = v128_load(input_base.add(8) as *const v128);
                        let v3 = v128_load(input_base.add(12) as *const v128);
                        
                        v128_store(output_base.add(0) as *mut v128, f32x4_neg(v0));
                        v128_store(output_base.add(4) as *mut v128, f32x4_neg(v1));
                        v128_store(output_base.add(8) as *mut v128, f32x4_neg(v2));
                        v128_store(output_base.add(12) as *mut v128, f32x4_neg(v3));
                    }
                    _ => {
                        // Process 1 SIMD vector (4 elements) at once
                        let v0 = v128_load(input_base as *const v128);
                        v128_store(output_base as *mut v128, f32x4_neg(v0));
                    }
                }
            }
            
            // Handle remaining elements with standard SIMD processing
            let processed = unrolled_chunks * unroll_size;
            let remaining_chunks = remaining / simd_width;
            
            for i in 0..remaining_chunks {
                let idx = processed + i * simd_width;
                let v = v128_load(input_ptr.add(idx) as *const v128);
                v128_store(output_ptr.add(idx) as *mut v128, f32x4_neg(v));
            }
            
            // Handle final scalar elements
            let final_processed = processed + remaining_chunks * simd_width;
            for i in final_processed..len {
                *output_ptr.add(i) = -*input_ptr.add(i);
            }
        }
    }
    
    #[inline]
    fn scalar_neg_impl(input: &[f32], output: &mut [f32]) {
        for (i, &val) in input.iter().enumerate() {
            output[i] = -val;
        }
    }
    
    /// SIMD-optimized absolute value for f32 arrays
    #[inline]
    pub fn simd_abs(input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        
        #[cfg(target_feature = "simd128")]
        {
            simd_abs_impl(input, output);
        }
        
        #[cfg(not(target_feature = "simd128"))]
        {
            scalar_abs_impl(input, output);
        }
    }
    
    #[cfg(target_feature = "simd128")]
    #[inline]
    fn simd_abs_impl(input: &[f32], output: &mut [f32]) {
        use std::arch::wasm32::*;
        
        let chunks = input.len() / 4;
        let remainder = input.len() % 4;
        
        // Process 4 elements at a time using SIMD
        for i in 0..chunks {
            let base_idx = i * 4;
            
            unsafe {
                // Load 4 f32 values into SIMD register
                let a = v128_load(input.as_ptr().add(base_idx) as *const v128);
                
                // Take absolute value of all 4 values simultaneously
                let result = f32x4_abs(a);
                
                // Store result back
                v128_store(output.as_mut_ptr().add(base_idx) as *mut v128, result);
            }
        }
        
        // Handle remaining elements with scalar operations
        if remainder > 0 {
            let start_idx = chunks * 4;
            for i in 0..remainder {
                output[start_idx + i] = input[start_idx + i].abs();
            }
        }
    }
    
    #[inline]
    fn scalar_abs_impl(input: &[f32], output: &mut [f32]) {
        for (i, &val) in input.iter().enumerate() {
            output[i] = val.abs();
        }
    }
    
    /// SIMD-optimized square root for f32 arrays
    #[inline]
    pub fn simd_sqrt(input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), output.len());
        
        #[cfg(target_feature = "simd128")]
        {
            simd_sqrt_impl(input, output);
        }
        
        #[cfg(not(target_feature = "simd128"))]
        {
            scalar_sqrt_impl(input, output);
        }
    }
    
    #[cfg(target_feature = "simd128")]
    #[inline]
    fn simd_sqrt_impl(input: &[f32], output: &mut [f32]) {
        use std::arch::wasm32::*;
        
        let chunks = input.len() / 4;
        let remainder = input.len() % 4;
        
        // Process 4 elements at a time using SIMD
        for i in 0..chunks {
            let base_idx = i * 4;
            
            unsafe {
                // Load 4 f32 values into SIMD register
                let a = v128_load(input.as_ptr().add(base_idx) as *const v128);
                
                // Take square root of all 4 values simultaneously
                let result = f32x4_sqrt(a);
                
                // Store result back
                v128_store(output.as_mut_ptr().add(base_idx) as *mut v128, result);
            }
        }
        
        // Handle remaining elements with scalar operations
        if remainder > 0 {
            let start_idx = chunks * 4;
            for i in 0..remainder {
                output[start_idx + i] = input[start_idx + i].sqrt();
            }
        }
    }
    
    #[inline]
    fn scalar_sqrt_impl(input: &[f32], output: &mut [f32]) {
        for (i, &val) in input.iter().enumerate() {
            output[i] = val.sqrt();
        }
    }
    
    /// SIMD-optimized addition for f32 arrays
    #[inline]
    pub fn simd_add(a: &[f32], b: &[f32], output: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), output.len());
        
        #[cfg(target_feature = "simd128")]
        {
            simd_add_impl(a, b, output);
        }
        
        #[cfg(not(target_feature = "simd128"))]
        {
            scalar_add_impl(a, b, output);
        }
    }
    
    #[cfg(target_feature = "simd128")]
    #[inline]
    fn simd_add_impl(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = a.len();
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let output_ptr = output.as_mut_ptr();
        
        let chunk_size = get_optimal_chunk_size(len);
        let simd_width = 4;
        let unroll_size = chunk_size * simd_width;
        
        let unrolled_chunks = len / unroll_size;
        let remaining = len % unroll_size;
        
        // Prefetch distance for cache optimization
        let prefetch_distance = 128; // Elements ahead to prefetch
        
        unsafe {
            // Process multiple SIMD vectors at once with prefetching
            for i in 0..unrolled_chunks {
                let base_idx = i * unroll_size;
                let a_base = a_ptr.add(base_idx);
                let b_base = b_ptr.add(base_idx);
                let output_base = output_ptr.add(base_idx);
                
                // Prefetch next cache lines if within bounds
                if base_idx + prefetch_distance < len {
                    prefetch_read(a_ptr, base_idx + prefetch_distance);
                    prefetch_read(b_ptr, base_idx + prefetch_distance);
                }
                
                // Unroll the loop based on chunk size for maximum ILP
                match chunk_size {
                    8 => {
                        // Process 8 SIMD vectors (32 elements) at once
                        let va0 = v128_load(a_base.add(0) as *const v128);
                        let va1 = v128_load(a_base.add(4) as *const v128);
                        let va2 = v128_load(a_base.add(8) as *const v128);
                        let va3 = v128_load(a_base.add(12) as *const v128);
                        let va4 = v128_load(a_base.add(16) as *const v128);
                        let va5 = v128_load(a_base.add(20) as *const v128);
                        let va6 = v128_load(a_base.add(24) as *const v128);
                        let va7 = v128_load(a_base.add(28) as *const v128);
                        
                        let vb0 = v128_load(b_base.add(0) as *const v128);
                        let vb1 = v128_load(b_base.add(4) as *const v128);
                        let vb2 = v128_load(b_base.add(8) as *const v128);
                        let vb3 = v128_load(b_base.add(12) as *const v128);
                        let vb4 = v128_load(b_base.add(16) as *const v128);
                        let vb5 = v128_load(b_base.add(20) as *const v128);
                        let vb6 = v128_load(b_base.add(24) as *const v128);
                        let vb7 = v128_load(b_base.add(28) as *const v128);
                        
                        v128_store(output_base.add(0) as *mut v128, f32x4_add(va0, vb0));
                        v128_store(output_base.add(4) as *mut v128, f32x4_add(va1, vb1));
                        v128_store(output_base.add(8) as *mut v128, f32x4_add(va2, vb2));
                        v128_store(output_base.add(12) as *mut v128, f32x4_add(va3, vb3));
                        v128_store(output_base.add(16) as *mut v128, f32x4_add(va4, vb4));
                        v128_store(output_base.add(20) as *mut v128, f32x4_add(va5, vb5));
                        v128_store(output_base.add(24) as *mut v128, f32x4_add(va6, vb6));
                        v128_store(output_base.add(28) as *mut v128, f32x4_add(va7, vb7));
                    }
                    4 => {
                        // Process 4 SIMD vectors (16 elements) at once
                        let va0 = v128_load(a_base.add(0) as *const v128);
                        let va1 = v128_load(a_base.add(4) as *const v128);
                        let va2 = v128_load(a_base.add(8) as *const v128);
                        let va3 = v128_load(a_base.add(12) as *const v128);
                        
                        let vb0 = v128_load(b_base.add(0) as *const v128);
                        let vb1 = v128_load(b_base.add(4) as *const v128);
                        let vb2 = v128_load(b_base.add(8) as *const v128);
                        let vb3 = v128_load(b_base.add(12) as *const v128);
                        
                        v128_store(output_base.add(0) as *mut v128, f32x4_add(va0, vb0));
                        v128_store(output_base.add(4) as *mut v128, f32x4_add(va1, vb1));
                        v128_store(output_base.add(8) as *mut v128, f32x4_add(va2, vb2));
                        v128_store(output_base.add(12) as *mut v128, f32x4_add(va3, vb3));
                    }
                    _ => {
                        // Process 1 SIMD vector (4 elements) at once
                        let va = v128_load(a_base as *const v128);
                        let vb = v128_load(b_base as *const v128);
                        v128_store(output_base as *mut v128, f32x4_add(va, vb));
                    }
                }
            }
            
            // Handle remaining elements with standard SIMD processing
            let processed = unrolled_chunks * unroll_size;
            let remaining_chunks = remaining / simd_width;
            
            for i in 0..remaining_chunks {
                let idx = processed + i * simd_width;
                let va = v128_load(a_ptr.add(idx) as *const v128);
                let vb = v128_load(b_ptr.add(idx) as *const v128);
                v128_store(output_ptr.add(idx) as *mut v128, f32x4_add(va, vb));
            }
            
            // Handle final scalar elements
            let final_processed = processed + remaining_chunks * simd_width;
            for i in final_processed..len {
                *output_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
            }
        }
    }
    
    #[inline]
    fn scalar_add_impl(a: &[f32], b: &[f32], output: &mut [f32]) {
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            output[i] = va + vb;
        }
    }
    
    /// SIMD-optimized multiplication for f32 arrays
    #[inline]
    pub fn simd_mul(a: &[f32], b: &[f32], output: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), output.len());
        
        #[cfg(target_feature = "simd128")]
        {
            simd_mul_impl(a, b, output);
        }
        
        #[cfg(not(target_feature = "simd128"))]
        {
            scalar_mul_impl(a, b, output);
        }
    }
    
    #[cfg(target_feature = "simd128")]
    #[inline]
    fn simd_mul_impl(a: &[f32], b: &[f32], output: &mut [f32]) {
        use std::arch::wasm32::*;
        
        let chunks = a.len() / 4;
        let remainder = a.len() % 4;
        
        // Process 4 elements at a time using SIMD
        for i in 0..chunks {
            let base_idx = i * 4;
            
            unsafe {
                // Load 4 f32 values from each array
                let va = v128_load(a.as_ptr().add(base_idx) as *const v128);
                let vb = v128_load(b.as_ptr().add(base_idx) as *const v128);
                
                // Multiply all 4 pairs simultaneously
                let result = f32x4_mul(va, vb);
                
                // Store result back
                v128_store(output.as_mut_ptr().add(base_idx) as *mut v128, result);
            }
        }
        
        // Handle remaining elements with scalar operations
        if remainder > 0 {
            let start_idx = chunks * 4;
            for i in 0..remainder {
                output[start_idx + i] = a[start_idx + i] * b[start_idx + i];
            }
        }
    }
    
    #[inline]
    fn scalar_mul_impl(a: &[f32], b: &[f32], output: &mut [f32]) {
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            output[i] = va * vb;
        }
    }
    
    /// SIMD-optimized subtraction for f32 arrays
    #[inline]
    pub fn simd_sub(a: &[f32], b: &[f32], output: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), output.len());
        
        #[cfg(target_feature = "simd128")]
        {
            simd_sub_impl(a, b, output);
        }
        
        #[cfg(not(target_feature = "simd128"))]
        {
            scalar_sub_impl(a, b, output);
        }
    }
    
    #[cfg(target_feature = "simd128")]
    #[inline]
    fn simd_sub_impl(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;
        
        unsafe {
            // Process 4 elements at a time using SIMD
            for i in 0..chunks {
                let base_idx = i * 4;
                
                // Load 4 f32 values from each array
                let va = v128_load(a.as_ptr().add(base_idx) as *const v128);
                let vb = v128_load(b.as_ptr().add(base_idx) as *const v128);
                
                // Subtract all 4 pairs simultaneously
                let result = f32x4_sub(va, vb);
                
                // Store result back
                v128_store(output.as_mut_ptr().add(base_idx) as *mut v128, result);
            }
        }
        
        // Handle remaining elements with scalar operations
        if remainder > 0 {
            let start_idx = chunks * 4;
            for i in 0..remainder {
                output[start_idx + i] = a[start_idx + i] - b[start_idx + i];
            }
        }
    }
    
    #[inline]
    fn scalar_sub_impl(a: &[f32], b: &[f32], output: &mut [f32]) {
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            output[i] = va - vb;
        }
    }
    
    /// SIMD-optimized division for f32 arrays
    #[inline]
    pub fn simd_div(a: &[f32], b: &[f32], output: &mut [f32]) {
        debug_assert_eq!(a.len(), b.len());
        debug_assert_eq!(a.len(), output.len());
        
        #[cfg(target_feature = "simd128")]
        {
            simd_div_impl(a, b, output);
        }
        
        #[cfg(not(target_feature = "simd128"))]
        {
            scalar_div_impl(a, b, output);
        }
    }
    
    #[cfg(target_feature = "simd128")]
    #[inline]
    fn simd_div_impl(a: &[f32], b: &[f32], output: &mut [f32]) {
        let len = a.len();
        let chunks = len / 4;
        let remainder = len % 4;
        
        unsafe {
            // Process 4 elements at a time using SIMD
            for i in 0..chunks {
                let base_idx = i * 4;
                
                // Load 4 f32 values from each array
                let va = v128_load(a.as_ptr().add(base_idx) as *const v128);
                let vb = v128_load(b.as_ptr().add(base_idx) as *const v128);
                
                // Divide all 4 pairs simultaneously
                let result = f32x4_div(va, vb);
                
                // Store result back
                v128_store(output.as_mut_ptr().add(base_idx) as *mut v128, result);
            }
        }
        
        // Handle remaining elements with scalar operations
        if remainder > 0 {
            let start_idx = chunks * 4;
            for i in 0..remainder {
                output[start_idx + i] = crate::utils::safe_div_f32(a[start_idx + i], b[start_idx + i]);
            }
        }
    }
    
    #[inline]
    fn scalar_div_impl(a: &[f32], b: &[f32], output: &mut [f32]) {
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            output[i] = crate::utils::safe_div_f32(va, vb);
        }
    }
}

/// SIMD-optimized operations for f64 arrays (processes 2 elements at a time)
pub mod float64 {
    /// SIMD-optimized negation for f64 arrays
    #[inline]
    pub fn simd_neg(input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());
        
        #[cfg(target_feature = "simd128")]
        {
            simd_neg_impl(input, output);
        }
        
        #[cfg(not(target_feature = "simd128"))]
        {
            scalar_neg_impl(input, output);
        }
    }
    
    #[cfg(target_feature = "simd128")]
    #[inline]
    fn simd_neg_impl(input: &[f64], output: &mut [f64]) {
        use std::arch::wasm32::*;
        
        let chunks = input.len() / 2;
        let remainder = input.len() % 2;
        
        // Process 2 elements at a time using SIMD
        for i in 0..chunks {
            let base_idx = i * 2;
            
            unsafe {
                // Load 2 f64 values into SIMD register
                let a = v128_load(input.as_ptr().add(base_idx) as *const v128);
                
                // Negate both values simultaneously
                let result = f64x2_neg(a);
                
                // Store result back
                v128_store(output.as_mut_ptr().add(base_idx) as *mut v128, result);
            }
        }
        
        // Handle remaining elements with scalar operations
        if remainder > 0 {
            let start_idx = chunks * 2;
            for i in 0..remainder {
                output[start_idx + i] = -input[start_idx + i];
            }
        }
    }
    
    #[inline]
    fn scalar_neg_impl(input: &[f64], output: &mut [f64]) {
        for (i, &val) in input.iter().enumerate() {
            output[i] = -val;
        }
    }
    
    /// SIMD-optimized absolute value for f64 arrays
    #[inline]
    pub fn simd_abs(input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());
        
        #[cfg(target_feature = "simd128")]
        {
            simd_abs_impl(input, output);
        }
        
        #[cfg(not(target_feature = "simd128"))]
        {
            scalar_abs_impl(input, output);
        }
    }
    
    #[cfg(target_feature = "simd128")]
    #[inline]
    fn simd_abs_impl(input: &[f64], output: &mut [f64]) {
        use std::arch::wasm32::*;
        
        let chunks = input.len() / 2;
        let remainder = input.len() % 2;
        
        // Process 2 elements at a time using SIMD
        for i in 0..chunks {
            let base_idx = i * 2;
            
            unsafe {
                // Load 2 f64 values into SIMD register
                let a = v128_load(input.as_ptr().add(base_idx) as *const v128);
                
                // Take absolute value of both values simultaneously
                let result = f64x2_abs(a);
                
                // Store result back
                v128_store(output.as_mut_ptr().add(base_idx) as *mut v128, result);
            }
        }
        
        // Handle remaining elements with scalar operations
        if remainder > 0 {
            let start_idx = chunks * 2;
            for i in 0..remainder {
                output[start_idx + i] = input[start_idx + i].abs();
            }
        }
    }
    
    #[inline]
    fn scalar_abs_impl(input: &[f64], output: &mut [f64]) {
        for (i, &val) in input.iter().enumerate() {
            output[i] = val.abs();
        }
    }
    
    /// SIMD-optimized square root for f64 arrays
    #[inline]
    pub fn simd_sqrt(input: &[f64], output: &mut [f64]) {
        debug_assert_eq!(input.len(), output.len());
        
        #[cfg(target_feature = "simd128")]
        {
            simd_sqrt_impl(input, output);
        }
        
        #[cfg(not(target_feature = "simd128"))]
        {
            scalar_sqrt_impl(input, output);
        }
    }
    
    #[cfg(target_feature = "simd128")]
    #[inline]
    fn simd_sqrt_impl(input: &[f64], output: &mut [f64]) {
        use std::arch::wasm32::*;
        
        let chunks = input.len() / 2;
        let remainder = input.len() % 2;
        
        // Process 2 elements at a time using SIMD
        for i in 0..chunks {
            let base_idx = i * 2;
            
            unsafe {
                // Load 2 f64 values into SIMD register
                let a = v128_load(input.as_ptr().add(base_idx) as *const v128);
                
                // Take square root of both values simultaneously
                let result = f64x2_sqrt(a);
                
                // Store result back
                v128_store(output.as_mut_ptr().add(base_idx) as *mut v128, result);
            }
        }
        
        // Handle remaining elements with scalar operations
        if remainder > 0 {
            let start_idx = chunks * 2;
            for i in 0..remainder {
                output[start_idx + i] = input[start_idx + i].sqrt();
            }
        }
    }
    
    #[inline]
    fn scalar_sqrt_impl(input: &[f64], output: &mut [f64]) {
        for (i, &val) in input.iter().enumerate() {
            output[i] = val.sqrt();
        }
    }
}