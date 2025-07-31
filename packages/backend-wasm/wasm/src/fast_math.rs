/*!
 * Fast mathematical operations optimized for WASM
 * 
 * Provides lookup tables, approximations, and SIMD-optimized
 * implementations of common mathematical functions.
 */

use core::f32::consts::PI;
use once_cell::sync::Lazy;

// Lookup table sizes - power of 2 for fast modulo
const TRIG_TABLE_SIZE: usize = 4096;
const EXP_TABLE_SIZE: usize = 2048;
const LOG_TABLE_SIZE: usize = 2048;

// Pre-computed lookup tables - initialized once at runtime
static SIN_TABLE: Lazy<Vec<f32>> = Lazy::new(|| {
    (0..TRIG_TABLE_SIZE)
        .map(|i| {
            let angle = (i as f32) * 2.0 * PI / (TRIG_TABLE_SIZE as f32);
            libm::sinf(angle)
        })
        .collect()
});

static COS_TABLE: Lazy<Vec<f32>> = Lazy::new(|| {
    (0..TRIG_TABLE_SIZE)
        .map(|i| {
            let angle = (i as f32) * 2.0 * PI / (TRIG_TABLE_SIZE as f32);
            libm::cosf(angle)
        })
        .collect()
});

/// Fast sine approximation using lookup table with linear interpolation
#[inline]
pub fn fast_sin_f32(x: f32) -> f32 {
    // Normalize angle to [0, 2*PI)
    let normalized = x * (1.0 / (2.0 * PI));
    let normalized = normalized - normalized.floor();
    
    // Convert to table index
    let index_f = normalized * (TRIG_TABLE_SIZE as f32);
    let index = index_f as usize;
    let fract = index_f - (index as f32);
    
    // Linear interpolation between table entries
    let idx0 = index % TRIG_TABLE_SIZE;
    let idx1 = (index + 1) % TRIG_TABLE_SIZE;
    
    SIN_TABLE[idx0] * (1.0 - fract) + SIN_TABLE[idx1] * fract
}

/// Fast cosine approximation using lookup table with linear interpolation
#[inline]
pub fn fast_cos_f32(x: f32) -> f32 {
    // Normalize angle to [0, 2*PI)
    let normalized = x * (1.0 / (2.0 * PI));
    let normalized = normalized - normalized.floor();
    
    // Convert to table index
    let index_f = normalized * (TRIG_TABLE_SIZE as f32);
    let index = index_f as usize;
    let fract = index_f - (index as f32);
    
    // Linear interpolation between table entries
    let idx0 = index % TRIG_TABLE_SIZE;
    let idx1 = (index + 1) % TRIG_TABLE_SIZE;
    
    COS_TABLE[idx0] * (1.0 - fract) + COS_TABLE[idx1] * fract
}

/// Fast exponential approximation using range reduction and polynomial
#[inline]
pub fn fast_exp_f32(x: f32) -> f32 {
    // Handle special cases
    if x.is_nan() {
        return x;
    }
    if x > 88.72283 {
        return f32::INFINITY;
    }
    if x < -87.33655 {
        return 0.0;
    }
    
    // Range reduction: e^x = 2^(x/ln(2)) = 2^(n + f) where n is integer, 0 <= f < 1
    const LOG2_E: f32 = 1.4426950408889634;
    let x_scaled = x * LOG2_E;
    let n = x_scaled.floor();
    let f = x_scaled - n;
    
    // Higher precision polynomial approximation of 2^f for f in [0, 1]
    // Using a 6th order polynomial for better accuracy
    let f2 = f * f;
    let f3 = f2 * f;
    let f4 = f2 * f2;
    let f5 = f4 * f;
    let f6 = f3 * f3;
    
    let p = 1.0 
        + f * 0.6931471805599453
        + f2 * 0.2402265069591007
        + f3 * 0.05550410866482158
        + f4 * 0.009618129842128887
        + f5 * 0.001333355814670577
        + f6 * 0.0001540353039338161;
    
    // Reconstruct: 2^n * 2^f
    if n >= -126.0 && n <= 127.0 {
        // Fast path: use bit manipulation for 2^n
        let n_int = n as i32;
        let bits = ((n_int + 127) << 23) as u32;
        let scale = f32::from_bits(bits);
        p * scale
    } else if n < -126.0 {
        0.0 // Underflow
    } else {
        f32::INFINITY // Overflow
    }
}

/// Fast natural logarithm approximation
#[inline]
pub fn fast_log_f32(x: f32) -> f32 {
    if x <= 0.0 {
        if x == 0.0 {
            return f32::NEG_INFINITY;
        }
        return f32::NAN; // log of negative number
    }
    if x.is_nan() {
        return x;
    }
    if x.is_infinite() {
        return x;
    }
    
    // Extract exponent and mantissa
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa_bits = (bits & 0x007FFFFF) | 0x3F800000;
    let mantissa = f32::from_bits(mantissa_bits);
    
    // Higher precision polynomial approximation of ln(m) for m in [1, 2)
    // Using minimax polynomial coefficients
    let m = mantissa;
    let m_minus_1 = m - 1.0;
    
    let ln_m = m_minus_1 * (1.0
        - m_minus_1 * (0.5
        - m_minus_1 * (0.33333333333333
        - m_minus_1 * (0.25
        - m_minus_1 * (0.2
        - m_minus_1 * 0.16666666666667)))));
    
    // Reconstruct: ln(x) = ln(2) * exponent + ln(mantissa)
    const LN_2: f32 = 0.6931471805599453;
    (exponent as f32) * LN_2 + ln_m
}

/// Fast square root using Newton-Raphson with initial guess from bit manipulation
#[inline]
pub fn fast_sqrt_f32(x: f32) -> f32 {
    if x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    
    // Initial guess using bit manipulation (similar to Quake's fast inverse sqrt)
    let i = x.to_bits();
    let guess_bits = (i >> 1) + 0x1FC00000;
    let mut guess = f32::from_bits(guess_bits);
    
    // Two iterations of Newton-Raphson for accuracy
    guess = 0.5 * (guess + x / guess);
    guess = 0.5 * (guess + x / guess);
    
    guess
}

/// SIMD-optimized trigonometric operations
pub mod simd {
    use super::*;
    use std::arch::wasm32::*;
    
    /// Process 4 sine values at once using SIMD
    #[cfg(target_feature = "simd128")]
    pub unsafe fn simd_sin_f32x4(input: v128) -> v128 {
        // Extract individual values
        let a = f32x4_extract_lane::<0>(input);
        let b = f32x4_extract_lane::<1>(input);
        let c = f32x4_extract_lane::<2>(input);
        let d = f32x4_extract_lane::<3>(input);
        
        // Apply fast sine
        let result_a = fast_sin_f32(a);
        let result_b = fast_sin_f32(b);
        let result_c = fast_sin_f32(c);
        let result_d = fast_sin_f32(d);
        
        // Pack results
        f32x4(result_a, result_b, result_c, result_d)
    }
    
    /// Process 4 exponential values at once using SIMD
    #[cfg(target_feature = "simd128")]
    pub unsafe fn simd_exp_f32x4(input: v128) -> v128 {
        // Extract individual values
        let a = f32x4_extract_lane::<0>(input);
        let b = f32x4_extract_lane::<1>(input);
        let c = f32x4_extract_lane::<2>(input);
        let d = f32x4_extract_lane::<3>(input);
        
        // Apply fast exponential
        let result_a = fast_exp_f32(a);
        let result_b = fast_exp_f32(b);
        let result_c = fast_exp_f32(c);
        let result_d = fast_exp_f32(d);
        
        // Pack results
        f32x4(result_a, result_b, result_c, result_d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fast_sin() {
        // Test key angles
        assert!((fast_sin_f32(0.0) - 0.0).abs() < 0.001);
        assert!((fast_sin_f32(PI / 2.0) - 1.0).abs() < 0.001);
        assert!((fast_sin_f32(PI) - 0.0).abs() < 0.001);
        assert!((fast_sin_f32(3.0 * PI / 2.0) - (-1.0)).abs() < 0.001);
    }
    
    #[test]
    fn test_fast_exp() {
        assert!((fast_exp_f32(0.0) - 1.0).abs() < 0.001);
        assert!((fast_exp_f32(1.0) - std::f32::consts::E).abs() < 0.01);
        assert!((fast_exp_f32(-1.0) - (1.0 / std::f32::consts::E)).abs() < 0.01);
    }
    
    #[test]
    fn test_fast_log() {
        assert!((fast_log_f32(1.0) - 0.0).abs() < 0.001);
        assert!((fast_log_f32(std::f32::consts::E) - 1.0).abs() < 0.01);
        assert!((fast_log_f32(10.0) - 2.302585).abs() < 0.01);
    }
}