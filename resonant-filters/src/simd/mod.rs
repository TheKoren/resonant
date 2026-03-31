//! SIMD-accelerated filter primitives with automatic platform dispatch.
//!
//! Each function dispatches to the best available backend at compile time:
//! x86 SSE/AVX, ARM NEON, or a scalar fallback.

mod scalar;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod neon;

/// Dot product of two equal-length slices: `Σ a[i] * b[i]`.
///
/// Used as the inner loop of FIR convolution.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[allow(dead_code)] // used once FIR SIMD path lands (commit 4)
#[inline]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    dispatch_dot_product(a, b)
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_dot_product(a: &[f32], b: &[f32]) -> f32 {
    x86::dot_product(a, b)
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_dot_product(a: &[f32], b: &[f32]) -> f32 {
    neon::dot_product(a, b)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_dot_product(a: &[f32], b: &[f32]) -> f32 {
    scalar::dot_product(a, b)
}

/// Scalar reference implementation, exposed for correctness tests.
#[cfg(test)]
pub(crate) fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    scalar::dot_product(a, b)
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;

    use super::*;

    #[test]
    fn dot_product_basic() {
        let a = [1.0_f32, 2.0, 3.0, 4.0];
        let b = [0.5_f32, 0.5, 0.5, 0.5];
        let result = dot_product(&a, &b);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_empty() {
        let result = dot_product(&[], &[]);
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn dot_product_matches_scalar() {
        let a = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let b = vec![2.0_f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let simd_result = dot_product(&a, &b);
        let scalar_result = dot_product_scalar(&a, &b);

        assert!(
            (simd_result - scalar_result).abs() < 1e-4,
            "mismatch: {simd_result} vs {scalar_result}"
        );
    }
}
