//! SIMD-accelerated primitives with automatic platform dispatch.
//!
//! Each function dispatches to the best available backend at compile time:
//! x86 SSE/AVX, ARM NEON, or a scalar fallback. The scalar path is always
//! compiled and serves as the reference for correctness tests.

mod scalar;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod neon;

/// Element-wise multiply: `a[i] *= b[i]` for each element.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[inline]
pub(crate) fn multiply_buffers(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "buffer length mismatch");
    dispatch_multiply_buffers(a, b);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_multiply_buffers(a: &mut [f32], b: &[f32]) {
    x86::multiply_buffers(a, b);
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_multiply_buffers(a: &mut [f32], b: &[f32]) {
    neon::multiply_buffers(a, b);
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_multiply_buffers(a: &mut [f32], b: &[f32]) {
    scalar::multiply_buffers(a, b);
}

/// Scalar reference implementation, exposed for correctness tests.
#[cfg(test)]
pub(crate) fn multiply_buffers_scalar(a: &mut [f32], b: &[f32]) {
    scalar::multiply_buffers(a, b);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_buffers_basic() {
        let mut a = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [0.5_f32, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0];
        multiply_buffers(&mut a, &b);
        assert_eq!(a, [0.5, 1.0, 1.5, 2.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn multiply_buffers_empty() {
        let mut a: [f32; 0] = [];
        let b: [f32; 0] = [];
        multiply_buffers(&mut a, &b);
    }

    #[test]
    fn multiply_buffers_matches_scalar() {
        let mut a_simd = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let mut a_scalar = a_simd;
        let b = [2.0_f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        multiply_buffers(&mut a_simd, &b);
        multiply_buffers_scalar(&mut a_scalar, &b);

        for (s, sc) in a_simd.iter().zip(a_scalar.iter()) {
            assert!((s - sc).abs() < 1e-6, "mismatch: {s} vs {sc}");
        }
    }
}
