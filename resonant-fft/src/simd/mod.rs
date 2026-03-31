//! SIMD-accelerated FFT utilities with automatic platform dispatch.

mod scalar;

#[cfg(target_arch = "x86_64")]
mod x86;

#[cfg(target_arch = "aarch64")]
mod neon;

/// Compute magnitude of interleaved complex data: `sqrt(re² + im²)`.
///
/// `flat` is `[re0, im0, re1, im1, ...]` with length `2 * out.len()`.
///
/// # Panics
///
/// Panics if `flat.len() != 2 * out.len()`.
#[inline]
pub(crate) fn magnitude(flat: &[f32], out: &mut [f32]) {
    assert_eq!(flat.len(), 2 * out.len(), "flat must be 2× output length");
    dispatch_magnitude(flat, out);
}

/// Compute squared magnitude of interleaved complex data: `re² + im²`.
///
/// # Panics
///
/// Panics if `flat.len() != 2 * out.len()`.
#[inline]
pub(crate) fn magnitude_squared(flat: &[f32], out: &mut [f32]) {
    assert_eq!(flat.len(), 2 * out.len(), "flat must be 2× output length");
    dispatch_magnitude_squared(flat, out);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_magnitude(flat: &[f32], out: &mut [f32]) {
    x86::magnitude(flat, out);
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_magnitude(flat: &[f32], out: &mut [f32]) {
    neon::magnitude(flat, out);
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_magnitude(flat: &[f32], out: &mut [f32]) {
    scalar::magnitude(flat, out);
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_magnitude_squared(flat: &[f32], out: &mut [f32]) {
    x86::magnitude_squared(flat, out);
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_magnitude_squared(flat: &[f32], out: &mut [f32]) {
    neon::magnitude_squared(flat, out);
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_magnitude_squared(flat: &[f32], out: &mut [f32]) {
    scalar::magnitude_squared(flat, out);
}

/// Scalar reference, exposed for correctness tests.
#[cfg(test)]
pub(crate) fn magnitude_scalar(flat: &[f32], out: &mut [f32]) {
    scalar::magnitude(flat, out);
}

/// Scalar reference, exposed for correctness tests.
#[cfg(test)]
pub(crate) fn magnitude_squared_scalar(flat: &[f32], out: &mut [f32]) {
    scalar::magnitude_squared(flat, out);
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;

    use super::*;

    #[test]
    fn magnitude_basic() {
        // (3, 4) -> 5, (1, 0) -> 1
        let flat = [3.0_f32, 4.0, 1.0, 0.0];
        let mut out = [0.0_f32; 2];
        magnitude(&flat, &mut out);
        assert!((out[0] - 5.0).abs() < 1e-4);
        assert!((out[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn magnitude_squared_basic() {
        let flat = [3.0_f32, 4.0, 1.0, 0.0];
        let mut out = [0.0_f32; 2];
        magnitude_squared(&flat, &mut out);
        assert!((out[0] - 25.0).abs() < 1e-4);
        assert!((out[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn magnitude_matches_scalar() {
        let flat: Vec<f32> = (0..18).map(|i| i as f32 * 0.1).collect();
        let n = flat.len() / 2;
        let mut out_simd = vec![0.0_f32; n];
        let mut out_scalar = vec![0.0_f32; n];

        magnitude(&flat, &mut out_simd);
        magnitude_scalar(&flat, &mut out_scalar);

        for (s, sc) in out_simd.iter().zip(out_scalar.iter()) {
            assert!((s - sc).abs() < 1e-4, "mismatch: {s} vs {sc}");
        }
    }

    #[test]
    fn magnitude_squared_matches_scalar() {
        let flat: Vec<f32> = (0..18).map(|i| i as f32 * 0.1).collect();
        let n = flat.len() / 2;
        let mut out_simd = vec![0.0_f32; n];
        let mut out_scalar = vec![0.0_f32; n];

        magnitude_squared(&flat, &mut out_simd);
        magnitude_squared_scalar(&flat, &mut out_scalar);

        for (s, sc) in out_simd.iter().zip(out_scalar.iter()) {
            assert!((s - sc).abs() < 1e-4, "mismatch: {s} vs {sc}");
        }
    }

    #[test]
    fn magnitude_empty() {
        let flat: [f32; 0] = [];
        let mut out: [f32; 0] = [];
        magnitude(&flat, &mut out);
        magnitude_squared(&flat, &mut out);
    }
}
