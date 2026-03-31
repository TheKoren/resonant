//! Scalar fallback implementations — always compiled, used as reference.

/// Compute magnitude: `sqrt(re² + im²)` for interleaved `[re, im, ...]` pairs.
///
/// `flat` must have even length. Output has `flat.len() / 2` elements.
#[allow(dead_code)] // fallback for non-x86/non-aarch64; also used in tests
pub(crate) fn magnitude(flat: &[f32], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let re = flat[i * 2];
        let im = flat[i * 2 + 1];
        *o = (re * re + im * im).sqrt();
    }
}

/// Compute squared magnitude: `re² + im²` for interleaved `[re, im, ...]` pairs.
#[allow(dead_code)]
pub(crate) fn magnitude_squared(flat: &[f32], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        let re = flat[i * 2];
        let im = flat[i * 2 + 1];
        *o = re * re + im * im;
    }
}
