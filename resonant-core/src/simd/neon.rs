//! ARM NEON accelerated primitives.
//!
//! Filled in by the SIMD implementation commits. Currently delegates to scalar.

/// Element-wise multiply using NEON.
#[inline]
pub(crate) fn multiply_buffers(a: &mut [f32], b: &[f32]) {
    // TODO: NEON implementation (commit 5)
    super::scalar::multiply_buffers(a, b);
}
