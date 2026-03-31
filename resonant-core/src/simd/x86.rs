//! x86_64 SSE/AVX accelerated primitives.
//!
//! Filled in by the SIMD implementation commits. Currently delegates to scalar.

/// Element-wise multiply using SSE/AVX.
#[inline]
pub(crate) fn multiply_buffers(a: &mut [f32], b: &[f32]) {
    // TODO: SSE/AVX implementation (commit 5)
    super::scalar::multiply_buffers(a, b);
}
