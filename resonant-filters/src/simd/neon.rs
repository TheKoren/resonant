//! ARM NEON accelerated filter primitives.
//!
//! Filled in by the SIMD implementation commits. Currently delegates to scalar.

/// Dot product using NEON.
#[inline]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // TODO: NEON implementation (commit 4)
    super::scalar::dot_product(a, b)
}
