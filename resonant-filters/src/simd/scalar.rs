//! Scalar fallback implementations — always compiled, used as reference.

/// Dot product: `Σ a[i] * b[i]`.
#[allow(dead_code)] // fallback for non-x86/non-aarch64; also used in tests
#[inline]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}
