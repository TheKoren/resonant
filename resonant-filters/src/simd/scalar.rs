//! Scalar fallback implementations — always compiled, used as reference.

/// Dot product: `Σ a[i] * b[i]`.
#[inline]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}
