//! Scalar fallback implementations — always compiled, used as reference.

/// Element-wise multiply: `a[i] *= b[i]`.
#[inline]
pub(crate) fn multiply_buffers(a: &mut [f32], b: &[f32]) {
    for (a, b) in a.iter_mut().zip(b.iter()) {
        *a *= *b;
    }
}
