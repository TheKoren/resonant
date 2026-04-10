//! ARM NEON accelerated filter primitives.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Dot product using NEON.
///
/// Processes 4 floats per iteration with a scalar tail.
#[inline]
#[cfg(target_arch = "aarch64")]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: NEON is guaranteed on all aarch64 targets.
    unsafe { dot_product_neon(a, b) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    // SAFETY: vld1q_f32 reads 4 contiguous f32s. Pointer range is valid via `chunks`.
    let mut acc = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        acc = vfmaq_f32(acc, va, vb);
    }

    // Horizontal sum
    let mut sum = vaddvq_f32(acc);

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        sum += a[tail_start + i] * b[tail_start + i];
    }

    sum
}

// Fallback if somehow compiled on non-aarch64
#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    super::scalar::dot_product(a, b)
}
