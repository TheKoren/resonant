//! x86_64 SSE accelerated filter primitives.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Dot product using SSE.
///
/// Processes 4 floats per iteration with a scalar tail.
#[inline]
#[cfg(target_arch = "x86_64")]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: SSE2 is guaranteed on all x86_64 targets.
    unsafe { dot_product_sse(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    // SAFETY: all intrinsics below operate on valid __m128 values.
    // _mm_loadu_ps reads 4 contiguous f32s; pointer ranges are bounded by `chunks`.
    let mut acc = _mm_setzero_ps();
    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm_loadu_ps(b.as_ptr().add(offset));
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }

    // Horizontal sum of the 4 lanes
    let shuf = _mm_movehdup_ps(acc); // [a1,a1,a3,a3]
    let sums = _mm_add_ps(acc, shuf); // [a0+a1, _, a2+a3, _]
    let shuf2 = _mm_movehl_ps(sums, sums); // [a2+a3, _, ...]
    let total = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(total);

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        sum += a[tail_start + i] * b[tail_start + i];
    }

    sum
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    super::scalar::dot_product(a, b)
}
