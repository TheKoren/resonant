//! x86_64 SSE accelerated primitives.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Element-wise multiply using SSE: `a[i] *= b[i]`.
///
/// Processes 4 floats per iteration with a scalar tail.
#[inline]
#[cfg(target_arch = "x86_64")]
pub(crate) fn multiply_buffers(a: &mut [f32], b: &[f32]) {
    // SAFETY: SSE2 is guaranteed on all x86_64 targets.
    unsafe { multiply_buffers_sse(a, b) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn multiply_buffers_sse(a: &mut [f32], b: &[f32]) {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    // SAFETY: _mm_loadu_ps / _mm_storeu_ps read/write 4 contiguous f32s.
    // Pointer ranges are bounded by `chunks`.
    for i in 0..chunks {
        let offset = i * 4;
        let va = _mm_loadu_ps(a.as_ptr().add(offset));
        let vb = _mm_loadu_ps(b.as_ptr().add(offset));
        _mm_storeu_ps(a.as_mut_ptr().add(offset), _mm_mul_ps(va, vb));
    }

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        a[tail_start + i] *= b[tail_start + i];
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn multiply_buffers(a: &mut [f32], b: &[f32]) {
    super::scalar::multiply_buffers(a, b);
}
