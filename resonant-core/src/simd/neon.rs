//! ARM NEON accelerated primitives.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Element-wise multiply using NEON: `a[i] *= b[i]`.
///
/// Processes 4 floats per iteration with a scalar tail.
#[inline]
#[cfg(target_arch = "aarch64")]
pub(crate) fn multiply_buffers(a: &mut [f32], b: &[f32]) {
    // SAFETY: NEON is guaranteed on all aarch64 targets.
    unsafe { multiply_buffers_neon(a, b) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn multiply_buffers_neon(a: &mut [f32], b: &[f32]) {
    let len = a.len();
    let chunks = len / 4;
    let remainder = len % 4;

    // SAFETY: vld1q_f32 / vst1q_f32 read/write 4 contiguous f32s.
    // Pointer ranges are bounded by `chunks`.
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a.as_ptr().add(offset));
        let vb = vld1q_f32(b.as_ptr().add(offset));
        vst1q_f32(a.as_mut_ptr().add(offset), vmulq_f32(va, vb));
    }

    // Scalar tail
    let tail_start = chunks * 4;
    for i in 0..remainder {
        a[tail_start + i] *= b[tail_start + i];
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn multiply_buffers(a: &mut [f32], b: &[f32]) {
    super::scalar::multiply_buffers(a, b);
}
