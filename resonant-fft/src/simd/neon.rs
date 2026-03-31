//! ARM NEON accelerated magnitude computation.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Compute magnitude: `sqrt(re² + im²)` for interleaved `[re, im, ...]` pairs.
#[cfg(target_arch = "aarch64")]
pub(crate) fn magnitude(flat: &[f32], out: &mut [f32]) {
    // SAFETY: NEON is guaranteed on all aarch64 targets.
    unsafe { magnitude_neon(flat, out) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn magnitude_neon(flat: &[f32], out: &mut [f32]) {
    let n = out.len();
    let chunks = n / 4;
    let remainder = n - chunks * 4;

    // SAFETY: vld2q_f32 deinterleaves 8 f32s into two 4-wide vectors.
    // We process 4 complex values (8 floats) per iteration.
    for i in 0..chunks {
        let f_off = i * 8;
        let o_off = i * 4;

        let pair = vld2q_f32(flat.as_ptr().add(f_off));
        let re = pair.0;
        let im = pair.1;
        let re_sq = vmulq_f32(re, re);
        let sum = vfmaq_f32(re_sq, im, im);
        let mag = vsqrtq_f32(sum);
        vst1q_f32(out.as_mut_ptr().add(o_off), mag);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = (tail_start + i) * 2;
        let re = flat[idx];
        let im = flat[idx + 1];
        out[tail_start + i] = (re * re + im * im).sqrt();
    }
}

/// Compute squared magnitude: `re² + im²`.
#[cfg(target_arch = "aarch64")]
pub(crate) fn magnitude_squared(flat: &[f32], out: &mut [f32]) {
    unsafe { magnitude_squared_neon(flat, out) }
}

#[cfg(target_arch = "aarch64")]
unsafe fn magnitude_squared_neon(flat: &[f32], out: &mut [f32]) {
    let n = out.len();
    let chunks = n / 4;
    let remainder = n - chunks * 4;

    for i in 0..chunks {
        let f_off = i * 8;
        let o_off = i * 4;

        let pair = vld2q_f32(flat.as_ptr().add(f_off));
        let re = pair.0;
        let im = pair.1;
        let re_sq = vmulq_f32(re, re);
        let sum = vfmaq_f32(re_sq, im, im);
        vst1q_f32(out.as_mut_ptr().add(o_off), sum);
    }

    let tail_start = chunks * 4;
    for i in 0..remainder {
        let idx = (tail_start + i) * 2;
        let re = flat[idx];
        let im = flat[idx + 1];
        out[tail_start + i] = re * re + im * im;
    }
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn magnitude(flat: &[f32], out: &mut [f32]) {
    super::scalar::magnitude(flat, out);
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn magnitude_squared(flat: &[f32], out: &mut [f32]) {
    super::scalar::magnitude_squared(flat, out);
}
