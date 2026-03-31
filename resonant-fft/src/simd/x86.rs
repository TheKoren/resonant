//! x86_64 SSE accelerated magnitude computation.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// Compute magnitude: `sqrt(re² + im²)` for interleaved `[re, im, ...]` pairs.
///
/// Processes 2 complex values (4 floats) per iteration.
#[cfg(target_arch = "x86_64")]
pub(crate) fn magnitude(flat: &[f32], out: &mut [f32]) {
    // SAFETY: SSE2 is guaranteed on all x86_64 targets.
    unsafe { magnitude_sse(flat, out) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn magnitude_sse(flat: &[f32], out: &mut [f32]) {
    let n = out.len();
    let chunks = n / 2;
    let remainder = n % 2;

    // SAFETY: we load 4 f32s (2 complex values) per iteration.
    // flat has 2*n elements, so 4*chunks <= 2*n is valid.
    for i in 0..chunks {
        let f_off = i * 4;
        let o_off = i * 2;

        // Load [re0, im0, re1, im1]
        let v = _mm_loadu_ps(flat.as_ptr().add(f_off));

        // Square: [re0², im0², re1², im1²]
        let sq = _mm_mul_ps(v, v);

        // Horizontal add pairs: re² + im² for each complex value
        // shuffle to get [im0², re0², im1², re1²]
        let shuf = _mm_shuffle_ps::<0b10_11_00_01>(sq, sq);
        // add: [re0²+im0², im0²+re0², re1²+im1², im1²+re1²]
        let sum = _mm_add_ps(sq, shuf);
        // sqrt
        let mag = _mm_sqrt_ps(sum);

        // Extract elements 0 and 2 (the unique values)
        out[o_off] = _mm_cvtss_f32(mag);
        out[o_off + 1] = _mm_cvtss_f32(_mm_shuffle_ps::<0b10_10_10_10>(mag, mag));
    }

    // Scalar tail
    let tail_start = chunks * 2;
    for i in 0..remainder {
        let idx = (tail_start + i) * 2;
        let re = flat[idx];
        let im = flat[idx + 1];
        out[tail_start + i] = (re * re + im * im).sqrt();
    }
}

/// Compute squared magnitude: `re² + im²` for interleaved `[re, im, ...]` pairs.
#[cfg(target_arch = "x86_64")]
pub(crate) fn magnitude_squared(flat: &[f32], out: &mut [f32]) {
    unsafe { magnitude_squared_sse(flat, out) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn magnitude_squared_sse(flat: &[f32], out: &mut [f32]) {
    let n = out.len();
    let chunks = n / 2;
    let remainder = n % 2;

    for i in 0..chunks {
        let f_off = i * 4;
        let o_off = i * 2;

        let v = _mm_loadu_ps(flat.as_ptr().add(f_off));
        let sq = _mm_mul_ps(v, v);
        let shuf = _mm_shuffle_ps::<0b10_11_00_01>(sq, sq);
        let sum = _mm_add_ps(sq, shuf);

        out[o_off] = _mm_cvtss_f32(sum);
        out[o_off + 1] = _mm_cvtss_f32(_mm_shuffle_ps::<0b10_10_10_10>(sum, sum));
    }

    let tail_start = chunks * 2;
    for i in 0..remainder {
        let idx = (tail_start + i) * 2;
        let re = flat[idx];
        let im = flat[idx + 1];
        out[tail_start + i] = re * re + im * im;
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn magnitude(flat: &[f32], out: &mut [f32]) {
    super::scalar::magnitude(flat, out);
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn magnitude_squared(flat: &[f32], out: &mut [f32]) {
    super::scalar::magnitude_squared(flat, out);
}
