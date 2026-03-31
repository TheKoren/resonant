//! Window functions for spectral analysis.
//!
//! Each function multiplies a sample buffer in-place by the corresponding
//! window shape. All windows are symmetric (DFT-even) with length equal to
//! the slice length.

use core::f32::consts::PI;

// num-traits with `libm` feature provides f32::cos/abs via libm on no_std targets.
#[allow(unused_imports)]
use num_traits::float::Float as _;

/// Applies a Hann window in-place.
///
/// The Hann window tapers to zero at both endpoints, reducing spectral
/// leakage at the cost of slightly wider main lobe.
///
/// `w[n] = 0.5 * (1 - cos(2π n / (N-1)))`
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let mut buf = [1.0_f32; 4];
/// window::hann(&mut buf);
/// assert!((buf[0] - 0.0).abs() < 1e-6);
/// assert!((buf[3] - 0.0).abs() < 1e-6);
/// ```
pub fn hann(samples: &mut [f32]) {
    let len = samples.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f32;
    for (i, s) in samples.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / n_minus_1).cos());
        *s *= w;
    }
}

/// Applies a Hamming window in-place.
///
/// Similar to Hann but with non-zero endpoints (~0.08), giving better
/// sidelobe suppression in exchange for a slightly wider main lobe.
///
/// `w[n] = 0.54 - 0.46 * cos(2π n / (N-1))`
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let mut buf = [1.0_f32; 4];
/// window::hamming(&mut buf);
/// assert!((buf[0] - 0.08).abs() < 1e-6);
/// ```
pub fn hamming(samples: &mut [f32]) {
    let len = samples.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f32;
    for (i, s) in samples.iter_mut().enumerate() {
        let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / n_minus_1).cos();
        *s *= w;
    }
}

/// Applies a Blackman window in-place.
///
/// Three-term cosine window with excellent sidelobe suppression (~-58 dB)
/// at the cost of a wider main lobe than Hann or Hamming.
///
/// `w[n] = 0.42 - 0.5 * cos(2π n / (N-1)) + 0.08 * cos(4π n / (N-1))`
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let mut buf = [1.0_f32; 4];
/// window::blackman(&mut buf);
/// assert!((buf[0] - 0.0).abs() < 0.01);
/// ```
pub fn blackman(samples: &mut [f32]) {
    let len = samples.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f32;
    for (i, s) in samples.iter_mut().enumerate() {
        let x = 2.0 * PI * i as f32 / n_minus_1;
        let w = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
        *s *= w;
    }
}

/// Applies a rectangular window in-place (identity — no modification).
///
/// Included for API completeness so that window selection can be uniform.
/// This is a no-op.
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let mut buf = [1.0_f32, 2.0, 3.0];
/// window::rectangular(&mut buf);
/// assert_eq!(buf, [1.0, 2.0, 3.0]);
/// ```
#[inline]
pub fn rectangular(_samples: &mut [f32]) {
    // intentional no-op
}

/// Applies a Bartlett (triangular) window in-place.
///
/// Linear taper from zero at both endpoints to one at the centre.
///
/// `w[n] = 1 - |2n/(N-1) - 1|`
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let mut buf = [1.0_f32; 5];
/// window::bartlett(&mut buf);
/// assert!((buf[0] - 0.0).abs() < 1e-6);
/// assert!((buf[2] - 1.0).abs() < 1e-6);
/// assert!((buf[4] - 0.0).abs() < 1e-6);
/// ```
pub fn bartlett(samples: &mut [f32]) {
    let len = samples.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f32;
    for (i, s) in samples.iter_mut().enumerate() {
        let w = 1.0 - (2.0 * i as f32 / n_minus_1 - 1.0).abs();
        *s *= w;
    }
}

/// Applies a precomputed window to a sample buffer via element-wise multiply.
///
/// This is useful when the same window shape is reused across many frames —
/// compute the window once, then call `apply` each time instead of
/// recomputing cosines.
///
/// Uses SIMD acceleration where available (SSE on x86_64, NEON on aarch64).
///
/// # Panics
///
/// Panics if `samples` and `window` have different lengths.
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let mut buf = [1.0_f32; 4];
/// let win = [0.0, 0.5, 1.0, 0.5];
/// window::apply(&mut buf, &win);
/// assert_eq!(buf, [0.0, 0.5, 1.0, 0.5]);
/// ```
pub fn apply(samples: &mut [f32], window: &[f32]) {
    crate::simd::multiply_buffers(samples, window);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_endpoints_are_zero() {
        let mut buf = [1.0_f32; 8];
        hann(&mut buf);
        assert!((buf[0]).abs() < 1e-6);
        assert!((buf[7]).abs() < 1e-6);
    }

    #[test]
    fn hann_is_symmetric() {
        let mut buf = [1.0_f32; 16];
        hann(&mut buf);
        for i in 0..8 {
            assert!((buf[i] - buf[15 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn hann_peak_at_centre() {
        let mut buf = [1.0_f32; 5];
        hann(&mut buf);
        assert!((buf[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hamming_endpoints_are_008() {
        let mut buf = [1.0_f32; 8];
        hamming(&mut buf);
        assert!((buf[0] - 0.08).abs() < 1e-6);
        assert!((buf[7] - 0.08).abs() < 1e-6);
    }

    #[test]
    fn hamming_is_symmetric() {
        let mut buf = [1.0_f32; 16];
        hamming(&mut buf);
        for i in 0..8 {
            assert!((buf[i] - buf[15 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn blackman_endpoints_near_zero() {
        let mut buf = [1.0_f32; 8];
        blackman(&mut buf);
        assert!(buf[0].abs() < 0.01);
        assert!(buf[7].abs() < 0.01);
    }

    #[test]
    fn blackman_is_symmetric() {
        let mut buf = [1.0_f32; 16];
        blackman(&mut buf);
        for i in 0..8 {
            assert!((buf[i] - buf[15 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn rectangular_is_identity() {
        let mut buf = [1.0_f32, 2.0, 3.0, 4.0];
        rectangular(&mut buf);
        assert_eq!(buf, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bartlett_endpoints_are_zero() {
        let mut buf = [1.0_f32; 8];
        bartlett(&mut buf);
        assert!((buf[0]).abs() < 1e-6);
        assert!((buf[7]).abs() < 1e-6);
    }

    #[test]
    fn bartlett_is_symmetric() {
        let mut buf = [1.0_f32; 16];
        bartlett(&mut buf);
        for i in 0..8 {
            assert!((buf[i] - buf[15 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn bartlett_peak_at_centre() {
        let mut buf = [1.0_f32; 5];
        bartlett(&mut buf);
        assert!((buf[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn empty_slice_is_noop() {
        let mut buf: [f32; 0] = [];
        hann(&mut buf);
        hamming(&mut buf);
        blackman(&mut buf);
        rectangular(&mut buf);
        bartlett(&mut buf);
    }

    #[test]
    fn single_element_unchanged() {
        for apply in [hann, hamming, blackman, rectangular, bartlett] {
            let mut buf = [42.0_f32];
            apply(&mut buf);
            // single element: all windows return early, value unchanged
            assert_eq!(buf[0], 42.0);
        }
    }

    #[test]
    fn all_windows_values_between_zero_and_one() {
        for apply in [hann, hamming, blackman, bartlett] {
            let mut buf = [1.0_f32; 64];
            apply(&mut buf);
            for &v in &buf {
                assert!(v >= -1e-6, "window value {v} below zero");
                assert!(v <= 1.0 + 1e-6, "window value {v} above one");
            }
        }
    }
}
