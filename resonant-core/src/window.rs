//! Window functions for spectral analysis.
//!
//! Each function multiplies a sample buffer in-place by the corresponding
//! window shape. All windows are symmetric (DFT-even) with length equal to
//! the slice length.
//!
//! All functions are generic over [`Sample`](crate::sample::Sample): they work on
//! `f32`, `f64`, `Q15`, `Q31`, and integer buffers without a manual conversion step.
//! Coefficients are computed in `f64` and converted back to `S` via
//! [`Sample::from_f64`](crate::sample::Sample::from_f64), preserving Q31 precision.

use core::f64::consts::PI;

// num-traits with `libm` feature provides f64::cos/abs via libm on no_std targets.
#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::sample::Sample;

#[cfg(feature = "alloc")]
extern crate alloc;

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
pub fn hann<S: Sample>(samples: &mut [S]) {
    let len = samples.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f64;
    for (i, s) in samples.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * PI * i as f64 / n_minus_1).cos());
        *s = S::from_f64(s.to_f64() * w);
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
/// assert!((buf[0] - 0.08).abs() < 1e-4);
/// ```
pub fn hamming<S: Sample>(samples: &mut [S]) {
    let len = samples.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f64;
    for (i, s) in samples.iter_mut().enumerate() {
        let w = 0.54 - 0.46 * (2.0 * PI * i as f64 / n_minus_1).cos();
        *s = S::from_f64(s.to_f64() * w);
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
pub fn blackman<S: Sample>(samples: &mut [S]) {
    let len = samples.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f64;
    for (i, s) in samples.iter_mut().enumerate() {
        let x = 2.0 * PI * i as f64 / n_minus_1;
        let w = 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos();
        *s = S::from_f64(s.to_f64() * w);
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
pub fn rectangular<S: Sample>(_samples: &mut [S]) {
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
pub fn bartlett<S: Sample>(samples: &mut [S]) {
    let len = samples.len();
    if len <= 1 {
        return;
    }
    let n_minus_1 = (len - 1) as f64;
    for (i, s) in samples.iter_mut().enumerate() {
        let w = 1.0 - (2.0 * i as f64 / n_minus_1 - 1.0).abs();
        *s = S::from_f64(s.to_f64() * w);
    }
}

/// Applies a precomputed `f32` window to an `f32` sample buffer via element-wise multiply.
///
/// Useful when the same window shape is reused across many frames — compute
/// the window once (e.g. with `hann` on a unit buffer), then call `apply`
/// each time instead of recomputing cosines.
///
/// Uses SIMD acceleration where available (SSE2 on x86_64, NEON on aarch64).
/// For non-`f32` sample types, use the typed window functions (`hann`, `hamming`,
/// etc.) directly with a unit-amplitude buffer as the precomputed template.
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

/// Returns a Hann window of length `n` as a `Vec<f32>`.
///
/// Equivalent to calling [`hann`] on a buffer of ones. Useful when the same
/// window must be applied across many frames — compute once, then pass to
/// [`apply`] on each frame.
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let w = window::hann_window(5);
/// assert_eq!(w.len(), 5);
/// assert!(w[0].abs() < 1e-6);
/// assert!(w[4].abs() < 1e-6);
/// assert!((w[2] - 1.0).abs() < 1e-6);
/// ```
#[cfg(feature = "alloc")]
pub fn hann_window(n: usize) -> alloc::vec::Vec<f32> {
    let mut buf = alloc::vec![1.0_f32; n];
    hann(&mut buf);
    buf
}

/// Returns a Hamming window of length `n` as a `Vec<f32>`.
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let w = window::hamming_window(5);
/// assert_eq!(w.len(), 5);
/// assert!((w[0] - 0.08).abs() < 1e-4);
/// ```
#[cfg(feature = "alloc")]
pub fn hamming_window(n: usize) -> alloc::vec::Vec<f32> {
    let mut buf = alloc::vec![1.0_f32; n];
    hamming(&mut buf);
    buf
}

/// Returns a Blackman window of length `n` as a `Vec<f32>`.
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let w = window::blackman_window(5);
/// assert_eq!(w.len(), 5);
/// assert!(w[0].abs() < 0.01);
/// assert!(w[4].abs() < 0.01);
/// ```
#[cfg(feature = "alloc")]
pub fn blackman_window(n: usize) -> alloc::vec::Vec<f32> {
    let mut buf = alloc::vec![1.0_f32; n];
    blackman(&mut buf);
    buf
}

/// Returns a rectangular (all-ones) window of length `n` as a `Vec<f32>`.
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let w = window::rectangular_window(4);
/// assert_eq!(w, vec![1.0_f32; 4]);
/// ```
#[cfg(feature = "alloc")]
pub fn rectangular_window(n: usize) -> alloc::vec::Vec<f32> {
    alloc::vec![1.0_f32; n]
}

/// Returns a Bartlett (triangular) window of length `n` as a `Vec<f32>`.
///
/// # Examples
///
/// ```
/// use resonant_core::window;
///
/// let w = window::bartlett_window(5);
/// assert_eq!(w.len(), 5);
/// assert!(w[0].abs() < 1e-6);
/// assert!((w[2] - 1.0).abs() < 1e-6);
/// assert!(w[4].abs() < 1e-6);
/// ```
#[cfg(feature = "alloc")]
pub fn bartlett_window(n: usize) -> alloc::vec::Vec<f32> {
    let mut buf = alloc::vec![1.0_f32; n];
    bartlett(&mut buf);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixed::Q15;

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
        assert!((buf[0] - 0.08).abs() < 1e-4);
        assert!((buf[7] - 0.08).abs() < 1e-4);
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
        let fns: [fn(&mut [f32]); 5] = [hann, hamming, blackman, rectangular, bartlett];
        for apply in fns {
            let mut buf = [42.0_f32];
            apply(&mut buf);
            // single element: all windows return early, value unchanged
            assert_eq!(buf[0], 42.0);
        }
    }

    #[test]
    fn all_windows_values_between_zero_and_one() {
        let fns: [fn(&mut [f32]); 4] = [hann, hamming, blackman, bartlett];
        for apply in fns {
            let mut buf = [1.0_f32; 64];
            apply(&mut buf);
            for &v in &buf {
                assert!(v >= -1e-6, "window value {v} below zero");
                assert!(v <= 1.0 + 1e-6, "window value {v} above one");
            }
        }
    }

    #[test]
    fn hann_on_q15_dc_tapers_correctly() {
        // A DC buffer of Q15::MAX. After hann, endpoints should be near zero
        // and the centre sample should be close to Q15::MAX.
        let mut buf = [crate::fixed::Q15::from_f32(0.5); 9];
        hann(&mut buf);
        assert!(
            buf[0].to_f32().abs() < 0.001,
            "hann Q15 endpoint not near zero"
        );
        assert!(
            buf[8].to_f32().abs() < 0.001,
            "hann Q15 endpoint not near zero"
        );
        // centre: w[4] = 0.5*(1 - cos(π)) = 1.0; sample * 1.0 = 0.5
        assert!((buf[4].to_f32() - 0.5).abs() < 0.002);
    }

    #[test]
    fn hamming_on_q15_non_zero_endpoints() {
        let mut buf = [Q15::from_f32(1.0); 9];
        hamming(&mut buf);
        // Hamming endpoint ≈ 0.08; Q15 has ~1/32768 precision
        let endpoint = buf[0].to_f32();
        assert!((endpoint - 0.08).abs() < 0.002);
    }

    #[test]
    fn bartlett_on_f64_endpoints_and_centre() {
        let mut buf = [1.0_f64; 5];
        bartlett(&mut buf);
        assert!(buf[0].abs() < 1e-10);
        assert!((buf[2] - 1.0).abs() < 1e-10);
        assert!(buf[4].abs() < 1e-10);
    }

    #[test]
    fn apply_f32() {
        let mut buf = [1.0_f32; 4];
        let win = [0.0_f32, 0.5, 1.0, 0.5];
        apply(&mut buf, &win);
        assert_eq!(buf, [0.0, 0.5, 1.0, 0.5]);
    }

    #[test]
    #[should_panic(expected = "buffer length mismatch")]
    fn apply_length_mismatch_panics() {
        let mut buf = [1.0_f32; 4];
        let win = [1.0_f32; 3];
        apply(&mut buf, &win);
    }

    #[test]
    fn applying_window_twice_is_not_idempotent() {
        let mut buf = [1.0_f32; 8];
        hann(&mut buf);
        let after_first = buf;
        hann(&mut buf);
        // second application squares the window: values further reduce
        for i in 1..7 {
            assert!(
                buf[i] <= after_first[i] + 1e-6,
                "second hann did not reduce value at {i}"
            );
        }
    }

    #[cfg(feature = "alloc")]
    mod alloc_tests {
        use super::*;

        #[test]
        fn hann_window_matches_inplace() {
            let w = hann_window(16);
            let mut buf = alloc::vec![1.0_f32; 16];
            hann(&mut buf);
            assert_eq!(w, buf);
        }

        #[test]
        fn hamming_window_matches_inplace() {
            let w = hamming_window(16);
            let mut buf = alloc::vec![1.0_f32; 16];
            hamming(&mut buf);
            assert_eq!(w, buf);
        }

        #[test]
        fn blackman_window_matches_inplace() {
            let w = blackman_window(16);
            let mut buf = alloc::vec![1.0_f32; 16];
            blackman(&mut buf);
            assert_eq!(w, buf);
        }

        #[test]
        fn rectangular_window_is_all_ones() {
            let w = rectangular_window(8);
            assert!(w.iter().all(|&v| v == 1.0));
        }

        #[test]
        fn bartlett_window_matches_inplace() {
            let w = bartlett_window(16);
            let mut buf = alloc::vec![1.0_f32; 16];
            bartlett(&mut buf);
            assert_eq!(w, buf);
        }

        #[test]
        fn window_functions_empty() {
            assert!(hann_window(0).is_empty());
            assert!(hamming_window(0).is_empty());
            assert!(blackman_window(0).is_empty());
            assert!(rectangular_window(0).is_empty());
            assert!(bartlett_window(0).is_empty());
        }
    }
}
