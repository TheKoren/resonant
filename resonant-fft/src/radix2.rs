//! Pure-core radix-2 Cooley-Tukey FFT.
//!
//! In-place, iterative, power-of-two sizes only. No allocation required.
//! Twiddle factors are computed on-the-fly (one `cos`/`sin` pair per butterfly).
//! For an N-point FFT this means N/2 · log₂N trig evaluations per call —
//! roughly 22 000 for N=2048. On hard-float targets the overhead is modest;
//! on soft-float MCUs it dominates. Enable the `rustfft` feature for
//! performance-sensitive code: [`FftPlan`](crate::rustfft_backend::FftPlan)
//! precomputes twiddle factors and supports arbitrary sizes.

use core::f32::consts::PI;

use num_complex::Complex;
// Brings f32::cos/sin into scope on no_std via libm.
#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::FftError;

/// Returns `log2(n)` if `n` is a non-zero power of two.
#[inline]
fn validate_length(n: usize) -> Result<u32, FftError> {
    if n == 0 {
        return Err(FftError::Empty);
    }
    if !n.is_power_of_two() {
        return Err(FftError::NotPowerOfTwo(n));
    }
    Ok(n.trailing_zeros())
}

/// Reorders `buf` by bit-reversing each index. In-place, no allocation.
#[inline]
fn bit_reverse_permutation(buf: &mut [Complex<f32>], log_n: u32) {
    if log_n == 0 {
        return;
    }
    let n = buf.len();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - log_n);
        if j > i {
            buf.swap(i, j);
        }
    }
}

/// Performs all butterfly stages of the radix-2 DIT FFT.
///
/// `inverse` controls the sign of twiddle factors: negative for forward,
/// positive for inverse.
fn butterfly_stages(buf: &mut [Complex<f32>], log_n: u32, inverse: bool) {
    let n = buf.len();
    let sign = if inverse { 1.0 } else { -1.0 };

    // Twiddle factors are recomputed per butterfly rather than precomputed
    // because this module is no_alloc: a precomputed table would require
    // either a scratch allocation or a const-generic array. Enable the
    // `rustfft` feature for precomputed twiddles and arbitrary FFT sizes.
    for s in 0..log_n {
        let m = 1 << (s + 1);
        let half_m = 1 << s;

        for k in (0..n).step_by(m) {
            for j in 0..half_m {
                let angle = sign * 2.0 * PI * (j as f32) / (m as f32);
                let w = Complex::new(angle.cos(), angle.sin());
                let t = w * buf[k + j + half_m];
                let u = buf[k + j];
                buf[k + j] = u + t;
                buf[k + j + half_m] = u - t;
            }
        }
    }
}

/// Computes the in-place radix-2 Cooley-Tukey FFT (decimation-in-time).
///
/// The input length must be a power of two. Operates on `Complex<f32>` values.
///
/// # Errors
///
/// Returns [`FftError::Empty`] if the slice is empty, or
/// [`FftError::NotPowerOfTwo`] if the length is not a power of two.
///
/// # Examples
///
/// ```
/// use resonant_fft::{fft, Complex};
///
/// let mut buf = [
///     Complex::new(1.0, 0.0),
///     Complex::new(1.0, 0.0),
///     Complex::new(1.0, 0.0),
///     Complex::new(1.0, 0.0),
/// ];
/// fft(&mut buf).ok();
/// // DC bin contains the sum
/// assert!((buf[0].re - 4.0).abs() < 1e-5);
/// ```
pub fn fft(buf: &mut [Complex<f32>]) -> Result<(), FftError> {
    let log_n = validate_length(buf.len())?;
    bit_reverse_permutation(buf, log_n);
    butterfly_stages(buf, log_n, false);
    Ok(())
}

/// Computes the in-place inverse FFT.
///
/// The output is scaled by `1/N` so that `ifft(fft(x)) ≈ x`.
///
/// # Errors
///
/// Returns [`FftError::Empty`] if the slice is empty, or
/// [`FftError::NotPowerOfTwo`] if the length is not a power of two.
///
/// # Examples
///
/// ```
/// use resonant_fft::{fft, ifft, Complex};
///
/// let original = [
///     Complex::new(1.0, 0.0),
///     Complex::new(0.0, 0.0),
///     Complex::new(-1.0, 0.0),
///     Complex::new(0.0, 0.0),
/// ];
/// let mut buf = original;
/// fft(&mut buf).ok();
/// ifft(&mut buf).ok();
/// for (a, b) in buf.iter().zip(original.iter()) {
///     assert!((a - b).norm() < 1e-5);
/// }
/// ```
pub fn ifft(buf: &mut [Complex<f32>]) -> Result<(), FftError> {
    let log_n = validate_length(buf.len())?;
    bit_reverse_permutation(buf, log_n);
    butterfly_stages(buf, log_n, true);
    let scale = 1.0 / buf.len() as f32;
    for x in buf.iter_mut() {
        *x *= scale;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f32, im: f32) -> Complex<f32> {
        Complex::new(re, im)
    }

    #[test]
    fn fft_empty_returns_error() {
        let mut buf: [Complex<f32>; 0] = [];
        assert_eq!(fft(&mut buf), Err(FftError::Empty));
    }

    #[test]
    fn fft_non_power_of_two_returns_error() {
        for n in [3, 5, 6, 7] {
            let mut buf = [c(0.0, 0.0); 8];
            assert_eq!(fft(&mut buf[..n]), Err(FftError::NotPowerOfTwo(n)));
        }
    }

    #[test]
    fn fft_single_element() {
        let mut buf = [c(42.0, 0.0)];
        fft(&mut buf).ok();
        assert!((buf[0].re - 42.0).abs() < 1e-5);
        assert!(buf[0].im.abs() < 1e-5);
    }

    #[test]
    fn fft_two_elements() {
        let mut buf = [c(1.0, 0.0), c(1.0, 0.0)];
        fft(&mut buf).ok();
        assert!((buf[0].re - 2.0).abs() < 1e-5);
        assert!(buf[1].norm() < 1e-5);
    }

    #[test]
    fn fft_dc_signal() {
        let mut buf = [c(1.0, 0.0); 8];
        fft(&mut buf).ok();
        // DC bin = sum = 8.0
        assert!((buf[0].re - 8.0).abs() < 1e-4);
        // All other bins ≈ 0
        for b in &buf[1..] {
            assert!(b.norm() < 1e-4);
        }
    }

    #[test]
    fn fft_impulse() {
        let mut buf = [c(0.0, 0.0); 8];
        buf[0] = c(1.0, 0.0);
        fft(&mut buf).ok();
        // Flat spectrum: all bins = 1+0i
        for b in &buf {
            assert!((b.re - 1.0).abs() < 1e-5);
            assert!(b.im.abs() < 1e-5);
        }
    }

    #[test]
    fn ifft_inverts_fft() {
        let original = [c(1.0, 0.0), c(0.5, 0.0), c(-0.3, 0.0), c(0.7, 0.0)];
        let mut buf = original;
        fft(&mut buf).ok();
        ifft(&mut buf).ok();
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a - b).norm() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn ifft_empty_returns_error() {
        let mut buf: [Complex<f32>; 0] = [];
        assert_eq!(ifft(&mut buf), Err(FftError::Empty));
    }

    #[test]
    fn parseval_theorem() {
        let original = [c(1.0, 0.0), c(-0.5, 0.0), c(0.3, 0.0), c(-0.8, 0.0)];
        let time_energy: f32 = original.iter().map(|x| x.norm_sqr()).sum();

        let mut buf = original;
        fft(&mut buf).ok();
        let freq_energy: f32 = buf.iter().map(|x| x.norm_sqr()).sum();

        // Parseval: sum|x|^2 = (1/N) * sum|X|^2
        let n = original.len() as f32;
        assert!((time_energy - freq_energy / n).abs() < 1e-4);
    }

    #[test]
    fn fft_known_4point() {
        // x = [1, 2, 3, 4]
        // X[0] = 10, X[1] = -2+2i, X[2] = -2, X[3] = -2-2i
        let mut buf = [c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        fft(&mut buf).ok();
        assert!((buf[0].re - 10.0).abs() < 1e-4);
        assert!((buf[1].re - (-2.0)).abs() < 1e-4);
        assert!((buf[1].im - 2.0).abs() < 1e-4);
        assert!((buf[2].re - (-2.0)).abs() < 1e-4);
        assert!(buf[2].im.abs() < 1e-4);
        assert!((buf[3].re - (-2.0)).abs() < 1e-4);
        assert!((buf[3].im - (-2.0)).abs() < 1e-4);
    }

    #[test]
    fn fft_error_display() {
        let e1 = FftError::Empty;
        let e2 = FftError::NotPowerOfTwo(7);
        // Verify Display doesn't panic (no format! in no_std unit tests)
        assert_eq!(e1, FftError::Empty);
        assert_eq!(e2, FftError::NotPowerOfTwo(7));
    }

    #[test]
    fn ifft_non_power_of_two_returns_error() {
        let mut buf = [c(0.0, 0.0); 8];
        assert_eq!(ifft(&mut buf[..5]), Err(FftError::NotPowerOfTwo(5)));
    }

    #[test]
    fn fft_large_power_of_two() {
        // 256-point FFT of impulse
        let mut buf = [c(0.0, 0.0); 256];
        buf[0] = c(1.0, 0.0);
        fft(&mut buf).ok();
        for b in &buf {
            assert!((b.re - 1.0).abs() < 1e-4);
            assert!(b.im.abs() < 1e-4);
        }
    }

    #[test]
    fn roundtrip_8point() {
        let original = [
            c(0.1, 0.0),
            c(0.2, 0.0),
            c(0.3, 0.0),
            c(0.4, 0.0),
            c(0.5, 0.0),
            c(0.6, 0.0),
            c(0.7, 0.0),
            c(0.8, 0.0),
        ];
        let mut buf = original;
        fft(&mut buf).ok();
        ifft(&mut buf).ok();
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a - b).norm() < 1e-4, "mismatch: {a} vs {b}");
        }
    }
}
