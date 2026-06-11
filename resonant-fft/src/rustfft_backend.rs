//! FFT backend using `rustfft` — supports arbitrary sizes, not just power-of-two.
//!
//! Enabled by the `rustfft` feature (on by default). Requires `std`.

extern crate alloc;

use alloc::sync::Arc;
use alloc::vec;

use num_complex::Complex;
use rustfft::FftPlanner;

use crate::FftError;

/// Computes the forward FFT of arbitrary length using `rustfft`.
///
/// Unlike [`radix2::fft`](crate::radix2::fft), this accepts any non-empty
/// input length — not just powers of two. Allocates internally for the
/// planner and scratch buffer.
///
/// # Errors
///
/// Returns [`FftError::Empty`] if the slice is empty.
///
/// # Examples
///
/// ```
/// use resonant_fft::{rustfft_backend, Complex};
///
/// let mut buf = vec![
///     Complex::new(1.0_f32, 0.0),
///     Complex::new(2.0, 0.0),
///     Complex::new(3.0, 0.0),
/// ];
/// rustfft_backend::fft(&mut buf).ok();
/// assert!((buf[0].re - 6.0).abs() < 1e-5);
/// ```
pub fn fft(buf: &mut [Complex<f32>]) -> Result<(), FftError> {
    if buf.is_empty() {
        return Err(FftError::Empty);
    }
    let mut planner = FftPlanner::new();
    let fft_impl = planner.plan_fft_forward(buf.len());
    let mut scratch = vec![Complex::new(0.0_f32, 0.0); fft_impl.get_inplace_scratch_len()];
    fft_impl.process_with_scratch(buf, &mut scratch);
    Ok(())
}

/// Computes the inverse FFT of arbitrary length using `rustfft`.
///
/// The output is scaled by `1/N` so that `ifft(fft(x)) ≈ x`.
///
/// # Errors
///
/// Returns [`FftError::Empty`] if the slice is empty.
///
/// # Examples
///
/// ```
/// use resonant_fft::{rustfft_backend, Complex};
///
/// let original = vec![
///     Complex::new(1.0_f32, 0.0),
///     Complex::new(2.0, 0.0),
///     Complex::new(3.0, 0.0),
/// ];
/// let mut buf = original.clone();
/// rustfft_backend::fft(&mut buf).ok();
/// rustfft_backend::ifft(&mut buf).ok();
/// for (a, b) in buf.iter().zip(original.iter()) {
///     assert!((a - b).norm() < 1e-4);
/// }
/// ```
pub fn ifft(buf: &mut [Complex<f32>]) -> Result<(), FftError> {
    if buf.is_empty() {
        return Err(FftError::Empty);
    }
    let mut planner = FftPlanner::new();
    let fft_impl = planner.plan_fft_inverse(buf.len());
    let mut scratch = vec![Complex::new(0.0_f32, 0.0); fft_impl.get_inplace_scratch_len()];
    fft_impl.process_with_scratch(buf, &mut scratch);
    let scale = 1.0 / buf.len() as f32;
    for x in buf.iter_mut() {
        *x *= scale;
    }
    Ok(())
}

/// Creates a reusable FFT plan for a given length.
///
/// When you need to compute many FFTs of the same size, creating the plan
/// once and reusing it avoids repeated planner overhead.
///
/// # Examples
///
/// ```
/// use resonant_fft::{rustfft_backend::FftPlan, Complex};
///
/// let plan = FftPlan::new(1024);
/// let mut buf = vec![Complex::new(0.0_f32, 0.0); 1024];
/// plan.fft(&mut buf).ok();
/// ```
pub struct FftPlan {
    forward: Arc<dyn rustfft::Fft<f32>>,
    inverse: Arc<dyn rustfft::Fft<f32>>,
    len: usize,
}

impl FftPlan {
    /// Creates a new plan for the given FFT length.
    ///
    /// # Panics
    ///
    /// Panics if `len` is zero.
    #[must_use]
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "FFT length must be > 0");
        let mut planner = FftPlanner::new();
        Self {
            forward: planner.plan_fft_forward(len),
            inverse: planner.plan_fft_inverse(len),
            len,
        }
    }

    /// Returns the planned FFT length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Always returns `false` — a plan is never empty (length is always > 0).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Computes the forward FFT using the precomputed plan.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::WrongLength`] if `buf.len() != self.len()`.
    pub fn fft(&self, buf: &mut [Complex<f32>]) -> Result<(), FftError> {
        if buf.len() != self.len {
            return Err(FftError::WrongLength {
                actual: buf.len(),
                planned: self.len,
            });
        }
        let mut scratch = vec![Complex::new(0.0_f32, 0.0); self.forward.get_inplace_scratch_len()];
        self.forward.process_with_scratch(buf, &mut scratch);
        Ok(())
    }

    /// Computes the inverse FFT using the precomputed plan, scaled by `1/N`.
    ///
    /// # Errors
    ///
    /// Returns [`FftError::WrongLength`] if `buf.len() != self.len()`.
    pub fn ifft(&self, buf: &mut [Complex<f32>]) -> Result<(), FftError> {
        if buf.len() != self.len {
            return Err(FftError::WrongLength {
                actual: buf.len(),
                planned: self.len,
            });
        }
        let mut scratch = vec![Complex::new(0.0_f32, 0.0); self.inverse.get_inplace_scratch_len()];
        self.inverse.process_with_scratch(buf, &mut scratch);
        let scale = 1.0 / self.len as f32;
        for x in buf.iter_mut() {
            *x *= scale;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f32, im: f32) -> Complex<f32> {
        Complex::new(re, im)
    }

    #[test]
    fn fft_empty_returns_error() {
        let mut buf: Vec<Complex<f32>> = vec![];
        assert_eq!(fft(&mut buf), Err(FftError::Empty));
    }

    #[test]
    fn ifft_empty_returns_error() {
        let mut buf: Vec<Complex<f32>> = vec![];
        assert_eq!(ifft(&mut buf), Err(FftError::Empty));
    }

    #[test]
    fn fft_dc_signal() {
        let mut buf = vec![c(1.0, 0.0); 8];
        fft(&mut buf).ok();
        assert!((buf[0].re - 8.0).abs() < 1e-4);
        for b in &buf[1..] {
            assert!(b.norm() < 1e-4);
        }
    }

    #[test]
    fn roundtrip_power_of_two() {
        let original = vec![c(1.0, 0.0), c(0.5, 0.0), c(-0.3, 0.0), c(0.7, 0.0)];
        let mut buf = original.clone();
        fft(&mut buf).ok();
        ifft(&mut buf).ok();
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a - b).norm() < 1e-4, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn roundtrip_non_power_of_two() {
        let original: Vec<_> = (0..7).map(|i| c(i as f32 / 7.0, 0.0)).collect();
        let mut buf = original.clone();
        fft(&mut buf).ok();
        ifft(&mut buf).ok();
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a - b).norm() < 1e-3, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn fft_3point_dc() {
        let mut buf = vec![c(1.0, 0.0); 3];
        fft(&mut buf).ok();
        assert!((buf[0].re - 3.0).abs() < 1e-4);
        for b in &buf[1..] {
            assert!(b.norm() < 1e-4);
        }
    }

    #[test]
    fn plan_reuse() {
        let plan = FftPlan::new(4);
        assert_eq!(plan.len(), 4);

        let mut buf = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
        let original = buf.clone();
        plan.fft(&mut buf).ok();
        plan.ifft(&mut buf).ok();
        for (a, b) in buf.iter().zip(original.iter()) {
            assert!((a - b).norm() < 1e-4);
        }
    }

    #[test]
    fn plan_wrong_length() {
        let plan = FftPlan::new(4);
        let mut buf = vec![c(0.0, 0.0); 8];
        let expected = FftError::WrongLength { actual: 8, planned: 4 };
        assert_eq!(plan.fft(&mut buf), Err(expected));
        assert_eq!(plan.ifft(&mut buf), Err(expected));
    }

    #[test]
    fn impulse_flat_spectrum() {
        let mut buf = vec![c(0.0, 0.0); 16];
        buf[0] = c(1.0, 0.0);
        fft(&mut buf).ok();
        for b in &buf {
            assert!((b.re - 1.0).abs() < 1e-4);
            assert!(b.im.abs() < 1e-4);
        }
    }
}
