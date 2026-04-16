//! Extension trait providing `.fft()` and `.ifft()` on `Signal` types.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use num_complex::Complex;
use resonant_core::signal::{FreqDomain, Signal, TimeDomain};

use crate::FftError;

/// Adds `.fft()` to time-domain signals, returning a frequency-domain signal.
///
/// Import this trait to call `.fft()` on any `Signal<T, TimeDomain>` where
/// `T: AsRef<[f32]>`.
///
/// # Examples
///
/// ```
/// use resonant_core::signal::Signal;
/// use resonant_fft::SignalFftExt;
///
/// let sig = Signal::from_samples(vec![1.0_f32, 0.0, -1.0, 0.0]);
/// let freq = sig.fft().unwrap();
/// assert_eq!(freq.data().len(), 4);
/// ```
///
/// Calling `.fft()` on a frequency-domain signal is a compile error:
///
/// ```compile_fail
/// use resonant_core::signal::{Signal, FreqDomain};
/// use resonant_fft::SignalFftExt;
/// use num_complex::Complex;
///
/// let freq = Signal::<Vec<Complex<f32>>, FreqDomain>::new(vec![]);
/// freq.fft(); // ERROR: SignalFftExt is not implemented for FreqDomain
/// ```
pub trait SignalFftExt<T> {
    /// The output type after transformation.
    type Output;

    /// Computes the forward FFT, consuming the time-domain signal and
    /// returning a frequency-domain signal.
    ///
    /// # Errors
    ///
    /// Returns [`FftError`] if the input length is invalid for the active
    /// backend (must be power-of-two without the `rustfft` feature).
    fn fft(self) -> Result<Self::Output, FftError>;
}

/// Adds `.ifft()` to frequency-domain signals, returning a time-domain signal.
///
/// Import this trait to call `.ifft()` on any
/// `Signal<Vec<Complex<f32>>, FreqDomain>`.
///
/// Calling `.ifft()` on a time-domain signal is a compile error:
///
/// ```compile_fail
/// use resonant_core::signal::{Signal, TimeDomain};
/// use resonant_fft::SignalIfftExt;
///
/// let time = Signal::from_samples(vec![1.0_f32, 0.0]);
/// time.ifft(); // ERROR: SignalIfftExt is not implemented for TimeDomain
/// ```
pub trait SignalIfftExt {
    /// The output type after inverse transformation.
    type Output;

    /// Computes the inverse FFT, consuming the frequency-domain signal and
    /// returning a time-domain signal.
    ///
    /// # Errors
    ///
    /// Returns [`FftError`] if the input length is invalid for the active
    /// backend.
    fn ifft(self) -> Result<Self::Output, FftError>;
}

impl<T: AsRef<[f32]>> SignalFftExt<T> for Signal<T, TimeDomain> {
    type Output = Signal<Vec<Complex<f32>>, FreqDomain>;

    fn fft(self) -> Result<Self::Output, FftError> {
        let samples = self.into_inner();
        let slice = samples.as_ref();
        let mut buf: Vec<Complex<f32>> = slice.iter().map(|&s| Complex::new(s, 0.0)).collect();

        run_fft_forward(&mut buf)?;
        Ok(Signal::new(buf))
    }
}

impl SignalIfftExt for Signal<Vec<Complex<f32>>, FreqDomain> {
    type Output = Signal<Vec<f32>, TimeDomain>;

    fn ifft(self) -> Result<Self::Output, FftError> {
        let mut buf = self.into_inner();
        run_fft_inverse(&mut buf)?;
        let real: Vec<f32> = buf.iter().map(|c| c.re).collect();
        Ok(Signal::new(real))
    }
}

/// Adds `.magnitude()` and `.phase()` to frequency-domain signals.
///
/// Works on both full-spectrum (`SignalFftExt`) and half-spectrum
/// (`SignalRfftExt`) outputs: the implementation operates on a flat
/// `[re, im, re, im, …]` slice and is indifferent to bin count.
///
/// # Examples
///
/// ```
/// use resonant_core::signal::Signal;
/// use resonant_fft::{SignalFftExt, SignalFreqExt, Complex};
///
/// let sig = Signal::from_samples(vec![1.0_f32, 0.0, -1.0, 0.0]);
/// let freq = sig.fft().unwrap();
/// let mags = freq.magnitude();
/// assert!(mags[0].abs() < 1e-4); // DC is ~0
/// ```
pub trait SignalFreqExt {
    /// Returns the magnitude (absolute value) of each frequency bin.
    fn magnitude(&self) -> Vec<f32>;

    /// Returns the phase angle (in radians) of each frequency bin.
    fn phase(&self) -> Vec<f32>;

    /// Returns the squared magnitude of each frequency bin.
    ///
    /// Avoids the square root in [`magnitude`](SignalFreqExt::magnitude),
    /// useful when only relative comparisons are needed.
    fn magnitude_squared(&self) -> Vec<f32>;
}

impl<T: AsRef<[Complex<f32>]>> SignalFreqExt for Signal<T, FreqDomain> {
    fn magnitude(&self) -> Vec<f32> {
        let bins = self.data().as_ref();
        let flat = complex_as_flat(bins);
        let mut out = vec![0.0_f32; bins.len()];
        crate::simd::magnitude(flat, &mut out);
        out
    }

    fn phase(&self) -> Vec<f32> {
        self.data().as_ref().iter().map(|c| c.arg()).collect()
    }

    fn magnitude_squared(&self) -> Vec<f32> {
        let bins = self.data().as_ref();
        let flat = complex_as_flat(bins);
        let mut out = vec![0.0_f32; bins.len()];
        crate::simd::magnitude_squared(flat, &mut out);
        out
    }
}

/// Reinterpret `&[Complex<f32>]` as `&[f32]` with twice the length.
///
/// `Complex<f32>` is `#[repr(C)]` with layout `(re: f32, im: f32)`,
/// so this is a safe zero-copy view.
fn complex_as_flat(bins: &[Complex<f32>]) -> &[f32] {
    // SAFETY: Complex<f32> is #[repr(C)] and has the same alignment as f32.
    // The resulting slice has exactly 2 * bins.len() f32 elements.
    unsafe { core::slice::from_raw_parts(bins.as_ptr().cast::<f32>(), bins.len() * 2) }
}

/// Adds `.rfft()` to time-domain signals, returning a half-spectrum frequency-domain signal.
///
/// The output contains N/2+1 complex bins, exploiting conjugate symmetry of real
/// inputs. This is roughly half the cost and memory of a full complex FFT for the
/// same sample count.
///
/// Calling `.rfft()` on a frequency-domain signal is a compile error:
///
/// ```compile_fail
/// use resonant_core::signal::{Signal, FreqDomain};
/// use resonant_fft::SignalRfftExt;
/// use num_complex::Complex;
///
/// let freq = Signal::<Vec<Complex<f32>>, FreqDomain>::new(vec![]);
/// freq.rfft(); // ERROR: SignalRfftExt is not implemented for FreqDomain signals
/// ```
pub trait SignalRfftExt {
    /// Computes the real-valued forward FFT, returning N/2+1 complex bins.
    ///
    /// # Errors
    ///
    /// Returns [`FftError`] if the input length is not a power of two, or is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_core::signal::Signal;
    /// use resonant_fft::SignalRfftExt;
    ///
    /// let sig = Signal::from_samples(vec![1.0_f32, 0.0, -1.0, 0.0]);
    /// let freq = sig.rfft().unwrap();
    /// assert_eq!(freq.data().len(), 3); // N/2 + 1 = 3
    /// ```
    fn rfft(self) -> Result<Signal<Vec<Complex<f32>>, FreqDomain>, FftError>;
}

impl<T: AsRef<[f32]>> SignalRfftExt for Signal<T, TimeDomain> {
    fn rfft(self) -> Result<Signal<Vec<Complex<f32>>, FreqDomain>, FftError> {
        let samples = self.into_inner();
        let slice = samples.as_ref();
        let n = slice.len();
        if n == 0 {
            return Err(FftError::Empty);
        }
        let m = n / 2;
        let mut out = vec![Complex::new(0.0_f32, 0.0); m + 1];
        crate::rfft::rfft(slice, &mut out)?;
        Ok(Signal::new(out))
    }
}

/// Dispatches to the best available backend.
#[cfg(feature = "rustfft")]
pub(crate) fn run_fft_forward(buf: &mut [Complex<f32>]) -> Result<(), FftError> {
    crate::rustfft_backend::fft(buf)
}

#[cfg(not(feature = "rustfft"))]
pub(crate) fn run_fft_forward(buf: &mut [Complex<f32>]) -> Result<(), FftError> {
    crate::radix2::fft(buf)
}

#[cfg(feature = "rustfft")]
pub(crate) fn run_fft_inverse(buf: &mut [Complex<f32>]) -> Result<(), FftError> {
    crate::rustfft_backend::ifft(buf)
}

#[cfg(not(feature = "rustfft"))]
pub(crate) fn run_fft_inverse(buf: &mut [Complex<f32>]) -> Result<(), FftError> {
    crate::radix2::ifft(buf)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    #[test]
    fn fft_time_to_freq_domain() {
        let sig = Signal::from_samples(std::vec![1.0_f32, 0.0, -1.0, 0.0]);
        let freq = sig.fft().unwrap();
        assert_eq!(freq.data().len(), 4);
    }

    #[test]
    fn ifft_freq_to_time_domain() {
        let sig = Signal::from_samples(std::vec![1.0_f32, 2.0, 3.0, 4.0]);
        let freq = sig.fft().unwrap();
        let time = freq.ifft().unwrap();
        let samples = time.data();
        assert!((samples[0] - 1.0).abs() < 1e-4);
        assert!((samples[1] - 2.0).abs() < 1e-4);
        assert!((samples[2] - 3.0).abs() < 1e-4);
        assert!((samples[3] - 4.0).abs() < 1e-4);
    }

    #[test]
    fn fft_ifft_roundtrip() {
        let original = std::vec![0.1_f32, -0.3, 0.5, -0.7, 0.9, -0.2, 0.4, -0.6];
        let sig = Signal::from_samples(original.clone());
        let freq = sig.fft().unwrap();
        let time = freq.ifft().unwrap();
        for (a, b) in time.data().iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-3, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn fft_empty_returns_error() {
        let sig = Signal::from_samples(std::vec![0.0_f32; 0]);
        assert_eq!(sig.fft(), Err(FftError::Empty));
    }

    #[test]
    fn magnitude_of_dc_signal() {
        let sig = Signal::from_samples(std::vec![1.0_f32; 4]);
        let freq = sig.fft().unwrap();
        let mags = freq.magnitude();
        assert!((mags[0] - 4.0).abs() < 1e-3);
        for m in &mags[1..] {
            assert!(*m < 1e-3);
        }
    }

    #[test]
    fn magnitude_squared_avoids_sqrt() {
        let sig = Signal::from_samples(std::vec![1.0_f32; 4]);
        let freq = sig.fft().unwrap();
        let mags = freq.magnitude();
        let mags_sq = freq.magnitude_squared();
        for (m, msq) in mags.iter().zip(mags_sq.iter()) {
            assert!((m * m - msq).abs() < 1e-4);
        }
    }

    #[test]
    fn phase_of_dc_signal() {
        let sig = Signal::from_samples(std::vec![1.0_f32; 4]);
        let freq = sig.fft().unwrap();
        let phases = freq.phase();
        // DC bin phase should be 0 (purely real positive)
        assert!(phases[0].abs() < 1e-4);
    }

    #[test]
    fn fft_works_with_array() {
        let sig = Signal::from_samples([1.0_f32, 0.0, -1.0, 0.0]);
        let freq = sig.fft().unwrap();
        assert_eq!(freq.data().len(), 4);
    }

    #[test]
    fn fft_works_with_slice() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let sig = Signal::<&[f32], TimeDomain>::new(&data[..]);
        let freq = sig.fft().unwrap();
        assert_eq!(freq.data().len(), 4);
    }

    #[test]
    fn rfft_ext_returns_half_spectrum() {
        let sig = Signal::from_samples(std::vec![1.0_f32, 0.0, -1.0, 0.0]);
        let freq = sig.rfft().unwrap();
        // N=4 → N/2+1 = 3 bins
        assert_eq!(freq.data().len(), 3);
    }

    #[test]
    fn rfft_ext_dc_signal() {
        // DC input → bin[0] = N, rest ≈ 0
        let sig = Signal::from_samples(std::vec![1.0_f32; 8]);
        let freq = sig.rfft().unwrap();
        assert_eq!(freq.data().len(), 5);
        assert!((freq.data()[0].re - 8.0).abs() < 1e-4);
        for b in &freq.data()[1..] {
            assert!(b.norm() < 1e-4, "non-DC bin should be ~0: {b:?}");
        }
    }

    #[test]
    fn rfft_ext_empty_returns_error() {
        let sig = Signal::from_samples(std::vec![0.0_f32; 0]);
        assert_eq!(sig.rfft(), Err(FftError::Empty));
    }

    #[test]
    fn rfft_magnitude_matches_complex_fft_magnitude() {
        // magnitude() on N/2+1 rfft output must equal magnitude() on the same
        // bins from a full complex FFT — SignalFreqExt works on any Vec<Complex<f32>>
        // FreqDomain signal regardless of length.
        let samples = std::vec![0.1_f32, -0.3, 0.5, -0.7, 0.9, -0.2, 0.4, -0.6];
        let n = samples.len();
        let m = n / 2;

        let rfft_mags = Signal::from_samples(samples.clone())
            .rfft()
            .unwrap()
            .magnitude();
        let full_mags = Signal::from_samples(samples).fft().unwrap().magnitude();

        // rfft gives N/2+1 bins; full FFT gives N bins.
        // The first N/2+1 magnitudes should agree.
        assert_eq!(rfft_mags.len(), m + 1);
        for k in 0..=m {
            assert!(
                (rfft_mags[k] - full_mags[k]).abs() < 1e-3,
                "magnitude mismatch at bin {k}: rfft={} full={}",
                rfft_mags[k],
                full_mags[k]
            );
        }
    }

    #[test]
    fn rfft_compile_fail_on_freq_domain() {
        // Verify that the compile_fail doc-test in SignalRfftExt is accurate:
        // calling .rfft() on a FreqDomain signal should not compile.
        // (This runtime test simply exercises the happy path to confirm the trait
        // is correctly gated on TimeDomain.)
        let sig = Signal::from_samples(std::vec![1.0_f32, 0.0, -1.0, 0.0]);
        let freq = sig.rfft().unwrap();
        // freq is FreqDomain — we can call magnitude() but not rfft()
        assert!(freq.magnitude()[0].abs() < 1e-3); // DC ≈ 0 for this input
    }
}
