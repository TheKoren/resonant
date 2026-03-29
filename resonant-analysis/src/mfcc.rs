//! Mel-frequency cepstral coefficients (MFCCs).
//!
//! MFCCs are a compact representation of the spectral envelope, widely used in
//! speech and music analysis. This module computes per-frame MFCCs from raw
//! audio via STFT → mel filterbank → log energy → DCT-II, and provides delta
//! and delta-delta (acceleration) coefficients.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use resonant_core::signal::Signal;
use resonant_core::window;
use resonant_fft::dct;
use resonant_fft::stft::Stft;
use resonant_fft::SignalFreqExt;

use crate::error::AnalysisError;

/// A single frame of MFCC coefficients.
///
/// # Examples
///
/// ```
/// use resonant_analysis::mfcc::MfccFrame;
///
/// let frame = MfccFrame { coefficients: vec![1.0, 0.5, -0.3] };
/// assert_eq!(frame.coefficients.len(), 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MfccFrame {
    /// The MFCC coefficients for this frame.
    pub coefficients: Vec<f32>,
}

/// MFCC extractor with configurable parameters.
///
/// # Examples
///
/// ```
/// use resonant_analysis::mfcc::MfccExtractor;
///
/// let extractor = MfccExtractor::new(44100.0);
/// let samples = vec![0.0_f32; 4096];
/// let frames = extractor.extract(&samples).unwrap();
/// // Each frame has 13 coefficients by default
/// for f in &frames {
///     assert_eq!(f.coefficients.len(), 13);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MfccExtractor {
    sample_rate: f32,
    num_coefficients: usize,
    num_mel_bands: usize,
    window_size: usize,
    hop_size: usize,
}

impl MfccExtractor {
    /// Creates an extractor with default parameters.
    ///
    /// Defaults: 13 coefficients, 26 mel bands, window 1024, hop 512.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            num_coefficients: 13,
            num_mel_bands: 26,
            window_size: 1024,
            hop_size: 512,
        }
    }

    /// Sets the number of MFCC coefficients to keep per frame.
    #[must_use]
    pub fn with_num_coefficients(mut self, n: usize) -> Self {
        self.num_coefficients = n;
        self
    }

    /// Sets the number of mel filterbank bands.
    #[must_use]
    pub fn with_num_mel_bands(mut self, n: usize) -> Self {
        self.num_mel_bands = n;
        self
    }

    /// Sets the STFT window size (should be a power of two).
    #[must_use]
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Sets the STFT hop size.
    #[must_use]
    pub fn with_hop_size(mut self, hop: usize) -> Self {
        self.hop_size = hop;
        self
    }

    /// Extracts MFCC frames from mono audio samples.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::EmptyInput`] if `samples` is empty, or
    /// [`AnalysisError::InvalidParameter`] if parameters are inconsistent.
    pub fn extract(&self, samples: &[f32]) -> Result<Vec<MfccFrame>, AnalysisError> {
        if samples.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }
        if self.num_coefficients == 0 || self.num_mel_bands == 0 {
            return Err(AnalysisError::InvalidParameter {
                name: "num_coefficients/num_mel_bands",
                reason: "must be positive",
            });
        }
        if self.num_coefficients > self.num_mel_bands {
            return Err(AnalysisError::InvalidParameter {
                name: "num_coefficients",
                reason: "must not exceed num_mel_bands",
            });
        }

        // If signal is shorter than one window, return a single zero frame
        if samples.len() < self.window_size {
            return Ok(vec![MfccFrame {
                coefficients: vec![0.0; self.num_coefficients],
            }]);
        }

        let signal = Signal::from_samples(samples.to_vec());
        let stft = Stft::builder(self.window_size, self.hop_size)
            .window_fn(window::hann)
            .build();

        let stft_frames = stft.analyze(&signal)?;
        if stft_frames.is_empty() {
            return Ok(vec![MfccFrame {
                coefficients: vec![0.0; self.num_coefficients],
            }]);
        }

        // Build mel filterbank (FFT bins → mel bands)
        let fft_size = self.window_size;
        let filterbank = build_mel_filterbank(self.num_mel_bands, fft_size, self.sample_rate);

        let mut frames = Vec::with_capacity(stft_frames.len());

        for stft_frame in &stft_frames {
            let magnitudes = stft_frame.magnitude();
            let mel_energies = apply_mel_filterbank(&magnitudes, &filterbank);
            let log_mel = log_mel_energy(&mel_energies);

            // DCT-II of log mel energies, then keep first num_coefficients
            let mut dct_out = vec![0.0_f32; self.num_mel_bands];
            dct::dct_ii(&log_mel, &mut dct_out)?;

            let coefficients = dct_out[..self.num_coefficients].to_vec();
            frames.push(MfccFrame { coefficients });
        }

        Ok(frames)
    }

    /// Computes delta (first derivative) coefficients from MFCC frames.
    ///
    /// Uses a regression window of `width` frames on each side.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::EmptyInput`] if `frames` is empty, or
    /// [`AnalysisError::InvalidParameter`] if `width` is zero.
    pub fn deltas(frames: &[MfccFrame], width: usize) -> Result<Vec<MfccFrame>, AnalysisError> {
        if frames.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }
        if width == 0 {
            return Err(AnalysisError::InvalidParameter {
                name: "width",
                reason: "must be positive",
            });
        }

        let n = frames.len();
        let num_coeffs = frames[0].coefficients.len();
        let mut result = Vec::with_capacity(n);

        // Denominator: 2 * Σ(w² for w in 1..=width)
        let denom: f32 = 2.0 * (1..=width).map(|w| (w * w) as f32).sum::<f32>();

        for t in 0..n {
            let mut coefficients = vec![0.0_f32; num_coeffs];
            if denom > f32::EPSILON {
                for w in 1..=width {
                    // Clamp indices to valid range (edge padding)
                    let prev = t.saturating_sub(w);
                    let next = (t + w).min(n - 1);
                    for (c, coeff) in coefficients.iter_mut().enumerate() {
                        *coeff += w as f32
                            * (frames[next].coefficients[c] - frames[prev].coefficients[c]);
                    }
                }
                for c in &mut coefficients {
                    *c /= denom;
                }
            }
            result.push(MfccFrame { coefficients });
        }

        Ok(result)
    }

    /// Computes delta-delta (second derivative / acceleration) coefficients.
    ///
    /// Equivalent to applying [`deltas`](Self::deltas) twice.
    ///
    /// # Errors
    ///
    /// Same as [`deltas`](Self::deltas).
    pub fn delta_deltas(
        frames: &[MfccFrame],
        width: usize,
    ) -> Result<Vec<MfccFrame>, AnalysisError> {
        let d = Self::deltas(frames, width)?;
        Self::deltas(&d, width)
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Converts frequency in Hz to mel scale.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Converts mel value to frequency in Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// A triangular mel filterbank: `num_bands` filters spanning [0, sample_rate/2].
///
/// Returns a `Vec<Vec<f32>>` where each inner vec has length `fft_size / 2 + 1`
/// (one weight per FFT magnitude bin).
fn build_mel_filterbank(num_bands: usize, fft_size: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let num_bins = fft_size / 2 + 1;
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate / 2.0);

    // num_bands + 2 equally spaced points on the mel scale
    let num_points = num_bands + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (num_points - 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz points to FFT bin indices (fractional)
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| hz * fft_size as f32 / sample_rate)
        .collect();

    let mut filterbank = Vec::with_capacity(num_bands);
    for band in 0..num_bands {
        let left = bin_points[band];
        let center = bin_points[band + 1];
        let right = bin_points[band + 2];

        let mut weights = vec![0.0_f32; num_bins];
        for (bin, weight) in weights.iter_mut().enumerate() {
            let b = bin as f32;
            if b > left && b < center && center > left {
                *weight = (b - left) / (center - left);
            } else if b >= center && b < right && right > center {
                *weight = (right - b) / (right - center);
            }
        }
        filterbank.push(weights);
    }

    filterbank
}

/// Applies the mel filterbank to a magnitude spectrum, producing mel-band energies.
fn apply_mel_filterbank(magnitudes: &[f32], filterbank: &[Vec<f32>]) -> Vec<f32> {
    filterbank
        .iter()
        .map(|weights| {
            let len = magnitudes.len().min(weights.len());
            magnitudes[..len]
                .iter()
                .zip(&weights[..len])
                .map(|(m, w)| m * m * w) // power spectrum weighted
                .sum()
        })
        .collect()
}

/// Converts mel-band energies to log scale, with a floor to avoid log(0).
fn log_mel_energy(energies: &[f32]) -> Vec<f32> {
    const FLOOR: f32 = 1e-10;
    energies.iter().map(|&e| (e.max(FLOOR)).ln()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    const SR: f32 = 44100.0;

    #[test]
    fn hz_to_mel_known_values() {
        // 0 Hz → 0 mel
        assert!((hz_to_mel(0.0)).abs() < 1e-4);
        // 1000 Hz → ~1000 mel (by design of the mel scale formula)
        let mel_1k = hz_to_mel(1000.0);
        assert!(
            (mel_1k - 1000.0).abs() < 100.0,
            "1000 Hz should be ~1000 mel, got {mel_1k}"
        );
    }

    #[test]
    fn mel_hz_round_trip() {
        for &hz in &[0.0, 100.0, 440.0, 1000.0, 8000.0, 22050.0] {
            let recovered = mel_to_hz(hz_to_mel(hz));
            assert!(
                (recovered - hz).abs() < 0.1,
                "round-trip failed for {hz} Hz: got {recovered}"
            );
        }
    }

    #[test]
    fn filterbank_shape() {
        let fb = build_mel_filterbank(26, 1024, SR);
        assert_eq!(fb.len(), 26);
        for band in &fb {
            assert_eq!(band.len(), 513); // 1024/2 + 1
                                         // All weights non-negative
            for &w in band {
                assert!(w >= 0.0, "negative filterbank weight: {w}");
            }
        }
    }

    #[test]
    fn filterbank_triangular_peaks_at_most_one() {
        let fb = build_mel_filterbank(26, 1024, SR);
        for band in &fb {
            for &w in band {
                assert!(w <= 1.0 + 1e-6, "weight exceeds 1.0: {w}");
            }
        }
    }

    #[test]
    fn silence_near_zero_mfccs() {
        let samples = vec![0.0_f32; 4096];
        let extractor = MfccExtractor::new(SR);
        let frames = extractor.extract(&samples).unwrap();
        for frame in &frames {
            assert_eq!(frame.coefficients.len(), 13);
            // All coefficients should be based on the log floor, roughly constant
            // The key point: no NaN or Inf
            for &c in &frame.coefficients {
                assert!(c.is_finite(), "non-finite MFCC: {c}");
            }
        }
    }

    #[test]
    fn extract_basic_sine() {
        let samples: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / SR).sin())
            .collect();
        let extractor = MfccExtractor::new(SR);
        let frames = extractor.extract(&samples).unwrap();
        assert!(!frames.is_empty());
        for frame in &frames {
            assert_eq!(frame.coefficients.len(), 13);
            for &c in &frame.coefficients {
                assert!(c.is_finite(), "non-finite MFCC: {c}");
            }
        }
    }

    #[test]
    fn empty_input_error() {
        let extractor = MfccExtractor::new(SR);
        assert_eq!(extractor.extract(&[]), Err(AnalysisError::EmptyInput));
    }

    #[test]
    fn zero_coefficients_error() {
        let extractor = MfccExtractor::new(SR).with_num_coefficients(0);
        let result = extractor.extract(&[1.0; 2048]);
        assert!(matches!(
            result,
            Err(AnalysisError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn coefficients_exceed_bands_error() {
        let extractor = MfccExtractor::new(SR)
            .with_num_coefficients(30)
            .with_num_mel_bands(13);
        let result = extractor.extract(&[1.0; 2048]);
        assert!(matches!(
            result,
            Err(AnalysisError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn short_signal_returns_zero_frame() {
        let extractor = MfccExtractor::new(SR).with_window_size(1024);
        let frames = extractor.extract(&[1.0; 512]).unwrap();
        assert_eq!(frames.len(), 1);
        assert!(frames[0].coefficients.iter().all(|&c| c == 0.0));
    }

    #[test]
    fn delta_of_constant_is_zero() {
        let frames: Vec<MfccFrame> = (0..10)
            .map(|_| MfccFrame {
                coefficients: vec![1.0, 2.0, 3.0],
            })
            .collect();
        let deltas = MfccExtractor::deltas(&frames, 2).unwrap();
        assert_eq!(deltas.len(), 10);
        for d in &deltas {
            for &c in &d.coefficients {
                assert!(c.abs() < 1e-6, "delta of constant should be zero, got {c}");
            }
        }
    }

    #[test]
    fn delta_of_linear_ramp() {
        // If coefficients increase linearly frame-to-frame, delta should be constant
        let frames: Vec<MfccFrame> = (0..10)
            .map(|i| MfccFrame {
                coefficients: vec![i as f32],
            })
            .collect();
        let deltas = MfccExtractor::deltas(&frames, 1).unwrap();
        // Interior frames (not edge-padded) should have delta ≈ 1.0
        for d in &deltas[1..9] {
            assert!(
                (d.coefficients[0] - 1.0).abs() < 1e-4,
                "expected delta ~1.0, got {}",
                d.coefficients[0]
            );
        }
    }

    #[test]
    fn delta_empty_error() {
        let result = MfccExtractor::deltas(&[], 2);
        assert_eq!(result, Err(AnalysisError::EmptyInput));
    }

    #[test]
    fn delta_zero_width_error() {
        let frames = vec![MfccFrame {
            coefficients: vec![1.0],
        }];
        let result = MfccExtractor::deltas(&frames, 0);
        assert!(matches!(
            result,
            Err(AnalysisError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn delta_delta_of_constant_is_zero() {
        let frames: Vec<MfccFrame> = (0..10)
            .map(|_| MfccFrame {
                coefficients: vec![5.0, 3.0],
            })
            .collect();
        let dd = MfccExtractor::delta_deltas(&frames, 2).unwrap();
        for d in &dd {
            for &c in &d.coefficients {
                assert!(c.abs() < 1e-6, "delta-delta of constant should be zero");
            }
        }
    }

    #[test]
    fn no_nan_or_inf_in_output() {
        // Pseudo-random signal
        let samples: Vec<f32> = (0..8192)
            .map(|i| (i as f32 * 7.3).sin() * (i as f32 * 13.7).cos())
            .collect();
        let extractor = MfccExtractor::new(SR);
        let frames = extractor.extract(&samples).unwrap();
        for frame in &frames {
            for &c in &frame.coefficients {
                assert!(c.is_finite(), "non-finite MFCC in noisy signal: {c}");
            }
        }
    }

    #[test]
    fn builder_methods() {
        let e = MfccExtractor::new(SR)
            .with_num_coefficients(20)
            .with_num_mel_bands(40)
            .with_window_size(2048)
            .with_hop_size(1024);
        assert_eq!(e.num_coefficients, 20);
        assert_eq!(e.num_mel_bands, 40);
        assert_eq!(e.window_size, 2048);
        assert_eq!(e.hop_size, 1024);
    }

    #[test]
    fn custom_num_coefficients() {
        let samples: Vec<f32> = (0..4096)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / SR).sin())
            .collect();
        let extractor = MfccExtractor::new(SR).with_num_coefficients(20);
        let frames = extractor.extract(&samples).unwrap();
        for frame in &frames {
            assert_eq!(frame.coefficients.len(), 20);
        }
    }

    // --- internal helper tests ---

    #[test]
    fn log_mel_energy_floor() {
        let energies = vec![0.0, 1e-20, 1.0];
        let log_e = log_mel_energy(&energies);
        for &v in &log_e {
            assert!(v.is_finite(), "log_mel_energy produced non-finite: {v}");
        }
        // log(1.0) = 0
        assert!((log_e[2]).abs() < 1e-4);
    }

    #[test]
    fn apply_filterbank_basic() {
        // Single-band filterbank that weights bin 1 fully
        let filterbank = vec![vec![0.0, 1.0, 0.0]];
        let magnitudes = [0.5, 2.0, 0.3];
        let result = apply_mel_filterbank(&magnitudes, &filterbank);
        // Power of bin 1 * weight 1.0 = 2.0² * 1.0 = 4.0
        assert!((result[0] - 4.0).abs() < 1e-4);
    }
}
