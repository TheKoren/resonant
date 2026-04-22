//! Filter frequency response — magnitude and phase over the audio spectrum.
//!
//! [`FilterResponseExt`] is implemented for [`BiquadCoeffs`] (analytic unit-circle
//! evaluation) and [`Fir`] (zero-padded rfft of the coefficient vector).
//!
//! Requires the `alloc` feature.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use core::f32::consts::PI;

#[allow(unused_imports)]
use num_traits::float::Float as _;

use crate::{BiquadCoeffs, Fir};

/// Errors returned by [`FilterResponseExt::frequency_response`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FilterError {
    /// The filter has no taps or coefficients (e.g. empty FIR).
    Empty,
    /// A parameter is out of the valid range (e.g. zero sample rate).
    InvalidParameter,
}

impl core::fmt::Display for FilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "filter is empty"),
            Self::InvalidParameter => write!(f, "invalid parameter"),
        }
    }
}

/// The computed frequency response of a filter.
///
/// All three vectors have the same length (`n_points`).
/// Frequencies run from 0 Hz (DC) to `sample_rate / 2` (Nyquist), inclusive.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde-alloc", derive(serde::Serialize, serde::Deserialize))]
pub struct FrequencyResponse {
    /// Frequency axis in Hz.
    pub frequencies: Vec<f32>,
    /// Linear magnitude at each frequency.
    pub magnitudes: Vec<f32>,
    /// Phase in radians at each frequency.
    pub phases: Vec<f32>,
}

impl FrequencyResponse {
    /// Returns magnitudes converted to decibels (20 · log₁₀(magnitude)).
    ///
    /// Values below −120 dB are clamped to −120 dB to avoid −∞ at true zeros.
    #[must_use]
    pub fn magnitudes_db(&self) -> Vec<f32> {
        self.magnitudes
            .iter()
            .map(|&m| {
                if m <= 0.0 {
                    -120.0_f32
                } else {
                    (20.0 * m.log10()).max(-120.0)
                }
            })
            .collect()
    }
}

/// Computes the frequency response of a filter over `n_points` equally-spaced
/// frequencies from 0 Hz to Nyquist.
///
/// ## Nonlinear processors
///
/// For nonlinear processors (`MoogLadder`, `SaturatingBiquad`) the response is
/// measured at `sweep_amplitude = 0.1` (−20 dBFS). The result approximates
/// small-signal behaviour. At higher drive levels the effective cutoff and
/// resonance shift due to the nonlinear state; this method does not capture
/// that dependence. To characterise a nonlinear processor at a specific drive
/// level, scale the input accordingly before calling `frequency_response`.
pub trait FilterResponseExt {
    /// Returns the frequency response sampled at `n_points` frequencies.
    ///
    /// # Arguments
    ///
    /// * `n_points` — number of frequency samples (must be ≥ 2)
    /// * `sample_rate` — sample rate in Hz (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns [`FilterError`] if the filter is empty or parameters are invalid.
    fn frequency_response(
        &self,
        n_points: usize,
        sample_rate: f32,
    ) -> Result<FrequencyResponse, FilterError>;
}

impl FilterResponseExt for BiquadCoeffs {
    /// Evaluates H(e^{jω}) analytically on the unit circle.
    ///
    /// ```text
    /// H(e^{jω}) = (b0 + b1·e^{-jω} + b2·e^{-j2ω})
    ///           / (1  + a1·e^{-jω} + a2·e^{-j2ω})
    /// ```
    ///
    /// Each frequency point is an independent O(1) evaluation — no FFT needed.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_filters::{design, response::FilterResponseExt};
    ///
    /// let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
    /// let resp = coeffs.frequency_response(512, 44100.0).unwrap();
    ///
    /// // Cutoff frequency should be near −3 dB
    /// let db = resp.magnitudes_db();
    /// // Find the bin closest to 1 kHz
    /// let bin = (1000.0 / (44100.0 / 2.0) * (512 - 1) as f32).round() as usize;
    /// assert!((db[bin] - (-3.0)).abs() < 1.0);
    /// ```
    fn frequency_response(
        &self,
        n_points: usize,
        sample_rate: f32,
    ) -> Result<FrequencyResponse, FilterError> {
        if n_points < 2 || sample_rate <= 0.0 {
            return Err(FilterError::InvalidParameter);
        }

        let mut frequencies = vec![0.0_f32; n_points];
        let mut magnitudes = vec![0.0_f32; n_points];
        let mut phases = vec![0.0_f32; n_points];

        let nyquist = sample_rate / 2.0;

        for i in 0..n_points {
            let freq = nyquist * i as f32 / (n_points - 1) as f32;
            let omega = 2.0 * PI * freq / sample_rate; // radians per sample

            // e^{-jω} = cos(ω) - j·sin(ω)
            let (sin_w, cos_w) = omega.sin_cos();
            let (sin_2w, cos_2w) = (2.0 * omega).sin_cos();

            // Numerator: b0 + b1·e^{-jω} + b2·e^{-j2ω}
            let num_re = self.b0 + self.b1 * cos_w + self.b2 * cos_2w;
            let num_im = -(self.b1 * sin_w + self.b2 * sin_2w);

            // Denominator: 1 + a1·e^{-jω} + a2·e^{-j2ω}
            let den_re = 1.0 + self.a1 * cos_w + self.a2 * cos_2w;
            let den_im = -(self.a1 * sin_w + self.a2 * sin_2w);

            // H = num / den (complex division)
            let den_norm_sq = den_re * den_re + den_im * den_im;
            let h_re = (num_re * den_re + num_im * den_im) / den_norm_sq;
            let h_im = (num_im * den_re - num_re * den_im) / den_norm_sq;

            frequencies[i] = freq;
            magnitudes[i] = (h_re * h_re + h_im * h_im).sqrt();
            phases[i] = h_im.atan2(h_re);
        }

        Ok(FrequencyResponse {
            frequencies,
            magnitudes,
            phases,
        })
    }
}

impl FilterResponseExt for Fir {
    /// Computes the FIR frequency response via zero-padded rfft of the
    /// coefficient vector.
    ///
    /// The coefficients are zero-padded to the smallest power-of-two length
    /// `N` such that `N/2 + 1 ≥ n_points`. The rfft output bins are then
    /// sub-sampled / truncated to `n_points` and converted to magnitude/phase.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_filters::fir::Fir;
    /// use resonant_filters::response::FilterResponseExt;
    ///
    /// // 3-tap averaging filter — expected sinc-like response
    /// let fir = Fir::new(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    /// let resp = fir.frequency_response(256, 44100.0).unwrap();
    ///
    /// // DC gain of an averaging filter = 1.0
    /// assert!((resp.magnitudes[0] - 1.0).abs() < 1e-4);
    /// ```
    fn frequency_response(
        &self,
        n_points: usize,
        sample_rate: f32,
    ) -> Result<FrequencyResponse, FilterError> {
        if n_points < 2 || sample_rate <= 0.0 {
            return Err(FilterError::InvalidParameter);
        }

        let n_taps = self.coeffs().len();
        if n_taps == 0 {
            return Err(FilterError::Empty);
        }

        // Find smallest power-of-two N so that N/2+1 >= n_points.
        let min_fft_len = (n_points - 1) * 2; // need N/2 >= n_points-1
        let fft_len = min_fft_len
            .next_power_of_two()
            .max(n_taps.next_power_of_two());

        // Zero-padded coefficient buffer
        let mut padded = vec![0.0_f32; fft_len];
        for (dst, &src) in padded.iter_mut().zip(self.coeffs().iter()) {
            *dst = src;
        }

        // rfft: input N real → N/2+1 complex bins
        let n_bins = fft_len / 2 + 1;
        let mut bins = vec![resonant_fft::Complex::new(0.0_f32, 0.0); n_bins];
        resonant_fft::rfft(&padded, &mut bins).map_err(|_| FilterError::InvalidParameter)?;

        // Build output — pick first n_points bins from the N/2+1 available
        let nyquist = sample_rate / 2.0;
        let mut frequencies = vec![0.0_f32; n_points];
        let mut magnitudes = vec![0.0_f32; n_points];
        let mut phases = vec![0.0_f32; n_points];

        for i in 0..n_points {
            // Map n_points indices to bins 0..n_bins
            let bin_idx_f32 = i as f32 * (n_bins - 1) as f32 / (n_points - 1) as f32;
            let bin_idx = bin_idx_f32.round() as usize;
            let complex_bin = bins[bin_idx.min(n_bins - 1)];
            frequencies[i] = nyquist * i as f32 / (n_points - 1) as f32;
            magnitudes[i] =
                (complex_bin.re * complex_bin.re + complex_bin.im * complex_bin.im).sqrt();
            phases[i] = complex_bin.im.atan2(complex_bin.re);
        }

        Ok(FrequencyResponse {
            frequencies,
            magnitudes,
            phases,
        })
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec;

    use super::*;
    use crate::design;

    #[test]
    fn filter_error_display() {
        let _ = std::format!("{}", FilterError::Empty);
        let _ = std::format!("{}", FilterError::InvalidParameter);
    }

    #[test]
    fn biquad_invalid_params() {
        let coeffs = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        assert_eq!(
            coeffs.frequency_response(1, 44100.0),
            Err(FilterError::InvalidParameter)
        );
        assert_eq!(
            coeffs.frequency_response(512, 0.0),
            Err(FilterError::InvalidParameter)
        );
        assert_eq!(
            coeffs.frequency_response(512, -1.0),
            Err(FilterError::InvalidParameter)
        );
    }

    #[test]
    fn biquad_passthrough_has_unity_magnitude() {
        // b0=1, all others 0 → flat magnitude 1.0 everywhere
        let coeffs = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let resp = coeffs.frequency_response(64, 44100.0).unwrap();
        for &m in &resp.magnitudes {
            assert!((m - 1.0).abs() < 1e-5, "expected 1.0, got {m}");
        }
    }

    #[test]
    fn biquad_butterworth_lowpass_cutoff_near_minus3db() {
        let cutoff = 1000.0_f64;
        let sr = 44100.0_f64;
        let coeffs = design::butterworth_lowpass(cutoff, sr).unwrap();
        let resp = coeffs.frequency_response(512, sr as f32).unwrap();
        let db = resp.magnitudes_db();

        // Bin closest to 1 kHz
        let bin = (cutoff as f32 / (sr as f32 / 2.0) * (512 - 1) as f32).round() as usize;
        assert!(
            (db[bin] - (-3.0)).abs() < 1.0,
            "Butterworth cutoff should be near -3 dB, got {:.2} dB",
            db[bin]
        );
    }

    #[test]
    fn biquad_butterworth_lowpass_passband_flat() {
        let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
        let resp = coeffs.frequency_response(512, 44100.0).unwrap();
        let db = resp.magnitudes_db();

        // Below 200 Hz the response should be within 0.1 dB of 0 dB
        let cutoff_bin = (200.0_f32 / 22050.0 * 511.0).round() as usize;
        for &d in &db[..cutoff_bin] {
            assert!(d.abs() < 0.1, "passband deviation too large: {d:.3} dB");
        }
    }

    #[test]
    fn biquad_butterworth_lowpass_stopband_attenuates() {
        let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
        let resp = coeffs.frequency_response(512, 44100.0).unwrap();
        let db = resp.magnitudes_db();

        // Above 5 kHz the response should be well below -20 dB
        let stopband_bin = (5000.0_f32 / 22050.0 * 511.0).round() as usize;
        for &d in &db[stopband_bin..] {
            assert!(d < -20.0, "stopband attenuation insufficient: {d:.1} dB");
        }
    }

    #[test]
    fn biquad_response_output_lengths_match() {
        let coeffs = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let resp = coeffs.frequency_response(128, 48000.0).unwrap();
        assert_eq!(resp.frequencies.len(), 128);
        assert_eq!(resp.magnitudes.len(), 128);
        assert_eq!(resp.phases.len(), 128);
    }

    #[test]
    fn biquad_dc_frequency_is_zero() {
        let coeffs = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let resp = coeffs.frequency_response(64, 44100.0).unwrap();
        assert!((resp.frequencies[0]).abs() < 1e-6);
    }

    #[test]
    fn biquad_nyquist_frequency_matches_sample_rate() {
        let coeffs = BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let resp = coeffs.frequency_response(64, 44100.0).unwrap();
        assert!((resp.frequencies[63] - 22050.0).abs() < 1.0);
    }

    #[test]
    fn magnitudes_db_floor_at_minus120() {
        let coeffs = BiquadCoeffs {
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let resp = coeffs.frequency_response(16, 44100.0).unwrap();
        for &d in &resp.magnitudes_db() {
            assert!(d >= -120.0, "dB floor violated: {d}");
        }
    }

    #[test]
    fn fir_invalid_params() {
        let fir = Fir::new(vec![1.0_f32]);
        assert_eq!(
            fir.frequency_response(1, 44100.0),
            Err(FilterError::InvalidParameter)
        );
        assert_eq!(
            fir.frequency_response(64, 0.0),
            Err(FilterError::InvalidParameter)
        );
    }

    #[test]
    fn fir_averaging_dc_gain_is_one() {
        // An averaging filter of any length has DC gain = 1
        let n = 16;
        let fir = Fir::new(vec![1.0 / n as f32; n]);
        let resp = fir.frequency_response(256, 44100.0).unwrap();
        assert!(
            (resp.magnitudes[0] - 1.0).abs() < 1e-3,
            "DC gain: {}",
            resp.magnitudes[0]
        );
    }

    #[test]
    fn fir_single_tap_is_flat() {
        // A 1-tap FIR with coefficient c has flat magnitude c
        let fir = Fir::new(vec![0.5_f32]);
        let resp = fir.frequency_response(64, 44100.0).unwrap();
        for &m in &resp.magnitudes {
            assert!((m - 0.5).abs() < 1e-3, "expected flat 0.5, got {m}");
        }
    }

    #[test]
    fn fir_response_output_lengths_match() {
        let fir = Fir::new(vec![1.0_f32; 8]);
        let resp = fir.frequency_response(128, 44100.0).unwrap();
        assert_eq!(resp.frequencies.len(), 128);
        assert_eq!(resp.magnitudes.len(), 128);
        assert_eq!(resp.phases.len(), 128);
    }

    #[test]
    fn fir_dc_frequency_is_zero() {
        let fir = Fir::new(vec![1.0_f32; 4]);
        let resp = fir.frequency_response(64, 44100.0).unwrap();
        assert!((resp.frequencies[0]).abs() < 1e-6);
    }

    #[test]
    fn fir_nyquist_frequency_matches_sample_rate() {
        let fir = Fir::new(vec![1.0_f32; 4]);
        let resp = fir.frequency_response(64, 44100.0).unwrap();
        assert!((resp.frequencies[63] - 22050.0).abs() < 1.0);
    }

    #[test]
    fn fir_averaging_attenuates_at_nyquist() {
        // An averaging FIR heavily attenuates high frequencies
        let n = 32;
        let fir = Fir::new(vec![1.0 / n as f32; n]);
        let resp = fir.frequency_response(256, 44100.0).unwrap();
        let db = resp.magnitudes_db();
        // Nyquist should be well below −20 dB
        assert!(
            db[db.len() - 1] < -20.0,
            "Nyquist not attenuated enough: {} dB",
            db[db.len() - 1]
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn filter_error_serde_roundtrip() {
        for e in [FilterError::Empty, FilterError::InvalidParameter] {
            let json = serde_json::to_string(&e)
                .unwrap_or_else(|err| panic!("serialize FilterError: {err}"));
            let back: FilterError = serde_json::from_str(&json)
                .unwrap_or_else(|err| panic!("deserialize FilterError: {err}"));
            assert_eq!(e, back);
        }
    }

    #[cfg(feature = "serde-alloc")]
    #[test]
    fn frequency_response_serde_roundtrip() {
        let fr = FrequencyResponse {
            frequencies: vec![0.0, 500.0, 1000.0],
            magnitudes: vec![1.0, 0.707, 0.5],
            phases: vec![0.0, -0.5, -1.0],
        };
        let json = serde_json::to_string(&fr)
            .unwrap_or_else(|e| panic!("serialize FrequencyResponse: {e}"));
        let back: FrequencyResponse = serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("deserialize FrequencyResponse: {e}"));
        assert_eq!(fr, back);
    }
}
