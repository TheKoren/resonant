//! Pitch estimation using the YIN algorithm.
//!
//! YIN is an autocorrelation-based fundamental frequency estimator designed
//! for monophonic audio. It works well for voiced speech and tonal instruments
//! but is unreliable for percussion or polyphonic signals.
//!
//! Reference: de Cheveigné & Kawahara (2002), doi:10.1121/1.1458024

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::AnalysisError;

/// Result of pitch estimation.
///
/// # Examples
///
/// ```
/// use resonant_analysis::pitch::PitchEstimate;
///
/// let est = PitchEstimate { frequency_hz: Some(440.0), confidence: 0.95 };
/// assert!(est.frequency_hz.is_some());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchEstimate {
    /// Estimated fundamental frequency in Hz, or `None` if no clear pitch.
    pub frequency_hz: Option<f32>,
    /// Confidence in `[0.0, 1.0]`. Higher means more periodic.
    pub confidence: f32,
}

/// YIN pitch estimator with configurable parameters.
///
/// # Examples
///
/// ```
/// use resonant_analysis::pitch::YinEstimator;
///
/// let estimator = YinEstimator::new(44100.0);
/// // Generate a 440 Hz sine wave (1024 samples)
/// let samples: Vec<f32> = (0..1024)
///     .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
///     .collect();
/// let est = estimator.estimate(&samples).unwrap();
/// assert!(est.frequency_hz.is_some());
/// assert!((est.frequency_hz.unwrap() - 440.0).abs() < 5.0);
/// ```
#[derive(Debug, Clone)]
pub struct YinEstimator {
    sample_rate: f32,
    threshold: f32,
    min_frequency: f32,
    max_frequency: f32,
}

impl YinEstimator {
    /// Creates a new estimator for the given sample rate.
    ///
    /// Defaults: threshold = 0.15, frequency range = 80–1000 Hz.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            threshold: 0.15,
            min_frequency: 80.0,
            max_frequency: 1000.0,
        }
    }

    /// Sets the CMND threshold (lower = stricter pitch detection).
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Sets the detectable frequency range in Hz.
    #[must_use]
    pub fn with_frequency_range(mut self, min_hz: f32, max_hz: f32) -> Self {
        self.min_frequency = min_hz;
        self.max_frequency = max_hz;
        self
    }

    /// Returns the configured sample rate.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Returns the configured threshold.
    #[inline]
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Estimates the fundamental frequency of `samples`.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::EmptyInput`] if `samples` is empty, or
    /// [`AnalysisError::InvalidParameter`] if the sample rate or frequency
    /// range is invalid.
    pub fn estimate(&self, samples: &[f32]) -> Result<PitchEstimate, AnalysisError> {
        if samples.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }
        if self.sample_rate <= 0.0 {
            return Err(AnalysisError::InvalidParameter {
                name: "sample_rate",
                reason: "must be positive",
            });
        }
        if self.min_frequency >= self.max_frequency || self.min_frequency <= 0.0 {
            return Err(AnalysisError::InvalidParameter {
                name: "frequency_range",
                reason: "min must be positive and less than max",
            });
        }

        // Lag range from frequency range
        let min_lag = (self.sample_rate / self.max_frequency).ceil() as usize;
        let max_lag = (self.sample_rate / self.min_frequency).floor() as usize;

        // Need at least max_lag samples to compute the difference function
        let half_len = samples.len() / 2;
        let effective_max = max_lag.min(half_len);

        if min_lag >= effective_max {
            return Ok(PitchEstimate {
                frequency_hz: None,
                confidence: 0.0,
            });
        }

        let diff = difference_function(samples, effective_max);
        let cmnd = cumulative_mean_normalized_difference(&diff);

        match absolute_threshold(&cmnd, min_lag, effective_max, self.threshold) {
            Some(tau) => {
                let refined = parabolic_interpolation(&cmnd, tau);
                let freq = self.sample_rate / refined;
                let conf = 1.0 - cmnd[tau];
                Ok(PitchEstimate {
                    frequency_hz: Some(freq),
                    confidence: conf.clamp(0.0, 1.0),
                })
            }
            None => Ok(PitchEstimate {
                frequency_hz: None,
                confidence: 0.0,
            }),
        }
    }
}

/// Step 2: Difference function d(τ) = Σ(x[i] - x[i+τ])².
fn difference_function(samples: &[f32], max_lag: usize) -> Vec<f32> {
    let mut d = vec![0.0_f32; max_lag + 1];
    // d[0] is always 0 by definition
    for tau in 1..=max_lag {
        let mut sum = 0.0_f32;
        for i in 0..samples.len() - tau {
            let delta = samples[i] - samples[i + tau];
            sum += delta * delta;
        }
        d[tau] = sum;
    }
    d
}

/// Step 3: Cumulative mean normalized difference (CMND).
fn cumulative_mean_normalized_difference(d: &[f32]) -> Vec<f32> {
    let mut cmnd = vec![0.0_f32; d.len()];
    if d.is_empty() {
        return cmnd;
    }
    cmnd[0] = 1.0; // by convention
    let mut running_sum = 0.0_f32;

    for tau in 1..d.len() {
        running_sum += d[tau];
        if running_sum > f32::EPSILON {
            cmnd[tau] = d[tau] * tau as f32 / running_sum;
        } else {
            cmnd[tau] = 1.0;
        }
    }
    cmnd
}

/// Step 4: Find the first τ in [min_lag, max_lag) where CMND dips below threshold,
/// then pick the minimum in that dip.
fn absolute_threshold(
    cmnd: &[f32],
    min_lag: usize,
    max_lag: usize,
    threshold: f32,
) -> Option<usize> {
    // Find first tau below threshold
    let mut tau = min_lag;
    while tau < max_lag {
        if cmnd[tau] < threshold {
            // Walk to the local minimum of the dip
            while tau + 1 < max_lag && cmnd[tau + 1] < cmnd[tau] {
                tau += 1;
            }
            return Some(tau);
        }
        tau += 1;
    }
    None
}

/// Step 5: Parabolic interpolation around the detected lag for sub-sample accuracy.
fn parabolic_interpolation(cmnd: &[f32], tau: usize) -> f32 {
    if tau == 0 || tau >= cmnd.len() - 1 {
        return tau as f32;
    }
    let s0 = cmnd[tau - 1];
    let s1 = cmnd[tau];
    let s2 = cmnd[tau + 1];
    let denominator = 2.0 * s1 - s2 - s0;
    if denominator.abs() < f32::EPSILON {
        return tau as f32;
    }
    tau as f32 + (s0 - s2) / (2.0 * denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    fn sine_wave(freq_hz: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn detect_440hz_sine() {
        let sr = 44100.0;
        let samples = sine_wave(440.0, sr, 2048);
        let est = YinEstimator::new(sr).estimate(&samples).ok();
        let freq = est.and_then(|e| e.frequency_hz);
        assert!(
            freq.is_some_and(|f| (f - 440.0).abs() < 5.0),
            "expected ~440 Hz, got {freq:?}"
        );
        let conf = est.map(|e| e.confidence);
        assert!(
            conf.is_some_and(|c| c > 0.8),
            "expected high confidence, got {conf:?}"
        );
    }

    #[test]
    fn detect_100hz_sine() {
        let sr = 44100.0;
        let samples = sine_wave(100.0, sr, 4096);
        let est = YinEstimator::new(sr).estimate(&samples).ok();
        let freq = est.and_then(|e| e.frequency_hz);
        assert!(
            freq.is_some_and(|f| (f - 100.0).abs() < 3.0),
            "expected ~100 Hz, got {freq:?}"
        );
    }

    #[test]
    fn detect_880hz_sine() {
        let sr = 44100.0;
        let samples = sine_wave(880.0, sr, 2048);
        let est = YinEstimator::new(sr)
            .with_frequency_range(80.0, 2000.0)
            .estimate(&samples)
            .ok();
        let freq = est.and_then(|e| e.frequency_hz);
        assert!(
            freq.is_some_and(|f| (f - 880.0).abs() < 10.0),
            "expected ~880 Hz, got {freq:?}"
        );
    }

    #[test]
    fn silence_no_pitch() {
        let sr = 44100.0;
        let samples = vec![0.0_f32; 2048];
        let est = YinEstimator::new(sr).estimate(&samples).ok();
        // Silence should yield no pitch or very low confidence
        if let Some(e) = est {
            if e.frequency_hz.is_some() {
                assert!(e.confidence < 0.5, "silence should have low confidence");
            }
        }
    }

    #[test]
    fn empty_input_returns_error() {
        let est = YinEstimator::new(44100.0).estimate(&[]);
        assert_eq!(est, Err(AnalysisError::EmptyInput));
    }

    #[test]
    fn invalid_sample_rate() {
        let est = YinEstimator::new(0.0).estimate(&[1.0; 100]);
        assert!(matches!(est, Err(AnalysisError::InvalidParameter { .. })));
    }

    #[test]
    fn invalid_frequency_range() {
        let est = YinEstimator::new(44100.0)
            .with_frequency_range(1000.0, 100.0)
            .estimate(&[1.0; 2048]);
        assert!(matches!(est, Err(AnalysisError::InvalidParameter { .. })));
    }

    #[test]
    fn very_short_signal() {
        let sr = 44100.0;
        let samples = sine_wave(440.0, sr, 16);
        let est = YinEstimator::new(sr).estimate(&samples).ok();
        // Too short to detect 80 Hz (min_lag > half_len)
        assert!(est.is_some_and(|e| e.frequency_hz.is_none()));
    }

    #[test]
    fn builder_methods() {
        let e = YinEstimator::new(48000.0).with_threshold(0.2);
        assert_eq!(e.sample_rate(), 48000.0);
        assert_eq!(e.threshold(), 0.2);
    }

    #[test]
    fn sawtooth_detects_fundamental() {
        let sr = 44100.0;
        let freq = 220.0_f32;
        let n = 4096;
        // Sawtooth: rich in harmonics but fundamental at 220 Hz
        let samples: Vec<f32> = (0..n)
            .map(|i| {
                let phase = (freq * i as f32 / sr).fract();
                2.0 * phase - 1.0
            })
            .collect();
        let est = YinEstimator::new(sr).estimate(&samples).ok();
        let detected = est.and_then(|e| e.frequency_hz);
        assert!(
            detected.is_some_and(|f| (f - 220.0).abs() < 5.0),
            "expected ~220 Hz, got {detected:?}"
        );
    }

    #[test]
    fn confidence_higher_for_periodic() {
        let sr = 44100.0;
        let sine = sine_wave(440.0, sr, 2048);
        let noise: Vec<f32> = (0..2048)
            .map(|i| (i as f32 * 7.3).sin() * (i as f32 * 13.7).cos())
            .collect();

        let est_sine = YinEstimator::new(sr).estimate(&sine).ok();
        let est_noise = YinEstimator::new(sr).estimate(&noise).ok();

        let conf_sine = est_sine.map(|e| e.confidence).unwrap_or(0.0);
        let conf_noise = est_noise.map(|e| e.confidence).unwrap_or(0.0);
        assert!(
            conf_sine > conf_noise,
            "sine confidence ({conf_sine}) should exceed noise ({conf_noise})"
        );
    }

    // --- internal helper tests ---

    #[test]
    fn difference_function_d0_is_zero() {
        let samples = [1.0_f32, 2.0, 3.0, 4.0];
        let d = difference_function(&samples, 1);
        assert_eq!(d[0], 0.0);
    }

    #[test]
    fn cmnd_first_element_is_one() {
        let d = vec![0.0, 1.0, 2.0, 3.0];
        let cmnd = cumulative_mean_normalized_difference(&d);
        assert_eq!(cmnd[0], 1.0);
    }

    #[test]
    fn parabolic_interpolation_at_boundary() {
        let cmnd = [0.5, 0.1, 0.3];
        let result = parabolic_interpolation(&cmnd, 0);
        assert_eq!(result, 0.0); // boundary, no interpolation
    }
}
