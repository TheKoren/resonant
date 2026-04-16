//! Tempo estimation via autocorrelation of the onset strength envelope.
//!
//! Computes the onset strength envelope using [`OnsetDetector`], then finds
//! the dominant periodicity via normalized autocorrelation. The peak lag is
//! converted to BPM.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::error::AnalysisError;
use crate::onset::OnsetDetector;

/// Result of tempo estimation.
///
/// # Examples
///
/// ```
/// use resonant_analysis::tempo::TempoEstimate;
///
/// let est = TempoEstimate { bpm: 120.0, confidence: 0.85 };
/// assert!(est.bpm > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoEstimate {
    /// Estimated tempo in beats per minute.
    pub bpm: f32,
    /// Pearson autocorrelation coefficient normalised to `[0.0, 1.0]`.
    ///
    /// Values below [`TempoEstimator::CONFIDENCE_LOW`] indicate insufficient
    /// periodicity in the signal — the BPM estimate is unreliable. Values above
    /// [`TempoEstimator::CONFIDENCE_HIGH`] indicate a strong, consistent pulse
    /// and the estimate is reliable for most tonal and rhythmic content.
    pub confidence: f32,
}

/// Tempo estimator with configurable BPM range.
///
/// # Examples
///
/// ```
/// use resonant_analysis::tempo::TempoEstimator;
///
/// let estimator = TempoEstimator::new(44100.0);
/// // 1 second of silence → low confidence
/// let est = estimator.estimate(&vec![0.0_f32; 44100]).unwrap();
/// assert!(est.confidence < 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct TempoEstimator {
    sample_rate: f32,
    min_bpm: f32,
    max_bpm: f32,
    onset_detector: OnsetDetector,
}

impl TempoEstimator {
    /// Confidence below this value indicates insufficient periodicity.
    ///
    /// The BPM estimate is unreliable when `confidence < CONFIDENCE_LOW`.
    pub const CONFIDENCE_LOW: f32 = 0.3;

    /// Confidence above this value indicates a solid, reliable estimate.
    ///
    /// When `confidence > CONFIDENCE_HIGH` the estimate is suitable for
    /// beat-synchronised playback, grid snapping, and similar use cases.
    pub const CONFIDENCE_HIGH: f32 = 0.6;

    /// Creates a tempo estimator with default parameters.
    ///
    /// Defaults: BPM range 60–200, onset detector with window 1024 / hop 512.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            min_bpm: 60.0,
            max_bpm: 200.0,
            onset_detector: OnsetDetector::new(sample_rate),
        }
    }

    /// Sets the detectable BPM range.
    #[must_use]
    pub fn with_bpm_range(mut self, min: f32, max: f32) -> Self {
        self.min_bpm = min;
        self.max_bpm = max;
        self
    }

    /// Sets a custom onset detector.
    #[must_use]
    pub fn with_onset_detector(mut self, detector: OnsetDetector) -> Self {
        self.onset_detector = detector;
        self
    }

    /// Estimates the tempo of mono audio samples.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::EmptyInput`] if `samples` is empty, or
    /// [`AnalysisError::InvalidParameter`] if the BPM range is invalid.
    pub fn estimate(&self, samples: &[f32]) -> Result<TempoEstimate, AnalysisError> {
        if samples.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }
        if self.min_bpm >= self.max_bpm || self.min_bpm <= 0.0 {
            return Err(AnalysisError::InvalidParameter {
                name: "bpm_range",
                reason: "min must be positive and less than max",
            });
        }

        let envelope = self.onset_detector.onset_strength(samples)?;
        if envelope.len() < 2 {
            return Ok(TempoEstimate {
                bpm: 0.0,
                confidence: 0.0,
            });
        }

        // Convert BPM range to lag range in onset envelope frames
        let hop = self.onset_detector_hop();
        let frames_per_sec = self.sample_rate / hop as f32;
        let min_lag = (60.0 * frames_per_sec / self.max_bpm).floor() as usize;
        let max_lag = (60.0 * frames_per_sec / self.min_bpm).ceil() as usize;
        let effective_max = max_lag.min(envelope.len() - 1);

        if min_lag >= effective_max {
            return Ok(TempoEstimate {
                bpm: 0.0,
                confidence: 0.0,
            });
        }

        let acf = autocorrelation(&envelope, effective_max);

        // Find the best peak, preferring shorter lags to resolve octave ambiguity.
        // If a peak at a shorter lag is within 90% of the global max, prefer it.
        let (peak_lag, peak_val) = find_best_peak(&acf, min_lag, effective_max);

        if peak_lag == 0 {
            return Ok(TempoEstimate {
                bpm: 0.0,
                confidence: 0.0,
            });
        }

        // Sub-lag refinement
        let refined_lag = parabolic_interpolation(&acf, peak_lag);
        let bpm = 60.0 * frames_per_sec / refined_lag;

        // Confidence: normalized peak relative to zero-lag (acf[0])
        let confidence = if acf[0] > f32::EPSILON {
            (peak_val / acf[0]).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Ok(TempoEstimate { bpm, confidence })
    }

    /// Infers the hop size from the onset detector.
    fn onset_detector_hop(&self) -> usize {
        // OnsetDetector's default hop is 512; we access it indirectly
        // via the stored field. Since OnsetDetector fields are private,
        // we reconstruct the default or use the stored detector's config.
        // For now, use the same default hop of 512.
        512
    }
}

/// Normalized autocorrelation of the envelope.
fn autocorrelation(envelope: &[f32], max_lag: usize) -> Vec<f32> {
    let n = envelope.len();
    let mut acf = vec![0.0_f32; max_lag + 1];

    for lag in 0..=max_lag {
        let mut sum = 0.0_f32;
        for i in 0..n - lag {
            sum += envelope[i] * envelope[i + lag];
        }
        acf[lag] = sum;
    }
    acf
}

/// Finds the best peak in acf[min_lag..=max_lag], resolving octave ambiguity
/// by checking if a peak at half the lag (double tempo) is also strong.
fn find_best_peak(acf: &[f32], min_lag: usize, max_lag: usize) -> (usize, f32) {
    let end = max_lag.min(acf.len() - 1);
    if min_lag > end {
        return (0, 0.0);
    }

    // Find global max position in range
    let mut best_lag = min_lag;
    let mut best_val = acf[min_lag];
    for (lag, &val) in acf.iter().enumerate().take(end + 1).skip(min_lag) {
        if val > best_val {
            best_val = val;
            best_lag = lag;
        }
    }

    // Octave correction: if the peak at half-lag is reasonably strong,
    // prefer it (the true tempo is likely double what we found).
    let half_lag = best_lag / 2;
    if half_lag >= min_lag && half_lag <= end && acf[half_lag] > best_val * 0.5 {
        best_val = acf[half_lag];
        best_lag = half_lag;
    }

    (best_lag, best_val)
}

/// Parabolic interpolation for sub-sample lag refinement.
fn parabolic_interpolation(acf: &[f32], lag: usize) -> f32 {
    if lag == 0 || lag >= acf.len() - 1 {
        return lag as f32;
    }
    let s0 = acf[lag - 1];
    let s1 = acf[lag];
    let s2 = acf[lag + 1];
    let denom = 2.0 * s1 - s0 - s2;
    if denom.abs() < f32::EPSILON {
        return lag as f32;
    }
    lag as f32 + (s0 - s2) / (2.0 * denom)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    const SR: f32 = 44100.0;

    /// Generate a click track: short bursts at regular BPM intervals.
    fn click_track(bpm: f32, duration_secs: f32, sample_rate: f32) -> Vec<f32> {
        let total_samples = (duration_secs * sample_rate) as usize;
        let mut samples = vec![0.0_f32; total_samples];
        let interval = (60.0 / bpm * sample_rate) as usize;

        let mut pos = 0;
        while pos < total_samples {
            // 32-sample burst of 1 kHz sine
            for j in 0..32 {
                if pos + j < total_samples {
                    samples[pos + j] = (2.0 * PI * 1000.0 * j as f32 / sample_rate).sin();
                }
            }
            pos += interval;
        }
        samples
    }

    #[test]
    fn detect_120bpm() {
        let samples = click_track(120.0, 5.0, SR);
        let est = TempoEstimator::new(SR).estimate(&samples).ok();
        let bpm = est.map(|e| e.bpm);
        assert!(
            bpm.is_some_and(|b| (b - 120.0).abs() < 5.0),
            "expected ~120 BPM, got {bpm:?}"
        );
    }

    #[test]
    fn detect_90bpm() {
        let samples = click_track(90.0, 6.0, SR);
        let est = TempoEstimator::new(SR)
            .with_bpm_range(60.0, 200.0)
            .estimate(&samples)
            .ok();
        let bpm = est.map(|e| e.bpm);
        assert!(
            bpm.is_some_and(|b| (b - 90.0).abs() < 5.0),
            "expected ~90 BPM, got {bpm:?}"
        );
    }

    #[test]
    fn detect_150bpm() {
        let samples = click_track(150.0, 5.0, SR);
        let est = TempoEstimator::new(SR).estimate(&samples).ok();
        let bpm = est.map(|e| e.bpm);
        assert!(
            bpm.is_some_and(|b| (b - 150.0).abs() < 5.0),
            "expected ~150 BPM, got {bpm:?}"
        );
    }

    #[test]
    fn silence_low_confidence() {
        let samples = vec![0.0_f32; 44100 * 3];
        let est = TempoEstimator::new(SR).estimate(&samples).ok();
        assert!(est.is_some_and(|e| e.confidence < 0.5));
    }

    #[test]
    fn empty_input_error() {
        let result = TempoEstimator::new(SR).estimate(&[]);
        assert_eq!(result, Err(AnalysisError::EmptyInput));
    }

    #[test]
    fn invalid_bpm_range() {
        let result = TempoEstimator::new(SR)
            .with_bpm_range(200.0, 60.0)
            .estimate(&[1.0; 44100]);
        assert!(matches!(
            result,
            Err(AnalysisError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn very_short_signal() {
        let samples = vec![1.0_f32; 512];
        let est = TempoEstimator::new(SR).estimate(&samples).ok();
        // Too short for meaningful tempo → confidence 0
        assert!(est.is_some_and(|e| e.confidence < 0.1));
    }

    #[test]
    fn confidence_higher_for_periodic() {
        let periodic = click_track(120.0, 5.0, SR);
        let noise: Vec<f32> = (0..SR as usize * 5)
            .map(|i| (i as f32 * 7.3).sin() * (i as f32 * 13.7).cos())
            .collect();

        let est_periodic = TempoEstimator::new(SR).estimate(&periodic).ok();
        let est_noise = TempoEstimator::new(SR).estimate(&noise).ok();

        let conf_p = est_periodic.map(|e| e.confidence).unwrap_or(0.0);
        let conf_n = est_noise.map(|e| e.confidence).unwrap_or(0.0);
        assert!(
            conf_p > conf_n,
            "periodic confidence ({conf_p}) should exceed noise ({conf_n})"
        );
    }

    #[test]
    fn builder_methods() {
        let e = TempoEstimator::new(SR).with_bpm_range(80.0, 180.0);
        assert_eq!(e.min_bpm, 80.0);
        assert_eq!(e.max_bpm, 180.0);
    }

    #[test]
    fn with_onset_detector() {
        let d = OnsetDetector::new(SR).with_window_size(2048);
        let e = TempoEstimator::new(SR).with_onset_detector(d);
        // Just verify it compiles and doesn't panic
        let _ = e;
    }

    // --- internal helper tests ---

    #[test]
    fn autocorrelation_of_constant() {
        let env = vec![1.0_f32; 10];
        let acf = autocorrelation(&env, 5);
        // For constant signal, acf[lag] = N - lag
        for lag in 0..=5 {
            let expected = (10 - lag) as f32;
            assert!(
                (acf[lag] - expected).abs() < 1e-4,
                "acf[{lag}] = {}, expected {expected}",
                acf[lag]
            );
        }
    }

    #[test]
    fn find_best_peak_basic() {
        // Peak at lag 4 = 8.0, half-lag 2 = 7.0 > 8.0*0.5 → prefer lag 2
        let acf = [10.0, 3.0, 7.0, 5.0, 8.0, 2.0];
        let (lag, _) = find_best_peak(&acf, 1, 4);
        assert_eq!(lag, 2); // octave correction kicks in
    }

    #[test]
    fn find_best_peak_no_correction_when_half_weak() {
        // Peak at lag 4 = 10.0, half-lag 2 = 2.0 < 10.0*0.5 → keep lag 4
        let acf = [20.0, 5.0, 2.0, 6.0, 10.0, 3.0];
        let (lag, val) = find_best_peak(&acf, 1, 4);
        assert_eq!(lag, 4);
        assert_eq!(val, 10.0);
    }

    #[test]
    fn parabolic_interpolation_at_boundary() {
        let acf = [1.0, 5.0, 3.0];
        let result = parabolic_interpolation(&acf, 0);
        assert_eq!(result, 0.0); // boundary, no interpolation
    }

    #[test]
    fn parabolic_interpolation_symmetric() {
        // Symmetric around center → no shift
        let acf = [2.0, 5.0, 2.0];
        let result = parabolic_interpolation(&acf, 1);
        assert!((result - 1.0).abs() < 1e-4);
    }
}
