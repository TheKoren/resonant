//! Onset detection via spectral flux and adaptive thresholding.
//!
//! An onset marks the beginning of a new musical event (note, drum hit, etc.).
//! This module computes the onset strength envelope using half-wave rectified
//! spectral flux, then picks peaks above an adaptive threshold.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use resonant_core::signal::Signal;
use resonant_core::window;
use resonant_fft::stft::Stft;
use resonant_fft::SignalFreqExt;

use crate::error::AnalysisError;

/// A detected onset event.
///
/// # Examples
///
/// ```
/// use resonant_analysis::onset::Onset;
///
/// let o = Onset { sample_index: 22050, time_secs: 0.5, strength: 1.2 };
/// assert_eq!(o.time_secs, 0.5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Onset {
    /// Sample index of the onset.
    pub sample_index: usize,
    /// Time in seconds.
    pub time_secs: f32,
    /// Strength of the onset (spectral flux value).
    pub strength: f32,
}

/// Onset detector with configurable STFT and threshold parameters.
///
/// # Examples
///
/// ```
/// use resonant_analysis::onset::OnsetDetector;
///
/// let detector = OnsetDetector::new(44100.0);
/// // Silence → no onsets
/// let onsets = detector.detect(&vec![0.0_f32; 4096]).unwrap();
/// assert!(onsets.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct OnsetDetector {
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
    threshold_multiplier: f32,
    median_window: usize,
}

impl OnsetDetector {
    /// Creates a detector with default parameters.
    ///
    /// Defaults: window 1024, hop 512, threshold multiplier 1.5, median window 10.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            window_size: 1024,
            hop_size: 512,
            threshold_multiplier: 1.5,
            median_window: 10,
        }
    }

    /// Sets the STFT window size (must be a power of two).
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

    /// Sets the adaptive threshold multiplier.
    ///
    /// Higher values → fewer onsets (stricter). Lower → more onsets.
    #[must_use]
    pub fn with_threshold_multiplier(mut self, m: f32) -> Self {
        self.threshold_multiplier = m;
        self
    }

    /// Sets the median filter window size for adaptive thresholding.
    #[must_use]
    pub fn with_median_window(mut self, w: usize) -> Self {
        self.median_window = w;
        self
    }

    /// Detects onsets in mono audio samples.
    ///
    /// Returns a list of [`Onset`] events sorted by time.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::EmptyInput`] if `samples` is empty, or
    /// an FFT error if the window size is invalid.
    pub fn detect(&self, samples: &[f32]) -> Result<Vec<Onset>, AnalysisError> {
        let envelope = self.onset_strength(samples)?;
        let threshold =
            adaptive_threshold(&envelope, self.median_window, self.threshold_multiplier);
        let peaks = pick_peaks(&envelope, &threshold);

        let onsets = peaks
            .into_iter()
            .map(|(frame_idx, strength)| {
                let sample_index = frame_idx * self.hop_size;
                Onset {
                    sample_index,
                    time_secs: sample_index as f32 / self.sample_rate,
                    strength,
                }
            })
            .collect();

        Ok(onsets)
    }

    /// Computes the onset strength envelope (spectral flux per STFT frame).
    ///
    /// Returns one value per frame. The first frame always has strength 0.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::EmptyInput`] if `samples` is empty.
    pub fn onset_strength(&self, samples: &[f32]) -> Result<Vec<f32>, AnalysisError> {
        if samples.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        // If the signal is shorter than one window, no frames can be extracted
        if samples.len() < self.window_size {
            return Ok(vec![0.0]);
        }

        let signal = Signal::from_samples(samples.to_vec());
        let stft = Stft::builder(self.window_size, self.hop_size)
            .window_fn(window::hann)
            .build();

        let frames = stft.analyze(&signal)?;
        if frames.is_empty() {
            return Ok(vec![0.0]);
        }

        let magnitudes: Vec<Vec<f32>> = frames.iter().map(|f| f.magnitude()).collect();

        let mut envelope = Vec::with_capacity(magnitudes.len());
        envelope.push(0.0); // first frame has no previous frame

        for i in 1..magnitudes.len() {
            envelope.push(spectral_flux(&magnitudes[i - 1], &magnitudes[i]));
        }

        Ok(envelope)
    }
}

/// Half-wave rectified spectral flux between consecutive magnitude frames.
fn spectral_flux(prev: &[f32], curr: &[f32]) -> f32 {
    prev.iter().zip(curr).map(|(p, c)| (c - p).max(0.0)).sum()
}

/// Adaptive threshold: local median + multiplier × local mean absolute deviation.
fn adaptive_threshold(envelope: &[f32], window: usize, multiplier: f32) -> Vec<f32> {
    let n = envelope.len();
    let mut thresholds = Vec::with_capacity(n);
    let half = window / 2;

    for i in 0..n {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let local = &envelope[start..end];

        let median = local_median(local);
        let mad = local_mad(local, median);
        thresholds.push(median + multiplier * mad);
    }

    thresholds
}

/// Median of a small slice (copies and sorts).
fn local_median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Mean absolute deviation from the median.
fn local_mad(values: &[f32], median: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: f32 = values.iter().map(|v| (v - median).abs()).sum();
    sum / values.len() as f32
}

/// Picks local maxima in the envelope that exceed the adaptive threshold.
fn pick_peaks(envelope: &[f32], threshold: &[f32]) -> Vec<(usize, f32)> {
    let mut peaks = Vec::new();
    for i in 1..envelope.len().saturating_sub(1) {
        let v = envelope[i];
        if v > threshold[i] && v >= envelope[i - 1] && v >= envelope[i + 1] {
            peaks.push((i, v));
        }
    }
    peaks
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    const SR: f32 = 44100.0;

    #[test]
    fn silence_no_onsets() {
        let detector = OnsetDetector::new(SR);
        let samples = vec![0.0_f32; 8192];
        let onsets = detector.detect(&samples).ok();
        assert_eq!(onsets.as_ref().map(|o| o.len()), Some(0));
    }

    #[test]
    fn empty_input_returns_error() {
        let detector = OnsetDetector::new(SR);
        let result = detector.detect(&[]);
        assert_eq!(result, Err(AnalysisError::EmptyInput));
    }

    #[test]
    fn onset_strength_empty_error() {
        let detector = OnsetDetector::new(SR);
        let result = detector.onset_strength(&[]);
        assert_eq!(result, Err(AnalysisError::EmptyInput));
    }

    #[test]
    fn onset_strength_nonnegative() {
        let detector = OnsetDetector::new(SR);
        let samples: Vec<f32> = (0..8192)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / SR).sin())
            .collect();
        let envelope = detector.onset_strength(&samples).ok();
        if let Some(env) = &envelope {
            for &v in env {
                assert!(v >= 0.0, "negative onset strength: {v}");
            }
        }
    }

    #[test]
    fn onset_strength_first_frame_zero() {
        let detector = OnsetDetector::new(SR);
        let samples = vec![1.0_f32; 4096];
        let envelope = detector.onset_strength(&samples).ok();
        assert!(envelope.as_ref().is_some_and(|e| e[0] == 0.0));
    }

    #[test]
    fn click_track_detects_onsets() {
        // Create a click track: silence with impulses at known positions
        // 1 second of silence; place clicks at 0.2s, 0.5s, 0.8s
        let mut samples = vec![0.0_f32; 44100];
        let click_positions = [8820_usize, 22050, 35280];
        for &pos in &click_positions {
            // Short burst (64 samples of sine at 1kHz)
            for j in 0..64 {
                if pos + j < samples.len() {
                    samples[pos + j] = (2.0 * PI * 1000.0 * j as f32 / SR).sin();
                }
            }
        }

        let detector = OnsetDetector::new(SR)
            .with_window_size(1024)
            .with_hop_size(512)
            .with_threshold_multiplier(1.0);

        let onsets = detector.detect(&samples).ok();
        let onsets = onsets.as_ref();

        // Should detect at least 2 of the 3 clicks (the first may be missed
        // due to edge effects)
        assert!(
            onsets.is_some_and(|o| o.len() >= 2),
            "expected >=2 onsets, got {:?}",
            onsets.map(|o| o.len())
        );

        // Each detected onset should be near one of the click positions
        if let Some(onsets) = onsets {
            for onset in onsets {
                let near_click = click_positions.iter().any(|&pos| {
                    let diff = (onset.sample_index as i64 - pos as i64).unsigned_abs() as usize;
                    diff < 2048 // within ~2 hops
                });
                assert!(
                    near_click,
                    "onset at sample {} not near any click",
                    onset.sample_index
                );
            }
        }
    }

    #[test]
    fn sine_onset_detected() {
        // Silence then sine: onset at the transition
        let mut samples = vec![0.0_f32; 8192];
        let onset_pos = 4096;
        for i in onset_pos..8192 {
            samples[i] = (2.0 * PI * 440.0 * (i - onset_pos) as f32 / SR).sin();
        }

        let detector = OnsetDetector::new(SR)
            .with_window_size(1024)
            .with_hop_size(512)
            .with_threshold_multiplier(1.0);

        let onsets = detector.detect(&samples).ok();
        assert!(
            onsets.as_ref().is_some_and(|o| !o.is_empty()),
            "should detect onset at silence→sine transition"
        );

        // The detected onset should be near the actual transition
        if let Some(onsets) = &onsets {
            let nearest = onsets
                .iter()
                .min_by_key(|o| (o.sample_index as i64 - onset_pos as i64).unsigned_abs())
                .map(|o| o.sample_index);
            assert!(
                nearest.is_some_and(|n| (n as i64 - onset_pos as i64).unsigned_abs() < 2048),
                "onset too far from transition: {nearest:?}"
            );
        }
    }

    #[test]
    fn short_signal_returns_single_zero() {
        let detector = OnsetDetector::new(SR).with_window_size(1024);
        let samples = vec![1.0_f32; 512]; // shorter than window
        let env = detector.onset_strength(&samples).ok();
        assert_eq!(env.as_deref(), Some([0.0].as_slice()));
    }

    #[test]
    fn builder_methods() {
        let d = OnsetDetector::new(SR)
            .with_window_size(2048)
            .with_hop_size(1024)
            .with_threshold_multiplier(2.0)
            .with_median_window(20);
        assert_eq!(d.window_size, 2048);
        assert_eq!(d.hop_size, 1024);
        assert_eq!(d.threshold_multiplier, 2.0);
        assert_eq!(d.median_window, 20);
    }

    #[test]
    fn onset_time_matches_sample_index() {
        // Check that time_secs = sample_index / sample_rate
        let mut samples = vec![0.0_f32; 8192];
        for i in 4096..8192 {
            samples[i] = (2.0 * PI * 1000.0 * i as f32 / SR).sin();
        }

        let detector = OnsetDetector::new(SR)
            .with_window_size(1024)
            .with_hop_size(512)
            .with_threshold_multiplier(1.0);

        let onsets = detector.detect(&samples).ok();
        if let Some(onsets) = &onsets {
            for o in onsets {
                let expected_time = o.sample_index as f32 / SR;
                assert!(
                    (o.time_secs - expected_time).abs() < 1e-6,
                    "time mismatch: {} vs {expected_time}",
                    o.time_secs
                );
            }
        }
    }

    // --- internal helper tests ---

    #[test]
    fn spectral_flux_identical_is_zero() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(spectral_flux(&a, &a), 0.0);
    }

    #[test]
    fn spectral_flux_increase_only() {
        let prev = [0.0, 0.0, 0.0];
        let curr = [1.0, 2.0, 3.0];
        assert_eq!(spectral_flux(&prev, &curr), 6.0);
    }

    #[test]
    fn spectral_flux_decrease_ignored() {
        let prev = [3.0, 2.0, 1.0];
        let curr = [0.0, 0.0, 0.0];
        assert_eq!(spectral_flux(&prev, &curr), 0.0);
    }

    #[test]
    fn local_median_odd() {
        assert_eq!(local_median(&[3.0, 1.0, 2.0]), 2.0);
    }

    #[test]
    fn local_median_even() {
        assert_eq!(local_median(&[1.0, 3.0, 2.0, 4.0]), 2.5);
    }

    #[test]
    fn local_median_empty() {
        assert_eq!(local_median(&[]), 0.0);
    }

    #[test]
    fn local_mad_uniform() {
        assert_eq!(local_mad(&[5.0, 5.0, 5.0], 5.0), 0.0);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn onset_serde_roundtrip() {
        let o = Onset {
            sample_index: 22050,
            time_secs: 0.5,
            strength: 1.2,
        };
        let json = serde_json::to_string(&o).unwrap_or_else(|e| panic!("serialize Onset: {e}"));
        let back: Onset =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("deserialize Onset: {e}"));
        assert_eq!(o, back);
    }
}
