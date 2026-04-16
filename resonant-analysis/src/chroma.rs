//! Chroma features — 12-bin pitch class profiles.
//!
//! A chroma vector maps spectral energy onto the 12 pitch classes
//! (C, C#, D, ..., B), collapsing octave information. Chroma features are
//! useful for chord recognition, key detection, and music similarity.
//!
//! The extractor computes an STFT, maps each FFT bin to a pitch class based
//! on its frequency, and accumulates energy per class.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use resonant_core::signal::Signal;
use resonant_core::window;
use resonant_fft::stft::Stft;
use resonant_fft::SignalFreqExt;

use crate::error::AnalysisError;

/// Names of the 12 pitch classes, indexed 0–11.
pub const PITCH_CLASS_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// A 12-bin chroma vector representing energy per pitch class.
///
/// Bins are ordered: C, C#, D, D#, E, F, F#, G, G#, A, A#, B.
///
/// # Examples
///
/// ```
/// use resonant_analysis::chroma::ChromaVector;
///
/// let cv = ChromaVector { bins: [0.0; 12] };
/// assert_eq!(cv.bins.len(), 12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChromaVector {
    /// Energy in each of the 12 pitch classes.
    pub bins: [f32; 12],
}

impl ChromaVector {
    /// Returns the index (0–11) of the dominant pitch class.
    #[must_use]
    pub fn dominant_class(&self) -> usize {
        self.bins
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Returns the name of the dominant pitch class.
    #[must_use]
    pub fn dominant_name(&self) -> &'static str {
        PITCH_CLASS_NAMES[self.dominant_class()]
    }
}

/// Chroma feature extractor with configurable STFT and tuning parameters.
///
/// # Examples
///
/// ```
/// use resonant_analysis::chroma::ChromaExtractor;
///
/// let extractor = ChromaExtractor::new(44100.0);
/// let samples = vec![0.0_f32; 8192];
/// let chroma = extractor.extract(&samples).unwrap();
/// for cv in &chroma {
///     assert_eq!(cv.bins.len(), 12);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ChromaExtractor {
    sample_rate: f32,
    window_size: usize,
    hop_size: usize,
    tuning_hz: f32,
}

impl ChromaExtractor {
    /// Creates an extractor with default parameters.
    ///
    /// Defaults: window 4096, hop 2048, A4 = 440 Hz tuning.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            window_size: 4096,
            hop_size: 2048,
            tuning_hz: 440.0,
        }
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

    /// Sets the tuning reference frequency for A4 (default 440 Hz).
    #[must_use]
    pub fn with_tuning(mut self, hz: f32) -> Self {
        self.tuning_hz = hz;
        self
    }

    /// Extracts chroma vectors from mono audio samples.
    ///
    /// Returns one [`ChromaVector`] per STFT frame.
    ///
    /// # Errors
    ///
    /// Returns [`AnalysisError::EmptyInput`] if `samples` is empty.
    pub fn extract(&self, samples: &[f32]) -> Result<Vec<ChromaVector>, AnalysisError> {
        if samples.is_empty() {
            return Err(AnalysisError::EmptyInput);
        }

        // If signal is shorter than one window, return a single zero chroma
        if samples.len() < self.window_size {
            return Ok(vec![ChromaVector { bins: [0.0; 12] }]);
        }

        let signal = Signal::from_samples(samples.to_vec());
        let stft = Stft::builder(self.window_size, self.hop_size)
            .window_fn(window::hann)
            .build();

        let stft_frames = stft.analyze(&signal)?;
        if stft_frames.is_empty() {
            return Ok(vec![ChromaVector { bins: [0.0; 12] }]);
        }

        // Precompute the pitch class map: FFT bin index → pitch class (0–11),
        // or None for bins below the audible range.
        let chroma_map = build_chroma_map(self.window_size, self.sample_rate, self.tuning_hz);

        let mut result = Vec::with_capacity(stft_frames.len());

        for frame in &stft_frames {
            let magnitudes = frame.magnitude();
            let mut bins = [0.0_f32; 12];

            for (bin_idx, &mag) in magnitudes.iter().enumerate() {
                if let Some(pitch_class) = chroma_map.get(bin_idx).copied().flatten() {
                    bins[pitch_class] += mag * mag; // accumulate power
                }
            }

            // Normalize: divide by max to get [0, 1] range, or leave zero
            let max = bins.iter().copied().fold(0.0_f32, f32::max);
            if max > f32::EPSILON {
                for b in &mut bins {
                    *b /= max;
                }
            }

            result.push(ChromaVector { bins });
        }

        Ok(result)
    }
}

/// Maps a frequency in Hz to a pitch class index (0 = C, 9 = A, etc.).
///
/// Returns `None` for frequencies below C1 (~32.7 Hz) or non-positive.
fn frequency_to_pitch_class(freq_hz: f32, tuning_hz: f32) -> Option<usize> {
    if freq_hz <= 0.0 || tuning_hz <= 0.0 {
        return None;
    }
    // A4 = tuning_hz, MIDI note 69. Pitch class = midi_note % 12.
    // midi_note = 69 + 12 * log2(freq / tuning)
    // We only need the fractional part mod 12.
    let semitones_from_a4 = 12.0 * (freq_hz / tuning_hz).log2();
    // A is pitch class 9. So pitch_class = (9 + round(semitones)) % 12.
    let midi_approx = 69.0 + semitones_from_a4;
    if midi_approx < 24.0 {
        // Below C1 — too low to assign meaningfully
        return None;
    }
    let pitch_class = midi_approx.round() as i32 % 12;
    // Ensure non-negative modulo
    Some(((pitch_class + 12) % 12) as usize)
}

/// Precomputes a lookup table mapping each FFT bin to its pitch class.
///
/// Returns `Vec<Option<usize>>` of length `fft_size / 2 + 1`.
fn build_chroma_map(fft_size: usize, sample_rate: f32, tuning_hz: f32) -> Vec<Option<usize>> {
    let num_bins = fft_size / 2 + 1;
    let bin_freq = sample_rate / fft_size as f32;

    (0..num_bins)
        .map(|bin| {
            let freq = bin as f32 * bin_freq;
            frequency_to_pitch_class(freq, tuning_hz)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    const SR: f32 = 44100.0;

    #[test]
    fn a4_maps_to_pitch_class_9() {
        let pc = frequency_to_pitch_class(440.0, 440.0);
        assert_eq!(pc, Some(9)); // A = 9
    }

    #[test]
    fn c4_maps_to_pitch_class_0() {
        // C4 ≈ 261.63 Hz
        let pc = frequency_to_pitch_class(261.63, 440.0);
        assert_eq!(pc, Some(0)); // C = 0
    }

    #[test]
    fn e4_maps_to_pitch_class_4() {
        // E4 ≈ 329.63 Hz
        let pc = frequency_to_pitch_class(329.63, 440.0);
        assert_eq!(pc, Some(4)); // E = 4
    }

    #[test]
    fn very_low_frequency_returns_none() {
        let pc = frequency_to_pitch_class(10.0, 440.0);
        assert_eq!(pc, None);
    }

    #[test]
    fn zero_frequency_returns_none() {
        assert_eq!(frequency_to_pitch_class(0.0, 440.0), None);
    }

    #[test]
    fn negative_frequency_returns_none() {
        assert_eq!(frequency_to_pitch_class(-100.0, 440.0), None);
    }

    #[test]
    fn octave_equivalence() {
        // A3 (220 Hz) and A5 (880 Hz) should both map to pitch class 9
        assert_eq!(frequency_to_pitch_class(220.0, 440.0), Some(9));
        assert_eq!(frequency_to_pitch_class(880.0, 440.0), Some(9));
    }

    #[test]
    fn chroma_map_length() {
        let map = build_chroma_map(4096, SR, 440.0);
        assert_eq!(map.len(), 2049); // 4096/2 + 1
    }

    #[test]
    fn chroma_map_dc_is_none() {
        let map = build_chroma_map(4096, SR, 440.0);
        assert_eq!(map[0], None); // DC bin = 0 Hz
    }

    #[test]
    fn pure_a4_dominates_a_bin() {
        let samples: Vec<f32> = (0..16384)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / SR).sin())
            .collect();
        let extractor = ChromaExtractor::new(SR);
        let chroma = extractor.extract(&samples).unwrap();
        assert!(!chroma.is_empty());
        // Most frames should have A (bin 9) as dominant
        let a_dominant = chroma.iter().filter(|cv| cv.dominant_class() == 9).count();
        assert!(
            a_dominant > chroma.len() / 2,
            "A4 should dominate, but only {a_dominant}/{} frames had A dominant",
            chroma.len()
        );
    }

    #[test]
    fn pure_c4_dominates_c_bin() {
        // C4 ≈ 261.63 Hz
        let samples: Vec<f32> = (0..16384)
            .map(|i| (2.0 * PI * 261.63 * i as f32 / SR).sin())
            .collect();
        let extractor = ChromaExtractor::new(SR);
        let chroma = extractor.extract(&samples).unwrap();
        assert!(!chroma.is_empty());
        let c_dominant = chroma.iter().filter(|cv| cv.dominant_class() == 0).count();
        assert!(
            c_dominant > chroma.len() / 2,
            "C4 should dominate, but only {c_dominant}/{} frames had C dominant",
            chroma.len()
        );
    }

    #[test]
    fn silence_all_near_zero() {
        let samples = vec![0.0_f32; 8192];
        let extractor = ChromaExtractor::new(SR);
        let chroma = extractor.extract(&samples).unwrap();
        for cv in &chroma {
            let sum: f32 = cv.bins.iter().sum();
            assert!(
                sum < 1e-6,
                "silence should have near-zero chroma, got {sum}"
            );
        }
    }

    #[test]
    fn empty_input_error() {
        let extractor = ChromaExtractor::new(SR);
        assert_eq!(extractor.extract(&[]), Err(AnalysisError::EmptyInput));
    }

    #[test]
    fn short_signal_returns_zero_chroma() {
        let extractor = ChromaExtractor::new(SR).with_window_size(4096);
        let chroma = extractor.extract(&[1.0; 2048]).unwrap();
        assert_eq!(chroma.len(), 1);
        assert_eq!(chroma[0].bins, [0.0; 12]);
    }

    #[test]
    fn normalized_bins() {
        // Non-silent signal: max bin should be 1.0 after normalization
        let samples: Vec<f32> = (0..16384)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / SR).sin())
            .collect();
        let extractor = ChromaExtractor::new(SR);
        let chroma = extractor.extract(&samples).unwrap();
        for cv in &chroma {
            let max = cv.bins.iter().copied().fold(0.0_f32, f32::max);
            if max > 0.0 {
                assert!(
                    (max - 1.0).abs() < 1e-6,
                    "max bin should be 1.0 after normalization, got {max}"
                );
            }
        }
    }

    #[test]
    fn dominant_name_matches_class() {
        let mut cv = ChromaVector { bins: [0.0; 12] };
        cv.bins[9] = 1.0; // A
        assert_eq!(cv.dominant_name(), "A");
        cv.bins[0] = 2.0; // C is now larger
        assert_eq!(cv.dominant_name(), "C");
    }

    #[test]
    fn builder_methods() {
        let e = ChromaExtractor::new(SR)
            .with_window_size(8192)
            .with_hop_size(4096)
            .with_tuning(442.0);
        assert_eq!(e.window_size, 8192);
        assert_eq!(e.hop_size, 4096);
        assert_eq!(e.tuning_hz, 442.0);
    }

    #[test]
    fn custom_tuning() {
        // A4 at 442 Hz tuning — A should still dominate
        let samples: Vec<f32> = (0..16384)
            .map(|i| (2.0 * PI * 442.0 * i as f32 / SR).sin())
            .collect();
        let extractor = ChromaExtractor::new(SR).with_tuning(442.0);
        let chroma = extractor.extract(&samples).unwrap();
        let a_dominant = chroma.iter().filter(|cv| cv.dominant_class() == 9).count();
        assert!(
            a_dominant > chroma.len() / 2,
            "442 Hz with 442 tuning should map to A"
        );
    }

    #[test]
    fn frame_count_matches_expected() {
        let n_samples = 16384;
        let window = 4096;
        let hop = 2048;
        let samples = vec![0.0_f32; n_samples];
        let extractor = ChromaExtractor::new(SR)
            .with_window_size(window)
            .with_hop_size(hop);
        let chroma = extractor.extract(&samples).unwrap();
        let expected_frames = (n_samples - window) / hop + 1;
        assert_eq!(
            chroma.len(),
            expected_frames,
            "expected {expected_frames} frames, got {}",
            chroma.len()
        );
    }

    #[test]
    fn no_nan_or_inf() {
        let samples: Vec<f32> = (0..8192)
            .map(|i| (i as f32 * 7.3).sin() * (i as f32 * 13.7).cos())
            .collect();
        let extractor = ChromaExtractor::new(SR);
        let chroma = extractor.extract(&samples).unwrap();
        for cv in &chroma {
            for &b in &cv.bins {
                assert!(b.is_finite(), "non-finite chroma bin: {b}");
            }
        }
    }
}
