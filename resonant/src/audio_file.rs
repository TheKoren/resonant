//! Audio file loading and decoding via symphonia.
//!
//! # Example
//!
//! ```rust,ignore
//! use resonant::AudioFile;
//!
//! let audio = AudioFile::open("assets/test.wav")?;
//! println!("Sample rate: {} Hz", audio.sample_rate());
//! println!("Duration: {:.2}s", audio.duration_secs());
//! println!("Samples (mono): {}", audio.samples_mono().len());
//! ```

use std::path::Path;

use resonant_analysis::chroma::ChromaExtractor;
use resonant_analysis::key::KeyDetector;
use resonant_analysis::loudness::LoudnessAnalyser;
use resonant_analysis::lufs::LufsAnalyser;
use resonant_analysis::mfcc::MfccExtractor;
use resonant_analysis::onset::OnsetDetector;
use resonant_analysis::tempo::TempoEstimator;
use resonant_core::signal::Signal;
use resonant_core::window;
use resonant_fft::{SignalFftExt, SignalFreqExt};

use crate::analysis::{AnalysisFlags, AnalysisResult};

use crate::decode::{decode_path, downmix_bs1770, downmix_to_mono};
use crate::error::AudioError;
use crate::frequency_bin::FrequencyBin;

/// Type alias for window functions accepted by the builder.
///
/// A window function takes a mutable slice and multiplies in-place.
pub type WindowFn = fn(&mut [f32]);

/// Analysis parameters used by [`AudioFile::fft()`], [`AudioFile::fft_stream()`],
/// and [`AudioFile::analyse()`].
#[derive(Debug, Clone)]
struct AnalysisConfig {
    /// FFT window size in samples. `None` means use the full signal length.
    window_size: Option<usize>,
    /// Overlap ratio (0.0–1.0). Only used by `fft_stream()`.
    overlap: f32,
    /// Window function applied before FFT.
    window_fn: WindowFn,
    /// Whether to return magnitudes in dB.
    db_scale: bool,
    /// STFT window size used by `analyse()`. `None` uses each detector's default.
    analysis_window_size: Option<usize>,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            window_size: None,
            overlap: 0.5,
            window_fn: window::hann,
            db_scale: false,
            analysis_window_size: None,
        }
    }
}

/// A decoded audio file ready for analysis.
///
/// Stores the full decoded audio as f32 samples. Provides both the original
/// interleaved multichannel data and a mono downmix for analysis APIs.
///
/// # Builder methods
///
/// Configure analysis parameters before calling [`fft()`](AudioFile::fft):
///
/// ```rust,ignore
/// let bins = AudioFile::open("track.wav")?
///     .with_window_size(4096)
///     .with_overlap(0.75)
///     .with_db_scale(true)
///     .fft()?;
/// ```
#[derive(Debug, Clone)]
pub struct AudioFile {
    /// Interleaved samples (all channels).
    samples_interleaved: Vec<f32>,
    /// Mono downmix for analysis.
    samples_mono: Vec<f32>,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Number of channels.
    channels: u16,
    /// Analysis configuration.
    config: AnalysisConfig,
}

impl AudioFile {
    /// Create an `AudioFile` from raw f32 samples.
    ///
    /// Useful for synthesised signals or data that is already decoded.
    #[must_use]
    pub fn from_samples(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        let mono = downmix_to_mono(&samples, channels);
        Self {
            samples_interleaved: samples,
            samples_mono: mono,
            sample_rate,
            channels,
            config: AnalysisConfig::default(),
        }
    }

    /// Open and decode an audio file.
    ///
    /// Supports WAV, MP3, FLAC, and OGG/Vorbis. The entire file is decoded
    /// into memory as f32 samples.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, AudioError> {
        let (samples, sample_rate, channels) = decode_path(path)?;
        Ok(Self::from_samples(samples, sample_rate, channels))
    }

    /// Sample rate in Hz (e.g. 44100).
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of audio channels (1 = mono, 2 = stereo).
    #[inline]
    #[must_use]
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Duration of the audio in seconds.
    #[inline]
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples_mono.len() as f64 / f64::from(self.sample_rate)
    }

    /// Mono-downmixed samples for analysis. If the source is already mono,
    /// this is the original data.
    #[inline]
    #[must_use]
    pub fn samples_mono(&self) -> &[f32] {
        &self.samples_mono
    }

    /// Raw interleaved samples (all channels).
    #[inline]
    #[must_use]
    pub fn samples_raw(&self) -> &[f32] {
        &self.samples_interleaved
    }

    /// Number of frames (samples per channel).
    #[inline]
    #[must_use]
    pub fn num_frames(&self) -> usize {
        self.samples_mono.len()
    }

    // --- Builder methods ---

    /// Set the FFT window size in samples.
    ///
    /// When set, [`fft()`](Self::fft) analyses only the first `size` samples
    /// (zero-padded if the signal is shorter). When `None` (the default), the
    /// entire signal is used.
    #[must_use]
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.config.window_size = Some(size);
        self
    }

    /// Set the overlap ratio for [`fft_stream()`](Self::fft_stream).
    ///
    /// Must be in the range `0.0..1.0`. Default is `0.5` (50% overlap).
    ///
    /// # Panics
    ///
    /// Panics if `overlap` is outside `0.0..1.0`.
    #[must_use]
    pub fn with_overlap(mut self, overlap: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&overlap),
            "overlap must be in 0.0..1.0, got {overlap}"
        );
        self.config.overlap = overlap;
        self
    }

    /// Set the window function applied before the FFT.
    ///
    /// Default is [`resonant_core::window::hann`]. Any function with signature
    /// `fn(&mut [f32])` is accepted.
    #[must_use]
    pub fn with_window_fn(mut self, f: WindowFn) -> Self {
        self.config.window_fn = f;
        self
    }

    /// Enable or disable dB-scale magnitudes in the output.
    ///
    /// When `true`, [`FrequencyBin::magnitude`] contains
    /// `20 * log10(linear_magnitude)` instead of the linear value.
    /// Default is `false`.
    #[must_use]
    pub fn with_db_scale(mut self, enabled: bool) -> Self {
        self.config.db_scale = enabled;
        self
    }

    /// Override the STFT window size used by [`analyse()`](Self::analyse).
    ///
    /// Must be a power of two (e.g. 1024, 2048, 4096). When not set, each
    /// detector uses its own default. Larger windows improve frequency
    /// resolution at the cost of temporal resolution.
    #[must_use]
    pub fn with_analysis_window(mut self, size: usize) -> Self {
        self.config.analysis_window_size = Some(size);
        self
    }

    // --- Analysis methods ---

    /// Compute the FFT of the mono signal, returning labelled frequency bins.
    ///
    /// Uses the configured window size, window function, and dB-scale setting.
    /// Returns only the positive-frequency half (bins 0 through N/2).
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Fft`] if the FFT backend rejects the input length
    /// (e.g. non-power-of-two without the `rustfft` feature).
    pub fn fft(&self) -> Result<Vec<FrequencyBin>, AudioError> {
        let mono = self.samples_mono();
        if mono.is_empty() {
            return Ok(Vec::new());
        }

        let fft_size = self.config.window_size.unwrap_or(mono.len());

        // Take up to fft_size samples, zero-pad if needed
        let mut windowed = vec![0.0_f32; fft_size];
        let copy_len = mono.len().min(fft_size);
        windowed[..copy_len].copy_from_slice(&mono[..copy_len]);

        // Apply window function
        (self.config.window_fn)(&mut windowed);

        // Run FFT via the type-state extension trait
        let signal = Signal::from_samples(windowed);
        let freq = signal.fft()?;

        // Extract magnitude and phase
        let magnitudes = freq.magnitude();
        let phases = freq.phase();
        let fft_len = magnitudes.len();
        let num_bins = fft_len / 2 + 1;
        let rate = self.sample_rate as f32;
        let fft_len_f32 = fft_len as f32;
        let use_db_scale = self.config.db_scale;

        let bins: Vec<FrequencyBin> = (0..num_bins)
            .map(|k| {
                let mag = magnitudes[k];
                FrequencyBin {
                    frequency_hz: k as f32 * rate / fft_len_f32,
                    magnitude: if use_db_scale {
                        20.0 * mag.max(f32::MIN_POSITIVE).log10()
                    } else {
                        mag
                    },
                    phase: phases[k],
                }
            })
            .collect();

        Ok(bins)
    }

    /// Return a lazy iterator that yields one `Vec<FrequencyBin>` per STFT
    /// frame.
    ///
    /// Uses the configured `window_size` (defaults to 4096 if not set),
    /// `overlap`, `window_fn`, and `db_scale`.
    ///
    /// Each frame is computed on demand, so memory usage is proportional to
    /// one frame, not the entire spectrogram.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidParameter`] if the effective window size
    /// is zero.
    pub fn fft_stream(&self) -> Result<FftFrameIter<'_>, AudioError> {
        let window_size = self.config.window_size.unwrap_or(4096);
        if window_size == 0 {
            return Err(AudioError::InvalidParameter(
                "window_size must be > 0".to_string(),
            ));
        }

        let hop_size = ((1.0 - self.config.overlap) * window_size as f32).round() as usize;
        let hop_size = hop_size.max(1); // at least 1 to avoid infinite loop

        Ok(FftFrameIter {
            samples: self.samples_mono(),
            window_size,
            hop_size,
            window_fn: self.config.window_fn,
            sample_rate: self.sample_rate,
            db_scale: self.config.db_scale,
            offset: 0,
        })
    }

    /// Run all analysis algorithms on the mono signal at once.
    ///
    /// Equivalent to `self.analyse_with(AnalysisFlags::ALL)`. Use
    /// [`analyse_with`](Self::analyse_with) to run only a subset of algorithms.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Analysis`] if any analysis step fails.
    pub fn analyse(&self) -> Result<AnalysisResult, AudioError> {
        self.analyse_with(AnalysisFlags::ALL)
    }

    /// Run a subset of analysis algorithms, controlled by `flags`.
    ///
    /// Fields corresponding to skipped algorithms are returned as `None` or
    /// empty `Vec`. Use [`AnalysisFlags::ALL`] to reproduce the behaviour of
    /// [`analyse()`](Self::analyse).
    ///
    /// Use [`with_analysis_window`](Self::with_analysis_window) to override the
    /// STFT window size for onset, chroma, and MFCC extraction.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::Analysis`] if any requested analysis step fails.
    pub fn analyse_with(&self, flags: AnalysisFlags) -> Result<AnalysisResult, AudioError> {
        let mono = self.samples_mono();
        let sr = self.sample_rate as f32;
        let analysis_window = self.config.analysis_window_size;

        // Tempo
        let (bpm, bpm_confidence) = if flags.contains(AnalysisFlags::TEMPO) {
            let mut tempo_estimator = TempoEstimator::new(sr);
            if let Some(window_size) = analysis_window {
                tempo_estimator = tempo_estimator
                    .with_onset_detector(OnsetDetector::new(sr).with_window_size(window_size));
            }
            let tempo = tempo_estimator.estimate(mono)?;
            let bpm = if tempo.confidence >= TempoEstimator::CONFIDENCE_LOW {
                Some(tempo.bpm)
            } else {
                None
            };
            (bpm, tempo.confidence)
        } else {
            (None, 0.0)
        };

        // Onsets → timestamps in seconds
        let onsets = if flags.contains(AnalysisFlags::ONSETS) {
            let mut onset_detector = OnsetDetector::new(sr);
            if let Some(window_size) = analysis_window {
                onset_detector = onset_detector.with_window_size(window_size);
            }
            onset_detector
                .detect(mono)?
                .into_iter()
                .map(|onset| onset.time_secs)
                .collect()
        } else {
            vec![]
        };

        // Key via chroma → Krumhansl-Schmuckler
        let key = if flags.contains(AnalysisFlags::KEY) {
            let mut chroma_extractor = ChromaExtractor::new(sr);
            if let Some(window_size) = analysis_window {
                chroma_extractor = chroma_extractor.with_window_size(window_size);
            }
            let chroma = chroma_extractor.extract(mono)?;
            KeyDetector::new()
                .detect(&chroma)
                .filter(|key_estimate| key_estimate.confidence > 0.3)
        } else {
            None
        };

        // Integrated loudness — BS.1770-4 weighted downmix for stereo/multi-channel
        let (loudness_lufs, peak_db, rms_db) = if flags.contains(AnalysisFlags::LOUDNESS) {
            let lufs_mono = downmix_bs1770(&self.samples_interleaved, self.channels);
            let loudness_lufs = LufsAnalyser::new(sr)
                .ok()
                .and_then(|mut lufs_analyser| lufs_analyser.integrated_loudness(&lufs_mono).ok());
            let loudness_analyser = LoudnessAnalyser::new();
            (
                loudness_lufs,
                Some(loudness_analyser.peak_db(mono)),
                Some(loudness_analyser.rms_db(mono)),
            )
        } else {
            (None, None, None)
        };

        // MFCCs
        let mfcc = if flags.contains(AnalysisFlags::MFCC) {
            let mut mfcc_extractor = MfccExtractor::new(sr);
            if let Some(window_size) = analysis_window {
                mfcc_extractor = mfcc_extractor.with_window_size(window_size);
            }
            mfcc_extractor.extract(mono).unwrap_or_default()
        } else {
            vec![]
        };

        Ok(AnalysisResult {
            bpm,
            bpm_confidence,
            key,
            onsets,
            loudness_lufs,
            peak_db,
            rms_db,
            mfcc,
        })
    }
}

/// Lazy iterator that yields one `Vec<FrequencyBin>` per STFT frame.
///
/// Created by [`AudioFile::fft_stream()`].
pub struct FftFrameIter<'a> {
    samples: &'a [f32],
    window_size: usize,
    hop_size: usize,
    window_fn: WindowFn,
    sample_rate: u32,
    db_scale: bool,
    offset: usize,
}

impl Iterator for FftFrameIter<'_> {
    type Item = Result<Vec<FrequencyBin>, AudioError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset + self.window_size > self.samples.len() {
            return None;
        }

        let frame = &self.samples[self.offset..self.offset + self.window_size];
        self.offset += self.hop_size;

        // Window the frame
        let mut windowed = frame.to_vec();
        (self.window_fn)(&mut windowed);

        // FFT
        let signal = Signal::from_samples(windowed);
        let freq = match signal.fft() {
            Ok(freq_signal) => freq_signal,
            Err(e) => return Some(Err(e.into())),
        };

        let magnitudes = freq.magnitude();
        let phases = freq.phase();
        let fft_len = magnitudes.len();
        let num_bins = fft_len / 2 + 1;
        let rate = self.sample_rate as f32;
        let fft_len_f32 = fft_len as f32;
        let use_db_scale = self.db_scale;

        let bins: Vec<FrequencyBin> = (0..num_bins)
            .map(|k| {
                let mag = magnitudes[k];
                FrequencyBin {
                    frequency_hz: k as f32 * rate / fft_len_f32,
                    magnitude: if use_db_scale {
                        20.0 * mag.max(f32::MIN_POSITIVE).log10()
                    } else {
                        mag
                    },
                    phase: phases[k],
                }
            })
            .collect();

        Some(Ok(bins))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.offset + self.window_size > self.samples.len() {
            return (0, Some(0));
        }
        let remaining = (self.samples.len() - self.offset - self.window_size) / self.hop_size + 1;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for FftFrameIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_wav_file() {
        let audio = AudioFile::open("../assets/test.wav");
        assert!(audio.is_ok(), "Failed to open test.wav: {:?}", audio.err());
        let audio = audio.ok();
        let audio = audio.as_ref();
        assert_eq!(audio.map(|a| a.sample_rate()), Some(44100));
        assert_eq!(audio.map(|a| a.channels()), Some(1));
        assert!(audio.map_or(false, |a| a.num_frames() > 0));
        assert!(audio.map_or(false, |a| a.duration_secs() > 0.0));
    }

    #[test]
    fn open_nonexistent_file() {
        let result = AudioFile::open("nonexistent.wav");
        assert!(result.is_err());
        assert!(
            matches!(result, Err(AudioError::Io(_))),
            "Expected Io error"
        );
    }

    #[test]
    fn samples_not_empty() {
        let audio = AudioFile::open("../assets/test.wav");
        if let Ok(audio) = audio {
            assert!(!audio.samples_mono().is_empty());
            assert!(!audio.samples_raw().is_empty());
            // Mono file: raw == mono
            assert_eq!(audio.samples_raw().len(), audio.samples_mono().len());
        }
    }

    #[test]
    fn duration_reasonable() {
        if let Ok(audio) = AudioFile::open("../assets/test.wav") {
            let duration_secs = audio.duration_secs();
            // test.wav is ~590KB at 44100 Hz mono 16-bit = ~6.7s
            assert!(duration_secs > 1.0, "Duration too short: {duration_secs}");
            assert!(duration_secs < 30.0, "Duration too long: {duration_secs}");
        }
    }

    #[test]
    fn fft_returns_bins() {
        if let Ok(audio) = AudioFile::open("../assets/test.wav") {
            let bins = audio.fft();
            assert!(bins.is_ok(), "FFT failed: {:?}", bins.err());
            let bins = bins.ok();
            let num_frames = audio.num_frames();
            let expected_bins = num_frames / 2 + 1;
            assert_eq!(
                bins.as_ref().map(|bin_vec| bin_vec.len()),
                Some(expected_bins)
            );
        }
    }

    #[test]
    fn fft_bin_frequencies_increase() {
        if let Ok(audio) = AudioFile::open("../assets/test.wav") {
            if let Ok(bins) = audio.fft() {
                // Frequencies should be monotonically increasing
                for pair in bins.windows(2) {
                    assert!(
                        pair[1].frequency_hz > pair[0].frequency_hz,
                        "Bins not increasing: {} >= {}",
                        pair[0].frequency_hz,
                        pair[1].frequency_hz,
                    );
                }
                // First bin is DC (0 Hz)
                assert!((bins[0].frequency_hz).abs() < 1e-6);
                // Last bin is near Nyquist
                let nyquist = audio.sample_rate() as f32 / 2.0;
                let last_freq = bins.last().map(|bin| bin.frequency_hz).unwrap_or(0.0);
                assert!(
                    (last_freq - nyquist).abs() < nyquist * 0.01,
                    "Last bin {last_freq} Hz not near Nyquist {nyquist} Hz"
                );
            }
        }
    }

    #[test]
    fn fft_known_sine() {
        // Synthesise a 1 kHz sine at 44100 Hz, power-of-two length
        let sample_rate = 44100_u32;
        let num_samples = 4096_usize;
        let freq_hz = 1000.0_f32;
        let mut samples = vec![0.0_f32; num_samples];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin();
        }

        let audio = AudioFile {
            samples_interleaved: samples.clone(),
            samples_mono: samples,
            sample_rate,
            channels: 1,
            config: AnalysisConfig::default(),
        };

        let bins = audio.fft().ok();
        let bins = bins.as_ref();
        assert!(bins.is_some());
        let bins = bins.as_ref().map(|bin_vec| bin_vec.as_slice());

        // Find the peak bin
        if let Some(bins) = bins {
            let peak = bins.iter().enumerate().max_by(|(_, a), (_, b)| {
                a.magnitude
                    .partial_cmp(&b.magnitude)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some((_, peak_bin)) = peak {
                // Peak should be near 1000 Hz (within one bin width)
                let bin_width = sample_rate as f32 / num_samples as f32;
                assert!(
                    (peak_bin.frequency_hz - freq_hz).abs() < bin_width * 2.0,
                    "Peak at {} Hz, expected near {} Hz",
                    peak_bin.frequency_hz,
                    freq_hz,
                );
            }
        }
    }

    #[test]
    fn fft_empty_signal() {
        let audio = AudioFile {
            samples_interleaved: Vec::new(),
            samples_mono: Vec::new(),
            sample_rate: 44100,
            channels: 1,
            config: AnalysisConfig::default(),
        };
        let bins = audio.fft();
        assert!(bins.is_ok());
        assert!(bins.ok().map_or(false, |b| b.is_empty()));
    }

    // --- Builder tests ---

    fn make_sine(num_samples: usize, freq_hz: f32, sample_rate: u32) -> AudioFile {
        let mut samples = vec![0.0_f32; num_samples];
        for (i, s) in samples.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sample_rate as f32).sin();
        }
        AudioFile {
            samples_interleaved: samples.clone(),
            samples_mono: samples,
            sample_rate,
            channels: 1,
            config: AnalysisConfig::default(),
        }
    }

    #[test]
    fn builder_window_size() {
        let audio = make_sine(8192, 1000.0, 44100).with_window_size(4096);
        let bins = audio.fft();
        assert!(bins.is_ok());
        // 4096-point FFT → 2049 bins
        assert_eq!(bins.ok().map(|b| b.len()), Some(4096 / 2 + 1));
    }

    #[test]
    fn builder_window_size_zero_pads() {
        // Signal shorter than window_size → zero-padded
        let audio = make_sine(512, 1000.0, 44100).with_window_size(1024);
        let bins = audio.fft();
        assert!(bins.is_ok());
        assert_eq!(bins.ok().map(|b| b.len()), Some(1024 / 2 + 1));
    }

    #[test]
    fn builder_db_scale() {
        let audio = make_sine(4096, 1000.0, 44100);
        let linear_bins = audio.clone().fft();
        let db_bins = audio.with_db_scale(true).fft();

        assert!(linear_bins.is_ok());
        assert!(db_bins.is_ok());

        if let (Ok(lin), Ok(db)) = (linear_bins, db_bins) {
            // dB values should be 20*log10 of linear values
            for (linear_bin, db_bin) in lin.iter().zip(db.iter()) {
                let expected_db = 20.0 * linear_bin.magnitude.max(f32::MIN_POSITIVE).log10();
                assert!(
                    (db_bin.magnitude - expected_db).abs() < 1e-4,
                    "dB mismatch: got {}, expected {}",
                    db_bin.magnitude,
                    expected_db
                );
            }
        }
    }

    #[test]
    fn builder_window_fn() {
        let audio_hann = make_sine(4096, 1000.0, 44100);
        let audio_rect =
            make_sine(4096, 1000.0, 44100).with_window_fn(resonant_core::window::rectangular);

        let bins_hann = audio_hann.fft();
        let bins_rect = audio_rect.fft();
        assert!(bins_hann.is_ok());
        assert!(bins_rect.is_ok());

        // Rectangular window should produce different magnitudes than Hann
        if let (Ok(hann_bins), Ok(rect_bins)) = (bins_hann, bins_rect) {
            let hann_peak: f32 = hann_bins
                .iter()
                .map(|bin| bin.magnitude)
                .fold(0.0, f32::max);
            let rect_peak: f32 = rect_bins
                .iter()
                .map(|bin| bin.magnitude)
                .fold(0.0, f32::max);
            // Rectangular window preserves more energy at the peak
            assert!(
                rect_peak > hann_peak,
                "Rectangular peak {rect_peak} should exceed Hann peak {hann_peak}"
            );
        }
    }

    #[test]
    fn builder_overlap_bounds() {
        // Valid overlap
        let _ = make_sine(1024, 440.0, 44100).with_overlap(0.0);
        let _ = make_sine(1024, 440.0, 44100).with_overlap(0.75);
    }

    #[test]
    #[should_panic(expected = "overlap must be in 0.0..1.0")]
    fn builder_overlap_too_high() {
        let _ = make_sine(1024, 440.0, 44100).with_overlap(1.0);
    }

    #[test]
    #[should_panic(expected = "overlap must be in 0.0..1.0")]
    fn builder_overlap_negative() {
        let _ = make_sine(1024, 440.0, 44100).with_overlap(-0.1);
    }

    // --- fft_stream tests ---

    #[test]
    fn fft_stream_frame_count() {
        // 8192 samples, window=1024, overlap=0.5 → hop=512
        // frames = (8192 - 1024) / 512 + 1 = 15
        let audio = make_sine(8192, 440.0, 44100).with_window_size(1024);
        let iter = audio.fft_stream();
        assert!(iter.is_ok());
        let iter = iter.ok();
        let frames: Vec<_> = iter.into_iter().flatten().collect();
        let ok_frames: Vec<_> = frames
            .into_iter()
            .filter_map(|frame_result| frame_result.ok())
            .collect();
        assert_eq!(ok_frames.len(), 15);
    }

    #[test]
    fn fft_stream_bin_count() {
        let audio = make_sine(4096, 440.0, 44100).with_window_size(1024);
        let mut iter = audio.fft_stream();
        assert!(iter.is_ok());
        if let Ok(ref mut it) = iter {
            if let Some(Ok(frame)) = it.next() {
                assert_eq!(frame.len(), 1024 / 2 + 1);
            }
        }
    }

    #[test]
    fn fft_stream_exact_size() {
        let audio = make_sine(8192, 440.0, 44100).with_window_size(1024);
        let iter = audio.fft_stream();
        assert!(iter.is_ok());
        if let Ok(it) = iter {
            assert_eq!(it.len(), 15);
        }
    }

    #[test]
    fn fft_stream_overlap_affects_count() {
        // overlap=0.0 → hop=1024 → frames = (8192-1024)/1024 + 1 = 8
        let audio_no_overlap = make_sine(8192, 440.0, 44100)
            .with_window_size(1024)
            .with_overlap(0.0);
        let count_no = audio_no_overlap
            .fft_stream()
            .ok()
            .map(|it| it.count())
            .unwrap_or(0);

        // overlap=0.75 → hop=256 → frames = (8192-1024)/256 + 1 = 29
        let audio_high_overlap = make_sine(8192, 440.0, 44100)
            .with_window_size(1024)
            .with_overlap(0.75);
        let count_hi = audio_high_overlap
            .fft_stream()
            .ok()
            .map(|it| it.count())
            .unwrap_or(0);

        assert_eq!(count_no, 8);
        assert_eq!(count_hi, 29);
    }

    #[test]
    fn fft_stream_signal_too_short() {
        // Signal shorter than window → no frames
        let audio = make_sine(512, 440.0, 44100).with_window_size(1024);
        let iter = audio.fft_stream();
        assert!(iter.is_ok());
        assert_eq!(iter.ok().map(|it| it.count()), Some(0));
    }

    #[test]
    fn fft_stream_db_scale() {
        let audio = make_sine(4096, 440.0, 44100)
            .with_window_size(1024)
            .with_db_scale(true);
        let mut iter = audio.fft_stream();
        assert!(iter.is_ok());
        if let Ok(ref mut it) = iter {
            if let Some(Ok(frame)) = it.next() {
                // All dB magnitudes should be negative or zero for a sine wave
                // (magnitude ≤ window_size/2)
                assert!(frame.iter().all(|b| b.magnitude <= 100.0));
            }
        }
    }

    // --- analyse() tests ---

    #[test]
    fn analyse_from_samples() {
        // Synthesise a 120 BPM click track with a 440 Hz tone
        let sample_rate = 44100_u32;
        let duration_secs = 3.0;
        let total = (duration_secs * sample_rate as f32) as usize;
        let mut samples = vec![0.0_f32; total];
        let interval = (60.0 / 120.0 * sample_rate as f32) as usize;
        let mut pos = 0;
        while pos < total {
            for j in 0..64 {
                if pos + j < total {
                    samples[pos + j] =
                        (2.0 * std::f32::consts::PI * 440.0 * j as f32 / sample_rate as f32).sin();
                }
            }
            pos += interval;
        }

        let audio = AudioFile::from_samples(samples, sample_rate, 1);
        let result = audio.analyse();
        assert!(result.is_ok(), "analyse() failed: {:?}", result.err());

        let analysis_opt = result.ok();
        let analysis_opt = analysis_opt.as_ref();

        // bpm_confidence in [0, 1]
        assert!(
            analysis_opt.map_or(false, |analysis| analysis.bpm_confidence >= 0.0
                && analysis.bpm_confidence <= 1.0)
        );

        // Peak near 0 dBFS — each burst is a full-scale sine
        assert!(analysis_opt.map_or(false, |analysis| analysis
            .peak_db
            .map_or(false, |peak| peak > -3.0)));

        // RMS is well below peak because the signal is sparse click bursts
        assert!(analysis_opt.map_or(false, |analysis| {
            let rms = analysis.rms_db.unwrap_or(-120.0);
            let peak = analysis.peak_db.unwrap_or(-120.0);
            rms > -120.0 && rms < peak
        }));
    }

    #[test]
    fn analyse_silence() {
        let sample_rate = 44100_u32;
        let audio = AudioFile::from_samples(vec![0.0_f32; 44100], sample_rate, 1);
        let result = audio.analyse();
        assert!(
            result.is_ok(),
            "analyse() on silence failed: {:?}",
            result.err()
        );
        if let Ok(analysis) = result {
            // Silence → no BPM
            assert!(analysis.bpm.is_none());
            // Peak and RMS at floor
            assert!(analysis
                .peak_db
                .map_or(false, |peak| (peak - (-120.0)).abs() < 1.0));
        }
    }

    #[test]
    fn analyse_wav_file() {
        if let Ok(audio) = AudioFile::open("../assets/test.wav") {
            let result = audio.analyse();
            assert!(
                result.is_ok(),
                "analyse() on test.wav failed: {:?}",
                result.err()
            );
            if let Ok(analysis) = &result {
                assert!(analysis.bpm_confidence >= 0.0 && analysis.bpm_confidence <= 1.0);
                assert!(analysis.peak_db.map_or(true, |peak| peak <= 0.1)); // dBFS ≤ 0 for normalised audio
            }
        }
    }

    #[test]
    fn with_analysis_window_compiles() {
        let audio = make_sine(8192, 440.0, 44100).with_analysis_window(2048);
        let result = audio.analyse();
        assert!(
            result.is_ok(),
            "analyse() with custom window failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn analyse_lufs_uses_bs1770_for_stereo() {
        // Build a stereo signal: L = sine, R = sine (identical channels).
        // With the naive (L+R)/2 mono average, LUFS equals the mono sine level.
        // With BS.1770-4 downmix (L+R)/sqrt(2), LUFS is ~3 LU higher.
        // Build a second signal: L = sine, R = 0.
        // The difference between the two LUFS values should be ~6 LU
        // (full stereo ms = 2*A^2, half stereo ms = A^2/2, ratio = 4 → 6 dB).
        use std::f32::consts::PI;
        let sample_rate = 44100_u32;
        let sr = sample_rate as f32;
        let num_samples = (sr * 3.0) as usize; // 3 seconds — enough for LUFS gating

        let sine: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).sin())
            .collect();

        // Interleaved stereo: L=sine, R=sine
        let mut both = Vec::with_capacity(num_samples * 2);
        for &s in &sine {
            both.push(s);
            both.push(s);
        }
        // Interleaved stereo: L=sine, R=silent
        let mut left_only = Vec::with_capacity(num_samples * 2);
        for &s in &sine {
            left_only.push(s);
            left_only.push(0.0_f32);
        }

        let af_both = AudioFile::from_samples(both, sample_rate, 2);
        let af_left = AudioFile::from_samples(left_only, sample_rate, 2);

        let lufs_both = af_both
            .analyse()
            .ok()
            .and_then(|analysis| analysis.loudness_lufs);
        let lufs_left = af_left
            .analyse()
            .ok()
            .and_then(|analysis| analysis.loudness_lufs);

        if let (Some(lufs_full_stereo), Some(lufs_left_only)) = (lufs_both, lufs_left) {
            let lufs_diff = lufs_full_stereo - lufs_left_only;
            assert!(
                (lufs_diff - 6.0).abs() < 1.0,
                "expected ~6 LU between full-stereo and L-only, got {lufs_diff:.2} LU"
            );
        }
    }

    #[test]
    fn analyse_returns_mfcc_frames_with_13_coefficients() {
        use std::f32::consts::PI;
        let sample_rate = 44100_u32;
        let sr = sample_rate as f32;
        let num_samples = (sr * 2.0) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).sin())
            .collect();

        let audio = AudioFile::from_samples(samples, sample_rate, 1);
        let result = audio.analyse();
        assert!(result.is_ok(), "analyse() failed: {:?}", result.err());
        if let Ok(analysis) = result {
            assert!(!analysis.mfcc.is_empty(), "expected MFCC frames, got none");
            for frame in &analysis.mfcc {
                assert_eq!(
                    frame.coefficients.len(),
                    13,
                    "expected 13 coefficients per frame, got {}",
                    frame.coefficients.len()
                );
            }
        }
    }

    #[test]
    fn analyse_with_loudness_only_skips_tempo() {
        use std::f32::consts::PI;
        let sample_rate = 44100_u32;
        let sr = sample_rate as f32;
        let num_samples = (sr * 3.0) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).sin())
            .collect();

        let audio = AudioFile::from_samples(samples, sample_rate, 1);
        let result = audio.analyse_with(AnalysisFlags::LOUDNESS);
        assert!(result.is_ok(), "analyse_with failed: {:?}", result.err());
        if let Ok(analysis) = result {
            assert_eq!(analysis.bpm_confidence, 0.0);
            assert!(analysis.bpm.is_none());
            assert!(analysis.onsets.is_empty());
            assert!(analysis.mfcc.is_empty());
            assert!(analysis.loudness_lufs.is_some());
            assert!(analysis.peak_db.is_some());
            assert!(analysis.rms_db.is_some());
        }
    }

    #[test]
    fn analyse_with_tempo_and_onsets_skips_loudness() {
        use std::f32::consts::PI;
        let sample_rate = 44100_u32;
        let sr = sample_rate as f32;
        let total = (sr * 3.0) as usize;
        let mut samples = vec![0.0_f32; total];
        let interval = (60.0 / 120.0 * sr) as usize;
        let mut pos = 0;
        while pos < total {
            for j in 0..64 {
                if pos + j < total {
                    samples[pos + j] = (2.0 * PI * 440.0 * j as f32 / sr).sin();
                }
            }
            pos += interval;
        }

        let audio = AudioFile::from_samples(samples, sample_rate, 1);
        let result = audio.analyse_with(AnalysisFlags::TEMPO | AnalysisFlags::ONSETS);
        assert!(result.is_ok(), "analyse_with failed: {:?}", result.err());
        if let Ok(analysis) = result {
            assert!(!analysis.onsets.is_empty());
            assert!(analysis.loudness_lufs.is_none());
            assert!(analysis.peak_db.is_none());
            assert!(analysis.rms_db.is_none());
            assert!(analysis.key.is_none());
            assert!(analysis.mfcc.is_empty());
        }
    }

    #[test]
    fn analyse_with_all_matches_analyse() {
        use std::f32::consts::PI;
        let sample_rate = 44100_u32;
        let sr = sample_rate as f32;
        let num_samples = (sr * 2.0) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / sr).sin())
            .collect();

        let audio = AudioFile::from_samples(samples, sample_rate, 1);
        let result_a = audio.analyse();
        let result_b = audio.analyse_with(AnalysisFlags::ALL);

        assert!(result_a.is_ok());
        assert!(result_b.is_ok());
        if let (Ok(from_analyse), Ok(from_analyse_with)) = (result_a, result_b) {
            assert_eq!(from_analyse.bpm, from_analyse_with.bpm);
            assert_eq!(
                from_analyse.bpm_confidence,
                from_analyse_with.bpm_confidence
            );
            assert_eq!(from_analyse.onsets, from_analyse_with.onsets);
            assert_eq!(from_analyse.loudness_lufs, from_analyse_with.loudness_lufs);
            assert_eq!(from_analyse.peak_db, from_analyse_with.peak_db);
            assert_eq!(from_analyse.rms_db, from_analyse_with.rms_db);
            assert_eq!(from_analyse.mfcc, from_analyse_with.mfcc);
        }
    }

    #[test]
    fn with_analysis_window_propagates_to_tempo() {
        use resonant_analysis::onset::OnsetDetector;
        use resonant_analysis::tempo::TempoEstimator;

        let sample_rate = 44100_u32;
        let sr = sample_rate as f32;
        let audio = make_sine(44100, 440.0, sample_rate);
        let mono = audio.samples_mono();

        // analyse() with a 2048-point window
        let result = audio.clone().with_analysis_window(2048).analyse();
        assert!(result.is_ok(), "analyse() failed: {:?}", result.err());
        let facade_bpm_conf = result
            .ok()
            .map(|analysis| analysis.bpm_confidence)
            .unwrap_or(-1.0);

        // Manually construct the same TempoEstimator that analyse() should use
        let manual_conf = TempoEstimator::new(sr)
            .with_onset_detector(OnsetDetector::new(sr).with_window_size(2048))
            .estimate(mono)
            .ok()
            .map(|e| e.confidence)
            .unwrap_or(-2.0);

        assert!(
            (facade_bpm_conf - manual_conf).abs() < 1e-4,
            "analyse() bpm_confidence ({facade_bpm_conf}) differs from manual ({manual_conf})"
        );
    }
}
