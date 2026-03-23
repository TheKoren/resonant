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

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use resonant_core::signal::Signal;
use resonant_core::window;
use resonant_fft::{SignalFftExt, SignalFreqExt};

use crate::error::AudioError;
use crate::frequency_bin::FrequencyBin;

/// Type alias for window functions accepted by the builder.
///
/// A window function takes a mutable slice and multiplies in-place.
pub type WindowFn = fn(&mut [f32]);

/// Analysis parameters used by [`AudioFile::fft()`] and
/// [`AudioFile::fft_stream()`].
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
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            window_size: None,
            overlap: 0.5,
            window_fn: window::hann,
            db_scale: false,
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
    /// Open and decode an audio file.
    ///
    /// Supports WAV, MP3, FLAC, and OGG/Vorbis. The entire file is decoded
    /// into memory as f32 samples.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, AudioError> {
        let path = path.as_ref();
        let file = std::fs::File::open(path)?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }

        let probed = symphonia::default::get_probe().format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )?;

        let mut format = probed.format;

        let track = format.default_track().ok_or(AudioError::NoTrack)?;
        let track_id = track.id;
        let codec_params = track.codec_params.clone();

        let sample_rate = codec_params
            .sample_rate
            .ok_or_else(|| AudioError::Decode("missing sample rate".to_string()))?;

        let channels = codec_params
            .channels
            .map(|ch| ch.count() as u16)
            .unwrap_or(1);

        let mut decoder =
            symphonia::default::get_codecs().make(&codec_params, &DecoderOptions::default())?;

        let mut all_samples: Vec<f32> = Vec::new();
        let mut sample_buf: Option<SampleBuffer<f32>> = None;

        loop {
            let packet = match format.next_packet() {
                Ok(p) => p,
                Err(symphonia::core::errors::Error::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break; // end of stream
                }
                Err(e) => return Err(e.into()),
            };

            if packet.track_id() != track_id {
                continue;
            }

            let audio_buf = match decoder.decode(&packet) {
                Ok(buf) => buf,
                Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
                Err(e) => return Err(e.into()),
            };

            if sample_buf.is_none() {
                let spec = *audio_buf.spec();
                let duration = audio_buf.capacity() as u64;
                sample_buf = Some(SampleBuffer::new(duration, spec));
            }

            if let Some(buf) = &mut sample_buf {
                buf.copy_interleaved_ref(audio_buf);
                all_samples.extend_from_slice(buf.samples());
            }
        }

        let mono = downmix_to_mono(&all_samples, channels);

        Ok(Self {
            samples_interleaved: all_samples,
            samples_mono: mono,
            sample_rate,
            channels,
            config: AnalysisConfig::default(),
        })
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
        let n = magnitudes.len();
        let num_bins = n / 2 + 1;
        let rate = self.sample_rate as f32;
        let n_f = n as f32;
        let db = self.config.db_scale;

        let bins: Vec<FrequencyBin> = (0..num_bins)
            .map(|k| {
                let mag = magnitudes[k];
                FrequencyBin {
                    frequency_hz: k as f32 * rate / n_f,
                    magnitude: if db {
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
}

/// Average all channels down to mono.
fn downmix_to_mono(interleaved: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let ch = channels as usize;
    let num_frames = interleaved.len() / ch;
    let scale = 1.0 / channels as f32;
    let mut mono = Vec::with_capacity(num_frames);
    for frame in 0..num_frames {
        let start = frame * ch;
        let mut sum = 0.0_f32;
        for c in 0..ch {
            sum += interleaved[start + c];
        }
        mono.push(sum * scale);
    }
    mono
}

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
    fn mono_downmix_passthrough() {
        let samples = vec![0.1, 0.2, 0.3, 0.4];
        let mono = downmix_to_mono(&samples, 1);
        assert_eq!(mono, samples);
    }

    #[test]
    fn mono_downmix_stereo() {
        // Stereo: L=1.0 R=0.0, L=0.0 R=1.0
        let interleaved = vec![1.0, 0.0, 0.0, 1.0];
        let mono = downmix_to_mono(&interleaved, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < 1e-6);
        assert!((mono[1] - 0.5).abs() < 1e-6);
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
            let dur = audio.duration_secs();
            // test.wav is ~590KB at 44100 Hz mono 16-bit = ~6.7s
            assert!(dur > 1.0, "Duration too short: {dur}");
            assert!(dur < 30.0, "Duration too long: {dur}");
        }
    }

    #[test]
    fn fft_returns_bins() {
        if let Ok(audio) = AudioFile::open("../assets/test.wav") {
            let bins = audio.fft();
            assert!(bins.is_ok(), "FFT failed: {:?}", bins.err());
            let bins = bins.ok();
            let n = audio.num_frames();
            let expected_bins = n / 2 + 1;
            assert_eq!(bins.as_ref().map(|b| b.len()), Some(expected_bins));
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
                let last_freq = bins.last().map(|b| b.frequency_hz).unwrap_or(0.0);
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
        let n = 4096_usize;
        let freq_hz = 1000.0_f32;
        let mut samples = vec![0.0_f32; n];
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
        let bins = bins.as_ref().map(|b| b.as_slice());

        // Find the peak bin
        if let Some(bins) = bins {
            let peak = bins.iter().enumerate().max_by(|(_, a), (_, b)| {
                a.magnitude
                    .partial_cmp(&b.magnitude)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some((_, peak_bin)) = peak {
                // Peak should be near 1000 Hz (within one bin width)
                let bin_width = sample_rate as f32 / n as f32;
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

    fn make_sine(n: usize, freq_hz: f32, sample_rate: u32) -> AudioFile {
        let mut samples = vec![0.0_f32; n];
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
            for (l, d) in lin.iter().zip(db.iter()) {
                let expected_db = 20.0 * l.magnitude.max(f32::MIN_POSITIVE).log10();
                assert!(
                    (d.magnitude - expected_db).abs() < 1e-4,
                    "dB mismatch: got {}, expected {}",
                    d.magnitude,
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
        if let (Ok(h), Ok(r)) = (bins_hann, bins_rect) {
            let hann_peak: f32 = h.iter().map(|b| b.magnitude).fold(0.0, f32::max);
            let rect_peak: f32 = r.iter().map(|b| b.magnitude).fold(0.0, f32::max);
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
}
