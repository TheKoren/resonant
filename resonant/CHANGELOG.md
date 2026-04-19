# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-04-19

### Added
- `AnalysisResult` — high-level analysis summary: `bpm`, `bpm_confidence`, `key`, `onsets`,
  `loudness_lufs`, `peak_db`, `rms_db`
- `AudioFile::analyse()` — single-pass: tempo, onset timestamps, key (Krumhansl-Schmuckler),
  integrated LUFS loudness, peak and RMS level
- `AudioFile::with_analysis_window()` — override STFT window size for `analyse()`
- Re-exports: `KeyEstimate`, `PitchClass`, `Mode` from `resonant-analysis`
- Opt-in `serde` feature: `AnalysisResult` and `FrequencyBin` derive `Serialize`/`Deserialize`;
  `AudioError` derives `Serialize` (no round-trip — `Io` variant serialises as display string)
- All inter-crate dependencies bumped to 0.3.0

## [0.2.0] - 2026-04-11

### Added
- Re-exports for `PolyphaseResampler`, `MoogLadder`, `SaturatingBiquad`, `StateVariableFilter`, `SvfOutputs`, `Oversample`, `FilterResponseExt`, `FrequencyResponse`, `FilterError`
- End-to-end Criterion benchmark (`benches/facade.rs`)

### Changed
- Audio decoding path extracted to internal `decode` module; `AudioFile::open()` API unchanged
- All inter-crate dependencies bumped to 0.2.0

## [0.1.0] - 2026-04-02

### Added
- SIMD stereo-to-mono downmix (SSE2 / NEON / scalar) in `AudioFile` decoding path
- `[package.metadata.docs.rs]` for full-feature docs.rs rendering

### Changed
- All inter-crate dependencies bumped to 0.1.0

## [0.0.2] - 2026-03-23

### Added
- `AudioFile::open()` — decode WAV, MP3, FLAC, OGG/Vorbis via symphonia
- `AudioFile::samples_mono()` — mono downmix for analysis
- `AudioFile::samples_raw()` — raw interleaved multichannel access
- `AudioFile::sample_rate()`, `channels()`, `duration_secs()`, `num_frames()`
- `AudioError` enum with `Io`, `UnsupportedFormat`, `Decode`, `NoTrack`, `InvalidParameter`, `Fft` variants
- `FrequencyBin` struct with `frequency_hz`, `magnitude`, `phase`, and `db()` method
- `AudioFile::fft()` — full-signal FFT with Hann window and labelled frequency bins
- Builder methods: `with_window_size()`, `with_overlap()`, `with_window_fn()`, `with_db_scale()`
- `WindowFn` type alias for window function pointers
- `AudioFile::fft_stream()` — lazy STFT frame iterator yielding `Vec<FrequencyBin>` per frame
- `FftFrameIter` implementing `Iterator` and `ExactSizeIterator`
- Re-exports: `Signal`, `TimeDomain`, `FreqDomain`, `window`, `SignalFftExt`, `SignalIfftExt`, `SignalFreqExt`, `Complex`, `Biquad`, `BiquadCoeffs`, `design`
- `AudioFile::from_samples()` — create from raw f32 samples without decoding a file
- `spectrum_visualiser` example — top-N frequency peaks from an audio file
- `beat_detector` example — spectral flux onset detection stub
- `spectrogram` example — time×frequency heatmap PNG via plotters
- `window_comparison` example — overlay plot of five window functions showing leakage trade-offs
