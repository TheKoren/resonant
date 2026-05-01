# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-05-01

### Added
- `LufsAnalyser::integrated_loudness_multichannel()` — ITU-R BS.1770-4 loudness measurement
  for interleaved multi-channel audio; each channel is K-weighted independently; gating
  operates on the weighted sum of per-channel mean-square levels
- `ChannelConfig` — per-channel gain weight configuration with presets: `mono()`,
  `stereo()` (L=1.0, R=1.0), and `surround_5_1()` (LFE excluded, surround at √2)
- `time_stretch()` — time-stretching and pitch-shifting via STFT phase vocoder; stretch
  factor and pitch shift (semitones) are independent controls; pitch shifting is implemented
  as phase-vocoder stretch followed by polyphase resampling

### Fixed
- `TempoEstimator` now correctly reads the hop size from the stored `OnsetDetector` rather
  than returning a hardcoded 512; callers using
  `with_onset_detector(detector.with_hop_size(n))` previously received silently incorrect
  BPM values when `n ≠ 512`

## [0.3.0] - 2026-04-19

### Added
- `KeyDetector` and associated types (`KeyEstimate`, `PitchClass`, `Mode`) — Krumhansl-Schmuckler
  key estimation from chroma frames; re-exported from crate root
- `LoudnessAnalyser` — RMS, peak, dBFS, and crest factor for mono buffers; re-exported from crate root
- `LufsAnalyser` — ITU-R BS.1770-4 integrated, momentary, and short-term loudness with 4× true-peak;
  K-weighting coefficients baked in for 44100/48000 Hz; re-exported from crate root
- Tempo confidence constants: `TempoEstimator::CONFIDENCE_LOW` (0.3) and `CONFIDENCE_HIGH` (0.6)
- Opt-in `serde` feature: `PitchEstimate`, `Onset`, `TempoEstimate`, `MfccFrame`, `ChromaVector`,
  `KeyEstimate`, `PitchClass`, `Mode`, and `AnalysisError` (serialize-only) derive serde traits
  when `features = ["serde"]`

## [0.2.0] - 2026-04-11

### Changed
- Mel-scale helpers (`hz_to_mel`, `mel_to_hz`, `build_mel_filterbank`, `apply_mel_filterbank`, `log_mel_energy`) extracted to internal `mel` module; public API unchanged

## [0.1.0] - 2026-04-02

### Added
- `[package.metadata.docs.rs]` for full-feature docs.rs rendering

## [0.0.2] - 2026-03-30

### Added

- `AnalysisError` — shared error type (EmptyInput, InvalidParameter, Fft)
- Spectral features: `spectral_centroid`, `spectral_spread`, `spectral_flatness`, `spectral_rolloff`
- `YinEstimator` — YIN pitch estimation with configurable frequency range and threshold
- `OnsetDetector` — onset detection via spectral flux + adaptive thresholding
- `TempoEstimator` — tempo estimation via autocorrelation of onset strength envelope with octave correction
- `MfccExtractor` — mel-frequency cepstral coefficients with delta and delta-delta support
- `ChromaExtractor` — 12-bin pitch class profiles with configurable tuning
