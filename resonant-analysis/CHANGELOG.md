# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.0.2] - 2026-03-30

### Added

- `AnalysisError` — shared error type (EmptyInput, InvalidParameter, Fft)
- Spectral features: `spectral_centroid`, `spectral_spread`, `spectral_flatness`, `spectral_rolloff`
- `YinEstimator` — YIN pitch estimation with configurable frequency range and threshold
- `OnsetDetector` — onset detection via spectral flux + adaptive thresholding
- `TempoEstimator` — tempo estimation via autocorrelation of onset strength envelope with octave correction
- `MfccExtractor` — mel-frequency cepstral coefficients with delta and delta-delta support
- `ChromaExtractor` — 12-bin pitch class profiles with configurable tuning

## [Unreleased]
