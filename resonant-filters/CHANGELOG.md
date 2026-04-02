# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-04-02

### Added
- SIMD dot product for `Fir::process_buf` — SSE2 / NEON / scalar dispatch via linearised delay line
- Criterion benchmark harness for FIR processing (`benches/fir.rs`)
- `[package.metadata.docs.rs]` for full-feature docs.rs rendering

### Changed
- `BiquadState::s1` and `BiquadState::s2` fields marked `#[doc(hidden)]`
- `simd` module gated on `#[cfg(feature = "alloc")]` (only used by `Fir`, which requires alloc)

## [0.0.2] - 2026-03-23

### Added
- `Biquad` filter — direct form II transposed with accessible state
- `BiquadCoeffs` and `BiquadState` types
- `BiquadCoeffs::PASSTHROUGH` constant for unity gain
- `Biquad::process_sample()`, `process_buf()`, `state()`, `reset()`, `set_coeffs()`
- `Fir` filter with arbitrary coefficient vector (requires `alloc` feature)
- `Fir::process_sample()`, `process_buf()`, `coeffs()`, `delay()`, `reset()`
- `design::butterworth_lowpass()` — second-order Butterworth lowpass coefficients
- `design::butterworth_highpass()` — second-order Butterworth highpass coefficients
- `resample::decimate()` — integer decimation with cascaded Butterworth anti-alias filter (requires `alloc`)
