# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-04-02

### Added
- `FftError::LengthMismatch { input, output }` — dedicated variant replacing misused `NotPowerOfTwo` for DCT length checks
- `alloc` feature gate — separates `no_alloc` core (radix-2, DCT, `FftError`) from Vec-returning APIs
- SIMD `magnitude()` and `magnitude_squared()` via interleaved `[re, im, ...]` f32 processing (SSE2 / NEON / scalar fallback)
- Criterion benchmark harness for magnitude computation (`benches/magnitude.rs`)
- `[package.metadata.docs.rs]` for full-feature docs.rs rendering

### Changed
- `ext`, `simd`, and `stft` modules gated on `#[cfg(feature = "alloc")]`

## [0.0.2] - 2026-03-23

### Added
- Pure-core radix-2 Cooley-Tukey FFT (`radix2::fft`, `radix2::ifft`)
- In-place, `no_std`, `no_alloc`, power-of-two sizes only
- `FftError` enum for non-power-of-two and empty input errors
- Re-export of `num_complex::Complex`
- `rustfft` backend for arbitrary-size FFTs (default feature, requires `std`)
- `FftPlan` struct for reusable FFT plans via `rustfft`
- `SignalFftExt` trait — `.fft()` on `Signal<T, TimeDomain>` with compile-time domain transition
- `SignalIfftExt` trait — `.ifft()` on `Signal<Vec<Complex<f32>>, FreqDomain>`
- `SignalFreqExt` trait — `.magnitude()`, `.phase()`, `.magnitude_squared()` on frequency-domain signals
- `compile_fail` doc-tests proving domain misuse is a compile error
- `Stft` with builder pattern for configurable STFT analysis and overlap-add synthesis
- Window function integration via `StftBuilder::window_fn()`
