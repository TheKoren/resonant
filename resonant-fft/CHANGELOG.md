# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-05-01

### Added
- `PhaseVocoder` — frame-by-frame phase vocoder for time-stretching and pitch-shifting;
  accumulates and propagates phase between overlapping STFT frames; configurable stretch
  factor with `hop_synthesis = hop_analysis × stretch_factor`; `reset()` clears the phase
  accumulator for seek operations

## [0.3.0] - 2026-04-19

### Added
- Opt-in `serde` feature: `FftError` derives `Serialize`/`Deserialize`
  when `features = ["serde"]`

## [0.2.0] - 2026-04-11

### Added
- `rfft` / `irfft` — real-valued FFT returning N/2+1 bins; `no_alloc` radix-2 path and `rustfft`-backed arbitrary-length path
- `SignalRfftExt` extension trait — `.rfft()` on `Signal<Vec<f32>, TimeDomain>`
- Criterion benchmark for rfft vs complex FFT (`benches/rfft.rs`, `benches/fft_comparison.rs`)

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
