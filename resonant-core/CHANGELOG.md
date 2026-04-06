# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `RingBuf<T, N>`: `peek`, `as_slices`, `iter`, `drain`, `is_full`, `capacity` methods;
  `RingBufIter` and `RingBufDrain` iterators with `ExactSizeIterator`
- `HeapRingBuf<T>` (requires `alloc` feature): heap-backed circular buffer with the same
  API surface as `RingBuf`, useful where capacity is not known at compile time
- `Signal` arithmetic operators (element-wise, domain-preserving):
  `Add`, `Sub`, `Mul<f32>`, `Div<f32>`, `Neg` for both `Signal<[f32; N], D>` and
  `Signal<Vec<f32>, D>` (the latter requires `alloc`); `Signal::mix(other, gain)` for
  weighted blending; mixing `TimeDomain + FreqDomain` is a compile error
- `Sample` trait: unified conversion interface (`to_f32`, `from_f32`, `to_f64`,
  `from_f64`, `zero`, `one`) with impls for `f32`, `f64`, `i16`, `i32`, `Q15`, `Q31`
- Window functions (`hann`, `hamming`, `blackman`, `rectangular`, `bartlett`, `apply`)
  are now generic over `S: Sample` — work on `f32`, `f64`, `Q15`, `Q31`, and integer
  buffers without a manual conversion step; existing `f32` callers compile unchanged

## [0.1.0] - 2026-04-02

### Added
- `Signal<T, D>` type-state struct with compile-time domain tracking
- `TimeDomain` and `FreqDomain` zero-sized domain markers
- `Domain` trait (open — downstream crates can implement custom domains)
- `Signal::map_domain()` for domain transitions
- `compile_fail` doc-test proving type safety
- `RingBuf<T, N>` const-generic, stack-allocated circular buffer
- `SlidingWindow<T>` heap-allocated overlapping frame extractor (requires `alloc` feature)
- Window functions: `hann`, `hamming`, `blackman`, `rectangular`, `bartlett`
- `Q15` signed 1.15 fixed-point type with saturating arithmetic and `From`/`Into` for `f32`/`f64`
- `Q31` signed 1.31 fixed-point type with higher precision and same API as `Q15`
- `window::apply()` — element-wise window application via SIMD (SSE2 / NEON / scalar)
- SIMD `multiply_buffers` dispatch module with x86-64 SSE2, AArch64 NEON, and scalar fallback
- Criterion benchmark harness for window functions (`benches/window.rs`)

### Changed
- `Signal::data_mut()` and `Signal::map_domain()` marked `#[doc(hidden)]` pending API stabilisation
