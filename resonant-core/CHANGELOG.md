# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
