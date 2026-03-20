# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
