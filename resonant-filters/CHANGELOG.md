# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `Biquad` filter — direct form II transposed with accessible state
- `BiquadCoeffs` and `BiquadState` types
- `BiquadCoeffs::PASSTHROUGH` constant for unity gain
- `Biquad::process_sample()`, `process_buf()`, `state()`, `reset()`, `set_coeffs()`
- `Fir` filter with arbitrary coefficient vector (requires `alloc` feature)
- `Fir::process_sample()`, `process_buf()`, `coeffs()`, `delay()`, `reset()`
