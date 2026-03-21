# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Pure-core radix-2 Cooley-Tukey FFT (`radix2::fft`, `radix2::ifft`)
- In-place, `no_std`, `no_alloc`, power-of-two sizes only
- `FftError` enum for non-power-of-two and empty input errors
- Re-export of `num_complex::Complex`
