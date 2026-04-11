# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-04-11

### Added
- Graph composition operators: `>>` (serial), `&` (parallel sum), `|` (stack) on any `DspNode`
- `Serial<A, B>`, `Parallel<A, B>`, `Stack<A, B>` concrete types implementing `DspNode`
- `Pipeline::from_graph()` — accepts any composed graph as a pipeline source

## [0.1.0] - 2026-04-02

### Changed
- `Chunk::set_data()` marked `#[doc(hidden)]`
- `[package.metadata.docs.rs]` added for full-feature docs.rs rendering

## [0.0.2] - 2026-03-27

### Added

- `Chunk` — owned buffer of interleaved f32 samples with sample rate and channel metadata
- `StreamError` — error type covering format mismatches and processing failures
- `DspNode` trait — `process(Chunk) -> Result<Chunk, StreamError>` + `reset()`
- `Pipeline` builder — sequential node chain with optional sample rate / channel validation
- `GainNode` — linear amplitude scaling with runtime-adjustable gain
- `FilterNode` — biquad IIR filter node wrapping `resonant_filters::Biquad`
- `TapNode` — side-channel observation via callback, does not modify audio
- `FftNode` — radix-2 forward FFT, outputs magnitude or power spectrum
- `MixNode` — multi-channel to mono downmix by averaging
- `ResampleNode` — integer decimation with 4th-order Butterworth anti-alias filter
