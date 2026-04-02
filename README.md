# resonant

A layered Rust workspace for digital signal processing — from `no_std` embedded
primitives to one-liner audio analysis.

```rust
use resonant::AudioFile;

let audio = AudioFile::open("track.wav")?;
let bins = audio.with_window_size(4096).fft()?;

for bin in &bins[..5] {
    println!("{:.1} Hz: {:.2} ({:.1} dB)", bin.frequency_hz, bin.magnitude, bin.db());
}
```

## Crates

| Crate | crates.io | What it provides |
|---|---|---|
| [`resonant-core`](resonant-core/) | [![crates.io](https://img.shields.io/crates/v/resonant-core)](https://crates.io/crates/resonant-core) | `no_std` foundation: type-state signals, ring buffer, fixed-point, window functions |
| [`resonant-fft`](resonant-fft/) | [![crates.io](https://img.shields.io/crates/v/resonant-fft)](https://crates.io/crates/resonant-fft) | FFT, IFFT, STFT, DCT with compile-time domain tracking |
| [`resonant-filters`](resonant-filters/) | [![crates.io](https://img.shields.io/crates/v/resonant-filters)](https://crates.io/crates/resonant-filters) | Biquad, FIR, Butterworth design, decimation |
| [`resonant-stream`](resonant-stream/) | [![crates.io](https://img.shields.io/crates/v/resonant-stream)](https://crates.io/crates/resonant-stream) | Pull-based streaming pipeline with composable DSP nodes |
| [`resonant-analysis`](resonant-analysis/) | [![crates.io](https://img.shields.io/crates/v/resonant-analysis)](https://crates.io/crates/resonant-analysis) | Onset detection, tempo, pitch (YIN), MFCCs, chroma, spectral features |
| [`resonant`](resonant/) | [![crates.io](https://img.shields.io/crates/v/resonant)](https://crates.io/crates/resonant) | Ergonomic facade: `AudioFile::open` → decode → FFT → analyse |

It is possible depend on `resonant-core` alone for embedded targets without pulling in any higher-level dependencies.

## Why resonant?

| Task | Without resonant | With resonant |
|---|---|---|
| Load + FFT an audio file | ~25 lines (symphonia + rustfft boilerplate) | `AudioFile::open("f.wav")?.fft()?` |
| Prevent FFT of a freq-domain signal | Runtime check or convention | Compile error — `Signal<T, FreqDomain>` has no `.fft()` |
| FIR filter on embedded (no heap) | Write your own circular buffer | `Fir::new(coeffs)` + `process_sample()`, `no_std` |
| Stereo → mono → window → magnitude | Four separate crates, manual plumbing | One workspace, consistent `Signal` type throughout |
| Run on Cortex-M4 / WASM | Careful feature-flag archaeology | `default-features = false` on core/fft/filters |

## Feature highlights

### Compile-time signal domain tracking

The `Signal<T, D>` type carries its domain (`TimeDomain` or `FreqDomain`) as a
type parameter. Operations that are only valid in one domain simply don't exist
on the other:

```rust,ignore
let time_signal: Signal<f32, TimeDomain> = Signal::from_samples(samples);
let freq_signal = time_signal.fft()?;

// This is a compile error — no .fft() on a FreqDomain signal:
// let oops = freq_signal.fft();
```

### `no_std` and zero-allocation

`resonant-core`, `resonant-fft`, and `resonant-filters` are `#![no_std]` by
default. The radix-2 FFT, all window functions, biquad and FIR filters run on
stack-allocated buffers — no global allocator required.

Verified in CI on `thumbv7em-none-eabihf` (Cortex-M4) via QEMU semihosting and
on `wasm32-unknown-unknown`.

### SIMD acceleration

Hot paths use `core::arch` intrinsics (SSE2 on x86-64, NEON on AArch64) with an
automatic scalar fallback for all other targets (including WASM and embedded):

- FIR filter dot product
- Window function application
- FFT magnitude computation (`sqrt(re² + im²)`)
- Stereo-to-mono downmix

### Audio analysis

`resonant-analysis` provides onset detection, autocorrelation tempo estimation,
YIN pitch estimation, MFCCs, chroma features, and spectral centroid / spread /
flatness / rolloff — all validated against known-good test inputs.

## Quick start

```toml
# Application developer
[dependencies]
resonant = "0.1.0"

# Embedded / WASM — pick only what you need
[dependencies]
resonant-core    = { version = "0.1.0", default-features = false }
resonant-fft     = { version = "0.1.0", default-features = false }
resonant-filters = { version = "0.1.0", default-features = false }
```

See each crate's `README.md` for usage examples, or [`docs/EDUCATIONAL.md`](docs/EDUCATIONAL.md)
for concept explanations (windowing, FFT, filters, spectral analysis, SIMD, embedded targets).

## Semver stability

Starting with v0.1.0, resonant follows [Semantic Versioning](https://semver.org/).
A breaking change to any public, documented item in any crate will increment the
**major** version.

Items marked `#[doc(hidden)]` are explicitly excluded from this guarantee — they
are internal implementation details that may change in any release.

**MSRV:** Rust 1.75. Increases will be treated as breaking changes and require a
minor version bump.

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option.
