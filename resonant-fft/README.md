# resonant-fft

Type-safe FFT, STFT, and DCT transforms with compile-time domain tracking.

Part of the [resonant](https://crates.io/crates/resonant) workspace.

## Features

- **Radix-2 FFT** — pure-core, `no_std`, `no_alloc`, in-place Cooley-Tukey (power-of-two sizes)
- **rustfft backend** — arbitrary-size FFTs via the `rustfft` crate (default feature, requires `std`)
- **Type-state extension traits** — `.fft()` and `.ifft()` on `Signal<T, D>` enforce domain transitions at compile time
- **Spectral helpers** — `.magnitude()`, `.phase()`, `.magnitude_squared()` on frequency-domain signals
- **STFT** — builder-pattern Short-Time Fourier Transform with configurable window, hop size, and overlap-add synthesis

## Usage

```rust,ignore
use resonant_core::Signal;
use resonant_fft::ext::{SignalFftExt, SignalIfftExt, SignalFreqExt};

let signal = Signal::from_samples(samples);
let spectrum = signal.fft().unwrap();

let magnitudes = spectrum.magnitude();
let reconstructed = spectrum.ifft().unwrap();
```

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `rustfft` | yes | Enables arbitrary-size FFTs via the `rustfft` crate |

Without `rustfft`, only the radix-2 backend is available (power-of-two sizes, no heap allocation).

## Roadmap

- DCT (Discrete Cosine Transform) for audio compression use cases
- Streaming FFT for real-time frame-by-frame processing
- WASM target verification
