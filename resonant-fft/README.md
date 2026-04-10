# resonant-fft

Type-safe FFT, STFT, and DCT transforms with compile-time domain tracking.

Part of the [resonant](https://crates.io/crates/resonant) workspace.

## Quick start

```rust
use resonant_core::signal::Signal;
use resonant_fft::{SignalFftExt, SignalIfftExt, SignalFreqExt};

let signal = Signal::from_samples(vec![1.0_f32, 0.0, -1.0, 0.0]);
let spectrum = signal.fft().unwrap();
let magnitudes = spectrum.magnitude();
let reconstructed = spectrum.ifft().unwrap();
```

## What's included

- **Radix-2 FFT** — pure-core, `no_std`, `no_alloc`, in-place Cooley-Tukey (power-of-two sizes)
- **Real-valued FFT** — `rfft`/`irfft` returning N/2+1 bins; roughly 40% faster than complex FFT for real inputs
- **rustfft backend** — arbitrary-size FFTs (default feature, requires `std`)
- **Type-state extension traits** — `.fft()` and `.ifft()` on `Signal<T, D>` enforce domain transitions at compile time
- **Spectral helpers** — `.magnitude()`, `.phase()`, `.magnitude_squared()` on frequency-domain signals
- **STFT** — builder-pattern Short-Time Fourier Transform with configurable window, hop size, and overlap-add synthesis
- **DCT** — types II and III for compression and spectral analysis workflows

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `rustfft` | yes | Arbitrary-size FFTs via `rustfft`; implies `alloc` |
| `alloc` | via rustfft | Enables extension traits, `SignalRfftExt`, and STFT |

Without `rustfft`, only the radix-2 and `rfft` backends are available (power-of-two sizes, no heap allocation).

## License

MIT OR Apache-2.0
