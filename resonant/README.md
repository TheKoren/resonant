# resonant

Ergonomic audio DSP for Rust — FFT, filters, analysis in one line of code.

```rust
use resonant::AudioFile;

let audio = AudioFile::open("track.wav")?;
let bins = audio.with_window_size(4096).fft()?;

for bin in &bins[..5] {
    println!("{:.1} Hz: {:.2} ({:.1} dB)", bin.frequency_hz, bin.magnitude, bin.db());
}
```

## Why resonant?

| Task | raw rustfft | resonant |
|------|-------------|----------|
| Load audio file | 15+ lines (symphonia boilerplate) | `AudioFile::open("f.wav")?` |
| FFT with window | 8 lines (alloc, convert, plan, execute) | `.with_window_size(4096).fft()?` |
| Labelled bins | Manual: `k * sr / N` per bin | Built-in `FrequencyBin { frequency_hz, magnitude, phase }` |
| Streaming STFT | Build your own frame loop | `.fft_stream()?` returns a lazy iterator |
| Domain safety | Nothing — you track it yourself | Compile-time `Signal<T, TimeDomain>` / `FreqDomain` |

## Quick start

```toml
[dependencies]
resonant = "0.0.1"
```

### One-shot FFT

```rust,ignore
use resonant::{AudioFile, FrequencyBin};

let audio = AudioFile::open("track.wav")?;
let bins: Vec<FrequencyBin> = audio
    .with_window_size(4096)
    .with_db_scale(true)
    .fft()?;
```

### Streaming STFT

```rust,ignore
use resonant::AudioFile;

let audio = AudioFile::open("track.wav")?;
for frame in audio.with_window_size(1024).with_overlap(0.75).fft_stream()? {
    let bins = frame?;
    // process each frame...
}
```

### Drop down to lower-level crates

```rust,ignore
use resonant::{Signal, SignalFftExt, SignalFreqExt};

let signal = Signal::from_samples(vec![0.0_f32; 1024]);
let spectrum = signal.fft()?;
let magnitudes = spectrum.magnitude();
```

```rust,ignore
use resonant::{Biquad, design};

let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
let mut filter = Biquad::new(coeffs);
for sample in audio_buf.iter_mut() {
    *sample = filter.process_sample(*sample);
}
```

## Feature matrix

| Feature | Crate | `no_std` | `no_alloc` |
|---------|-------|----------|------------|
| Type-state `Signal<T, D>` | resonant-core | yes | yes |
| Window functions | resonant-core | yes | yes |
| Q15/Q31 fixed-point | resonant-core | yes | yes |
| Radix-2 FFT | resonant-fft | yes | yes |
| Arbitrary-size FFT (rustfft) | resonant-fft | no | no |
| STFT (analysis + overlap-add) | resonant-fft | no | no |
| Biquad filter | resonant-filters | yes | yes |
| FIR filter | resonant-filters | yes | no |
| Butterworth design | resonant-filters | yes | yes |
| Integer decimation | resonant-filters | yes | no |
| Audio file decoding | **resonant** | no | no |
| One-liner FFT with labelled bins | **resonant** | no | no |

## Examples

Examples live in the workspace-level [`examples/`](../examples/) crate:

```bash
cargo run -p resonant-examples --bin spectrum_visualiser -- assets/test.wav
cargo run -p resonant-examples --bin beat_detector -- assets/test.wav
cargo run -p resonant-examples --bin spectrogram -- assets/test.wav
cargo run -p resonant-examples --bin window_comparison
```

See [`examples/README.md`](../examples/README.md) for details and sample output.

## Workspace crates

| Crate | Description |
|-------|-------------|
| [resonant-core](https://crates.io/crates/resonant-core) | `no_std`, zero-allocation DSP foundation |
| [resonant-fft](https://crates.io/crates/resonant-fft) | Type-safe FFT, STFT, DCT |
| [resonant-filters](https://crates.io/crates/resonant-filters) | FIR/IIR filters and design helpers |
| [resonant-stream](https://crates.io/crates/resonant-stream) | Async streaming DSP pipeline (planned) |
| [resonant-analysis](https://crates.io/crates/resonant-analysis) | Onset, beat, pitch, MFCC analysis (planned) |
| [resonant](https://crates.io/crates/resonant) | Ergonomic facade — this crate |

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
