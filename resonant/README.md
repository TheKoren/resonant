# resonant

Ergonomic audio DSP for Rust — FFT, filters, analysis in one line of code.

```rust
let result = AudioFile::open("track.mp3")?.fft()?;
```

## Workspace crates

| Crate | Description |
|-------|-------------|
| [resonant-core](https://crates.io/crates/resonant-core) | `no_std`, zero-allocation DSP foundation |
| [resonant-fft](https://crates.io/crates/resonant-fft) | Type-safe FFT, STFT, DCT |
| [resonant-filters](https://crates.io/crates/resonant-filters) | FIR/IIR filters and design helpers |
| [resonant-stream](https://crates.io/crates/resonant-stream) | Async streaming DSP pipeline |
| [resonant-analysis](https://crates.io/crates/resonant-analysis) | Onset, beat, pitch, MFCC analysis |
| [resonant](https://crates.io/crates/resonant) | Ergonomic facade for application developers |

## License

This project is licensed under [MIT](LICENSE) OR [Apache-2.0](LICENSE).
