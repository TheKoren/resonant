# resonant-filters

FIR/IIR filters, design helpers, and sample-rate conversion for audio DSP.

Part of the [resonant](https://crates.io/crates/resonant) workspace.

## Biquad filter — `no_std`, `no_alloc`

Second-order IIR in direct form II transposed. Works on bare-metal targets.

```rust
use resonant_filters::{Biquad, design};

let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
let mut filter = Biquad::new(coeffs);

for sample in audio.iter_mut() {
    *sample = filter.process_sample(*sample);
}
```

## FIR filter — requires `alloc`

Arbitrary coefficient vector with a circular delay line. SIMD-accelerated dot product (SSE2 / NEON / scalar).

## Nonlinear filters

Analog-modelled processors for saturation and tonal shaping:

- `SaturatingBiquad` — biquad with tanh waveshaper on the state variables
- `MoogLadder` — four cascaded 1-pole sections with nonlinear feedback
- `StateVariableFilter` — Chamberlin SVF with simultaneous lowpass/highpass/bandpass outputs

## Decimation and resampling — requires `alloc`

```rust
use resonant_filters::resample;

// Integer decimation (48 kHz → 16 kHz)
let output = resample::decimate(&input, 3, 48000.0).unwrap();

// Polyphase resampling (arbitrary rational ratios)
use resonant_filters::resample::PolyphaseResampler;
let mut resampler = PolyphaseResampler::new(44100, 48000, 32)?;
resampler.process_into(&input, &mut output);
```

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `alloc` | yes | Enables `Fir`, `decimate()`, `PolyphaseResampler`, nonlinear filters |

The `Biquad` filter and Butterworth design helpers work without `alloc` on any target with `core`.

## License

MIT OR Apache-2.0
