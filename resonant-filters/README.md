# resonant-filters

FIR/IIR filters and design helpers for audio DSP.

Part of the [resonant](https://crates.io/crates/resonant) workspace.

## Features

- **Biquad filter** — direct form II transposed, second-order IIR with accessible state
- **FIR filter** — arbitrary coefficient vector with circular delay line (requires `alloc`)
- **Butterworth design** — `butterworth_lowpass()` and `butterworth_highpass()` with bilinear transform and frequency pre-warping
- **Integer decimation** — `decimate()` with cascaded 4th-order Butterworth anti-alias filter (requires `alloc`)

## Usage

```rust,ignore
use resonant_filters::{Biquad, design};

// Design a 1 kHz lowpass at 44.1 kHz sample rate
let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
let mut filter = Biquad::new(coeffs);

for sample in audio.iter_mut() {
    *sample = filter.process_sample(*sample);
}
```

```rust,ignore
use resonant_filters::resample;

// Decimate from 48 kHz to 16 kHz (factor 3)
let output = resample::decimate(&input, 3, 48000.0).unwrap();
```

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `alloc` | yes | Enables `Fir`, `decimate()`, and other heap-allocating APIs |

The `Biquad` filter and Butterworth design helpers work without `alloc` on bare-metal targets.

## Roadmap

- Interpolation (integer upsample with anti-image filtering)
- Polyphase resampler for efficient non-integer rate conversion
- Higher-order filter design (Chebyshev, elliptic)
- Notch / bandpass / allpass design helpers
