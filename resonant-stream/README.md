# resonant-stream

Streaming DSP pipeline with pull-based processing and in-place chunks.

Part of the [resonant](https://crates.io/crates/resonant) workspace.

## Features

- **`Chunk`** — owned buffer of interleaved `f32` samples with sample rate and channel metadata
- **`DspNode` trait** — implement `process` and `reset` to create custom processors
- **`Pipeline` builder** — chain nodes sequentially with optional format validation
- **Built-in nodes** — gain, biquad filter, FFT, downmix, decimation, and tap (observation)

## Quick start

```rust
use resonant_stream::{Chunk, DspNode, Pipeline};
use resonant_stream::nodes::{GainNode, FilterNode, MixNode};
use resonant_filters::design::butterworth_lowpass;
use resonant_filters::Biquad;

// Build a 3-node pipeline: downmix -> lowpass -> gain
let coeffs = butterworth_lowpass(1000.0, 44100.0).unwrap();
let mut pipeline = Pipeline::builder()
    .sample_rate(44100)
    .channels(2)
    .node(MixNode::new())
    .node(FilterNode::new(Biquad::new(coeffs)))
    .node(GainNode::new(0.8))
    .build();

let chunk = Chunk::new(vec![0.5, -0.5; 1024], 44100, 2);
let output = pipeline.process(chunk).unwrap();
```

## Built-in nodes

| Node | Description |
|------|-------------|
| `GainNode` | Linear amplitude scaling (adjustable at runtime) |
| `FilterNode` | Biquad IIR filter (lowpass, highpass, bandpass, etc.) |
| `TapNode` | Side-channel observation via callback |
| `FftNode` | Forward FFT — outputs magnitude or power spectrum |
| `MixNode` | Multi-channel to mono downmix |
| `ResampleNode` | Integer decimation with anti-alias filtering |

## Custom nodes

Implement `DspNode` to create your own:

```rust
use resonant_stream::{Chunk, DspNode, StreamError};

struct HardClip { threshold: f32 }

impl DspNode for HardClip {
    fn process(&mut self, mut input: Chunk) -> Result<Chunk, StreamError> {
        for s in input.data_mut() {
            *s = s.clamp(-self.threshold, self.threshold);
        }
        Ok(input)
    }
    fn reset(&mut self) {}
}
```

## License

MIT OR Apache-2.0
