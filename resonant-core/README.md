# resonant-core

`no_std`, zero-allocation DSP foundation with compile-time signal domain tracking.

Part of the [resonant](https://crates.io/crates/resonant) workspace.

## What you get

### Signal type-state (`Signal<T, D>`)

A generic container that tracks whether your data is in the **time domain** or
**frequency domain** at the type level. Passing a frequency-domain signal to a
function that expects time-domain input is a compile error, not a runtime bug.

```rust
use resonant_core::{Signal, TimeDomain, FreqDomain};

let sig = Signal::<[f32; 4], TimeDomain>::new([0.0, 0.5, 1.0, 0.5]);

// Transitions are explicit — higher-level crates (resonant-fft) provide
// .fft() / .ifft() that consume one domain and return the other.
let freq: Signal<[f32; 4], FreqDomain> = sig.map_domain();
```

### Ring buffer (`RingBuf<T, N>`)

Const-generic, stack-allocated circular buffer. No heap, no panics in the hot
path. Useful for delay lines, moving averages, and streaming sample ingestion.

```rust
use resonant_core::RingBuf;

let mut rb = RingBuf::<f32, 4>::new();
rb.push(1.0);
rb.push(2.0);
assert_eq!(rb.pop(), Some(1.0));
```

### Sliding window (`SlidingWindow<T>`) — requires `alloc` feature

Heap-allocated overlapping frame extractor for STFT-style workflows. Configure
window size and hop size, push samples in, and read complete frames.

```rust,ignore
use resonant_core::SlidingWindow;

let mut sw = SlidingWindow::new(1024, 512); // 50% overlap
sw.push_slice(&audio_chunk);
if sw.is_ready() {
    let frame = sw.current_frame().unwrap();
    // process frame...
    sw.advance();
}
```

### Window functions

Five standard window functions that multiply a buffer in-place — no allocation:

| Function | Use case |
|----------|----------|
| `window::hann` | General-purpose spectral analysis |
| `window::hamming` | Speech processing, filter design |
| `window::blackman` | Maximum side-lobe suppression |
| `window::rectangular` | No-op (API completeness) |
| `window::bartlett` | Simple triangular taper |

```rust
use resonant_core::window;

let mut buf = [1.0_f32; 1024];
window::hann(&mut buf);
```

### Fixed-point types (`Q15`, `Q31`)

Signed fixed-point types for targets without an FPU. Saturating arithmetic
prevents wrapping distortion. Converts to/from `f32` and `f64`.

```rust
use resonant_core::fixed::{Q15, Q31};

let a = Q15::from_f32(0.5);
let b = Q15::from_f32(0.25);
let c = a.saturating_add(b); // 0.75, no overflow risk
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `alloc` | no | Enables `SlidingWindow<T>` (requires a heap allocator) |

## Targets

- `no_std` by default — works on bare-metal and WASM
- Tested on `x86_64` and should work on any target with `core`
- No floating-point requirement unless you use `window` or `f32`-based `Signal`

## Roadmap (resonant-core)

- **v0.1.0** — SIMD-friendly numeric traits (`core::ops` wrappers), additional
  domain markers for user-defined domains
- **v0.2.0** — `const` window function variants for compile-time lookup tables,
  interpolation helpers for fixed-point types
- **v1.0.0** — API stabilisation, verified `thumbv7em-none-eabihf` and
  `wasm32-unknown-unknown` builds

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT license](../LICENSE-MIT) at your option.
