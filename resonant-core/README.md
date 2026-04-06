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

### Ring buffers

**`RingBuf<T, N>`** — const-generic, stack-allocated circular buffer. No heap, no
panics in the hot path. Useful for delay lines, moving averages, and streaming sample
ingestion.

```rust
use resonant_core::RingBuf;

let mut rb = RingBuf::<f32, 4>::new();
rb.push(1.0);
rb.push(2.0);
assert_eq!(rb.pop(), Some(1.0));

// View contents without consuming
let (a, b) = rb.as_slices(); // two contiguous slices covering oldest→newest
let oldest = rb.peek();      // borrow without removing
for s in rb.iter() { /* oldest first */ }
for s in rb.drain() { /* consume all in order */ }
```

**`HeapRingBuf<T>`** (requires `alloc` feature) — same API, capacity set at runtime.
Useful in pipeline nodes whose buffer size comes from a config file or user input.

```rust,ignore
use resonant_core::HeapRingBuf;

let mut rb = HeapRingBuf::<f32>::new(frame_size);
rb.push(sample);
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

### Signal arithmetic operators

Element-wise arithmetic on `Signal<T, D>` preserves domain safety at compile time:

```rust
use resonant_core::signal::{Signal, TimeDomain};

let a = Signal::<[f32; 4], TimeDomain>::new([1.0, 2.0, 3.0, 4.0]);
let b = Signal::<[f32; 4], TimeDomain>::new([0.5; 4]);
let c = a + b;            // element-wise add
let d = c * 0.5;          // scale

let dry = Signal::<[f32; 4], TimeDomain>::new([1.0; 4]);
let wet = Signal::<[f32; 4], TimeDomain>::new([0.0; 4]);
let e = dry.mix(&wet, 0.5); // dry + wet * gain
```

Mixing signals from different domains is a compile error:

```rust,compile_fail
use resonant_core::signal::{Signal, TimeDomain, FreqDomain};
let t = Signal::<[f32; 4], TimeDomain>::new([0.0; 4]);
let f = Signal::<[f32; 4], FreqDomain>::new([0.0; 4]);
let _ = t + f; // ERROR — domain mismatch
```

Operators are available for both array-backed (`no_alloc`) and `Vec`-backed
(`alloc` feature) signals.

### `Sample` trait

Unified conversion between scalar sample formats — useful when writing generic
DSP algorithms that work across `f32`, `f64`, `Q15`, `Q31`, `i16`, and `i32`:

```rust
use resonant_core::{Sample, fixed::Q15};

fn scale_half<S: Sample>(v: S) -> S {
    S::from_f32(v.to_f32() * 0.5)
}

let result = scale_half(Q15::from_f32(0.8));
assert!((result.to_f32() - 0.4).abs() < 0.002);
```

### Window functions

Five standard window functions that multiply a buffer in-place — no allocation.
All are generic over `S: Sample`, so they work on `f32`, `f64`, `Q15`, `Q31`,
and integer buffers without a manual conversion step.

| Function | Use case |
|----------|----------|
| `window::hann` | General-purpose spectral analysis |
| `window::hamming` | Speech processing, filter design |
| `window::blackman` | Maximum side-lobe suppression |
| `window::rectangular` | No-op (API completeness) |
| `window::bartlett` | Simple triangular taper |

```rust
use resonant_core::{window, fixed::Q15};

// f32 — most common case
let mut f32_buf = [1.0_f32; 1024];
window::hann(&mut f32_buf);

// Same API for fixed-point buffers — no conversion needed
let mut q15_buf = [Q15::from_f32(1.0); 1024];
window::hann(&mut q15_buf);

// Precomputed window (SIMD-accelerated, f32 only)
let mut win = [1.0_f32; 1024];
window::hann(&mut win);
let mut frame = [0.5_f32; 1024];
window::apply(&mut frame, &win); // fast path for repeated application
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
| `alloc` | no | Enables `SlidingWindow<T>`, `HeapRingBuf<T>`, and `Signal<Vec<f32>, D>` operators |

## Targets

- `no_std` by default — works on bare-metal and WASM
- Tested on `x86_64` and should work on any target with `core`
- No floating-point requirement unless you use `window` or `f32`-based `Signal`

## Roadmap (resonant-core)

- **v0.1.0** ✓ — `Signal<T, D>` type-state, ring buffers, window functions, fixed-point types, SIMD dispatch
- **v0.2.0** — `Sample` trait (done), signal arithmetic (done), generic window functions (done),
  real-valued FFT support, polyphase resampling primitives

## License

Licensed under either of [Apache License, Version 2.0](../LICENSE-APACHE) or
[MIT license](../LICENSE-MIT) at your option.
