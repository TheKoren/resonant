# DSP Concepts for Rust Developers

> This document explains the signal processing concepts used in the resonant library.

---

## Table of Contents

- [Windowing](#windowing)
- [The Fourier Transform (FFT)](#the-fourier-transform-fft)
- [Short-Time Fourier Transform (STFT)](#short-time-fourier-transform-stft)
- [Fixed-Point Arithmetic](#fixed-point-arithmetic)
- [Filters](#filters)
- [Spectral Analysis](#spectral-analysis)

---

## Windowing

### What is windowing?

When we analyse a chunk of audio, we're slicing a continuous signal into a finite block.
The edges of that block create artificial discontinuities — the signal doesn't naturally
start and stop at our chunk boundaries. These discontinuities cause **spectral leakage**:
energy smears across frequency bins in the FFT output, making it harder to identify the
true frequencies present.

A **window function** tapers the signal smoothly to zero at both ends, reducing leakage.

### Window functions in resonant

| Window | Main-lobe width | Side-lobe level | Best for |
|--------|----------------|-----------------|----------|
| Rectangular | Narrowest | Highest (-13 dB) | When you need maximum frequency resolution and can tolerate leakage |
| Hann | Moderate | Good (-31 dB) | General-purpose spectral analysis |
| Hamming | Moderate | Better (-43 dB) | Speech processing, filter design |
| Blackman | Wide | Excellent (-58 dB) | When side-lobe suppression matters more than resolution |
| Bartlett | Moderate | Moderate (-27 dB) | Simple triangular taper; sometimes used in averaged spectra |

### The trade-off

There is always a trade-off between **frequency resolution** (narrow main lobe) and
**spectral leakage** (low side lobes). No window is universally best — choose based on
what matters for your application.

### Code example

```rust
use resonant_core::window;

let mut samples = [0.5_f32; 1024];
window::hann(&mut samples);
// `samples` is now tapered to zero at both ends — ready for FFT
```

Each window function takes `&mut [f32]` and multiplies in-place, so there is no
allocation. You can also apply a window to a subset of a buffer:

```rust
use resonant_core::window;

let mut buf = [1.0_f32; 2048];
window::hamming(&mut buf[..1024]); // only window the first half
```

---

## The Fourier Transform (FFT)

The Fourier Transform decomposes a time-domain signal into its constituent frequencies.
Given N samples of audio, the FFT produces N/2+1 complex values, each representing the
amplitude and phase of a specific frequency.

The naive Discrete Fourier Transform (DFT) requires O(N²) operations. The Fast Fourier
Transform (FFT) computes the same result in O(N log N) — making real-time audio analysis
practical.

### Frequency bins

Each output bin k corresponds to a frequency:

```
f_k = k × sample_rate / N
```

Where:
- `k` is the bin index (0 to N/2)
- `sample_rate` is in Hz (e.g. 44100)
- `N` is the FFT size (number of input samples)

Bin 0 is the DC component (average value). Bin N/2 is the Nyquist frequency (half the
sample rate) — the highest frequency that can be represented.

### Type-state safety in resonant

In resonant, a signal carries its domain in the type system:

```rust,ignore
let time_signal: Signal<f32, TimeDomain> = Signal::from_samples(&samples);
let freq_signal: Signal<Complex32, FreqDomain> = time_signal.fft();

// This would be a compile error — you can't FFT a frequency-domain signal:
// let oops = freq_signal.fft(); // ERROR: no method `fft` on Signal<_, FreqDomain>
```

This prevents an entire class of bugs at compile time.

---

## Short-Time Fourier Transform (STFT)

The FFT gives you the frequency content of an entire signal — but not *when* those
frequencies occur. The STFT solves this by applying the FFT to overlapping windows
of the signal, producing a **spectrogram**: a 2D representation of frequency vs. time.

### How it works

1. **Slice** the signal into overlapping frames of length `window_size`
2. **Window** each frame (e.g. Hann) to taper the edges
3. **FFT** each windowed frame to get its frequency content
4. Collect the results into a 2D grid: one axis is time (frame index), the other is
   frequency (bin index)

### Parameters

- **Window size**: determines frequency resolution (larger = more bins, better resolution)
- **Hop size**: how far the window advances each step (smaller = more time resolution)
- **Overlap**: `window_size - hop_size`, typically 50–75%

### The time-frequency trade-off

You cannot have perfect resolution in both time and frequency simultaneously — this
is the **uncertainty principle** of signal processing.

| Window size | Frequency resolution | Time resolution |
|-------------|---------------------|-----------------|
| Large (4096) | Excellent — narrow bins | Poor — each frame spans ~93 ms at 44.1 kHz |
| Small (256) | Poor — wide bins | Excellent — each frame spans ~5.8 ms |

A hop size of 50% overlap (hop = window_size / 2) is a common starting point.

### Synthesis: overlap-add

The inverse STFT reconstructs the time-domain signal from its frames:

1. **IFFT** each frequency-domain frame back to the time domain
2. **Overlap-add**: sum the reconstructed frames at their original positions

With a good window and sufficient overlap, the original signal is recovered exactly
(up to floating-point precision). This is the basis for effects like time-stretching
and pitch-shifting.

### Code example

```rust,ignore
use resonant_fft::stft::StftBuilder;
use resonant_core::Signal;

let signal = Signal::from_samples(audio_samples);
let stft = StftBuilder::new(1024).hop_size(512).build();

let frames = stft.analyze(&signal).unwrap();
// `frames` is a Vec of frequency-domain Signal frames

let reconstructed = stft.synthesize(&frames, original_length).unwrap();
```

---

## Fixed-Point Arithmetic

### Why fixed-point?

Many microcontrollers (Cortex-M0, M3, most RISC-V cores) have no floating-point
unit (FPU). On these targets every `f32` operation is emulated in software —
slow, unpredictable, and power-hungry. Fixed-point arithmetic uses plain integers
with an implicit scaling factor, giving deterministic performance and
bit-exact results across platforms.

Even on targets *with* an FPU, fixed-point is sometimes preferred in safety-critical
or latency-sensitive paths because it has no NaN, no infinity, and no rounding
mode surprises.

### The Q format

The **Q format** is a convention for interpreting an integer as a fractional value.
The name tells you how the bits are split:

| Format | Storage | Signed bits | Fractional bits | Range | Resolution |
|--------|---------|-------------|-----------------|-------|------------|
| Q15 | `i16` | 1 | 15 | \[−1.0, 1.0) | 1/32768 ≈ 3.05 × 10⁻⁵ |
| Q31 | `i32` | 1 | 31 | \[−1.0, 1.0) | 1/2147483648 ≈ 4.66 × 10⁻¹⁰ |

The conversion is straightforward:

```
real_value = raw_integer / 2^fractional_bits
```

So the `i16` value `16384` in Q15 represents `16384 / 32768 = 0.5`.

### Saturation vs wrapping

In audio DSP, wrapping overflow produces harsh clicks and distortion. Hardware
DSP chips use **saturating arithmetic**: when a result exceeds the representable
range, it clamps to the maximum (or minimum) value instead of wrapping around.

resonant-core's `Q15` and `Q31` types follow this convention — all arithmetic
methods are named `saturating_*` to make the behaviour explicit:

```rust
use resonant_core::fixed::Q15;

let loud = Q15::from_f32(0.9);
let result = loud.saturating_add(loud);
assert_eq!(result, Q15::MAX); // clamped, not wrapped
```

### Multiplication

Multiplying two Q15 values produces a Q30 result (15 + 15 fractional bits) in a
32-bit intermediate. To get back to Q15 we shift right by 15 bits, discarding the
extra precision. This is equivalent to:

```
result = (a * b) >> 15
```

The same logic applies to Q31 using a 64-bit intermediate with a right-shift of 31.

### When to use Q15 vs Q31

| | Q15 | Q31 |
|---|-----|-----|
| **Storage** | 2 bytes | 4 bytes |
| **Precision** | ~16-bit audio quality | ~32-bit, exceeds 24-bit audio |
| **Speed** | Faster on 16-bit MCUs | Faster on 32-bit MCUs |
| **Use case** | Telephony, simple filters, Cortex-M0 | High-fidelity audio, Cortex-M4/M7 |

If you are targeting a 32-bit MCU, prefer Q31 — the extra precision costs nothing
on a 32-bit data path. Use Q15 when memory bandwidth or storage is the bottleneck
(e.g. large delay buffers on a 16-bit target).

### Converting between fixed and floating point

resonant-core provides `From`/`Into` conversions in both directions:

```rust
use resonant_core::fixed::{Q15, Q31};

// f32 → Q15
let q: Q15 = Q15::from_f32(0.5);

// Q15 → f32
let f: f32 = q.to_f32();

// f64 → Q31 (higher precision during conversion)
let q31: Q31 = Q31::from_f64(0.123456789);
let back: f64 = q31.to_f64();
assert!((back - 0.123456789).abs() < 1e-7);
```

Values outside \[−1.0, 1.0) are clamped automatically — no panics, no undefined
behaviour.

---

## Filters

A **digital filter** modifies a signal by attenuating or amplifying certain frequencies.
Filters are the workhorses of audio DSP — equalizers, crossovers, anti-alias stages, and
effects like wah-wah are all built from filters.

### FIR vs IIR

There are two fundamental filter architectures:

| | FIR (Finite Impulse Response) | IIR (Infinite Impulse Response) |
|---|---|---|
| **Feedback** | None — output depends only on current and past *inputs* | Uses feedback — output depends on past *outputs* too |
| **Impulse response** | Finite: dies out after N taps | Infinite: decays but never truly reaches zero |
| **Phase** | Can be exactly linear phase | Generally non-linear phase |
| **Order for steep rolloff** | High (many taps needed) | Low (a 2nd-order section gives 12 dB/oct) |
| **Stability** | Always stable | Can be unstable if poles are outside the unit circle |

**Rule of thumb**: Use IIR (biquad) when you need efficient, steep filtering and don't
care about phase linearity. Use FIR when you need linear phase or a very specific impulse
response shape (e.g. a matched filter or a Hilbert transformer).

### The biquad filter

The **biquad** (bi-quadratic) is a second-order IIR filter — the most common building
block in audio processing. It has five coefficients:

```
y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] - a1·y[n-1] - a2·y[n-2]
```

Where `x[n]` is the input, `y[n]` is the output, and the `b`/`a` coefficients define
the filter's frequency response.

resonant uses **direct form II transposed**, which has better numerical properties
(less rounding error) than the direct form shown above:

```
y[n] = b0·x[n] + s1
s1   = b1·x[n] - a1·y[n] + s2
s2   = b2·x[n] - a2·y[n]
```

Here `s1` and `s2` are the filter's internal state — just two numbers that capture the
filter's "memory" of past samples.

### Why direct form II transposed?

There are several ways to implement the same biquad transfer function. Direct form II
transposed is preferred in audio because:

- **Fewer delay elements**: only 2 state variables instead of 4
- **Better numerical behaviour**: accumulation happens on the output path, reducing
  the chance of intermediate overflow with f32 arithmetic
- **Coefficient changes**: updating coefficients mid-stream is smoother because the
  state represents filtered (not raw) history

### Cascading biquads

A single biquad gives 12 dB/octave rolloff. For steeper filters, cascade multiple
sections. Two cascaded biquads give a 4th-order filter (24 dB/oct), which is what
resonant's `decimate()` function uses for anti-aliasing.

### Butterworth design

The **Butterworth filter** has the flattest possible magnitude response in the passband —
no ripples. At the cutoff frequency, the gain is exactly −3 dB (≈ 0.707 amplitude).

resonant computes Butterworth coefficients using the **bilinear transform**: a mapping
from the analog (continuous-time) filter design to the digital (discrete-time) domain.

The steps are:

1. **Pre-warp** the cutoff frequency to compensate for the bilinear transform's
   frequency compression near Nyquist:
   ```
   k = tan(π × cutoff / sample_rate)
   ```

2. **Map** the analog Butterworth prototype (which has a simple formula) through the
   bilinear transform to get digital coefficients `b0, b1, b2, a1, a2`.

The design arithmetic uses `f64` for precision; the resulting coefficients are cast to
`f32` for runtime efficiency.

```rust,ignore
use resonant_filters::design;
use resonant_filters::Biquad;

// Design a lowpass at 1 kHz
let coeffs = design::butterworth_lowpass(1000.0, 44100.0).unwrap();
let mut filter = Biquad::new(coeffs);

// Filter audio sample-by-sample
for sample in audio.iter_mut() {
    *sample = filter.process_sample(*sample);
}
```

### FIR filters

An FIR filter computes each output sample as a weighted sum of the most recent N input
samples:

```
y[n] = h[0]·x[n] + h[1]·x[n-1] + ... + h[N-1]·x[n-N+1]
```

The weights `h[0..N]` are called the filter **coefficients** or **taps**. The FIR's
impulse response *is* the coefficient vector — feed in a single `1.0` followed by
zeros, and you get the coefficients back out.

resonant's `Fir` struct uses a circular delay line internally, avoiding the need to
shift the entire buffer each sample.

### Decimation (sample-rate reduction)

**Decimation** reduces the sample rate by an integer factor *M*: keep every M-th sample,
discard the rest. But you can't just throw samples away — the original signal may
contain frequencies above the new Nyquist limit (half the new sample rate), which would
fold back as **aliasing** artifacts.

The solution: **lowpass filter first**, then downsample.

```
input (48 kHz) → [anti-alias LP at 8 kHz] → [keep every 3rd] → output (16 kHz)
```

resonant's `decimate()` cascades two Butterworth biquad sections for a 4th-order
anti-alias filter, then picks every M-th sample.

```rust,ignore
use resonant_filters::resample;

// Decimate from 48 kHz to 16 kHz (factor 3)
let output = resample::decimate(&input, 3, 48000.0).unwrap();
```

---

## Spectral Analysis

TODO: Explain spectral features (centroid, flatness, rolloff) and their musical meaning.
