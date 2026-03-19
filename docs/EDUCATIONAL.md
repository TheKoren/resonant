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

```rust,ignore
use resonant_core::window::WindowFn;

let samples: Vec<f32> = get_audio_chunk();
let windowed = WindowFn::Hann.apply(&samples);
// `windowed` is now tapered — ready for FFT
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

### Parameters

- **Window size**: determines frequency resolution (larger = more bins, better resolution)
- **Hop size**: how far the window advances each step (smaller = more time resolution)
- **Overlap**: `window_size - hop_size`, typically 50–75%

More detail will be added as the STFT implementation is completed.

---

## Fixed-Point Arithmetic

TODO: Explain Q15/Q31 formats, why they matter for embedded targets without FPU,
and how resonant-core's types work.

---

## Filters

TODO: Explain biquad filters, FIR vs IIR, and the design helpers in resonant-filters.

---

## Spectral Analysis

TODO: Explain spectral features (centroid, flatness, rolloff) and their musical meaning.
