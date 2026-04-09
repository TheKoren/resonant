# DSP Concepts for Rust Developers

> This document explains the signal processing concepts used in the resonant library.

---

## Table of Contents

- [Windowing](#windowing)
- [The Fourier Transform (FFT)](#the-fourier-transform-fft)
- [Real-Valued FFT (rfft)](#real-valued-fft-rfft)
- [Short-Time Fourier Transform (STFT)](#short-time-fourier-transform-stft)
- [Fixed-Point Arithmetic](#fixed-point-arithmetic)
- [Filters](#filters)
- [Nonlinear Filters](#nonlinear-filters)
- [Polyphase Resampling](#polyphase-resampling)
- [Spectral Analysis](#spectral-analysis)
- [Building for Embedded and WASM](#building-for-embedded-and-wasm)
- [SIMD Acceleration](#simd-acceleration)

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

Each window function multiplies a buffer in-place with no allocation. The named window
functions (`hann`, `hamming`, `blackman`, `bartlett`, `rectangular`) are generic over
the [`Sample`](https://docs.rs/resonant-core) trait and work equally on `f32`, `f64`,
`Q15`, and `Q31` buffers without a manual conversion step. Existing `f32` callers
require no changes.

`window::apply` is a separate SIMD-accelerated function for the common case of
repeatedly applying a precomputed `f32` window — it skips the cosine computation.

```rust
use resonant_core::{window, fixed::Q15};

// f32 — most common
let mut f32_buf = [1.0_f32; 1024];
window::hann(&mut f32_buf);

// Q15 fixed-point — no conversion needed
let mut q15_buf = [Q15::from_f32(1.0); 1024];
window::hann(&mut q15_buf);

// Precomputed window for repeated application (SIMD-accelerated, f32 only)
let mut win = [1.0_f32; 1024];
window::hann(&mut win); // compute once

let mut frame = [0.5_f32; 1024];
window::apply(&mut frame, &win); // fast path — no cosines, uses SSE2/NEON

// Window a subset of a buffer
let mut buf = [1.0_f32; 2048];
window::hamming(&mut buf[..1024]);
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

## Real-Valued FFT (rfft)

### Conjugate symmetry of real inputs

When the input to an FFT is purely real (no imaginary part), the output is not
arbitrary — it has **conjugate symmetry**:

```
X[k] = conj(X[N - k])
```

This means bin `k` and bin `N - k` carry the same information (one is the complex
conjugate of the other). Half the FFT output is redundant. The real-valued FFT (rfft)
exploits this to return only the **unique** bins: indices 0 through N/2, giving
**N/2 + 1** complex values instead of N.

All audio signals are real — there is no physical interpretation of an imaginary
sample value. Using `rfft` instead of the full complex FFT gives roughly:

- **~40% lower CPU** — the transform works on half the data internally
- **Half the output memory** — N/2+1 bins vs N bins
- **Identical results** for the unique half of the spectrum

### Bin count and frequency mapping

For N input samples at sample rate `fs`, `rfft` returns N/2+1 bins. The frequency
of bin k is:

```
f_k = k × fs / N     (k = 0, 1, …, N/2)
```

Bin 0 is DC (0 Hz). Bin N/2 is the Nyquist frequency (fs/2). Bins beyond N/2 are
the conjugate mirror and are not returned.

### When to use rfft vs fft

| Situation | Use |
|---|---|
| Audio analysis, spectral features, STFT frames | `rfft` — input is always real |
| Signal has a complex component (e.g. analytic signal after Hilbert) | `fft` |
| You need the full complex spectrum for convolution in frequency domain | `rfft` (the product of two rfft outputs is their circular convolution) |
| Computing on non-power-of-two lengths with the `rustfft` backend | `fft` — rfft is radix-2 only in the no-alloc path |

### Code example

```rust,ignore
use resonant_fft::SignalRfftExt;
use resonant_core::Signal;

let samples: Vec<f32> = (0..1024)
    .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44100.0).sin())
    .collect();

let signal = Signal::from_samples(samples);
let freq = signal.rfft().unwrap();
// freq contains 513 complex bins (1024/2 + 1)

// Magnitude at bin k
let mags = freq.magnitude();
assert_eq!(mags.len(), 513);
```

For the inverse transform, `irfft` reconstructs N real samples from the N/2+1
complex bins. The original signal length must be provided because the bin count
alone is ambiguous (both N=1024 and N=1023 produce 512 bins):

```rust,ignore
use resonant_fft::irfft;

let mut reconstructed = vec![0.0_f32; 1024];
irfft(&freq_bins, &mut reconstructed).unwrap();
```

### No-alloc usage

The underlying `rfft` and `irfft` functions are `no_std`, `no_alloc`, operating on
caller-provided output buffers:

```rust,ignore
use resonant_fft::rfft;
use resonant_fft::Complex;

let input = [0.0_f32; 1024];
let mut out = [Complex::new(0.0, 0.0); 513]; // N/2 + 1
rfft(&input, &mut out).unwrap();
```

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

## Nonlinear Filters

### Why linear filters aren't enough

A linear filter obeys superposition: `filter(a + b) = filter(a) + filter(b)`. This is
mathematically clean, but it means a linear filter can only attenuate or amplify
frequencies — it cannot create new ones.

Real analog circuits, magnetic tape, and vacuum tubes are *nonlinear*. When audio is
pushed hard through them, the output contains **harmonics** — frequencies that were not
present in the input. A pure 440 Hz sine through a driven tube amplifier produces energy
at 880 Hz, 1320 Hz, and so on. This harmonic saturation is the sound of "warmth" and
"character" that musicians seek.

Software models of these effects must incorporate nonlinearity.

### The tanh waveshaper

The simplest nonlinearity is a **waveshaper**: a function applied point-by-point to
the signal. The hyperbolic tangent (`tanh`) is the standard choice because:

- It is smooth and differentiable (no discontinuities)
- It is bounded: `tanh(x) → ±1` as `x → ±∞` (soft clip, not hard clip)
- It matches the saturation curve of real transistors reasonably well
- It is cheap to compute on modern hardware

```
output = tanh(drive × input) / tanh(drive)
```

The `drive` parameter controls how far into saturation the signal is pushed:

- `drive → 0`: nearly linear — `tanh(x)/x → 1` for small x
- `drive = 1`: noticeable harmonic generation, peak amplitude unchanged
- `drive >> 1`: hard limiting, output approaches ±1

resonant's `SaturatingBiquad` blends the linear and nonlinear paths:

```
output = (1 − drive) × linear_output + drive × tanh(linear_output)
```

At `drive = 0.0` the filter is identical to a plain `Biquad`. At `drive = 1.0` the
output is fully saturated.

```rust
use resonant_filters::{design, nonlinear::SaturatingBiquad};

let coeffs = design::butterworth_lowpass(2000.0, 44100.0).unwrap();
let mut filter = SaturatingBiquad::new(coeffs, 0.7); // 70% saturation

let y = filter.process_sample(0.8); // pushed into soft saturation
assert!(y.is_finite());
assert!(y.abs() <= 1.0 + 1e-4);
```

### Moog ladder filter topology

The **Moog ladder** is a 4th-order resonant lowpass designed to model the transistor
ladder circuit in the Minimoog synthesizer. It produces a distinctively warm, musical
lowpass that self-oscillates at high resonance — a feature with no equivalent in linear
filter design.

The topology is four cascaded first-order sections with nonlinear (tanh) feedback
from the output back to the input:

```
x[n] → [tanh] → [pole 1] → [pole 2] → [pole 3] → [pole 4] → y[n]
          ↑_______________resonance × y[n]_____________________↑
```

Each pole adds 6 dB/octave rolloff; the cascade gives 24 dB/octave. The resonance
feedback creates a peak at the cutoff frequency — increasing resonance narrows and
amplifies this peak until the loop gain exceeds unity and the filter **self-oscillates**:
it produces a sustained tone at the cutoff frequency even with zero input.

Parameters:
- `cutoff` — normalised frequency: 0.0 = DC, 1.0 = Nyquist
- `resonance` — feedback amount: 0.0 = damped, 4.0 = self-oscillation threshold

```rust
use resonant_filters::nonlinear::MoogLadder;

let mut ladder = MoogLadder::new(0.3, 2.5); // cutoff 30%, moderate resonance

// Process a block
let output: Vec<f32> = (0..256)
    .map(|i| ladder.process_sample((i as f32 * 0.05).sin()))
    .collect();
```

At `resonance = 4.0` the filter self-oscillates — feeding it a single impulse and
then silence produces a sustained sinusoid at the cutoff frequency.

### State-variable filter (SVF): four outputs from one pass

A conventional filter computes one output type (lowpass, or highpass, etc.). The
**state-variable filter** (SVF) computes all four simultaneously from a single pass
through the signal, at the same CPU cost as one:

- **Lowpass** (LP) — attenuates above cutoff
- **Highpass** (HP) — attenuates below cutoff
- **Bandpass** (BP) — peak at cutoff, rolls off in both directions
- **Notch** — dip at cutoff (= LP + HP)

The Chamberlin SVF formulation uses two integrators in a loop:

```
hp  = x − lp − damp × bp
bp += f0 × hp
lp += f0 × bp
notch = lp + hp
```

Where `f0 = 2 × sin(π × fc / fs)` and `damp = 1/Q`.

The LP + HP identity (`lp + hp = x − damp × bp`) means the sum of the lowpass and
highpass outputs approximates the input when Q is high — an exact allpass property
in the limit.

```rust
use resonant_filters::nonlinear::StateVariableFilter;

let mut svf = StateVariableFilter::new(
    1000.0, // cutoff Hz
    1.0,    // Q
    44100.0 // sample rate Hz
);

let out = svf.process(0.5);
println!("LP={:.3} HP={:.3} BP={:.3} Notch={:.3}",
    out.lp, out.hp, out.bp, out.notch);
```

The SVF is ideal when you need to morph between filter types smoothly (e.g. an
LP → HP crossfade) or extract the bandpass for a resonance detector, without paying
the CPU cost of running three separate filters.

### Filter frequency response with nonlinear filters

Because nonlinear filters have no closed-form transfer function `H(z)`, their
frequency response is estimated via a **sine sweep**: drive the filter with a pure
sine at each analysis frequency, let it settle, then measure the peak output
amplitude relative to the input peak.

```rust,ignore
use resonant_filters::nonlinear::MoogLadder;
use resonant_filters::response::FilterResponseExt;

let filter = MoogLadder::new(0.4, 1.0);
let resp = filter.frequency_response(256, 44100.0).unwrap();

// Print the frequency response table
for (f, m) in resp.frequencies.iter().zip(resp.magnitudes.iter()) {
    println!("{:8.1} Hz  {:+.1} dB", f, 20.0 * m.log10());
}
```

The result reflects the operating point at unity-gain input — a driven filter at
high input amplitude will measure differently because the saturation state changes.

---

## Polyphase Resampling

### The problem with integer decimation

Simple decimation keeps every M-th sample after lowpass filtering. This is efficient
for integer ratios (e.g. 48 kHz → 16 kHz, M = 3) but cannot express arbitrary
rational ratios like 44100 Hz → 48000 Hz (ratio 147:160).

A naïve approach would upsample by 160 (insert 159 zeros between each sample),
apply a lowpass filter, then downsample by 147 — but this requires processing at
160× the original rate, which is prohibitively expensive.

### The polyphase decomposition

The **polyphase decomposition** restructures the problem to process at the *output*
rate, not at the intermediate upsampled rate.

Starting from a single FIR lowpass filter `h[n]` of length `L = P × taps_per_phase`:

1. Split `h` into `P` **sub-filters** (polyphase branches), one per upsample phase:
   ```
   E_p[k] = h[p + k × P]     (p = 0, 1, …, P-1;  k = 0, 1, …, taps_per_phase-1)
   ```
   Each sub-filter `E_p` has `taps_per_phase` coefficients.

2. To produce output sample `m`, identify which phase `p = m mod P` is active and
   how many input samples have been consumed: `m // P × Q` samples per output period.

3. Convolve the input history with sub-filter `E_p` only — never the full FIR.

The result: each output sample costs `taps_per_phase` multiplications, regardless of
the upsample factor P. Processing 44100→48000 (P=160, Q=147) costs the same per
sample as 2:1 upsampling.

### Anti-alias filter design

The anti-alias FIR is designed as a **windowed-sinc lowpass** at the stricter of the
two Nyquist frequencies: `cutoff = 0.5 / max(P, Q)`. resonant uses the Blackman
window, which gives ~74 dB stopband attenuation.

`taps_per_phase` controls the trade-off between quality and CPU:

| `taps_per_phase` | Stopband | Startup latency | Use case |
|---|---|---|---|
| 8 | ~55 dB | Short | Real-time, modest quality |
| 16 (default) | ~74 dB | Medium | General audio |
| 32 | ~90 dB | Long | Mastering, archival |

### Using PolyphaseResampler

```rust
use resonant_filters::resample::PolyphaseResampler;

// 44100 → 48000 Hz  (up=160, down=147 after GCD reduction)
let mut r = PolyphaseResampler::new(160, 147).unwrap();

let input: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.1).sin()).collect();
let output = r.process(&input);

// Expect approximately 44100 × 160/147 = 48000 samples
assert!((output.len() as i64 - 48000).abs() <= 2);
```

For streaming (zero-alloc inner loop), use `process_into` which appends to an
existing buffer:

```rust,ignore
let mut out_buf = Vec::with_capacity(expected_len);
r.process_into(&input_chunk, &mut out_buf);
```

`reset()` clears the ring buffer and phase index, making the resampler behave as if
newly constructed — useful when switching between unrelated audio segments.

### SNR considerations

The signal-to-noise ratio of a resampled signal is limited by the stopband attenuation
of the anti-alias FIR. With the default 16 taps per phase and Blackman window, you
can expect:

- **Passband ripple**: < 0.01 dB for frequencies below 90% of the lower Nyquist
- **Stopband attenuation**: ~74 dB (Blackman window)
- **THD+N**: well below -70 dBFS for a full-scale sine at 1 kHz

For reference, 16-bit CD audio requires ~96 dB dynamic range. The default quality is
sufficient for 16-bit distribution; use `with_quality(up, down, 32)` for 24-bit
mastering.

### Oversampling for nonlinear processing

Running a saturating processor at 2× or 4× the nominal sample rate reduces aliasing
from the nonlinear harmonics back into the audible band. The `Oversample<N>` wrapper
handles the upsample → process → downsample cycle:

```rust,ignore
use resonant_filters::{design, nonlinear::SaturatingBiquad, oversample::Oversample};

let coeffs = design::butterworth_lowpass(3000.0, 44100.0).unwrap();
let mut drive = SaturatingBiquad::new(coeffs, 0.9);

let mut os: Oversample<4> = Oversample::new(44100.0);
let output = os.process(&input, |x| drive.process_sample(x));
// `output` is at 44100 Hz; the distortion was computed at 176400 Hz
```

For `N = 1`, `Oversample` is a zero-overhead passthrough — the upsamplers are
never created.

---

## Spectral Analysis

Spectral features summarise the shape of a magnitude spectrum as a single number. They
are the building blocks of music information retrieval (MIR) — used to classify timbre,
detect changes in sound texture, and drive effects like auto-EQ.

All four features in resonant operate on a **magnitude spectrum**: a slice of non-negative
values where each element is the magnitude of a frequency bin. Compute this from a
frequency-domain signal with `SignalFreqExt::magnitude()`.

### Spectral centroid

The centroid is the **weighted mean frequency** of the spectrum — its "centre of mass".

```
centroid = Σ(magnitude[k] × frequency[k]) / Σ(magnitude[k])
```

- A **bright, treble-heavy** sound (cymbal, violin harmonics) has a high centroid.
- A **dark, bass-heavy** sound (kick drum, cello) has a low centroid.
- In music information retrieval, centroid correlates strongly with perceived brightness.

```rust
use resonant_analysis::spectral;

let magnitudes = [0.1_f32, 0.3, 0.9, 0.4, 0.1];
let frequencies = [0.0_f32, 1000.0, 2000.0, 3000.0, 4000.0];

let centroid = spectral::spectral_centroid(&magnitudes, &frequencies).unwrap();
// centroid ≈ 2105 Hz — pulled toward the dominant 2 kHz peak
```

### Spectral spread

Spread is the **standard deviation** of frequency around the centroid — it measures how
"wide" the spectrum is.

```
spread = sqrt( Σ(magnitude[k] × (frequency[k] − centroid)²) / Σ(magnitude[k]) )
```

- A **pure sine** at a single frequency has near-zero spread.
- **White noise** (equal energy at all frequencies) has maximum spread.
- A **rich harmonic** sound (full orchestra) has intermediate spread.

```rust
use resonant_analysis::spectral;

// Pure sine at 1 kHz
let mags_sine = [0.0_f32, 1.0, 0.0, 0.0, 0.0];
let freqs     = [0.0_f32, 1000.0, 2000.0, 3000.0, 4000.0];
let spread_sine = spectral::spectral_spread(&mags_sine, &freqs).unwrap();

// Flat noise
let mags_noise = [1.0_f32; 5];
let spread_noise = spectral::spectral_spread(&mags_noise, &freqs).unwrap();

assert!(spread_noise > spread_sine); // noise is much wider than a pure tone
```

### Spectral flatness

Flatness is the ratio of the **geometric mean** to the **arithmetic mean** of the
magnitudes. It ranges from 0.0 (perfectly tonal — one sharp peak) to 1.0 (perfectly
flat — noise-like energy across all bins).

```
flatness = geometric_mean(magnitudes) / arithmetic_mean(magnitudes)
         = exp(mean(ln(magnitudes))) / mean(magnitudes)
```

The log-domain computation avoids numerical underflow for long spectra.

- **Tonal sounds** (sine wave, tuned instrument): flatness close to 0.
- **Noise and unpitched percussion**: flatness close to 1.
- **Flatness is used in codec design** (e.g. MP3 bit allocation) and tonality detection.

```rust
use resonant_analysis::spectral;

let tonal = [0.0_f32, 0.0, 10.0, 0.0, 0.0]; // single peak
let noisy = [1.0_f32, 1.1, 0.9, 1.0, 1.0];  // roughly flat

let f_tonal = spectral::spectral_flatness(&tonal).unwrap();
let f_noisy = spectral::spectral_flatness(&noisy).unwrap();

assert!(f_tonal < 0.1);  // strongly tonal
assert!(f_noisy > 0.9);  // nearly flat
```

### Spectral rolloff

Rolloff is the frequency below which a given **percentage of the total spectral energy**
is contained. The most common threshold is 85%.

```
rolloff_85 = min frequency f such that: Σ_{k: freq[k] ≤ f} magnitude[k] ≥ 0.85 × Σ magnitude[k]
```

- A **bass-heavy** mix has a low rolloff — most energy is in the low frequencies.
- A **high-pitched** or bright sound has a high rolloff.
- Rolloff is used to distinguish **speech from music** and to estimate the high-frequency
  boundary of content (useful for adaptive bit allocation).

```rust
use resonant_analysis::spectral;

let magnitudes = [3.0_f32, 2.0, 1.0, 0.5, 0.2]; // most energy is low
let frequencies = [200.0_f32, 500.0, 1000.0, 2000.0, 4000.0];

let rolloff = spectral::spectral_rolloff(&magnitudes, &frequencies, 0.85).unwrap();
// rolloff ≈ 500 Hz — 85% of energy is below 500 Hz
```

### Putting it together — a real FFT pipeline

```rust,ignore
use resonant_core::window;
use resonant_fft::{SignalFftExt, SignalFreqExt};
use resonant_analysis::spectral;
use resonant_core::Signal;

// 1. Build a windowed signal
let mut samples = audio_frame.to_vec(); // 1024 f32 samples at 44100 Hz
window::hann(&mut samples);
let signal = Signal::from_samples(samples);

// 2. FFT → magnitude spectrum
let freq_signal = signal.fft().unwrap();
let magnitudes = freq_signal.magnitude();

// 3. Build a frequency axis (bin k → Hz)
let n = 1024_f32;
let sr = 44100.0_f32;
let frequencies: Vec<f32> = (0..magnitudes.len()).map(|k| k as f32 * sr / n).collect();

// 4. Compute spectral features
let centroid = spectral::spectral_centroid(&magnitudes, &frequencies).unwrap();
let spread   = spectral::spectral_spread(&magnitudes, &frequencies).unwrap();
let flatness = spectral::spectral_flatness(&magnitudes).unwrap();
let rolloff  = spectral::spectral_rolloff(&magnitudes, &frequencies, 0.85).unwrap();
```

---

## Building for Embedded and WASM

### Feature flags

| Crate | Feature | Effect |
|---|---|---|
| `resonant-core` | *(none required)* | `no_std`, `no_alloc` by default |
| `resonant-core` | `alloc` | Enables `SlidingWindow` and other heap-backed APIs |
| `resonant-fft` | *(none)* | `no_std`, `no_alloc` — radix-2 FFT and DCT only |
| `resonant-fft` | `alloc` | Enables extension traits (`SignalFftExt`, `SignalFreqExt`) |
| `resonant-fft` | `rustfft` (default) | Arbitrary-length FFT; implies `alloc` |
| `resonant-filters` | *(none)* | `no_std`, `no_alloc` — biquad and FIR filters |
| `resonant-filters` | `alloc` (default) | Enables `IirChain` and design helpers |

### Embedded (Cortex-M4 / `thumbv7em-none-eabihf`)

```toml
# Cargo.toml — your embedded crate
[dependencies]
resonant-core    = { version = "0.0.2", default-features = false }
resonant-fft     = { version = "0.0.3", default-features = false }
resonant-filters = { version = "0.0.2", default-features = false }
```

No global allocator is required. The radix-2 FFT, all window functions, biquad, and FIR
filters work on stack-allocated buffers.

```sh
# Build for Cortex-M4 with hardware FPU
cargo build --release --target thumbv7em-none-eabihf
```

You can verify the build runs correctly in QEMU:

```sh
qemu-system-arm -machine mps2-an386 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel target/thumbv7em-none-eabihf/release/your-app
echo "Exit: $?"
```

The `ci/no-std-check/` crate in the resonant repository is a complete smoke-test that
exercises all three crates on this exact target.

### WASM (`wasm32-unknown-unknown`)

```sh
rustup target add wasm32-unknown-unknown

# Core is always compatible
cargo build -p resonant-core --target wasm32-unknown-unknown

# FFT without the std rustfft backend
cargo build -p resonant-fft --target wasm32-unknown-unknown --no-default-features

# Filters compile cleanly
cargo build -p resonant-filters --target wasm32-unknown-unknown
```

The SIMD dispatch modules use `#[cfg(target_arch = "x86_64")]` and
`#[cfg(target_arch = "aarch64")]`, so the scalar fallback is automatically selected for
`wasm32` — no conditional compilation is needed in your own code.

---

## SIMD Acceleration

resonant uses `core::arch` intrinsics for hot-path acceleration, with a scalar fallback
that is always compiled and used on targets without SIMD support (including `wasm32`
and embedded Cortex-M).

### Dispatch pattern

Each accelerated function follows the same pattern:

```rust,ignore
pub(crate) fn multiply_buffers(a: &mut [f32], b: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        dispatch_multiply_buffers_x86(a, b);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        dispatch_multiply_buffers_neon(a, b);
        return;
    }
    scalar::multiply_buffers(a, b);
}
```

The `dispatch_*` functions are separate so that clippy's `needless_return` lint is
not triggered. The scalar implementation is always compiled — it serves as the reference
for correctness tests and is the only path on unsupported targets.

### Accelerated paths

| Path | Intrinsic set | Description |
|---|---|---|
| `window::apply(samples, window)` | SSE2 / NEON | Element-wise multiply of two f32 slices |
| `Fir::process_buf(block)` | SSE2 / NEON | FIR dot product over linearised delay line |
| `SignalFreqExt::magnitude()` | SSE2 / NEON | `sqrt(re² + im²)` over complex bins |
| `SignalFreqExt::magnitude_squared()` | SSE2 / NEON | `re² + im²`, no sqrt |
| Stereo→mono downmix in `resonant` facade | SSE2 / NEON | Deinterleave, add, scale |

### Safety

All intrinsic functions are `unsafe`. Each unsafe block carries a `// SAFETY:` comment
explaining the invariant. Runtime CPU feature detection is not used — each accelerated
function requires its target feature to be enabled at compile time via
`#[target_feature(enable = "sse2")]`. This is safe when the binary is compiled for a
specific target (e.g. `-C target-feature=+sse2`) or when the caller has verified support
via `is_x86_feature_detected!`.

### Benchmarking

Criterion benchmarks live in each crate's `benches/` directory:

```sh
cargo bench --bench window     # resonant-core: window application
cargo bench --bench fir        # resonant-filters: FIR filter
cargo bench --bench magnitude  # resonant-fft: magnitude computation
```
