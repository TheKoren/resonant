# Benchmarks

Performance measurements for the `resonant` workspace.

Run locally with:

```bash
cargo bench --workspace --exclude resonant-examples
```

Results land in `target/criterion/` as interactive HTML reports.
CI only verifies that bench binaries compile (`cargo bench --no-run`); actual numbers are local-only to avoid shared-runner noise.

---

## Methodology

- Hardware: results will vary by machine; record your own with the command above.
- All benchmarks use [Criterion.rs](https://bheisler.github.io/criterion.rs/book/) with the default statistical model (bootstrap, 95% CI).
- Throughput figures are in **samples/sec** unless stated otherwise.
- Where two implementations are compared the faster one is the baseline.

---

## FFT (`resonant-fft`)

### `fft_comparison` — rfft vs complex FFT

Real FFT (`rfft`) exploits conjugate symmetry to process N real samples by
running an N/2-point complex FFT and a O(N) post-processing step.
Expected speedup: ~1.5–2× over the full complex FFT, plus half the output memory.

| Size | rfft | complex_fft | Speedup |
|------|------|-------------|---------|
| 64   | —    | —           | run locally |
| 256  | —    | —           | run locally |
| 1024 | —    | —           | run locally |
| 4096 | —    | —           | run locally |
| 16384| —    | —           | run locally |
| 65536| —    | —           | run locally |

### `rfft` — real FFT vs complex FFT (existing bench)

Covers sizes 1024 / 4096 / 16384 with throughput measurement.

### `magnitude` — SIMD magnitude over complex slices

`sqrt(re² + im²)` vectorised with SSE2/NEON over N/2+1 rfft output bins.

---

## Filters (`resonant-filters`)

### `nonlinear` — linear vs nonlinear filter CPU cost

Block size: 4096 samples.

| Filter | drive/resonance/Q | Expected cost vs linear biquad |
|--------|-------------------|-------------------------------|
| `Biquad` (linear) | — | baseline |
| `SaturatingBiquad` | drive=0.0 | ~1× (tanh(x)≈x at zero drive) |
| `SaturatingBiquad` | drive=1.0 | ~2–3× (full tanh per sample) |
| `MoogLadder` | resonance=0.0 | ~4× (4 cascaded stages) |
| `MoogLadder` | resonance=3.5 | ~4× |
| `StateVariableFilter` | Q=1.0 | ~1.5× (2 integrators, 4 outputs) |

Run `cargo bench --bench nonlinear -p resonant-filters` for actual numbers.

### `resample` — polyphase vs integer decimation

Input: 44100 samples (1 second at 44.1 kHz).

| Operation | Implementation | Notes |
|-----------|---------------|-------|
| 2× downsample | `decimate` | Single biquad cascade |
| 2× downsample | `PolyphaseResampler` | 16 taps/phase, Blackman window |
| 2× upsample | `PolyphaseResampler` | — |
| 44100 → 48000 | `PolyphaseResampler` (160:147) | Real-world CD→DAC |
| 44100 → 16000 | `PolyphaseResampler` (160:441) | Speech downsampling |

Run `cargo bench --bench resample -p resonant-filters` for actual numbers.

### `fir` — FIR dot product (existing bench)

32 / 128 / 512 taps on 4096 samples.

---

## Core (`resonant-core`)

### `window` — window function application (existing bench)

`hann`, `hamming`, `blackman` on 1024 / 4096 / 16384 samples.
SIMD `multiply_buffers` (SSE2/NEON) vs scalar for the `apply` step.

---

## Facade (`resonant`)

### `facade` — end-to-end AudioFile → FFT

Input: 10 seconds of synthetic mono audio at 44100 Hz (~441 000 samples).
Covers: `AudioFile::from_samples` construction, mono normalisation, FFT dispatch.

| Benchmark | Window size | Notes |
|-----------|-------------|-------|
| `audio_file_fft_default` | full buffer | Single FFT over entire signal |
| `audio_file_fft_4096` | 4096 | Single FFT with explicit window |
| `consume_all_frames_1024` | 1024 | Iterator over all STFT frames |

Run `cargo bench --bench facade -p resonant` for actual numbers.
