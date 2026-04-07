//! Sample-rate conversion — integer decimation with anti-alias filtering.
//!
//! Decimation reduces the sample rate by an integer factor *M*: for every
//! *M* input samples, one output sample is produced. Before downsampling,
//! a lowpass anti-alias filter removes frequencies above the new Nyquist
//! limit to prevent aliasing.
//!
//! The implementation cascades two second-order Butterworth sections for a
//! fourth-order rolloff (~24 dB/octave), which provides adequate alias
//! rejection for most audio applications.
//!
//! # Examples
//!
//! ```
//! use resonant_filters::resample;
//!
//! // Decimate a 48 kHz signal down to 16 kHz (factor 3)
//! let input: Vec<f32> = (0..480).map(|i| (i as f32 * 0.1).sin()).collect();
//! let output = resample::decimate(&input, 3, 48000.0).unwrap();
//! assert_eq!(output.len(), 160);
//! ```

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::design::butterworth_lowpass;
use crate::Biquad;

/// Decimates `input` by keeping every `factor`-th sample after anti-alias
/// lowpass filtering.
///
/// The anti-alias filter is a cascade of two second-order Butterworth
/// lowpass sections with cutoff at `sample_rate / (2 * factor)` — i.e. the
/// Nyquist frequency of the decimated output.
///
/// # Arguments
///
/// * `input` — input samples at the original sample rate
/// * `factor` — decimation factor (must be ≥ 2)
/// * `sample_rate` — original sample rate in Hz
///
/// # Errors
///
/// Returns `None` if:
/// - `factor` is less than 2
/// - `input` is empty
/// - `sample_rate` is zero or negative
/// - the anti-alias cutoff falls outside valid Butterworth design range
///
/// # Examples
///
/// ```
/// use resonant_filters::resample;
///
/// let dc: Vec<f32> = vec![1.0; 1000];
/// let out = resample::decimate(&dc, 4, 44100.0).unwrap();
/// // DC should pass through with near-unity gain
/// assert!((out.last().copied().unwrap() - 1.0).abs() < 0.01);
/// ```
#[must_use]
pub fn decimate(input: &[f32], factor: usize, sample_rate: f64) -> Option<Vec<f32>> {
    if factor < 2 || input.is_empty() || sample_rate <= 0.0 {
        return None;
    }

    // Anti-alias cutoff: Nyquist of the output rate, with a small margin
    // to stay within the Butterworth design range.
    let cutoff = sample_rate / (2.0 * factor as f64) * 0.9;

    let coeffs = butterworth_lowpass(cutoff, sample_rate)?;

    // Cascade two second-order sections for 4th-order (~24 dB/oct) rolloff.
    let mut stage1 = Biquad::new(coeffs);
    let mut stage2 = Biquad::new(coeffs);

    let output_len = input.len().div_ceil(factor);
    let mut output = Vec::with_capacity(output_len);

    for (i, &x) in input.iter().enumerate() {
        let y = stage2.process_sample(stage1.process_sample(x));
        if i % factor == 0 {
            output.push(y);
        }
    }

    Some(output)
}

// ── PolyphaseResampler ────────────────────────────────────────────────────────

/// Default taps per polyphase sub-filter: good quality / speed trade-off.
const DEFAULT_TAPS_PER_PHASE: usize = 16;

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Design a windowed-sinc lowpass FIR with `total_taps` coefficients.
///
/// `cutoff` is normalised (0 = DC, 0.5 = Nyquist).  The Blackman window
/// gives ~74 dB stopband attenuation.  `gain` scales every coefficient by
/// a constant (use `up as f32` to compensate for polyphase interleaving).
fn design_lowpass_sinc(cutoff: f64, total_taps: usize, gain: f32) -> Vec<f32> {
    use core::f64::consts::PI;
    #[allow(unused_imports)]
    use num_traits::float::Float as _;

    let n = total_taps;
    let half = (n - 1) as f64 / 2.0;
    let mut h = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 - half;
        let sinc = if t.abs() < 1e-12 {
            2.0 * cutoff
        } else {
            (2.0 * PI * cutoff * t).sin() / (PI * t)
        };
        // Blackman window
        let w = 0.42 - 0.5 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()
            + 0.08 * (4.0 * PI * i as f64 / (n - 1) as f64).cos();
        h.push((sinc * w * gain as f64) as f32);
    }
    h
}

/// Polyphase resampler for rational sample-rate conversion.
///
/// Converts between two sample rates whose ratio is expressible as
/// `up / down` (integers).  The internal lowpass sinc filter is designed
/// automatically; no external coefficient tables are needed.
///
/// The conversion ratio is reduced by the GCD of `up` and `down` before
/// any computation, so `new(4, 2)` is equivalent to `new(2, 1)`.
///
/// # Examples
///
/// ```
/// use resonant_filters::resample::PolyphaseResampler;
///
/// // Convert 44100 Hz → 48000 Hz  (up=160, down=147)
/// let mut r = PolyphaseResampler::new(160, 147).unwrap();
/// let input: Vec<f32> = (0..441).map(|i| (i as f32 * 0.1).sin()).collect();
/// let output = r.process(&input);
/// // Expect approximately 441 * 160/147 ≈ 480 samples
/// assert!((output.len() as i32 - 480).abs() <= 2);
/// ```
#[derive(Clone)]
pub struct PolyphaseResampler {
    /// Upsample factor (after GCD reduction).
    up: usize,
    /// Downsample factor (after GCD reduction).
    down: usize,
    /// Number of taps per polyphase sub-filter.
    taps_per_phase: usize,
    /// Flat polyphase coefficients: sub-filter `p` occupies
    /// `coeffs[p * taps_per_phase .. (p + 1) * taps_per_phase]`.
    coeffs: Vec<f32>,
    /// Circular delay buffer of input samples.
    ring: Vec<f32>,
    /// Write position in `ring` (next slot to be written).
    ring_pos: usize,
    /// Current polyphase phase index, in `[0, up)`.
    phase: usize,
}

impl PolyphaseResampler {
    /// Creates a resampler with the default quality setting (16 taps per phase).
    ///
    /// Returns `None` if either factor is zero.
    #[must_use]
    pub fn new(up: usize, down: usize) -> Option<Self> {
        Self::with_quality(up, down, DEFAULT_TAPS_PER_PHASE)
    }

    /// Creates a resampler with a custom number of taps per polyphase sub-filter.
    ///
    /// Higher values give sharper alias rejection at the cost of more CPU and
    /// a longer startup transient.  Must be ≥ 4.
    ///
    /// Returns `None` if either factor is zero or `taps_per_phase` < 4.
    #[must_use]
    pub fn with_quality(up: usize, down: usize, taps_per_phase: usize) -> Option<Self> {
        if up == 0 || down == 0 || taps_per_phase < 4 {
            return None;
        }
        let g = gcd(up, down);
        let up = up / g;
        let down = down / g;

        // Anti-alias cutoff: min(1/up, 1/down) * 0.5 — just below both Nyquists.
        let cutoff = 0.5 / up.max(down) as f64;
        let total_taps = up * taps_per_phase;
        let h = design_lowpass_sinc(cutoff, total_taps, up as f32);

        // Decompose h into polyphase sub-filters.
        // Sub-filter p: h[p], h[p + up], h[p + 2*up], ...
        let mut coeffs = vec![0.0_f32; up * taps_per_phase];
        for p in 0..up {
            for k in 0..taps_per_phase {
                coeffs[p * taps_per_phase + k] = h[p + k * up];
            }
        }

        Some(Self {
            up,
            down,
            taps_per_phase,
            coeffs,
            ring: vec![0.0_f32; taps_per_phase],
            ring_pos: 0,
            phase: 0,
        })
    }

    /// Upsample factor (after GCD reduction).
    #[must_use]
    #[inline]
    pub fn up(&self) -> usize {
        self.up
    }

    /// Downsample factor (after GCD reduction).
    #[must_use]
    #[inline]
    pub fn down(&self) -> usize {
        self.down
    }

    /// Number of taps per polyphase sub-filter.
    #[must_use]
    #[inline]
    pub fn taps_per_phase(&self) -> usize {
        self.taps_per_phase
    }

    /// Resets the delay buffer and phase to zero.
    pub fn reset(&mut self) {
        for x in self.ring.iter_mut() {
            *x = 0.0;
        }
        self.ring_pos = 0;
        self.phase = 0;
    }

    /// Processes `input` samples and returns the resampled output.
    ///
    /// The output length will be approximately `input.len() * up / down`.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let capacity = input.len() * self.up / self.down + 2;
        let mut output = Vec::with_capacity(capacity);
        self.process_into(input, &mut output);
        output
    }

    /// Processes `input` samples and appends results to `output`.
    ///
    /// Avoids an intermediate allocation when the caller already has a buffer.
    pub fn process_into(&mut self, input: &[f32], output: &mut Vec<f32>) {
        let n = self.taps_per_phase;
        for &x in input {
            // Insert sample into circular ring buffer.
            self.ring[self.ring_pos] = x;
            self.ring_pos = (self.ring_pos + 1) % n;

            // Emit one output sample per polyphase phase that falls in [0, up).
            while self.phase < self.up {
                let base = self.phase * n;
                let sub = &self.coeffs[base..base + n];
                let mut y = 0.0_f32;
                for (k, &coeff) in sub.iter().enumerate() {
                    // x[n − k] lives at (ring_pos − 1 − k) mod n.
                    let idx = (self.ring_pos + n - 1 - k) % n;
                    y += coeff * self.ring[idx];
                }
                output.push(y);
                self.phase += self.down;
            }
            self.phase -= self.up;
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    const SR: f64 = 48000.0;

    #[test]
    fn returns_none_for_factor_below_2() {
        let input = vec![1.0; 100];
        assert!(decimate(&input, 0, SR).is_none());
        assert!(decimate(&input, 1, SR).is_none());
    }

    #[test]
    fn returns_none_for_empty_input() {
        assert!(decimate(&[], 2, SR).is_none());
    }

    #[test]
    fn returns_none_for_invalid_sample_rate() {
        let input = vec![1.0; 100];
        assert!(decimate(&input, 2, 0.0).is_none());
        assert!(decimate(&input, 2, -1.0).is_none());
    }

    #[test]
    fn output_length_is_correct() {
        let input = vec![0.0; 1000];
        let out = decimate(&input, 3, SR).unwrap();
        assert_eq!(out.len(), 334); // ceil(1000/3)
    }

    #[test]
    fn output_length_exact_multiple() {
        let input = vec![0.0; 1000];
        let out = decimate(&input, 5, SR).unwrap();
        assert_eq!(out.len(), 200);
    }

    #[test]
    fn dc_passes_through() {
        let input = vec![1.0; 4000];
        let out = decimate(&input, 4, SR).unwrap();
        // After transient settles, DC should be near 1.0
        let last = out[out.len() - 1];
        assert!(
            (last - 1.0).abs() < 0.01,
            "DC output = {last}, expected ~1.0"
        );
    }

    #[test]
    fn high_frequency_is_attenuated() {
        // Generate a tone at 20 kHz, decimate by 4 (output rate 12 kHz).
        // 20 kHz is well above the 6 kHz output Nyquist — should be killed.
        let n = 8000;
        let freq = 20000.0_f32;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * core::f32::consts::PI * freq * i as f32 / SR as f32).sin())
            .collect();

        let out = decimate(&input, 4, SR).unwrap();

        // Check tail samples (skip transient)
        let tail = &out[out.len() / 2..];
        let peak: f32 = tail.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(peak < 0.05, "High-freq peak after decimation = {peak}");
    }

    #[test]
    fn low_frequency_preserved() {
        // 100 Hz tone, decimate by 2 (output rate 24 kHz). Should pass.
        let n = 8000;
        let freq = 100.0_f32;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * core::f32::consts::PI * freq * i as f32 / SR as f32).sin())
            .collect();

        let out = decimate(&input, 2, SR).unwrap();

        let tail = &out[out.len() / 2..];
        let peak: f32 = tail.iter().map(|x| x.abs()).fold(0.0, f32::max);
        assert!(
            peak > 0.85,
            "Low-freq peak after decimation = {peak}, expected > 0.85"
        );
    }

    #[test]
    fn different_factors_produce_different_lengths() {
        let input = vec![0.0; 1200];
        let out2 = decimate(&input, 2, SR).unwrap();
        let out3 = decimate(&input, 3, SR).unwrap();
        let out4 = decimate(&input, 4, SR).unwrap();
        assert_eq!(out2.len(), 600);
        assert_eq!(out3.len(), 400);
        assert_eq!(out4.len(), 300);
    }

    // ── PolyphaseResampler ────────────────────────────────────────────────────

    #[test]
    fn polyphase_invalid_factors_return_none() {
        assert!(PolyphaseResampler::new(0, 1).is_none());
        assert!(PolyphaseResampler::new(1, 0).is_none());
        assert!(PolyphaseResampler::with_quality(2, 1, 3).is_none()); // taps_per_phase < 4
    }

    #[test]
    fn polyphase_gcd_reduction() {
        // 4/2 reduces to 2/1
        let r = PolyphaseResampler::new(4, 2).unwrap();
        assert_eq!(r.up(), 2);
        assert_eq!(r.down(), 1);
    }

    #[test]
    fn polyphase_upsample_2x_output_length() {
        let mut r = PolyphaseResampler::new(2, 1).unwrap();
        let input = vec![0.0_f32; 100];
        let out = r.process(&input);
        // Upsample by 2: expect approximately 200 samples
        assert_eq!(out.len(), 200);
    }

    #[test]
    fn polyphase_downsample_2x_output_length() {
        let mut r = PolyphaseResampler::new(1, 2).unwrap();
        let input = vec![0.0_f32; 100];
        let out = r.process(&input);
        // Downsample by 2: expect approximately 50 samples
        assert_eq!(out.len(), 50);
    }

    #[test]
    fn polyphase_44100_to_48000_output_length() {
        // 44100 → 48000: up=160, down=147
        let mut r = PolyphaseResampler::new(160, 147).unwrap();
        let input: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = r.process(&input);
        // Expected: 44100 * 160 / 147 = 48000 samples (exact)
        let expected = 44100 * 160 / 147;
        assert!(
            (out.len() as i64 - expected as i64).abs() <= 2,
            "expected ≈{expected}, got {}",
            out.len()
        );
    }

    #[test]
    fn polyphase_dc_passes_through_upsample() {
        let mut r = PolyphaseResampler::new(2, 1).unwrap();
        // DC signal — after startup transient, output should settle near 1.0
        let input = vec![1.0_f32; 500];
        let out = r.process(&input);
        let tail = &out[out.len() - 100..];
        let mean: f32 = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!((mean - 1.0).abs() < 0.05, "DC mean after upsample = {mean}");
    }

    #[test]
    fn polyphase_dc_passes_through_downsample() {
        let mut r = PolyphaseResampler::new(1, 2).unwrap();
        let input = vec![1.0_f32; 500];
        let out = r.process(&input);
        let tail = &out[out.len() - 50..];
        let mean: f32 = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(
            (mean - 1.0).abs() < 0.05,
            "DC mean after downsample = {mean}"
        );
    }

    #[test]
    fn polyphase_high_frequency_aliased_out_on_downsample() {
        // Downsample by 4: signal at 80% of original Nyquist (40% of output Nyquist)
        // should be suppressed by the anti-alias filter.
        let mut r = PolyphaseResampler::new(1, 4).unwrap();
        let input: Vec<f32> = (0..4000)
            .map(|i| (core::f32::consts::PI * 0.8 * i as f32).sin())
            .collect();
        let out = r.process(&input);
        let tail = &out[out.len() / 2..];
        let peak: f32 = tail.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        assert!(peak < 0.1, "alias should be suppressed, peak = {peak}");
    }

    #[test]
    fn polyphase_low_frequency_preserved_on_downsample() {
        // 5% of Nyquist — well within passband
        let mut r = PolyphaseResampler::new(1, 2).unwrap();
        let input: Vec<f32> = (0..2000)
            .map(|i| (core::f32::consts::PI * 0.05 * i as f32).sin())
            .collect();
        let out = r.process(&input);
        let tail = &out[out.len() / 2..];
        let peak: f32 = tail.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        assert!(peak > 0.7, "low-freq should be preserved, peak = {peak}");
    }

    #[test]
    fn polyphase_reset_is_idempotent() {
        let mut r = PolyphaseResampler::new(2, 1).unwrap();
        let kick = vec![1.0_f32; 100];
        r.process(&kick);
        r.reset();

        let mut r2 = PolyphaseResampler::new(2, 1).unwrap();
        let signal: Vec<f32> = (0..200).map(|i| (i as f32 * 0.1).sin()).collect();
        let out1 = r.process(&signal);
        let out2 = r2.process(&signal);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6, "reset mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn polyphase_process_into_matches_process() {
        let mut r1 = PolyphaseResampler::new(3, 2).unwrap();
        let mut r2 = PolyphaseResampler::new(3, 2).unwrap();
        let input: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let out1 = r1.process(&input);
        let mut out2 = Vec::new();
        r2.process_into(&input, &mut out2);
        assert_eq!(out1, out2);
    }
}
