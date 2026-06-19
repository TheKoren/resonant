extern crate alloc;
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

    let coeffs = butterworth_lowpass(cutoff, sample_rate).ok()?;

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
}
