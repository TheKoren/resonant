//! Transparent oversampling wrapper for nonlinear processors.
//!
//! Oversampling reduces aliasing in nonlinear signal processing by running the
//! processor at a higher internal sample rate.  The input is upsampled by `N`,
//! the processor runs on every oversampled sample, and the output is decimated
//! back to the original rate.  The polyphase anti-alias filter handles both the
//! interpolation and decimation.
//!
//! `N = 1` is a pure passthrough: no resampling occurs, and the processor runs
//! at the nominal sample rate.

extern crate alloc;
use alloc::vec::Vec;

use crate::resample::PolyphaseResampler;

/// Transparent `N`× oversampling wrapper.
///
/// Upsamples input by `N`, applies the caller-supplied processor to each
/// oversampled sample, then decimates back to the original rate.
///
/// # Type parameters
///
/// * `N` — oversampling factor.  Must be ≥ 1.
///   Common values: 2, 4, 8.  `N = 1` is a zero-overhead passthrough.
///
/// # Examples
///
/// ```
/// use resonant_filters::{design, nonlinear::SaturatingBiquad, oversample::Oversample};
///
/// let coeffs = design::butterworth_lowpass(5000.0, 44100.0).unwrap();
/// let mut drive = SaturatingBiquad::new(coeffs, 0.9);
/// let mut os = Oversample::<4>::new(44100.0);
///
/// let input: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
/// let output = os.process(&input, |x| drive.process_sample(x));
/// assert_eq!(output.len(), input.len());
/// ```
pub struct Oversample<const N: usize> {
    upsampler: PolyphaseResampler,
    downsampler: PolyphaseResampler,
}

impl<const N: usize> Oversample<N> {
    /// Creates a new oversampler at the given nominal sample rate.
    ///
    /// The `sample_rate` argument is informational; the polyphase resamplers
    /// operate on integer ratios and do not require it for coefficient design.
    ///
    /// # Panics
    ///
    /// Panics if `N == 0`.
    #[must_use]
    pub fn new(_sample_rate: f32) -> Self {
        assert!(N >= 1, "Oversample factor N must be >= 1");
        if N == 1 {
            // Dummy 1:1 resamplers — process() bypasses them for N=1.
            // SAFETY: new(1, 1) always returns Some; only None for factor=0.
            let r = PolyphaseResampler::new(1, 1).unwrap_or_else(|| unreachable!());
            return Self {
                upsampler: r.clone(),
                downsampler: r,
            };
        }
        // SAFETY: N >= 2 guaranteed by the assert above; both ratios are valid.
        let upsampler = PolyphaseResampler::new(N, 1).unwrap_or_else(|| unreachable!());
        let downsampler = PolyphaseResampler::new(1, N).unwrap_or_else(|| unreachable!());
        Self {
            upsampler,
            downsampler,
        }
    }

    /// Resets internal filter state.
    ///
    /// Call this between unrelated audio segments to avoid transient artefacts.
    pub fn reset(&mut self) {
        self.upsampler.reset();
        self.downsampler.reset();
    }

    /// Processes `input` through the oversampled processor.
    ///
    /// Each sample is upsampled by `N`, passed to `f`, then decimated back.
    /// The output length equals the input length.
    ///
    /// For `N = 1` the input is passed directly to `f` with no resampling.
    pub fn process(&mut self, input: &[f32], mut f: impl FnMut(f32) -> f32) -> Vec<f32> {
        if N == 1 {
            return input.iter().map(|&x| f(x)).collect();
        }

        // Upsample
        let upsampled = self.upsampler.process(input);

        // Apply processor at the oversampled rate
        let processed: Vec<f32> = upsampled.iter().map(|&x| f(x)).collect();

        // Decimate back to nominal rate
        self.downsampler.process(&processed)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn oversample_1x_is_passthrough() {
        let mut os = Oversample::<1>::new(44100.0);
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let output = os.process(&input, |x| x);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6, "1x passthrough mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn oversample_output_length_matches_input_2x() {
        let mut os = Oversample::<2>::new(44100.0);
        let input = vec![0.0_f32; 128];
        let output = os.process(&input, |x| x);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn oversample_output_length_matches_input_4x() {
        let mut os = Oversample::<4>::new(44100.0);
        let input = vec![0.0_f32; 256];
        let output = os.process(&input, |x| x);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn oversample_output_length_matches_input_8x() {
        let mut os = Oversample::<8>::new(44100.0);
        let input = vec![0.0_f32; 512];
        let output = os.process(&input, |x| x);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn oversample_dc_preserved_2x() {
        // DC signal through identity processor — output should settle near 1.0
        let mut os = Oversample::<2>::new(44100.0);
        let input = vec![1.0_f32; 512];
        let output = os.process(&input, |x| x);
        let tail = &output[output.len() - 64..];
        let mean: f32 = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!((mean - 1.0).abs() < 0.05, "DC not preserved: mean = {mean}");
    }

    #[test]
    fn oversample_dc_preserved_4x() {
        let mut os = Oversample::<4>::new(44100.0);
        let input = vec![1.0_f32; 512];
        let output = os.process(&input, |x| x);
        let tail = &output[output.len() - 64..];
        let mean: f32 = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!(
            (mean - 1.0).abs() < 0.05,
            "DC not preserved at 4x: mean = {mean}"
        );
    }

    #[test]
    fn oversample_low_frequency_preserved() {
        // Low-frequency tone through identity — should pass through near-unity
        let mut os = Oversample::<4>::new(44100.0);
        let input: Vec<f32> = (0..1024)
            .map(|i| (core::f32::consts::PI * 0.02 * i as f32).sin())
            .collect();
        let output = os.process(&input, |x| x);
        let tail_in: &[f32] = &input[input.len() - 128..];
        let tail_out: &[f32] = &output[output.len() - 128..];
        let peak_in: f32 = tail_in.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let peak_out: f32 = tail_out.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        // Output peak should be within 20% of input peak
        assert!(
            (peak_out - peak_in).abs() < 0.2,
            "low-freq not preserved: in={peak_in:.3} out={peak_out:.3}"
        );
    }

    #[test]
    fn oversample_reset_gives_same_output_as_fresh() {
        let mut os = Oversample::<2>::new(44100.0);
        let mut os2 = Oversample::<2>::new(44100.0);

        // Dirty up os with some signal
        let noise: Vec<f32> = (0..256).map(|i| (i as f32 * 0.3).sin()).collect();
        os.process(&noise, |x| x);
        os.reset();

        // Both should now produce the same output
        let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let out1 = os.process(&signal, |x| x);
        let out2 = os2.process(&signal, |x| x);
        for (a, b) in out1.iter().zip(out2.iter()) {
            assert!((a - b).abs() < 1e-6, "reset mismatch: {a} vs {b}");
        }
    }

    #[test]
    #[should_panic(expected = "N must be >= 1")]
    fn oversample_zero_panics() {
        let _ = Oversample::<0>::new(44100.0);
    }
}
