extern crate alloc;

use alloc::string::ToString;
use alloc::vec::Vec;

use resonant_fft::radix2;
use resonant_fft::Complex;

use crate::chunk::Chunk;
use crate::error::StreamError;
use crate::node::DspNode;

/// What the FFT node outputs per frequency bin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftOutput {
    /// Linear magnitude: `sqrt(re² + im²)`.
    Magnitude,
    /// Power spectrum: `re² + im²`.
    Power,
}

/// Computes a forward FFT on each chunk, outputting frequency-domain magnitudes.
///
/// Input chunks must have a power-of-two number of samples. The output chunk
/// has the same length, with each sample replaced by its frequency bin value
/// according to the configured [`FftOutput`] mode.
///
/// Uses the `no_alloc` radix-2 backend from `resonant-fft`.
///
/// # Examples
///
/// ```
/// use resonant_stream::{Chunk, DspNode};
/// use resonant_stream::nodes::FftNode;
///
/// let mut fft = FftNode::new();
/// // DC signal: all energy in bin 0
/// let chunk = Chunk::new(vec![1.0; 4], 44100, 1);
/// let out = fft.process(chunk).unwrap();
/// assert!(out.data()[0] > 0.0); // bin 0 has energy
/// ```
#[derive(Debug, Clone)]
pub struct FftNode {
    output: FftOutput,
}

impl FftNode {
    /// Creates an FFT node with default magnitude output.
    #[must_use]
    pub fn new() -> Self {
        Self {
            output: FftOutput::Magnitude,
        }
    }

    /// Creates an FFT node with the specified output mode.
    #[must_use]
    pub fn with_output(output: FftOutput) -> Self {
        Self { output }
    }

    /// Returns the current output mode.
    #[inline]
    #[must_use]
    pub fn output(&self) -> FftOutput {
        self.output
    }
}

impl Default for FftNode {
    fn default() -> Self {
        Self::new()
    }
}

impl DspNode for FftNode {
    fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> {
        let sample_rate = input.sample_rate();
        let channels = input.channels();
        let samples = input.into_data();
        let n = samples.len();

        if n == 0 {
            return Ok(Chunk::empty(sample_rate, channels));
        }

        // Convert real samples to complex
        let mut buf: Vec<Complex<f32>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();

        radix2::fft(&mut buf).map_err(|e| StreamError::ProcessingError(e.to_string()))?;

        // Extract the requested representation
        let out: Vec<f32> = match self.output {
            FftOutput::Magnitude => buf.iter().map(|c: &Complex<f32>| c.norm()).collect(),
            FftOutput::Power => buf.iter().map(|c: &Complex<f32>| c.norm_sqr()).collect(),
        };

        Ok(Chunk::new(out, sample_rate, channels))
    }

    fn reset(&mut self) {
        // Stateless — nothing to reset.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_signal_energy_in_bin_zero() {
        let mut node = FftNode::new();
        let chunk = Chunk::new(vec![1.0; 8], 44100, 1);
        let out = node.process(chunk).ok();
        let data = out.as_ref().map(|c| c.data());
        // Bin 0 (DC) should have all the energy
        assert!(data.is_some());
        let data = data.map(|d| d.to_vec());
        let data = data.as_deref();
        let dc = data.map(|d| d[0]);
        assert!(dc.is_some_and(|v| v > 0.0));
        // All other bins should be ~0 for a DC signal
        if let Some(d) = data {
            for &val in &d[1..] {
                assert!(val.abs() < 1e-4, "non-DC bin has energy: {val}");
            }
        }
    }

    #[test]
    fn power_output_mode() {
        let mut node = FftNode::with_output(FftOutput::Power);
        let chunk = Chunk::new(vec![1.0; 4], 44100, 1);
        let out = node.process(chunk).ok();
        let data = out.as_ref().map(|c| c.data());
        // Power of DC bin = magnitude²
        assert!(data.is_some());
        let dc_power = data.map(|d| d[0]);
        // For 4 samples of 1.0, FFT DC bin = 4.0, power = 16.0
        assert!(dc_power.is_some_and(|v| (v - 16.0).abs() < 1e-4));
    }

    #[test]
    fn non_power_of_two_fails() {
        let mut node = FftNode::new();
        let chunk = Chunk::new(vec![1.0; 3], 44100, 1);
        let err = node.process(chunk).err();
        assert!(err.is_some());
        if let Some(StreamError::ProcessingError(msg)) = &err {
            assert!(msg.contains("not a power of two"));
        }
    }

    #[test]
    fn empty_chunk_passthrough() {
        let mut node = FftNode::new();
        let chunk = Chunk::empty(44100, 1);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.is_empty()), Some(true));
    }

    #[test]
    fn preserves_metadata() {
        let mut node = FftNode::new();
        let chunk = Chunk::new(vec![0.0; 8], 48000, 2);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.sample_rate()), Some(48000));
        assert_eq!(out.map(|c| c.channels()), Some(2));
    }

    #[test]
    fn output_length_matches_input() {
        let mut node = FftNode::new();
        let chunk = Chunk::new(vec![0.0; 16], 44100, 1);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.len()), Some(16));
    }

    #[test]
    fn default_is_magnitude() {
        let node = FftNode::default();
        assert_eq!(node.output(), FftOutput::Magnitude);
    }

    #[test]
    fn sine_wave_has_peak() {
        let mut node = FftNode::new();
        let n = 16;
        // Generate a single-cycle sine (bin 1)
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * core::f32::consts::PI * i as f32 / n as f32).sin())
            .collect();
        let chunk = Chunk::new(samples, 44100, 1);
        let out = node.process(chunk).ok();
        let data = out.as_ref().map(|c| c.data().to_vec());
        if let Some(d) = data {
            // Bin 1 should have the peak (and its mirror bin n-1)
            let peak_bin = d[1..n / 2]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
                .map(|(i, _)| i);
            assert_eq!(peak_bin, Some(0)); // bin 1 minus offset = index 0
        }
    }
}
