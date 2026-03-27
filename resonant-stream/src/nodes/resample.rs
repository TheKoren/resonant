extern crate alloc;

use alloc::format;

use resonant_filters::resample;

use crate::chunk::Chunk;
use crate::error::StreamError;
use crate::node::DspNode;

/// Integer decimation node — reduces sample rate by an integer factor.
///
/// Wraps [`resonant_filters::resample::decimate`], which applies a
/// fourth-order Butterworth anti-alias filter before downsampling.
/// The output chunk has `ceil(input_len / factor)` samples and the
/// sample rate is divided by the factor.
///
/// # Examples
///
/// ```
/// use resonant_stream::{Chunk, DspNode};
/// use resonant_stream::nodes::ResampleNode;
///
/// let mut node = ResampleNode::new(2);
/// let chunk = Chunk::new(vec![1.0; 1000], 44100, 1);
/// let out = node.process(chunk).unwrap();
/// assert_eq!(out.sample_rate(), 22050);
/// assert_eq!(out.len(), 500);
/// ```
#[derive(Debug, Clone)]
pub struct ResampleNode {
    factor: usize,
}

impl ResampleNode {
    /// Creates a decimation node with the given integer factor.
    ///
    /// # Panics
    ///
    /// Panics if `factor` is less than 2.
    #[must_use]
    pub fn new(factor: usize) -> Self {
        assert!(factor >= 2, "decimation factor must be >= 2");
        Self { factor }
    }

    /// Returns the decimation factor.
    #[inline]
    #[must_use]
    pub fn factor(&self) -> usize {
        self.factor
    }
}

impl DspNode for ResampleNode {
    fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> {
        if input.is_empty() {
            return Ok(Chunk::empty(
                input.sample_rate() / self.factor as u32,
                input.channels(),
            ));
        }

        let sr = input.sample_rate();
        let channels = input.channels();
        let data = input.into_data();

        let out = resample::decimate(&data, self.factor, f64::from(sr)).ok_or_else(|| {
            StreamError::ProcessingError(format!(
                "decimation failed (factor={}, sr={sr}, len={})",
                self.factor,
                data.len()
            ))
        })?;

        let new_sr = sr / self.factor as u32;
        Ok(Chunk::new(out, new_sr, channels))
    }

    fn reset(&mut self) {
        // The underlying decimate() is stateless (creates fresh filters each call).
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn decimate_by_2() {
        let mut node = ResampleNode::new(2);
        let chunk = Chunk::new(vec![1.0; 1000], 44100, 1);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.sample_rate()), Some(22050));
        assert_eq!(out.map(|c| c.len()), Some(500));
    }

    #[test]
    fn decimate_by_3() {
        let mut node = ResampleNode::new(3);
        let chunk = Chunk::new(vec![0.0; 900], 48000, 1);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.sample_rate()), Some(16000));
        assert_eq!(out.map(|c| c.len()), Some(300));
    }

    #[test]
    fn dc_preserved() {
        let mut node = ResampleNode::new(2);
        let chunk = Chunk::new(vec![1.0; 4000], 44100, 1);
        let out = node.process(chunk).ok();
        if let Some(c) = out.as_ref() {
            let last = c.data()[c.len() - 1];
            assert!(
                (last - 1.0).abs() < 0.01,
                "DC should pass through, got {last}"
            );
        }
    }

    #[test]
    fn empty_chunk() {
        let mut node = ResampleNode::new(4);
        let chunk = Chunk::empty(48000, 1);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.is_empty()), Some(true));
        assert_eq!(out.map(|c| c.sample_rate()), Some(12000));
    }

    #[test]
    fn preserves_channel_count() {
        let mut node = ResampleNode::new(2);
        let chunk = Chunk::new(vec![0.0; 1000], 44100, 2);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.channels()), Some(2));
    }

    #[test]
    fn factor_accessor() {
        let node = ResampleNode::new(3);
        assert_eq!(node.factor(), 3);
    }

    #[test]
    #[should_panic(expected = "decimation factor must be >= 2")]
    fn factor_below_2_panics() {
        let _ = ResampleNode::new(1);
    }

    #[test]
    fn high_freq_attenuated() {
        let mut node = ResampleNode::new(4);
        let sr = 48000_u32;
        let n = 8000;
        let freq = 20000.0_f32;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * core::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();
        let chunk = Chunk::new(samples, sr, 1);
        let out = node.process(chunk).ok();
        if let Some(c) = out.as_ref() {
            let tail = &c.data()[c.len() / 2..];
            let peak: f32 = tail.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
            assert!(peak < 0.05, "High freq should be attenuated, peak={peak}");
        }
    }

    #[test]
    fn in_pipeline() {
        use crate::nodes::GainNode;
        use crate::Pipeline;

        let mut p = Pipeline::builder()
            .sample_rate(48000)
            .channels(1)
            .node(GainNode::new(1.0))
            .node(ResampleNode::new(2))
            .build();

        let chunk = Chunk::new(vec![1.0; 2000], 48000, 1);
        let out = p.process(chunk);
        assert!(out.is_ok());
        let out = out.ok();
        assert_eq!(out.as_ref().map(|c| c.sample_rate()), Some(24000));
        assert_eq!(out.as_ref().map(|c| c.len()), Some(1000));
    }
}
