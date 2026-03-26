use resonant_filters::Biquad;

use crate::chunk::Chunk;
use crate::error::StreamError;
use crate::node::DspNode;

/// Applies a biquad filter to each sample in the chunk.
///
/// Wraps a [`Biquad`] from `resonant-filters`, processing each channel's
/// samples through the same filter. For independent per-channel filtering,
/// use one `FilterNode` per channel with separate state.
///
/// # Examples
///
/// ```
/// use resonant_stream::{Chunk, DspNode};
/// use resonant_stream::nodes::FilterNode;
/// use resonant_filters::{Biquad, BiquadCoeffs};
///
/// // Unity pass-through filter
/// let coeffs = BiquadCoeffs { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };
/// let mut node = FilterNode::new(Biquad::new(coeffs));
///
/// let chunk = Chunk::new(vec![1.0, 0.5, -0.5], 44100, 1);
/// let out = node.process(chunk).unwrap();
/// // Unity filter preserves samples exactly
/// assert!((out.data()[0] - 1.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct FilterNode {
    filter: Biquad,
}

impl FilterNode {
    /// Creates a filter node wrapping the given biquad.
    #[must_use]
    pub fn new(filter: Biquad) -> Self {
        Self { filter }
    }

    /// Returns a reference to the inner biquad filter.
    #[inline]
    #[must_use]
    pub fn filter(&self) -> &Biquad {
        &self.filter
    }

    /// Returns a mutable reference to the inner biquad filter.
    #[inline]
    pub fn filter_mut(&mut self) -> &mut Biquad {
        &mut self.filter
    }
}

impl DspNode for FilterNode {
    fn process(&mut self, mut input: Chunk) -> Result<Chunk, StreamError> {
        self.filter.process_buf(input.data_mut());
        Ok(input)
    }

    fn reset(&mut self) {
        self.filter.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resonant_filters::BiquadCoeffs;

    fn unity_coeffs() -> BiquadCoeffs {
        BiquadCoeffs {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        }
    }

    #[test]
    fn unity_filter_passthrough() {
        let mut node = FilterNode::new(Biquad::new(unity_coeffs()));
        let chunk = Chunk::new(vec![1.0, 0.5, -0.3, 0.0], 44100, 1);
        let out = node.process(chunk).ok();
        let data = out.as_ref().map(|c| c.data());
        let data = data.as_ref().map(|d| *d);
        assert!(data.is_some());
        for (i, &s) in data.into_iter().flatten().enumerate() {
            assert!(
                (s - [1.0, 0.5, -0.3, 0.0][i]).abs() < 1e-6,
                "sample {i} mismatch"
            );
        }
    }

    #[test]
    fn zero_coeffs_silences() {
        let coeffs = BiquadCoeffs {
            b0: 0.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        let mut node = FilterNode::new(Biquad::new(coeffs));
        let chunk = Chunk::new(vec![1.0, 2.0, 3.0], 44100, 1);
        let out = node.process(chunk).ok();
        let data = out.as_ref().map(|c| c.data());
        assert_eq!(data, Some([0.0, 0.0, 0.0].as_slice()));
    }

    #[test]
    fn reset_clears_state() {
        let mut node = FilterNode::new(Biquad::new(unity_coeffs()));
        // Process something to build up state
        let chunk = Chunk::new(vec![1.0, 0.5], 44100, 1);
        let _ = node.process(chunk);
        node.reset();
        let state = node.filter().state();
        assert_eq!(state.s1, 0.0);
        assert_eq!(state.s2, 0.0);
    }

    #[test]
    fn empty_chunk() {
        let mut node = FilterNode::new(Biquad::new(unity_coeffs()));
        let chunk = Chunk::empty(44100, 1);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.is_empty()), Some(true));
    }

    #[test]
    fn filter_accessor() {
        let node = FilterNode::new(Biquad::new(unity_coeffs()));
        assert_eq!(node.filter().coeffs().b0, 1.0);
    }

    #[test]
    fn filter_mut_accessor() {
        let mut node = FilterNode::new(Biquad::new(unity_coeffs()));
        let new_coeffs = BiquadCoeffs {
            b0: 0.5,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
        };
        node.filter_mut().set_coeffs(new_coeffs);
        assert_eq!(node.filter().coeffs().b0, 0.5);
    }

    #[test]
    fn preserves_metadata() {
        let mut node = FilterNode::new(Biquad::new(unity_coeffs()));
        let chunk = Chunk::new(vec![1.0, 2.0], 48000, 2);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.sample_rate()), Some(48000));
        assert_eq!(out.map(|c| c.channels()), Some(2));
    }
}
