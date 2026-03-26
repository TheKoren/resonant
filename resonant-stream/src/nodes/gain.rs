use crate::chunk::Chunk;
use crate::error::StreamError;
use crate::node::DspNode;

/// Applies a constant gain (volume scaling) to all samples.
///
/// A gain of `1.0` is unity (no change), `0.5` halves the amplitude,
/// and `2.0` doubles it. Negative gains invert the phase.
///
/// # Examples
///
/// ```
/// use resonant_stream::{Chunk, DspNode};
/// use resonant_stream::nodes::GainNode;
///
/// let mut gain = GainNode::new(0.5);
/// let chunk = Chunk::new(vec![1.0, -1.0, 0.4], 44100, 1);
/// let out = gain.process(chunk).unwrap();
/// assert_eq!(out.data(), &[0.5, -0.5, 0.2]);
/// ```
#[derive(Debug, Clone)]
pub struct GainNode {
    gain: f32,
}

impl GainNode {
    /// Creates a gain node with the given linear multiplier.
    #[must_use]
    pub fn new(gain: f32) -> Self {
        Self { gain }
    }

    /// Returns the current gain value.
    #[inline]
    #[must_use]
    pub fn gain(&self) -> f32 {
        self.gain
    }

    /// Sets the gain value.
    #[inline]
    pub fn set_gain(&mut self, gain: f32) {
        self.gain = gain;
    }
}

impl DspNode for GainNode {
    fn process(&mut self, mut input: Chunk) -> Result<Chunk, StreamError> {
        for s in input.data_mut() {
            *s *= self.gain;
        }
        Ok(input)
    }

    fn reset(&mut self) {
        // Stateless — nothing to reset.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unity_gain() {
        let mut node = GainNode::new(1.0);
        let chunk = Chunk::new(vec![0.5, -0.3, 1.0], 44100, 1);
        let out = node.process(chunk).ok();
        assert_eq!(
            out.as_ref().map(|c| c.data()),
            Some([0.5, -0.3, 1.0].as_slice())
        );
    }

    #[test]
    fn half_gain() {
        let mut node = GainNode::new(0.5);
        let chunk = Chunk::new(vec![2.0, -4.0], 44100, 1);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.data()), Some([1.0, -2.0].as_slice()));
    }

    #[test]
    fn zero_gain_silences() {
        let mut node = GainNode::new(0.0);
        let chunk = Chunk::new(vec![1.0, -1.0], 44100, 1);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.data()), Some([0.0, 0.0].as_slice()));
    }

    #[test]
    fn negative_gain_inverts() {
        let mut node = GainNode::new(-1.0);
        let chunk = Chunk::new(vec![0.5, -0.5], 44100, 1);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.data()), Some([-0.5, 0.5].as_slice()));
    }

    #[test]
    fn empty_chunk() {
        let mut node = GainNode::new(2.0);
        let chunk = Chunk::empty(44100, 1);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.is_empty()), Some(true));
    }

    #[test]
    fn stereo_data() {
        let mut node = GainNode::new(0.5);
        let chunk = Chunk::new(vec![1.0, 2.0, 3.0, 4.0], 48000, 2);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.data()), Some([0.5, 1.0, 1.5, 2.0].as_slice()));
        assert_eq!(out.map(|c| c.channels()), Some(2));
    }

    #[test]
    fn set_gain_changes_value() {
        let mut node = GainNode::new(1.0);
        assert_eq!(node.gain(), 1.0);
        node.set_gain(0.25);
        assert_eq!(node.gain(), 0.25);
    }

    #[test]
    fn preserves_metadata() {
        let mut node = GainNode::new(1.0);
        let chunk = Chunk::new(vec![1.0], 96000, 4);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.sample_rate()), Some(96000));
        assert_eq!(out.map(|c| c.channels()), Some(4));
    }
}
