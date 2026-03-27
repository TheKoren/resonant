extern crate alloc;

use alloc::vec::Vec;

use crate::chunk::Chunk;
use crate::error::StreamError;
use crate::node::DspNode;

/// Downmixes multi-channel audio to mono by averaging across channels.
///
/// For mono input, this is a no-op. For stereo, each frame `[L, R]` becomes
/// `(L + R) / 2`. The output chunk always has `channels = 1`.
///
/// # Examples
///
/// ```
/// use resonant_stream::{Chunk, DspNode};
/// use resonant_stream::nodes::MixNode;
///
/// let mut mix = MixNode::new();
/// let stereo = Chunk::new(vec![1.0, 0.0, 0.0, 1.0], 44100, 2);
/// let mono = mix.process(stereo).unwrap();
/// assert_eq!(mono.channels(), 1);
/// assert_eq!(mono.data(), &[0.5, 0.5]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MixNode;

impl MixNode {
    /// Creates a new downmix node.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DspNode for MixNode {
    fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> {
        let channels = input.channels() as usize;

        // Already mono — pass through without allocation
        if channels <= 1 {
            return Ok(input);
        }

        let sample_rate = input.sample_rate();
        let data = input.into_data();
        let frames = data.len() / channels;
        let scale = 1.0 / channels as f32;

        let mut mono = Vec::with_capacity(frames);
        for frame in data.chunks_exact(channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum * scale);
        }

        Ok(Chunk::new(mono, sample_rate, 1))
    }

    fn reset(&mut self) {
        // Stateless — nothing to reset.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mono_passthrough() {
        let mut node = MixNode::new();
        let chunk = Chunk::new(vec![0.5, -0.3, 1.0], 44100, 1);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.channels()), Some(1));
        assert_eq!(out.map(|c| c.data()), Some([0.5, -0.3, 1.0].as_slice()));
    }

    #[test]
    fn stereo_to_mono() {
        let mut node = MixNode::new();
        // L=1.0, R=0.0 | L=0.0, R=1.0
        let chunk = Chunk::new(vec![1.0, 0.0, 0.0, 1.0], 44100, 2);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.channels()), Some(1));
        assert_eq!(out.map(|c| c.frames()), Some(2));
        assert_eq!(out.map(|c| c.data()), Some([0.5, 0.5].as_slice()));
    }

    #[test]
    fn quad_to_mono() {
        let mut node = MixNode::new();
        // 4 channels: [1, 2, 3, 4] -> avg = 2.5
        let chunk = Chunk::new(vec![1.0, 2.0, 3.0, 4.0], 44100, 4);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.channels()), Some(1));
        assert_eq!(out.map(|c| c.frames()), Some(1));
        let data = out.map(|c| c.data()[0]);
        assert!(data.is_some_and(|v| (v - 2.5).abs() < 1e-6));
    }

    #[test]
    fn empty_chunk() {
        let mut node = MixNode::new();
        let chunk = Chunk::empty(44100, 2);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.is_empty()), Some(true));
        assert_eq!(out.map(|c| c.channels()), Some(1));
    }

    #[test]
    fn preserves_sample_rate() {
        let mut node = MixNode::new();
        let chunk = Chunk::new(vec![1.0, 0.0], 96000, 2);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.sample_rate()), Some(96000));
    }

    #[test]
    fn stereo_silence() {
        let mut node = MixNode::new();
        let chunk = Chunk::new(vec![0.0, 0.0, 0.0, 0.0], 44100, 2);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.data()), Some([0.0, 0.0].as_slice()));
    }

    #[test]
    fn multiple_frames_stereo() {
        let mut node = MixNode::new();
        // 3 frames of stereo: [0.2, 0.8], [1.0, 0.0], [-0.5, 0.5]
        let chunk = Chunk::new(vec![0.2, 0.8, 1.0, 0.0, -0.5, 0.5], 44100, 2);
        let out = node.process(chunk).ok();
        let data = out.as_ref().map(|c| c.data().to_vec());
        if let Some(d) = data {
            assert_eq!(d.len(), 3);
            assert!((d[0] - 0.5).abs() < 1e-6);
            assert!((d[1] - 0.5).abs() < 1e-6);
            assert!((d[2] - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn default_trait() {
        let node = MixNode::default();
        let _ = node; // just ensure it compiles
    }
}
