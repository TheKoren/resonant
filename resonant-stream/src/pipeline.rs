extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::chunk::Chunk;
use crate::error::StreamError;
use crate::node::DspNode;

/// A chain of [`DspNode`]s that processes audio sequentially.
///
/// `Pipeline` owns its nodes and feeds each chunk through them in order.
/// It validates sample rate and channel count on every call to
/// [`process`](Pipeline::process) if configured to do so.
///
/// # Building
///
/// ```
/// use resonant_stream::{Pipeline, Chunk};
///
/// let pipeline = Pipeline::builder()
///     .sample_rate(44100)
///     .channels(2)
///     .build();
///
/// assert_eq!(pipeline.len(), 0);
/// ```
pub struct Pipeline {
    nodes: Vec<Box<dyn DspNode>>,
    sample_rate: Option<u32>,
    channels: Option<u16>,
}

impl Pipeline {
    /// Returns a new [`PipelineBuilder`].
    #[must_use]
    pub fn builder() -> PipelineBuilder {
        PipelineBuilder {
            nodes: Vec::new(),
            sample_rate: None,
            channels: None,
        }
    }

    /// Feeds a chunk through every node in sequence, returning the final output.
    ///
    /// If `sample_rate` or `channels` were set at build time, the input chunk is
    /// validated before processing begins.
    ///
    /// # Errors
    ///
    /// Returns [`StreamError::SampleRateMismatch`] or [`StreamError::ChannelMismatch`]
    /// if the chunk does not match the configured format, or any error returned by
    /// a node's `process` method.
    ///
    /// # Examples
    ///
    /// ```
    /// use resonant_stream::{Pipeline, Chunk, DspNode, StreamError};
    ///
    /// struct Double;
    /// impl DspNode for Double {
    ///     fn process(&mut self, mut input: Chunk) -> Result<Chunk, StreamError> {
    ///         for s in input.data_mut() { *s *= 2.0; }
    ///         Ok(input)
    ///     }
    ///     fn reset(&mut self) {}
    /// }
    ///
    /// let mut pipeline = Pipeline::builder()
    ///     .node(Double)
    ///     .node(Double)
    ///     .build();
    ///
    /// let chunk = Chunk::new(vec![1.0, 0.5], 44100, 1);
    /// let out = pipeline.process(chunk).unwrap();
    /// assert_eq!(out.data(), &[4.0, 2.0]);
    /// ```
    pub fn process(&mut self, chunk: Chunk) -> Result<Chunk, StreamError> {
        if let Some(expected) = self.sample_rate {
            let got = chunk.sample_rate();
            if got != expected {
                return Err(StreamError::SampleRateMismatch { expected, got });
            }
        }
        if let Some(expected) = self.channels {
            let got = chunk.channels();
            if got != expected {
                return Err(StreamError::ChannelMismatch { expected, got });
            }
        }

        let mut current = chunk;
        for node in &mut self.nodes {
            current = node.process(current)?;
        }
        Ok(current)
    }

    /// Resets all nodes in the pipeline, clearing internal state.
    pub fn reset(&mut self) {
        for node in &mut self.nodes {
            node.reset();
        }
    }

    /// The number of nodes in the pipeline.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the pipeline has no nodes.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The expected sample rate, if configured.
    #[inline]
    #[must_use]
    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }

    /// The expected channel count, if configured.
    #[inline]
    #[must_use]
    pub fn channels(&self) -> Option<u16> {
        self.channels
    }

    /// Appends a node to the end of the pipeline.
    pub fn push(&mut self, node: impl DspNode + 'static) {
        self.nodes.push(Box::new(node));
    }
}

/// Builder for constructing a [`Pipeline`].
///
/// # Examples
///
/// ```
/// use resonant_stream::{Pipeline, Chunk, DspNode, StreamError};
///
/// struct Noop;
/// impl DspNode for Noop {
///     fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> { Ok(input) }
///     fn reset(&mut self) {}
/// }
///
/// let pipeline = Pipeline::builder()
///     .sample_rate(48000)
///     .channels(1)
///     .node(Noop)
///     .node(Noop)
///     .build();
///
/// assert_eq!(pipeline.len(), 2);
/// assert_eq!(pipeline.sample_rate(), Some(48000));
/// assert_eq!(pipeline.channels(), Some(1));
/// ```
pub struct PipelineBuilder {
    nodes: Vec<Box<dyn DspNode>>,
    sample_rate: Option<u32>,
    channels: Option<u16>,
}

impl PipelineBuilder {
    /// Sets the expected sample rate for format validation.
    #[must_use]
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Sets the expected channel count for format validation.
    #[must_use]
    pub fn channels(mut self, channels: u16) -> Self {
        self.channels = Some(channels);
        self
    }

    /// Appends a processing node to the pipeline.
    #[must_use]
    pub fn node(mut self, node: impl DspNode + 'static) -> Self {
        self.nodes.push(Box::new(node));
        self
    }

    /// Consumes the builder and returns the configured [`Pipeline`].
    #[must_use]
    pub fn build(self) -> Pipeline {
        Pipeline {
            nodes: self.nodes,
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scale(f32);
    impl DspNode for Scale {
        fn process(&mut self, mut input: Chunk) -> Result<Chunk, StreamError> {
            for s in input.data_mut() {
                *s *= self.0;
            }
            Ok(input)
        }
        fn reset(&mut self) {}
    }

    struct Fail;
    impl DspNode for Fail {
        fn process(&mut self, _: Chunk) -> Result<Chunk, StreamError> {
            Err(StreamError::ProcessingError("boom".into()))
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn empty_pipeline_passthrough() {
        let mut p = Pipeline::builder().build();
        let chunk = Chunk::new(vec![1.0, 2.0], 44100, 1);
        let out = p.process(chunk);
        assert!(out.is_ok());
        assert_eq!(out.ok().map(|c| c.into_data()), Some(vec![1.0, 2.0]));
    }

    #[test]
    fn single_node() {
        let mut p = Pipeline::builder().node(Scale(0.5)).build();
        let chunk = Chunk::new(vec![2.0, 4.0], 44100, 1);
        let out = p.process(chunk);
        assert_eq!(out.ok().map(|c| c.into_data()), Some(vec![1.0, 2.0]));
    }

    #[test]
    fn chained_nodes() {
        let mut p = Pipeline::builder()
            .node(Scale(2.0))
            .node(Scale(3.0))
            .build();
        let chunk = Chunk::new(vec![1.0], 44100, 1);
        let out = p.process(chunk);
        assert_eq!(out.ok().map(|c| c.into_data()), Some(vec![6.0]));
    }

    #[test]
    fn sample_rate_validation_pass() {
        let mut p = Pipeline::builder().sample_rate(44100).build();
        let chunk = Chunk::new(vec![1.0], 44100, 1);
        assert!(p.process(chunk).is_ok());
    }

    #[test]
    fn sample_rate_validation_fail() {
        let mut p = Pipeline::builder().sample_rate(44100).build();
        let chunk = Chunk::new(vec![1.0], 48000, 1);
        let err = p.process(chunk).err();
        assert_eq!(
            err,
            Some(StreamError::SampleRateMismatch {
                expected: 44100,
                got: 48000,
            })
        );
    }

    #[test]
    fn channel_validation_pass() {
        let mut p = Pipeline::builder().channels(2).build();
        let chunk = Chunk::new(vec![1.0, 2.0], 44100, 2);
        assert!(p.process(chunk).is_ok());
    }

    #[test]
    fn channel_validation_fail() {
        let mut p = Pipeline::builder().channels(2).build();
        let chunk = Chunk::new(vec![1.0], 44100, 1);
        let err = p.process(chunk).err();
        assert_eq!(
            err,
            Some(StreamError::ChannelMismatch {
                expected: 2,
                got: 1,
            })
        );
    }

    #[test]
    fn node_error_propagates() {
        let mut p = Pipeline::builder()
            .node(Scale(2.0))
            .node(Fail)
            .node(Scale(3.0))
            .build();
        let chunk = Chunk::new(vec![1.0], 44100, 1);
        let err = p.process(chunk).err();
        assert_eq!(err, Some(StreamError::ProcessingError("boom".into())));
    }

    #[test]
    fn reset_all_nodes() {
        let mut p = Pipeline::builder()
            .node(Scale(1.0))
            .node(Scale(2.0))
            .build();
        p.reset(); // should not panic
    }

    #[test]
    fn len_and_is_empty() {
        let p = Pipeline::builder().build();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);

        let p = Pipeline::builder().node(Scale(1.0)).build();
        assert!(!p.is_empty());
        assert_eq!(p.len(), 1);
    }

    #[test]
    fn push_after_build() {
        let mut p = Pipeline::builder().build();
        assert!(p.is_empty());
        p.push(Scale(2.0));
        assert_eq!(p.len(), 1);

        let chunk = Chunk::new(vec![3.0], 44100, 1);
        let out = p.process(chunk);
        assert_eq!(out.ok().map(|c| c.into_data()), Some(vec![6.0]));
    }

    #[test]
    fn accessors() {
        let p = Pipeline::builder().sample_rate(48000).channels(2).build();
        assert_eq!(p.sample_rate(), Some(48000));
        assert_eq!(p.channels(), Some(2));
    }

    #[test]
    fn no_validation_when_unconfigured() {
        let mut p = Pipeline::builder().build();
        let chunk = Chunk::new(vec![1.0], 96000, 6);
        assert!(p.process(chunk).is_ok());
    }

    #[test]
    fn both_validations_rate_fails_first() {
        let mut p = Pipeline::builder().sample_rate(44100).channels(2).build();
        let chunk = Chunk::new(vec![1.0], 48000, 1);
        // Sample rate is checked first
        let err = p.process(chunk).err();
        assert_eq!(
            err,
            Some(StreamError::SampleRateMismatch {
                expected: 44100,
                got: 48000,
            })
        );
    }
}
