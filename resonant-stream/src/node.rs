use crate::chunk::Chunk;
use crate::error::StreamError;

/// A processing node in a DSP pipeline.
///
/// Each node receives a [`Chunk`] of audio, transforms it, and returns the
/// result. Nodes may be stateful (e.g. filters with delay-line memory) and
/// should implement [`reset`](DspNode::reset) to clear that state.
///
/// # In-place processing
///
/// `process` takes ownership of the input `Chunk`. Nodes that operate in-place
/// (e.g. gain, biquad filter) can mutate the buffer directly and return it
/// without allocation. Nodes that change the buffer length (e.g. FFT,
/// resampler) allocate a new buffer internally.
///
/// # Examples
///
/// ```
/// use resonant_stream::{Chunk, DspNode, StreamError};
///
/// struct Invert;
///
/// impl DspNode for Invert {
///     fn process(&mut self, mut input: Chunk) -> Result<Chunk, StreamError> {
///         for s in input.data_mut() {
///             *s = -*s;
///         }
///         Ok(input)
///     }
///
///     fn reset(&mut self) {}
/// }
///
/// let mut node = Invert;
/// let chunk = Chunk::new(vec![1.0, -0.5], 44100, 1);
/// let out = node.process(chunk).unwrap();
/// assert_eq!(out.data(), &[-1.0, 0.5]);
/// ```
pub trait DspNode: Send {
    /// Process an input chunk, returning the transformed output.
    ///
    /// Implementations should return `Err` only for unrecoverable problems
    /// (format mismatches, invalid state). Transient conditions like an
    /// empty input should be handled gracefully (e.g. return the empty chunk).
    fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError>;

    /// Reset all internal state (e.g. filter delay lines, buffers).
    ///
    /// Called between songs, after a seek, or when the pipeline is restarted.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trivial passthrough node for testing the trait.
    struct Passthrough;

    impl DspNode for Passthrough {
        fn process(&mut self, input: Chunk) -> Result<Chunk, StreamError> {
            Ok(input)
        }
        fn reset(&mut self) {}
    }

    #[test]
    fn passthrough_preserves_data() {
        let mut node = Passthrough;
        let chunk = Chunk::new(vec![1.0, 2.0, 3.0], 44100, 1);
        let out = node.process(chunk).ok();
        assert_eq!(
            out.as_ref().map(|c| c.data()),
            Some([1.0, 2.0, 3.0].as_slice())
        );
    }

    #[test]
    fn passthrough_preserves_metadata() {
        let mut node = Passthrough;
        let chunk = Chunk::new(vec![0.0; 4], 48000, 2);
        let out = node.process(chunk).ok();
        let out = out.as_ref();
        assert_eq!(out.map(|c| c.sample_rate()), Some(48000));
        assert_eq!(out.map(|c| c.channels()), Some(2));
    }

    #[test]
    fn passthrough_handles_empty() {
        let mut node = Passthrough;
        let chunk = Chunk::empty(44100, 1);
        let out = node.process(chunk).ok();
        assert_eq!(out.as_ref().map(|c| c.is_empty()), Some(true));
    }

    /// Verify that DspNode is object-safe (can be boxed).
    #[test]
    fn trait_is_object_safe() {
        let node: Box<dyn DspNode> = Box::new(Passthrough);
        let _ = node;
    }

    /// Verify that DspNode: Send (required for potential async use).
    #[test]
    fn trait_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Passthrough>();
    }
}
